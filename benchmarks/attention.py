import math

import torch
import torch.nn.functional as F
import triton

from kernels import flash_attention_v1, flash_attention_v2

DEVICE = "cuda"

BATCH = 16
HEAD_DIM = 128
PAGE_SIZE = 16

KERNELS = {
    "v1": flash_attention_v1,
    "v2": flash_attention_v2,
}
DTYPES = [
    torch.float16,
    torch.bfloat16,
]

LAYOUTS = [
    "dense",
    "ragged",
    "paged",
]

ATTENTION_TYPES = [
    "self",
    "cross",
]

TOPOLOGIES = {
    "mha": (8, 8),
    "gqa": (8, 2),
    "mqa": (8, 1),
}

CAUSAL = [
    False,
    True,
]

Q_LENS = [
    128,
    512,
]

KV_LENS = [
    128,
    512,
    2048,
]

INCLUDE_NAIVE_LOOP_REFERENCE = True


def build_padded_mask(q_lens, kv_lens, max_q, max_kv, causal, device):
    B = len(q_lens)

    q_idx = torch.arange(max_q, device=device).view(1, max_q, 1)
    kv_idx = torch.arange(max_kv, device=device).view(1, 1, max_kv)

    q_len_t = torch.tensor(q_lens, device=device).view(B, 1, 1)
    kv_len_t = torch.tensor(kv_lens, device=device).view(B, 1, 1)

    valid = (q_idx < q_len_t) & (kv_idx < kv_len_t)

    if causal:
        # Right-aligned causal masking isn't appropriate here: each
        # sequence's query position i may attend to kv position j iff
        # j <= i, using LOCAL (per-sequence) indices, matching how the
        # Triton kernel computes causal masking per-sequence.
        causal_mask = q_idx >= kv_idx
        valid = valid & causal_mask

    return valid.unsqueeze(1)  # (B, 1, max_q, max_kv)


def print_header():

    print(
        f"{'Layout':8}"
        f"{'Type':8}"
        f"{'Topo':8}"
        f"{'Mask':8}"
        f"{'DType':10}"
        f"{'Q':>6}"
        f"{'KV':>6}"
        f"{'V1(ms)':>12}"
        f"{'V2(ms)':>12}"
        f"{'V1/V2':>10}"
        f"{'Torch(ms)':>12}"
        f"{'Torch/V2':>11}"
        f"{'Torch/V1':>11}"
    )


def print_row(
    layout,
    attn_type,
    topology,
    causal,
    dtype,
    q_len,
    kv_len,
    v1_ms,
    v2_ms,
    torch_ms,
    torch_naive_ms=None,
):

    mask = "causal" if causal else "full"

    print(
        f"{layout:<8}"
        f"{attn_type:<8}"
        f"{topology:<8}"
        f"{mask:<8}"
        f"{str(dtype).split('.')[-1]:<10}"
        f"{q_len:>8}"
        f"{kv_len:>8}"
        f"{v1_ms:12.3f}"
        f"{v2_ms:12.3f}"
        f"{v1_ms / v2_ms:9.2f}x"
        f"{torch_ms:12.3f}"
        f"{torch_ms / v2_ms:10.2f}x",
        f"{torch_ms / v1_ms:10.2f}x",
    )


def benchmark_dense(
    kernel,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    cross: bool,
):
    q = torch.randn(
        (BATCH, q_heads, q_len, HEAD_DIM),
        device=DEVICE,
        dtype=dtype,
    )

    if cross:
        k = torch.randn(
            (BATCH, kv_heads, kv_len, HEAD_DIM),
            device=DEVICE,
            dtype=dtype,
        )
    else:
        kv_len = q_len
        k = torch.randn(
            (BATCH, kv_heads, q_len, HEAD_DIM),
            device=DEVICE,
            dtype=dtype,
        )

    v = torch.randn_like(k)

    def torch_call():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            enable_gqa=(q_heads != kv_heads),
        )

    # Warmup
    kernel(
        q,
        k,
        v,
        is_causal=causal,
    )

    torch_call()

    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: kernel(
            q,
            k,
            v,
            is_causal=causal,
        )
    )

    torch_ms = triton.testing.do_bench(torch_call)

    return triton_ms, torch_ms, None


def _make_variable_length_batch(q_heads, kv_heads, q_len, kv_len, cross, dtype):
    """
    Builds BATCH independent sequences of length q_len (query) and kv_len
    (key/value) for self or cross attention. Lengths are currently uniform
    across the batch (matching the original benchmark's shapes) — this
    keeps the comparison focused on layout/kernel overhead rather than
    variable-length effects, but the helper accepts per-sequence lengths so
    it is easy to extend to truly ragged shapes later.
    """
    qs, ks, vs = [], [], []

    for _ in range(BATCH):
        q = torch.randn((q_len, q_heads, HEAD_DIM), device=DEVICE, dtype=dtype)

        if cross:
            k = torch.randn((kv_len, kv_heads, HEAD_DIM), device=DEVICE, dtype=dtype)
        else:
            k = torch.randn((q_len, kv_heads, HEAD_DIM), device=DEVICE, dtype=dtype)

        v = torch.randn_like(k)

        qs.append(q)
        ks.append(k)
        vs.append(v)

    return qs, ks, vs


def _dense_pad_and_mask_reference(qs, ks, vs, q_heads, kv_heads, causal, dtype):
    B = len(qs)
    q_lens = [q.shape[0] for q in qs]
    kv_lens = [k.shape[0] for k in ks]

    max_q = max(q_lens)
    max_kv = max(kv_lens)

    q_dense = torch.zeros((B, q_heads, max_q, HEAD_DIM), device=DEVICE, dtype=dtype)
    k_dense = torch.zeros((B, kv_heads, max_kv, HEAD_DIM), device=DEVICE, dtype=dtype)
    v_dense = torch.zeros((B, kv_heads, max_kv, HEAD_DIM), device=DEVICE, dtype=dtype)

    for b in range(B):
        q_dense[b, :, : q_lens[b], :] = qs[b].transpose(0, 1)
        k_dense[b, :, : kv_lens[b], :] = ks[b].transpose(0, 1)
        v_dense[b, :, : kv_lens[b], :] = vs[b].transpose(0, 1)

    if q_heads != kv_heads:
        groups = q_heads // kv_heads
        k_dense = k_dense.repeat_interleave(groups, dim=1)
        v_dense = v_dense.repeat_interleave(groups, dim=1)

    attn_mask = build_padded_mask(q_lens, kv_lens, max_q, max_kv, causal, DEVICE)
    # attn_mask is (B, 1, max_q, max_kv) boolean; broadcast over heads.

    def torch_call():
        return F.scaled_dot_product_attention(
            q_dense,
            k_dense,
            v_dense,
            attn_mask=attn_mask,
            is_causal=False,  # causal already baked into attn_mask
        )

    return torch_call


def _naive_loop_reference(qs, ks, vs, q_heads, kv_heads, causal):

    def torch_call():
        for q, k, v in zip(qs, ks, vs):
            q_ref = q.transpose(0, 1).unsqueeze(0)
            k_ref = k.transpose(0, 1).unsqueeze(0)
            v_ref = v.transpose(0, 1).unsqueeze(0)

            F.scaled_dot_product_attention(
                q_ref,
                k_ref,
                v_ref,
                is_causal=causal,
                enable_gqa=(q_heads != kv_heads),
            )

    return torch_call


def benchmark_ragged(
    kernel,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    cross: bool,
):
    if not cross:
        kv_len = q_len

    qs, ks, vs = _make_variable_length_batch(q_heads, kv_heads, q_len, kv_len, cross, dtype)

    q = torch.cat(qs, dim=0)
    k = torch.cat(ks, dim=0)
    v = torch.cat(vs, dim=0)

    q_indptr = torch.arange(
        0,
        (BATCH + 1) * q_len,
        q_len,
        device=DEVICE,
        dtype=torch.int32,
    )

    kv_indptr = torch.arange(
        0,
        (BATCH + 1) * kv_len,
        kv_len,
        device=DEVICE,
        dtype=torch.int32,
    )

    def triton_call():
        return kernel(
            q,
            k,
            v,
            is_causal=causal,
            q_indptr=q_indptr,
            kv_indptr=kv_indptr,
        )

    fair_call = _dense_pad_and_mask_reference(qs, ks, vs, q_heads, kv_heads, causal, dtype)

    # Warmup
    triton_call()
    fair_call()
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(triton_call)

    torch_ms = triton.testing.do_bench(fair_call)

    torch_naive_ms = None
    if INCLUDE_NAIVE_LOOP_REFERENCE:
        naive_call = _naive_loop_reference(qs, ks, vs, q_heads, kv_heads, causal)
        naive_call()
        torch.cuda.synchronize()
        torch_naive_ms = triton.testing.do_bench(naive_call)

    return triton_ms, torch_ms, torch_naive_ms


def build_paged_cache(
    seqs: list[torch.Tensor],
    page_size: int,
):
    """
    seqs:
        List[(N_i, KVH, D)]

    Returns
    -------
    cache:
        (total_pages * page_size, KVH, D)

    lengths:
        (B,)

    block_table:
        (B, max_pages)
    """

    device = seqs[0].device
    dtype = seqs[0].dtype

    kv_heads = seqs[0].shape[1]
    head_dim = seqs[0].shape[2]

    lengths = torch.tensor(
        [x.shape[0] for x in seqs],
        device=device,
        dtype=torch.int32,
    )

    pages = []
    tables = []

    next_page = 0

    for seq in seqs:
        n = seq.shape[0]
        num_pages = math.ceil(n / page_size)

        table = []

        for page in range(num_pages):
            start = page * page_size
            end = min(start + page_size, n)

            buf = torch.zeros(
                (page_size, kv_heads, head_dim),
                device=device,
                dtype=dtype,
            )

            buf[: end - start] = seq[start:end]

            pages.append(buf)

            table.append(next_page)
            next_page += 1

        tables.append(table)

    max_pages = max(len(x) for x in tables)

    block_table = torch.full(
        (len(seqs), max_pages),
        -1,
        device=device,
        dtype=torch.int32,
    )

    for i, table in enumerate(tables):
        block_table[i, : len(table)] = torch.tensor(
            table,
            device=device,
            dtype=torch.int32,
        )

    cache = torch.cat(pages, dim=0)

    return cache, lengths, block_table


def benchmark_paged(
    kernel,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    cross: bool,
):
    if not cross:
        kv_len = q_len

    qs, ks, vs = _make_variable_length_batch(q_heads, kv_heads, q_len, kv_len, cross, dtype)

    q = torch.stack(qs, dim=0)

    k_cache, lengths, k_table = build_paged_cache(ks, PAGE_SIZE)
    v_cache, _, v_table = build_paged_cache(vs, PAGE_SIZE)

    def triton_call():
        return kernel(
            q,
            k_cache,
            v_cache,
            is_causal=causal,
            kv_indptr=lengths,
            k_block_table=k_table,
            v_block_table=v_table,
            kv_page_size=PAGE_SIZE,
        )

    fair_call = _dense_pad_and_mask_reference(qs, ks, vs, q_heads, kv_heads, causal, dtype)

    # Warmup
    triton_call()
    fair_call()
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(triton_call)

    torch_ms = triton.testing.do_bench(fair_call)

    torch_naive_ms = None
    if INCLUDE_NAIVE_LOOP_REFERENCE:
        naive_call = _naive_loop_reference(qs, ks, vs, q_heads, kv_heads, causal)
        naive_call()
        torch.cuda.synchronize()
        torch_naive_ms = triton.testing.do_bench(naive_call)

    return triton_ms, torch_ms, torch_naive_ms


def main():

    print_header()

    for layout in LAYOUTS:
        for attention_type in ATTENTION_TYPES:
            cross = attention_type == "cross"

            for topology, (q_heads, kv_heads) in TOPOLOGIES.items():
                for dtype in DTYPES:
                    for causal in CAUSAL:
                        for q_len in Q_LENS:
                            if cross:
                                kv_lengths = KV_LENS
                            else:
                                kv_lengths = [q_len]

                            for kv_len in kv_lengths:
                                if causal and q_len > kv_len:
                                    continue

                                if layout == "dense":
                                    v1_ms, torch_ms, torch_native_ms = benchmark_dense(
                                        kernel=KERNELS.get("v1"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                    v2_ms, torch_ms, torch_naive_ms = benchmark_dense(
                                        kernel=KERNELS.get("v2"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                elif layout == "ragged":
                                    v1_ms, torch_ms, torch_naive_ms = benchmark_ragged(
                                        kernel=KERNELS.get("v1"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                    v2_ms, torch_ms, torch_naive_ms = benchmark_ragged(
                                        kernel=KERNELS.get("v2"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                else:
                                    v1_ms, torch_ms, torch_naive_ms = benchmark_paged(
                                        kernel=KERNELS.get("v1"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                    v2_ms, torch_ms, torch_naive_ms = benchmark_paged(
                                        kernel=KERNELS.get("v2"),
                                        dtype=dtype,
                                        q_heads=q_heads,
                                        kv_heads=kv_heads,
                                        q_len=q_len,
                                        kv_len=kv_len,
                                        causal=causal,
                                        cross=cross,
                                    )

                                print_row(
                                    layout=layout,
                                    attn_type=attention_type,
                                    topology=topology,
                                    causal=causal,
                                    dtype=dtype,
                                    q_len=q_len,
                                    kv_len=kv_len,
                                    v1_ms=v1_ms,
                                    v2_ms=v2_ms,
                                    torch_ms=torch_ms,
                                    torch_naive_ms=torch_naive_ms,
                                )


if __name__ == "__main__":
    main()
