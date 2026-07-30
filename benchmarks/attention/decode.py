import math

import torch
import torch.nn.functional as F
import triton.testing

from kernels import decode_attention

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
]

LAYOUTS = [
    "dense",
    "ragged",
    "paged",
]

TOPOLOGIES = [
    ("mha", 8, 8),
    ("gqa", 8, 2),
    ("mqa", 8, 1),
]

SEQ_LENS = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]

PAGE_SIZE = 16
BATCH = 16
HEAD_DIM = 64


def allocate_paged_cache(tensors, lengths, kv_heads, head_dim, page_size):
    B = len(lengths)

    max_blocks = max(math.ceil(x / page_size) for x in lengths)
    total_blocks = sum(math.ceil(x / page_size) for x in lengths)

    cache = torch.zeros(
        (total_blocks * page_size, kv_heads, head_dim),
        device=DEVICE,
        dtype=tensors[0].dtype,
    )

    table = torch.zeros(
        (B, max_blocks),
        device=DEVICE,
        dtype=torch.int32,
    )

    current = 0

    for b in range(B):
        blocks = math.ceil(lengths[b] / page_size)

        for logical in range(blocks):
            table[b, logical] = current

            start = logical * page_size
            end = min(start + page_size, lengths[b])

            cache[
                current * page_size:
                current * page_size + (end - start)
            ] = tensors[b][start:end]

            current += 1

    return cache, table


def benchmark_dense(dtype, qh, kvh, N):

    q = torch.randn((BATCH, qh, HEAD_DIM), device=DEVICE, dtype=dtype)

    k = torch.randn(
        (BATCH, kvh, N, HEAD_DIM),
        device=DEVICE,
        dtype=dtype,
    )

    v = torch.randn_like(k)

    # warmup
    decode_attention(q, k, v)
    F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k,
        v,
        enable_gqa=(qh != kvh),
    )
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: decode_attention(q, k, v)
    )

    torch_ms = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(
            q.unsqueeze(2),
            k,
            v,
            enable_gqa=(qh != kvh),
        )
    )

    return triton_ms, torch_ms


def benchmark_ragged(dtype, qh, kvh, N):

    q = torch.randn((BATCH, qh, HEAD_DIM), device=DEVICE, dtype=dtype)

    ks = [
        torch.randn((N, kvh, HEAD_DIM), device=DEVICE, dtype=dtype)
        for _ in range(BATCH)
    ]

    vs = [
        torch.randn((N, kvh, HEAD_DIM), device=DEVICE, dtype=dtype)
        for _ in range(BATCH)
    ]

    k = torch.cat(ks, dim=0)
    v = torch.cat(vs, dim=0)

    kv_indptr = torch.arange(
        0,
        (BATCH + 1) * N,
        N,
        device=DEVICE,
        dtype=torch.int32,
    )

    decode_attention(
        q,
        k,
        v,
        kv_indptr=kv_indptr,
    )

    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: decode_attention(
            q,
            k,
            v,
            kv_indptr=kv_indptr,
        )
    )

    def reference():

        for b in range(BATCH):

            F.scaled_dot_product_attention(
                q[b].unsqueeze(0).unsqueeze(2),
                ks[b].transpose(0, 1).unsqueeze(0),
                vs[b].transpose(0, 1).unsqueeze(0),
                enable_gqa=(qh != kvh),
            )

    reference_ms = triton.testing.do_bench(reference)

    return triton_ms, reference_ms


def benchmark_paged(dtype, qh, kvh, N):

    lengths = [N] * BATCH

    q = torch.randn((BATCH, qh, HEAD_DIM), device=DEVICE, dtype=dtype)

    ks = [
        torch.randn((N, kvh, HEAD_DIM), device=DEVICE, dtype=dtype)
        for _ in range(BATCH)
    ]

    vs = [
        torch.randn((N, kvh, HEAD_DIM), device=DEVICE, dtype=dtype)
        for _ in range(BATCH)
    ]

    k_cache, k_table = allocate_paged_cache(
        ks,
        lengths,
        kvh,
        HEAD_DIM,
        PAGE_SIZE,
    )

    v_cache, v_table = allocate_paged_cache(
        vs,
        lengths,
        kvh,
        HEAD_DIM,
        PAGE_SIZE,
    )

    kv_lengths = torch.tensor(
        lengths,
        device=DEVICE,
        dtype=torch.int32,
    )

    decode_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr=kv_lengths,
        k_block_table=k_table,
        v_block_table=v_table,
        kv_page_size=PAGE_SIZE,
    )

    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: decode_attention(
            q,
            k_cache,
            v_cache,
            kv_indptr=kv_lengths,
            k_block_table=k_table,
            v_block_table=v_table,
            kv_page_size=PAGE_SIZE,
        )
    )

    def reference():

        for b in range(BATCH):

            F.scaled_dot_product_attention(
                q[b].unsqueeze(0).unsqueeze(2),
                ks[b].transpose(0, 1).unsqueeze(0),
                vs[b].transpose(0, 1).unsqueeze(0),
                enable_gqa=(qh != kvh),
            )

    reference_ms = triton.testing.do_bench(reference)

    return triton_ms, reference_ms


if __name__ == "__main__":

    print(
        f"{'Layout':8}"
        f"{'Topology':8}"
        f"{'DType':10}"
        f"{'SeqLen':8}"
        f"{'Triton(ms)':>14}"
        f"{'Torch(ms)':>14}"
        f"{'Speedup':>12}"
    )

    for layout in LAYOUTS:

        for topo, qh, kvh in TOPOLOGIES:

            for dtype in DTYPES:

                for N in SEQ_LENS:

                    if layout == "dense":
                        t_ms, p_ms = benchmark_dense(dtype, qh, kvh, N)

                    elif layout == "ragged":
                        t_ms, p_ms = benchmark_ragged(dtype, qh, kvh, N)

                    else:
                        t_ms, p_ms = benchmark_paged(dtype, qh, kvh, N)

                    print(
                        f"{layout:8}"
                        f"{topo:8}"
                        f"{str(dtype).split('.')[-1]:10}"
                        f"{N:<8}"
                        f"{t_ms:14.3f}"
                        f"{p_ms:14.3f}"
                        f"{p_ms / t_ms:12.2f}x"
                    )