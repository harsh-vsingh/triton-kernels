import math
import random

import pytest
import torch
import torch.nn.functional as F

from kernels import speculative_decode_attention

torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

BASE_TOL = {
    torch.float16: dict(base_atol=1e-2, rtol=1e-2, eps=2e-3),
    torch.bfloat16: dict(base_atol=2e-2, rtol=2e-2, eps=4e-3),
    torch.float32: dict(base_atol=1e-3, rtol=1e-2, eps=2e-3),
}


def decode_tolerances(dtype, N):
    cfg = BASE_TOL[dtype]
    atol = cfg["base_atol"] * math.sqrt(max(N, 1)) + cfg["eps"]
    return dict(atol=atol, rtol=cfg["rtol"])


LAYOUTS = ["dense", "ragged", "paged"]
TOPOLOGIES = ["mha", "gqa", "mqa"]

PAGE_SIZE = 16

M_VALUES = [2, 3, 4, 6, 8, 10, 12, 16]
CAUSAL = [True, False]


def get_topology_heads(topology):
    if topology == "mha":
        return 8, 8
    if topology == "gqa":
        return 8, 2
    if topology == "mqa":
        return 8, 1
    raise ValueError


def generate_sequences(B, QH, KVH, M, D, dtype, layout):
    qs = []
    ks = []
    vs = []
    draft_ks = []
    draft_vs = []
    kv_lens = []

    fixed_kv = random.randint(16, 256)

    for _ in range(B):
        l_kv = fixed_kv if layout == "dense" else random.randint(1, 256)

        kv_lens.append(l_kv)

        qs.append(torch.randn(QH, M, D, device=DEVICE, dtype=dtype))
        ks.append(torch.randn(l_kv, KVH, D, device=DEVICE, dtype=dtype))
        vs.append(torch.randn(l_kv, KVH, D, device=DEVICE, dtype=dtype))

        draft_ks.append(torch.randn(M, KVH, D, device=DEVICE, dtype=dtype))
        draft_vs.append(torch.randn(M, KVH, D, device=DEVICE, dtype=dtype))

    return qs, ks, vs, kv_lens, draft_ks, draft_vs

def get_reference_outputs(qs, ks, vs, draft_ks, draft_vs, QH, KVH, causal):
    refs = []

    for q, k, v, dk, dv in zip(qs, ks, vs, draft_ks, draft_vs):
        q_ref = q.unsqueeze(0).float()

        k_ref = k.transpose(0, 1).unsqueeze(0).float()
        v_ref = v.transpose(0, 1).unsqueeze(0).float()

        if causal:
            k_ref = torch.cat(
                [k_ref, dk.transpose(0, 1).unsqueeze(0).float()],
                dim=2,
            )

            v_ref = torch.cat(
                [v_ref, dv.transpose(0, 1).unsqueeze(0).float()],
                dim=2,
            )

        if QH != KVH:
            groups = QH // KVH
            k_ref = k_ref.repeat_interleave(groups, dim=1)
            v_ref = v_ref.repeat_interleave(groups, dim=1)

        mask = None

        if causal:
            M = q_ref.shape[2]
            N = k_ref.shape[2] - M

            mask = torch.ones(
                (M, N + M),
                dtype=torch.bool,
                device=q_ref.device,
            )

            for i in range(M):
                mask[i, N + i + 1 :] = False

        out = F.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            attn_mask=mask,
            is_causal=False,
        )

        refs.append(out.squeeze(0))

    return refs


def allocate_paged_cache(tensors, lens, heads, dim, page_size):

    B = len(lens)

    max_blocks = max((length + page_size - 1) // page_size for length in lens)

    total_blocks = sum((length + page_size - 1) // page_size for length in lens)

    cache = torch.zeros(
        (total_blocks * page_size, heads, dim),
        device=DEVICE,
        dtype=tensors[0].dtype,
    )

    table = torch.zeros(
        (B, max_blocks),
        dtype=torch.int32,
        device=DEVICE,
    )

    current = 0

    for b in range(B):
        blocks = (lens[b] + page_size - 1) // page_size

        for logical in range(blocks):
            table[b, logical] = current

            start = logical * page_size
            end = min(start + page_size, lens[b])

            size = end - start

            cache[current * page_size : current * page_size + size] = tensors[b][start:end]

            current += 1

    return cache, table


@pytest.mark.parametrize("causal", CAUSAL)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_speculative_decode(M, layout, topology, dtype, causal):

    B = 4
    D = 64

    QH, KVH = get_topology_heads(topology)

    qs, ks, vs, kv_lens, draft_ks, draft_vs = generate_sequences(
        B,
        QH,
        KVH,
        M,
        D,
        dtype,
        layout,
    )

    refs = get_reference_outputs(
        qs,
        ks,
        vs,
        draft_ks,
        draft_vs,
        QH,
        KVH,
        causal,
    )

    if layout == "dense":
        kv_lens_kernel = [
            l + (M if causal else 0)
            for l in kv_lens
        ]
        max_kv = max(kv_lens_kernel)

        q = torch.stack(qs)

        k = torch.zeros(
            (B, KVH, max_kv, D),
            device=DEVICE,
            dtype=dtype,
        )
        v = torch.zeros_like(k)

        for b in range(B):
            k[b, :, :kv_lens[b]] = ks[b].transpose(0, 1)
            v[b, :, :kv_lens[b]] = vs[b].transpose(0, 1)

            if causal:
                k[b, :, kv_lens[b]:kv_lens[b] + M] = draft_ks[b].transpose(0, 1)
                v[b, :, kv_lens[b]:kv_lens[b] + M] = draft_vs[b].transpose(0, 1)

        out = speculative_decode_attention(
            q,
            k,
            v,
            causal=causal,
        )

        outputs = [out[b] for b in range(B)]

    elif layout == "ragged":
        q = torch.stack(qs)

        k_list = []
        v_list = []
        kv_lens_kernel = []

        for b, (k_i, v_i, l) in enumerate(zip(ks, vs, kv_lens)):

            if causal:
                k_i = torch.cat([k_i, draft_ks[b]], dim=0)
                v_i = torch.cat([v_i, draft_vs[b]], dim=0)
                l += M

            k_list.append(k_i)
            v_list.append(v_i)
            kv_lens_kernel.append(l)


        k = torch.cat(k_list, dim=0)
        v = torch.cat(v_list, dim=0)

        kv_indptr = torch.tensor(
            [0] + kv_lens_kernel,
            device=DEVICE,
            dtype=torch.int32,
        ).cumsum(0)

        out = speculative_decode_attention(
            q,
            k,
            v,
            kv_indptr=kv_indptr,
            causal=causal,
        )

        outputs = [out[b] for b in range(B)]

    else:
        q = torch.stack(qs)

        k_list = []
        v_list = []
        kv_lens_kernel = []

        for b, (k_i, v_i, l) in enumerate(zip(ks, vs, kv_lens)):

            if causal:
                k_i = torch.cat([k_i, draft_ks[b]], dim=0)
                v_i = torch.cat([v_i, draft_vs[b]], dim=0)
                l += M

            k_list.append(k_i)
            v_list.append(v_i)
            kv_lens_kernel.append(l)


        k_cache, k_table = allocate_paged_cache(
            k_list,
            kv_lens_kernel,
            KVH,
            D,
            PAGE_SIZE,
        )

        v_cache, v_table = allocate_paged_cache(
            v_list,
            kv_lens_kernel,
            KVH,
            D,
            PAGE_SIZE,
        )

        kv_lengths = torch.tensor(
            kv_lens_kernel,
            dtype=torch.int32,
            device=DEVICE,
        )

        out = speculative_decode_attention(
            q,
            k_cache,
            v_cache,
            kv_indptr=kv_lengths,
            k_block_table=k_table,
            v_block_table=v_table,
            kv_page_size=PAGE_SIZE,
            causal=causal,
        )

        outputs = [out[b] for b in range(B)]

    for b in range(B):
        torch.testing.assert_close(
            outputs[b].float(),
            refs[b],
            **decode_tolerances(dtype, max(kv_lens_kernel)),
        )