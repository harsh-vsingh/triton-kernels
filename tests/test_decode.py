import math
import random

import pytest
import torch
import torch.nn.functional as F

from kernels import decode_attention

torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

BASE_TOL = {
    torch.float16:  dict(base_atol=1e-2, rtol=1e-2, eps=2e-3),
    torch.bfloat16: dict(base_atol=2e-2, rtol=2e-2, eps=4e-3),
    torch.float32:  dict(base_atol=1e-3, rtol=1e-2, eps=2e-3),
}

def decode_tolerances(dtype, N):
    cfg = BASE_TOL[dtype]
    atol = cfg["base_atol"] * math.sqrt(max(N, 1)) + cfg["eps"]
    return dict(atol=atol, rtol=cfg["rtol"])


LAYOUTS = ["dense", "ragged", "paged"]
TOPOLOGIES = ["mha", "gqa", "mqa"]

PAGE_SIZE = 16


def get_topology_heads(topology):
    if topology == "mha":
        return 8, 8
    if topology == "gqa":
        return 8, 2
    if topology == "mqa":
        return 8, 1
    raise ValueError


def generate_sequences(B, QH, KVH, D, dtype, layout):
    qs = []
    ks = []
    vs = []
    kv_lens = []

    fixed_kv = random.randint(16, 256)

    for _ in range(B):

        l_kv = fixed_kv if layout == "dense" else random.randint(1, 256)

        kv_lens.append(l_kv)

        qs.append(torch.randn(QH, D, device=DEVICE, dtype=dtype))
        ks.append(torch.randn(l_kv, KVH, D, device=DEVICE, dtype=dtype))
        vs.append(torch.randn(l_kv, KVH, D, device=DEVICE, dtype=dtype))

    return qs, ks, vs, kv_lens


def get_reference_outputs(qs, ks, vs, QH, KVH):
    refs = []

    for q, k, v in zip(qs, ks, vs):

        q_ref = q.unsqueeze(0).unsqueeze(2).float()
        k_ref = k.transpose(0, 1).unsqueeze(0).float()
        v_ref = v.transpose(0, 1).unsqueeze(0).float()

        if QH != KVH:
            groups = QH // KVH
            k_ref = k_ref.repeat_interleave(groups, dim=1)
            v_ref = v_ref.repeat_interleave(groups, dim=1)

        out = F.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
        )

        refs.append(out.squeeze(0).squeeze(1))

    return refs


def allocate_paged_cache(tensors, lens, heads, dim, page_size):

    B = len(lens)

    max_blocks = max((l + page_size - 1) // page_size for l in lens)

    total_blocks = sum((l + page_size - 1) // page_size for l in lens)

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

            cache[
                current * page_size:
                current * page_size + size
            ] = tensors[b][start:end]

            current += 1

    return cache, table


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_decode(layout, topology, dtype):

    B = 4
    D = 64

    QH, KVH = get_topology_heads(topology)

    qs, ks, vs, kv_lens = generate_sequences(
        B,
        QH,
        KVH,
        D,
        dtype,
        layout,
    )

    refs = get_reference_outputs(
        qs,
        ks,
        vs,
        QH,
        KVH,
    )

    if layout == "dense":

        max_kv = max(kv_lens)

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

        out = decode_attention(q, k, v)

        outputs = [out[b] for b in range(B)]

    elif layout == "ragged":

        q = torch.stack(qs)

        k = torch.cat(ks, dim=0)
        v = torch.cat(vs, dim=0)

        kv_indptr = torch.tensor(
            [0] + kv_lens,
            device=DEVICE,
            dtype=torch.int32,
        ).cumsum(0)

        out = decode_attention(
            q,
            k,
            v,
            kv_indptr=kv_indptr,
        )

        outputs = [out[b] for b in range(B)]

    else:

        q = torch.stack(qs)

        k_cache, k_table = allocate_paged_cache(
            ks,
            kv_lens,
            KVH,
            D,
            PAGE_SIZE,
        )

        v_cache, v_table = allocate_paged_cache(
            vs,
            kv_lens,
            KVH,
            D,
            PAGE_SIZE,
        )

        kv_lengths = torch.tensor(
            kv_lens,
            dtype=torch.int32,
            device=DEVICE,
        )

        out = decode_attention(
            q,
            k_cache,
            v_cache,
            kv_indptr=kv_lengths,
            k_block_table=k_table,
            v_block_table=v_table,
            kv_page_size=PAGE_SIZE,
        )

        outputs = [out[b] for b in range(B)]

    for b in range(B):

        torch.testing.assert_close(
            outputs[b].float(),
            refs[b],
            **decode_tolerances(dtype, max(kv_lens)),
        )