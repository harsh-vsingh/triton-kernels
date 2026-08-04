import math
import random

import pytest
import torch
import torch.nn.functional as F

from kernels.attention import flash_attention_v2

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


def attn_tolerances(dtype, N):
    cfg = BASE_TOL[dtype]
    atol = cfg["base_atol"] * math.sqrt(max(N, 1)) + cfg["eps"]
    return dict(atol=atol, rtol=cfg["rtol"])


# Test dimensions
LAYOUTS = ["dense", "paged", "ragged"]
ATTN_TYPES = ["self", "cross"]
TOPOLOGIES = ["mha", "gqa", "mqa"]
CAUSAL_FLAGS = [False, True]

PAGE_SIZE = 16


def get_topology_heads(topology):
    if topology == "mha":
        return 8, 8
    elif topology == "gqa":
        return 8, 2
    elif topology == "mqa":
        return 8, 1
    raise ValueError(f"Unknown topology {topology}")


def generate_sequences(B, QH, KVH, D, attn_type, dtype, layout):
    """Generates independent sequences to be packed into various layouts."""
    qs, ks, vs = [], [], []
    q_lens, kv_lens = [], []

    # Pre-determine a uniform length for the dense batch to prevent padding artifacts
    fixed_q = random.randint(16, 128)
    fixed_kv = fixed_q if attn_type == "self" else random.randint(16, 256)

    for _ in range(B):
        # Dense layouts require uniform lengths since the dense kernel doesn't take a mask
        l_q = fixed_q if layout == "dense" else random.randint(1, 128)

        if attn_type == "self":
            l_kv = l_q
        else:
            l_kv = fixed_kv if layout == "dense" else random.randint(1, 256)

        q_lens.append(l_q)
        kv_lens.append(l_kv)

        qs.append(torch.randn((l_q, QH, D), device=DEVICE, dtype=dtype))
        ks.append(torch.randn((l_kv, KVH, D), device=DEVICE, dtype=dtype))
        vs.append(torch.randn((l_kv, KVH, D), device=DEVICE, dtype=dtype))

    return qs, ks, vs, q_lens, kv_lens


def get_reference_outputs(qs, ks, vs, QH, KVH, is_causal):
    """Computes exact PyTorch SDPA reference for each sequence independently."""
    refs = []
    for q, k, v in zip(qs, ks, vs):
        # SDPA expects (Batch, Heads, SeqLen, Dim) -> (1, H, L, D)
        q_ref = q.transpose(0, 1).unsqueeze(0).float()
        k_ref = k.transpose(0, 1).unsqueeze(0).float()
        v_ref = v.transpose(0, 1).unsqueeze(0).float()

        if QH != KVH:
            groups = QH // KVH
            k_ref = k_ref.repeat_interleave(groups, dim=1)
            v_ref = v_ref.repeat_interleave(groups, dim=1)

        out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)

        # Convert back to (L, H, D)
        refs.append(out.squeeze(0).transpose(0, 1))
    return refs


def allocate_paged_cache(tensors, lens, heads, dim, page_size):
    """Takes a list of sequence tensors and scatters them into a physical paged cache."""
    B = len(lens)
    max_blocks = max((length + page_size - 1) // page_size for length in lens)

    # Pre-allocate total blocks needed (sum of all sequences' blocks)
    total_blocks = sum((length + page_size - 1) // page_size for length in lens)

    # Create the physical cache flattened by token: (total_blocks * page_size, Heads, Dim)
    cache = torch.zeros(
        (total_blocks * page_size, heads, dim), device=DEVICE, dtype=tensors[0].dtype
    )
    block_table = torch.zeros((B, max_blocks), dtype=torch.int32, device=DEVICE)

    current_physical_block = 0
    for b in range(B):
        seq_len = lens[b]
        seq_tensor = tensors[b]
        blocks_needed = (seq_len + page_size - 1) // page_size

        for logical_idx in range(blocks_needed):
            block_table[b, logical_idx] = current_physical_block

            # Copy data into the cache
            start_token = logical_idx * page_size
            end_token = min((logical_idx + 1) * page_size, seq_len)
            chunk_size = end_token - start_token

            phys_start = current_physical_block * page_size
            cache[phys_start : phys_start + chunk_size] = seq_tensor[start_token:end_token]

            current_physical_block += 1

    return cache, block_table


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("attn_type", ATTN_TYPES)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_comprehensive_attention(layout, attn_type, topology, causal, dtype):

    # SDPA causal on cross-attention behaves weirdly if Q_len != KV_len and isn't standard in LLMs.
    if causal and attn_type == "cross":
        pytest.skip("Causal masking on cross attention is generally undefined/uncommon.")

    B = 4
    D = 64
    QH, KVH = get_topology_heads(topology)

    # 1. Generate Raw Sequences - Layout passed explicitly!
    qs, ks, vs, q_lens, kv_lens = generate_sequences(B, QH, KVH, D, attn_type, dtype, layout)

    # 2. Get Ground Truth References
    refs = get_reference_outputs(qs, ks, vs, QH, KVH, causal)

    # 3. Format Data & Run Kernel based on Layout
    if layout == "dense":
        max_q = max(q_lens)
        max_kv = max(kv_lens)

        q_dense = torch.zeros((B, QH, max_q, D), device=DEVICE, dtype=dtype)
        k_dense = torch.zeros((B, KVH, max_kv, D), device=DEVICE, dtype=dtype)
        v_dense = torch.zeros((B, KVH, max_kv, D), device=DEVICE, dtype=dtype)

        for b in range(B):
            q_dense[b, :, : q_lens[b], :] = qs[b].transpose(0, 1)
            k_dense[b, :, : kv_lens[b], :] = ks[b].transpose(0, 1)
            v_dense[b, :, : kv_lens[b], :] = vs[b].transpose(0, 1)

        out_dense = flash_attention_v2(q_dense, k_dense, v_dense, is_causal=causal)

        # Unpack dense output for validation
        extracted_outs = [out_dense[b, :, : q_lens[b], :].transpose(0, 1) for b in range(B)]

    elif layout == "ragged":
        q_ragged = torch.cat(qs, dim=0)
        k_ragged = torch.cat(ks, dim=0)
        v_ragged = torch.cat(vs, dim=0)

        q_indptr = torch.tensor([0] + q_lens, device=DEVICE, dtype=torch.int32).cumsum(dim=0)
        kv_indptr = torch.tensor([0] + kv_lens, device=DEVICE, dtype=torch.int32).cumsum(dim=0)

        out_ragged = flash_attention_v2(
            q_ragged, k_ragged, v_ragged, is_causal=causal, q_indptr=q_indptr, kv_indptr=kv_indptr
        )

        # Unpack ragged output for validation
        extracted_outs = []
        for b in range(B):
            start = q_indptr[b].item()
            end = q_indptr[b + 1].item()
            extracted_outs.append(out_ragged[start:end])

    elif layout == "paged":
        # Q stays dense.
        max_q = max(q_lens)

        q_dense = torch.zeros(
            (B, QH, max_q, D),
            device=DEVICE,
            dtype=dtype,
        )

        for b in range(B):
            q_dense[b, :, : q_lens[b], :] = qs[b].transpose(0, 1)

        # Only KV are stored in the paged cache.
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

        kv_lens_tensor = torch.tensor(
            kv_lens,
            dtype=torch.int32,
            device=DEVICE,
        )

        out_dense = flash_attention_v2(
            q_dense,
            k_cache,
            v_cache,
            is_causal=causal,
            kv_indptr=kv_lens_tensor,
            k_block_table=k_table,
            v_block_table=v_table,
            kv_page_size=PAGE_SIZE,
        )

        extracted_outs = [out_dense[b, :, : q_lens[b], :].transpose(0, 1) for b in range(B)]

    # 4. Assert correctness sequence by sequence
    for b in range(B):
        torch.testing.assert_close(
            extracted_outs[b].float(),
            refs[b],
            **attn_tolerances(dtype, max(kv_lens)),
        )
