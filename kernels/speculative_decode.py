import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=1),
        # triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=2),
        # triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
    ],
    key=["N", "TYPE", "kv_page_size"],
)
@triton.jit
def _speculative_decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    N,
    M,
    TYPE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    kv_page_size: tl.constexpr = 1,
    indptr=None,
    k_block_table=None,
    v_block_table=None,
    causal: tl.constexpr = False,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    batch = pid1 // Q_HEADS
    q_head = pid1 % Q_HEADS
    kv_head = q_head * KV_HEADS // Q_HEADS

    if TYPE == 0:
        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM
        k_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM
        v_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM

        seq_len = N

    elif TYPE == 1:
        start = tl.load(indptr + batch)
        end = tl.load(indptr + batch + 1)
        length = end - start

        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM
        k_ptr += (start * KV_HEADS + kv_head) * HEAD_DIM
        v_ptr += (start * KV_HEADS + kv_head) * HEAD_DIM

        seq_len = length

    elif TYPE == 2:
        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM

        seq_len = tl.load(indptr + batch)
        kv_max_blocks = tl.cdiv(N, kv_page_size)

        k_block_table_start = k_block_table + batch * kv_max_blocks
        v_block_table_start = v_block_table + batch * kv_max_blocks

    total = tl.cdiv(seq_len, BLOCK_N)

    running_sum = tl.zeros([BLOCK_M], tl.float32)
    running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    running_acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    draft_start = seq_len - M

    q = tl.load(q_ptr + offs_d[None, :] + offs_m[:, None] * HEAD_DIM, mask=mask_m[:, None], other=0.0).to(tl.float32)

    split_id = pid0

    while pid0 < total:
        blk_n = pid0 * BLOCK_N

        offs_n = blk_n + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len

        if TYPE == 0:
            k = tl.load(
                k_ptr + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

            v = tl.load(
                v_ptr + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

        elif TYPE == 1:
            tmp = offs_n[:, None] * (KV_HEADS * HEAD_DIM) + offs_d[None, :]
            k = tl.load(
                k_ptr + tmp,
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

            v = tl.load(
                v_ptr + offs_n[:, None] * (KV_HEADS * HEAD_DIM) + offs_d[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

        elif TYPE == 2:
            logical_kv_pages = offs_n // kv_page_size
            kv_page_offs = offs_n % kv_page_size

            phys_k_pages = tl.load(k_block_table_start + logical_kv_pages, mask=mask, other=0)
            phys_v_pages = tl.load(v_block_table_start + logical_kv_pages, mask=mask, other=0)

            k_ptrs = (
                k_ptr
                + (phys_k_pages[:, None] * kv_page_size + kv_page_offs[:, None])
                * (KV_HEADS * HEAD_DIM)
                + (kv_head * HEAD_DIM)
                + offs_d[None, :]
            )
            v_ptrs = (
                v_ptr
                + (phys_v_pages[:, None] * kv_page_size + kv_page_offs[:, None])
                * (KV_HEADS * HEAD_DIM)
                + (kv_head * HEAD_DIM)
                + offs_d[None, :]
            )

            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        qk = tl.sum(k[None, :, :] * q[:, None, :], axis=2)
        qk *= SM_SCALE

        if causal:
            draft_pos = offs_n - draft_start
            future_draft = (offs_n[None, :] >= draft_start) & (draft_pos[None, :] > offs_m[:, None])

            valid = mask[None, :] & ~future_draft
            qk = tl.where(valid, qk, float("-inf"))

        else:
            qk = tl.where(mask[None, :], qk, float("-inf"))

        block_max = tl.max(qk, axis=1)
        new_max = tl.maximum(running_max, block_max)

        new_mask = (new_max == float("-inf"))
        safe_new_max = tl.where(new_mask, 0.0, new_max)
 
        alpha = tl.where(new_mask, 0.0, tl.exp(running_max - safe_new_max))
        weights = tl.where(new_mask[:, None], 0.0, tl.exp(qk - safe_new_max[:, None]))
        weights = tl.where(mask_m[:, None], weights, 0.0)
 
        prod = weights[:, :, None] * v[None, :, :]
 
        running_acc = running_acc * alpha[:, None] + tl.sum(prod, axis=1)
        running_sum = running_sum * alpha + tl.sum(weights, axis=1)
        running_max = safe_new_max
        pid0 += SPLIT_K


    workspace = (batch * Q_HEADS + q_head) * SPLIT_K + split_id
    offs = tl.arange(0, BLOCK_M)

    tl.store(running_sum_ptr + workspace * BLOCK_M + offs, running_sum, mask=mask_m)
    tl.store(running_max_ptr + workspace * BLOCK_M + offs, running_max, mask=mask_m)
    tl.store(running_acc_ptr + workspace * BLOCK_M * HEAD_DIM + offs_d[None, :] + offs[:, None] * HEAD_DIM, running_acc, mask=mask_m[:, None])


@triton.jit
def _speculative_decode_merge_kernel(
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    o_ptr,
    M,
    Q_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid0 = tl.program_id(0)

    batch = pid0 // Q_HEADS
    q_head = pid0 % Q_HEADS

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    workspace = ((batch * Q_HEADS + q_head) * SPLIT_K + tl.arange(0, SPLIT_K)[:, None]) * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]

    running_max = tl.load(running_max_ptr + workspace, mask=mask_m[None, :], other=float("-inf"))
    running_sum = tl.load(running_sum_ptr + workspace, mask=mask_m[None, :], other=0.0)
    running_acc = tl.load(running_acc_ptr + workspace[:, :, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, None, :], mask=mask_m[None, :, None], other=0.0)

    global_max = tl.max(running_max, axis=0)
    alpha_i = tl.exp(running_max - global_max[None, :])
    total_sum = tl.sum(running_sum * alpha_i, axis=0)
    total_acc = tl.sum(running_acc * alpha_i[:, :, None], axis=0)

    output = total_acc / total_sum[:, None]

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < M

    tl.store(o_ptr + (batch * Q_HEADS + q_head) * M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :], output, mask=mask_m[:, None],)

def speculative_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    kv_indptr: torch.Tensor | None = None,
    k_block_table: torch.Tensor | None = None,
    v_block_table: torch.Tensor | None = None,
    kv_page_size: int | None = None,
) -> torch.Tensor:
    """
    Speculative decode attention with Dense, Ragged, and Paged KV Cache support.

    Dense (mode 0):
        q : (B, QH, M, D)
        k : (B, KVH, N, D)
        v : (B, KVH, N, D)

    Ragged (mode 1):
        q : (B, QH, M, D)
        k : (total_kv, KVH, D)
        v : (total_kv, KVH, D)
        kv_indptr : (B + 1,)

    Paged (mode 2):
        q : (B, QH, M, D)
        k : (total_kv_tokens, KVH, D)
        v : (total_kv_tokens, KVH, D)
        kv_indptr : (B,) -> sequence context length per batch
        k_block_table : (B, max_blocks_per_seq)
        v_block_table : (B, max_blocks_per_seq)
    """

    assert q.dtype == k.dtype == v.dtype, "Tensors must have the same dtype"
    assert q.device == k.device == v.device, "Tensors must be on the same device"
    assert q.is_cuda, "Tensors must be on a CUDA device"

    paged = k_block_table is not None
    ragged = (kv_indptr is not None) and not paged

    if paged:
        assert k_block_table is not None and v_block_table is not None
        assert kv_page_size is not None
        assert kv_indptr is not None, (
            "kv_indptr (sequence context lengths) must be provided for paged mode."
        )

        B, Q_HEADS, M, HEAD_DIM = q.shape
        KV_HEADS = k.shape[2] if k.ndim == 4 else k.shape[1]

        N = k_block_table.shape[1] * kv_page_size
        mode = 2

    elif ragged:
        if kv_indptr is None:
            raise ValueError("kv_indptr must be provided for ragged mode.")

        B, Q_HEADS, M, HEAD_DIM = q.shape
        total_kv, KV_HEADS, _ = k.shape

        N = total_kv
        mode = 1

    else:
        B, Q_HEADS, M, HEAD_DIM = q.shape
        _, KV_HEADS, N, _ = k.shape
        mode = 0

    assert M <= 16, ("Speculative decode is intended for small draft lengths (M <= 16).")

    BLOCK_M = triton.next_power_of_2(M)

    blocks = math.ceil(N / 64) * BLOCK_M
    programs = B * Q_HEADS
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count


    if programs > num_sms:
        split_k = 1
    else:
        split_k = min(max(1, num_sms // programs), max(1, blocks // 64))

    running_sum = torch.empty(
        (B, Q_HEADS, split_k, BLOCK_M),
        device=q.device,
        dtype=torch.float32,
    )

    running_max = torch.empty_like(running_sum)

    running_acc = torch.empty(
        (B, Q_HEADS, split_k, BLOCK_M, HEAD_DIM),
        device=q.device,
        dtype=torch.float32,
    )

    out = torch.empty_like(q)

    grid = (
        split_k,
        B * Q_HEADS,
    )
    _speculative_decode_split_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        N=N,
        M=M,
        TYPE=mode,
        Q_HEADS=Q_HEADS,
        BLOCK_M=BLOCK_M,
        SPLIT_K=split_k,
        KV_HEADS=KV_HEADS,
        SM_SCALE=1.0 / math.sqrt(HEAD_DIM),
        HEAD_DIM=HEAD_DIM,
        kv_page_size=kv_page_size if kv_page_size else 1,
        indptr=kv_indptr,
        k_block_table=k_block_table,
        v_block_table=v_block_table,
        causal=causal,
    )

    _speculative_decode_merge_kernel[(B * Q_HEADS,)](
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        M=M,
        o_ptr=out,
        Q_HEADS=Q_HEADS,
        BLOCK_M=BLOCK_M,
        SPLIT_K=split_k,
        HEAD_DIM=HEAD_DIM,
    )

    return out