import triton
import triton.language as tl
import torch
import math

@triton.autotune(
configs = [
    triton.Config({"BLOCK_N": 32},  num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 32},  num_warps=2, num_stages=2),

    triton.Config({"BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=2),

    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),

    triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
],
    key=["N", "TYPE", "kv_page_size"],
)
@triton.jit
def _decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    N,
    TYPE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    kv_page_size: tl.constexpr = 1,
    indptr=None,
    k_block_table=None,
    v_block_table=None,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    batch = pid1 // Q_HEADS
    q_head = pid1 % Q_HEADS
    kv_head = q_head * KV_HEADS // Q_HEADS

    if TYPE == 0:
        q_ptr += (batch * Q_HEADS + q_head) * HEAD_DIM
        k_ptr += (batch * N * KV_HEADS + kv_head) * HEAD_DIM
        v_ptr += (batch * N * KV_HEADS + kv_head) * HEAD_DIM

        seq_len = N

    elif TYPE == 1:
        start = tl.load(indptr + batch)
        end = tl.load(indptr + batch + 1)
        length = end - start

        q_ptr += (batch * Q_HEADS + q_head) * HEAD_DIM        
        k_ptr += (start * KV_HEADS + kv_head) * HEAD_DIM
        v_ptr += (start * KV_HEADS + kv_head) * HEAD_DIM

        seq_len = length

    elif TYPE == 2:
        q_ptr += (batch * Q_HEADS + q_head) * HEAD_DIM

        seq_len = tl.load(indptr + batch)
        kv_max_blocks = tl.cdiv(N, kv_page_size)

        k_block_table_start = k_block_table + batch * kv_max_blocks
        v_block_table_start = v_block_table + batch * kv_max_blocks

    total = tl.cdiv(seq_len, BLOCK_N)

    running_sum = tl.zeros((), tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_acc = tl.zeros((HEAD_DIM,), tl.float32)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + offs_d).to(tl.float32)

    split_id = pid0

    while pid0 < total:
        blk_n = pid0 * BLOCK_N

        offs_n = blk_n + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len

        if TYPE == 0 or TYPE == 1:
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

            k_ptrs = k_ptr + (phys_k_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + (kv_head * HEAD_DIM) + offs_d[None, :]
            v_ptrs = v_ptr + (phys_v_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + (kv_head * HEAD_DIM) + offs_d[None, :]

            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1)
        qk *= SM_SCALE

        qk = tl.where(mask, qk, float("-inf"))

        block_max = tl.max(qk)
        new_max = tl.maximum(running_max, block_max)

        alpha = tl.exp(running_max - new_max)
        weights = tl.exp(qk - new_max)

        running_acc = (
            running_acc * alpha +
            tl.sum(weights[:, None] * v, axis=0)
        )

        running_sum = running_sum * alpha + tl.sum(weights)
        running_max = new_max

        pid0 += SPLIT_K

    workspace = (batch * Q_HEADS + q_head) * SPLIT_K + split_id

    tl.store(running_sum_ptr + workspace, running_sum)
    tl.store(running_max_ptr + workspace, running_max)
    tl.store(running_acc_ptr + workspace * HEAD_DIM + offs_d, running_acc)

@triton.jit
def _decode_merge_kernel(
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    o_ptr,
    Q_HEADS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid0 = tl.program_id(0)

    batch = pid0 // Q_HEADS
    q_head = pid0 % Q_HEADS
    
    workspace = (batch * Q_HEADS + q_head) * SPLIT_K + tl.arange(0, SPLIT_K)

    running_max = tl.load(running_max_ptr + workspace)
    running_sum = tl.load(running_sum_ptr + workspace)
    running_acc = tl.load(running_acc_ptr + workspace[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM))

    global_max = tl.max(running_max, axis=0)
    alpha_i = tl.exp(running_max - global_max)
    total_sum = tl.sum(running_sum * alpha_i, axis=0)
    total_acc = tl.sum(running_acc * alpha_i[:, None], axis=0)

    output = total_acc / total_sum

    offs_d = tl.arange(0, HEAD_DIM)
    tl.store(o_ptr + (batch * Q_HEADS + q_head) * HEAD_DIM + offs_d, output)

def decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_indptr: torch.Tensor | None = None,
    k_block_table: torch.Tensor | None = None,
    v_block_table: torch.Tensor | None = None,
    kv_page_size: int | None = None,
) -> torch.Tensor:
    """
    Decode attention with Dense, Ragged, and Paged KV Cache support.

    Dense (mode 0):
        q : (B, QH, D)
        k : (B, KVH, N, D)
        v : (B, KVH, N, D)

    Ragged (mode 1):
        q : (B, QH, D)
        k : (total_kv, KVH, D)
        v : (total_kv, KVH, D)
        kv_indptr : (B + 1,)

    Paged (mode 2):
        q : (B, QH, D)
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
        assert kv_indptr is not None, "kv_indptr (sequence context lengths) must be provided for paged mode."

        B, Q_HEADS, HEAD_DIM = q.shape
        KV_HEADS = k.shape[2] if k.ndim == 4 else k.shape[1]

        N = k_block_table.shape[1] * kv_page_size
        mode = 2

    elif ragged:
        if kv_indptr is None:
            raise ValueError("kv_indptr must be provided for ragged mode.")

        B, Q_HEADS, HEAD_DIM = q.shape
        total_kv, KV_HEADS, _ = k.shape

        N = total_kv
        mode = 1

    else:
        B, Q_HEADS, HEAD_DIM = q.shape
        _, KV_HEADS, N, _ = k.shape

        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        mode = 0

    blocks = math.ceil(N / 64)
    programs = B * Q_HEADS

    if programs >= 256:
        split_k = 1
    elif programs >= 128:
        split_k = min(4, max(1, blocks // 16))
    else:
        split_k = min(16, max(1, blocks // 8))

    running_sum = torch.empty(
        (B, Q_HEADS, split_k),
        device=q.device,
        dtype=torch.float32,
    )

    running_max = torch.empty_like(running_sum)

    running_acc = torch.empty(
        (B, Q_HEADS, split_k, HEAD_DIM),
        device=q.device,
        dtype=torch.float32,
    )

    out = torch.empty_like(q)

    
    grid = (
        split_k,
        B * Q_HEADS,
    )

    _decode_split_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        N=N,
        TYPE=mode,
        Q_HEADS=Q_HEADS,
        SPLIT_K=split_k,
        KV_HEADS=KV_HEADS,
        SM_SCALE=1.0 / math.sqrt(HEAD_DIM),
        HEAD_DIM=HEAD_DIM,
        kv_page_size=kv_page_size if kv_page_size else 1,
        indptr=kv_indptr,
        k_block_table=k_block_table,
        v_block_table=v_block_table,
    )

    _decode_merge_kernel[(B * Q_HEADS,)](
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        o_ptr=out,
        Q_HEADS=Q_HEADS,
        SPLIT_K=split_k,
        HEAD_DIM=HEAD_DIM,
    )

    return out