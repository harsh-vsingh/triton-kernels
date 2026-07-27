import triton
import triton.language as tl
import torch

@triton.jit
def _flash_attentionv1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    KV_HEADS,
    Q_HEADS,
    M,
    N,
    TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr,
    kv_page_size: tl.constexpr,
    q_indptr=None,
    kv_indptr=None,
    k_block_table=None,
    v_block_table=None,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    programs = tl.num_programs(0)

    batch = pid1 // Q_HEADS
    q_head = pid1 % Q_HEADS
    kv_head = q_head * KV_HEADS // Q_HEADS

    if TYPE == 0:
        q_ptr += (batch * M * Q_HEADS + q_head) * HEAD_DIM
        k_ptr += (batch * N * KV_HEADS + kv_head) * HEAD_DIM
        v_ptr += (batch * N * KV_HEADS + kv_head) * HEAD_DIM
        o_ptr += (batch * M * Q_HEADS + q_head) * HEAD_DIM

        q_shape = (M, HEAD_DIM)
        kv_shape = (N, HEAD_DIM)

        seq_len_q = M
        seq_len_k = N

    elif TYPE == 1:
        q_start = tl.load(q_indptr + batch)
        q_end = tl.load(q_indptr + batch + 1)
        q_length = q_end - q_start

        kv_start = tl.load(kv_indptr + batch)
        kv_end = tl.load(kv_indptr + batch + 1)
        kv_length = kv_end - kv_start

        q_ptr += (q_start * Q_HEADS + q_head) * HEAD_DIM
        k_ptr += (kv_start * KV_HEADS + kv_head) * HEAD_DIM
        v_ptr += (kv_start * KV_HEADS + kv_head) * HEAD_DIM
        o_ptr += (q_start * Q_HEADS + q_head) * HEAD_DIM

        q_shape = (q_length, HEAD_DIM)
        kv_shape = (kv_length, HEAD_DIM)

        seq_len_q = q_length
        seq_len_k = kv_length

    elif TYPE == 2:
        seq_len_k = tl.load(kv_indptr + batch)
        kv_max_blocks = tl.cdiv(N, kv_page_size)
        
        k_block_table_start = k_block_table + batch * kv_max_blocks
        v_block_table_start = v_block_table + batch * kv_max_blocks

        q_start = tl.load(q_indptr + batch)
        q_end = tl.load(q_indptr + batch + 1)
        q_length = q_end - q_start

        q_ptr += (q_start * Q_HEADS + q_head) * HEAD_DIM
        out_ptr += (q_start * Q_HEADS + q_head) * HEAD_DIM
        q_shape = (q_length, HEAD_DIM)
        seq_len_q = q_length

    if TYPE == 0 or TYPE == 1:
        k_desc = tl.make_tensor_descriptor(
            base=k_ptr,
            shape=kv_shape,
            strides=(KV_HEADS * HEAD_DIM, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

        v_desc = tl.make_tensor_descriptor(
            base=v_ptr,
            shape=kv_shape,
            strides=(KV_HEADS * HEAD_DIM, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

        o_desc = tl.make_tensor_descriptor(
            base=o_ptr,
            shape=q_shape,
            strides=(Q_HEADS * HEAD_DIM, 1),
            block_shape=(BLOCK_M, HEAD_DIM),
        )

    q_desc = tl.make_tensor_descriptor(
        base=q_ptr,
        shape=q_shape,
        strides=(Q_HEADS * HEAD_DIM, 1),
        block_shape=(BLOCK_M, HEAD_DIM),
    )

    o_desc = tl.make_tensor_descriptor(
        base=o_ptr,
        shape=q_shape,
        strides=(Q_HEADS * HEAD_DIM, 1),
        block_shape=(BLOCK_M, HEAD_DIM),
    )
    
    total_blk = tl.cdiv(seq_len_q, BLOCK_M)
    blk = pid0
    offs_d = tl.arange(0, HEAD_DIM)

    while blk < total_blk:
        blk_m = blk * BLOCK_M
        blk_m_int = blk_m.to(tl.int32)

        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        running_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        running_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        q_idx = blk_m + tl.arange(0, BLOCK_M)
        q_mask = q_idx < seq_len_q

        q = q_desc.load([blk_m_int, 0]).to(tl.float32)

        if IS_CAUSAL:
            last_q = tl.minimum(blk_m + BLOCK_M, seq_len_q)
            max_block = tl.cdiv(last_q, BLOCK_N) * BLOCK_N
        else:
            max_block = seq_len_k

        for block_n in tl.range(0, max_block, BLOCK_N, num_stages=num_stages):
            k_idx = block_n + tl.arange(0, BLOCK_N)
            k_mask_1d = k_idx < seq_len_k
            block_n_int = block_n.to(tl.int32)

            if IS_CAUSAL:
                mask = k_mask_1d[None, :] & (q_idx[:, None] >= k_idx[None, :])
            else:
                mask = k_mask_1d[None, :]

            if TYPE==0 or TYPE==1:
                k = k_desc.load([block_n_int, 0]).to(tl.float32)
                v = v_desc.load([block_n_int, 0]).to(tl.float32)

            elif TYPE==2:
                logical_kv_pages = k_idx // kv_page_size
                kv_page_offs = k_idx % kv_page_size

                phys_k_pages = tl.load(k_block_table_start + logical_kv_pages, mask=k_mask_1d, other=0)
                phys_v_pages = tl.load(v_block_table_start + logical_kv_pages, mask=k_mask_1d, other=0)
                
                k_ptrs = k_ptr + (phys_k_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + (kv_head * HEAD_DIM) + offs_d[None, :]
                v_ptrs = v_ptr + (phys_v_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + (kv_head * HEAD_DIM) + offs_d[None, :]
                
                k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0).to(tl.float32)
                v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k))
            qk *= SM_SCALE
            qk = tl.where(mask, qk, float("-inf"))

            old_max = running_max
            block_max = tl.max(qk, axis=1)
            new_max = tl.maximum(old_max, block_max)

            alpha = tl.exp(old_max - new_max)
            prob = tl.exp(qk - new_max[:, None])

            running_sum = running_sum * alpha + tl.sum(prob, axis=1)
            running_acc = running_acc * alpha[:, None] + tl.dot(prob, v)
            running_max = new_max

        output = running_acc / running_sum[:, None]
        o_desc.store([blk_m_int, 0], output)

        blk += programs

def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    q_indptr: torch.Tensor | None = None,
    kv_indptr: torch.Tensor | None = None,
    q_block_table: torch.Tensor | None = None,
    k_block_table: torch.Tensor | None = None,
    v_block_table: torch.Tensor | None = None,
    q_page_size: int | None = None,
    kv_page_size: int | None = None,
) -> torch.Tensor:
    
    assert q.dtype == k.dtype == v.dtype, "Tensors must have the same dtype"
    assert q.device == k.device == v.device, "Tensors must be on the same device"
    assert q.is_cuda, "Tensors must be on a CUDA device"

    paged = (k_block_table is not None)
    
    ragged = (q_indptr is not None) and not paged

    if paged:
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        assert kv_page_size is not None
        assert k_block_table is not None and v_block_table is not None
        
        batch = k_block_table.shape[0]
        q_heads, head_dim = q.shape[1], q.shape[2]
        kv_heads = k.shape[1]

        if q_indptr is None:
            q_len = q.shape[0] // batch
            q_indptr = torch.full((batch,), q_len, dtype=torch.int32, device=q.device)
            M = q_len
        else:
            M = q_indptr.max().item()

        if kv_indptr is None:
            raise ValueError("kv_indptr (context lengths) must be provided for paged mode.")
            
        N = k_block_table.shape[1] * kv_page_size

        out = torch.empty_like(q)
        mode = 2

    elif ragged:
        if (q_indptr is None) != (kv_indptr is None):
            raise ValueError("Either provide both q_indptr and kv_indptr or neither.")

        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3

        total_q, q_heads, head_dim = q.shape
        total_kv, kv_heads, _ = k.shape

        batch = q_indptr.numel() - 1
        
        M = total_q
        N = total_kv

        out = torch.empty_like(q)
        mode = 1

    else:
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        batch, M, q_heads, head_dim = q.shape
        _, N, kv_heads, _ = k.shape

        out = torch.empty_like(q)
        mode = 0

    grid = (
        min(
            triton.cdiv(M, 64),
            torch.cuda.get_device_properties(q.device).multi_processor_count,
        ),
        batch * q_heads,
    )

    _flash_attentionv1_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=out,
        KV_HEADS=kv_heads,
        Q_HEADS=q_heads,
        M=M,
        N=N,
        TYPE=mode,
        IS_CAUSAL=is_causal,
        SM_SCALE=head_dim**-0.5,
        HEAD_DIM=head_dim,
        BLOCK_M=64,
        BLOCK_N=64,
        num_stages=2,
        q_page_size=q_page_size if q_page_size else 1,
        kv_page_size=kv_page_size if kv_page_size else 1,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        q_block_table=q_block_table,
        k_block_table=k_block_table,
        v_block_table=v_block_table,
    )

    if ragged or paged:
        return out

    return out.transpose(1, 2).contiguous()