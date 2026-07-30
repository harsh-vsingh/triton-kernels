import triton
import triton.language as tl
import torch


@triton.autotune(
    configs = [

        triton.Config(
            {"BLOCK_M":64,"BLOCK_N":32},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_M":64,"BLOCK_N":64},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_M":64,"BLOCK_N":64},
            num_warps=8,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_M":64,"BLOCK_N":128},
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {"BLOCK_M":128,"BLOCK_N":64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=[
        "HEAD_DIM",
        "TYPE",
        "IS_CAUSAL",
        "kv_page_size",
    ],
)
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
        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM
        k_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM
        v_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM
        o_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM

        q_shape = (M, HEAD_DIM)
        kv_shape = (N, HEAD_DIM)

        seq_len_q = M
        seq_len_k = N

        q_stride = HEAD_DIM
        k_stride = HEAD_DIM

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

        q_stride = HEAD_DIM * Q_HEADS
        k_stride = HEAD_DIM * KV_HEADS

    elif TYPE == 2:
        seq_len_k = tl.load(kv_indptr + batch)

        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM
        o_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM

        q_shape = (M, HEAD_DIM)
        seq_len_q = M

        kv_max_blocks = tl.cdiv(N, kv_page_size)

        k_block_table_start = k_block_table + batch * kv_max_blocks
        v_block_table_start = v_block_table + batch * kv_max_blocks

        q_stride = HEAD_DIM

    if TYPE == 0 or TYPE == 1:
        k_desc = tl.make_tensor_descriptor(
            base=k_ptr,
            shape=kv_shape,
            strides=(k_stride, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

        v_desc = tl.make_tensor_descriptor(
            base=v_ptr,
            shape=kv_shape,
            strides=(k_stride, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

    q_desc = tl.make_tensor_descriptor(
        base=q_ptr,
        shape=q_shape,
        strides=(q_stride, 1),
        block_shape=(BLOCK_M, HEAD_DIM),
    )

    o_desc = tl.make_tensor_descriptor(
        base=o_ptr,
        shape=q_shape,
        strides=(q_stride, 1),
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
    k_block_table: torch.Tensor | None = None,
    v_block_table: torch.Tensor | None = None,
    kv_page_size: int | None = None,
) -> torch.Tensor:

    assert q.dtype == k.dtype == v.dtype
    assert q.device == k.device == v.device
    assert q.is_cuda

    paged = k_block_table is not None
    ragged = (q_indptr is not None) and not paged

    if not ragged and not paged:

        assert q.ndim == 4
        assert k.ndim == 4
        assert v.ndim == 4

        batch, q_heads, M, head_dim = q.shape
        batch_k, kv_heads, N, head_dim_k = k.shape
        batch_v, kv_heads_v, N_v, head_dim_v = v.shape

        assert batch == batch_k == batch_v
        assert kv_heads == kv_heads_v
        assert N == N_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        out = torch.empty_like(q)
        mode = 0

    elif ragged:

        assert q.ndim == 3
        assert k.ndim == 3
        assert v.ndim == 3

        total_q, q_heads, head_dim = q.shape
        total_kv, kv_heads, head_dim_k = k.shape
        total_v, kv_heads_v, head_dim_v = v.shape

        assert total_kv == total_v
        assert kv_heads == kv_heads_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        assert q_indptr is not None
        assert kv_indptr is not None

        batch = q_indptr.numel() - 1

        M = total_q
        N = total_kv

        out = torch.empty_like(q)
        mode = 1

    else:

        assert q.ndim == 4
        assert k.ndim == 3
        assert v.ndim == 3

        batch, q_heads, M, head_dim = q.shape
        total_cache, kv_heads, head_dim_k = k.shape
        total_cache_v, kv_heads_v, head_dim_v = v.shape

        assert total_cache == total_cache_v
        assert kv_heads == kv_heads_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        assert kv_page_size is not None
        assert kv_indptr is not None
        assert k_block_table is not None
        assert v_block_table is not None

        out = torch.empty_like(q)
        mode = 2

        N = k_block_table.shape[1] * kv_page_size

    grid = lambda META: (
        min(
            triton.cdiv(M, META["BLOCK_M"]),
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
        SM_SCALE=head_dim ** -0.5,
        HEAD_DIM=head_dim,
        kv_page_size=kv_page_size if kv_page_size is not None else 1,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        k_block_table=k_block_table,
        v_block_table=v_block_table,
    )

    return out


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_N": 64},
            num_warps=8,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_N": 128},
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {"BLOCK_N": 64},
            num_warps=8,
            num_stages=3,
        ),

        triton.Config(
            {"BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),

        triton.Config(
            {"BLOCK_N": 32},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_N": 128},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_N": 64},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=[
        "HEAD_DIM",
        "TYPE",
        "IS_CAUSAL",
        "kv_page_size",
        "SPLIT_K",
        "M",
        "N",
    ],
)
@triton.jit
def _flash_attentionv2_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    KV_HEADS,
    Q_HEADS,
    M,
    N,
    NUM_Q_BLOCKS,
    TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPLIT_K: tl.constexpr,
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
    pid2 = tl.program_id(2)

    batch = pid1 // Q_HEADS
    q_head = pid1 % Q_HEADS
    kv_head = q_head * KV_HEADS // Q_HEADS

    if TYPE == 0:
        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM
        k_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM
        v_ptr += (batch * KV_HEADS + kv_head) * N * HEAD_DIM

        q_shape = (M, HEAD_DIM)
        kv_shape = (N, HEAD_DIM)

        seq_len_q = M
        seq_len_k = N

        q_stride = HEAD_DIM
        k_stride = HEAD_DIM

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

        q_shape = (q_length, HEAD_DIM)
        kv_shape = (kv_length, HEAD_DIM)

        seq_len_q = q_length
        seq_len_k = kv_length

        q_stride = HEAD_DIM * Q_HEADS
        k_stride = HEAD_DIM * KV_HEADS

    elif TYPE == 2:
        seq_len_k = tl.load(kv_indptr + batch)

        q_ptr += (batch * Q_HEADS + q_head) * M * HEAD_DIM

        q_shape = (M, HEAD_DIM)
        seq_len_q = M

        kv_max_blocks = tl.cdiv(N, kv_page_size)

        k_block_table_start = k_block_table + batch * kv_max_blocks
        v_block_table_start = v_block_table + batch * kv_max_blocks

        q_stride = HEAD_DIM

    if TYPE == 0 or TYPE == 1:
        k_desc = tl.make_tensor_descriptor(
            base=k_ptr,
            shape=kv_shape,
            strides=(k_stride, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

        v_desc = tl.make_tensor_descriptor(
            base=v_ptr,
            shape=kv_shape,
            strides=(k_stride, 1),
            block_shape=(BLOCK_N, HEAD_DIM),
        )

    q_desc = tl.make_tensor_descriptor(
        base=q_ptr,
        shape=q_shape,
        strides=(q_stride, 1),
        block_shape=(BLOCK_M, HEAD_DIM),
    )

    total_q_blocks = tl.cdiv(seq_len_q, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    blk = pid0

    while blk < total_q_blocks:

        blk_m = blk * BLOCK_M
        blk_m_int = blk_m.to(tl.int32)

        q_idx = blk_m + tl.arange(0, BLOCK_M)
        q = q_desc.load([blk_m_int, 0]).to(tl.float16)

        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        running_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        running_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        if IS_CAUSAL:
            last_q = tl.minimum(blk_m + BLOCK_M, seq_len_q)
            max_k_blocks = tl.cdiv(last_q, BLOCK_N)
        else:
            max_k_blocks = tl.cdiv(seq_len_k, BLOCK_N)

        for block_k in tl.range(pid2, max_k_blocks, SPLIT_K, num_stages=num_stages):

            block_n = block_k * BLOCK_N
            block_n_int = block_n.to(tl.int32)

            k_idx = block_n + tl.arange(0, BLOCK_N)
            k_mask_1d = k_idx < seq_len_k

            if IS_CAUSAL:
                mask = k_mask_1d[None, :] & (q_idx[:, None] >= k_idx[None, :])
            else:
                mask = k_mask_1d[None, :]

            if TYPE == 0 or TYPE == 1:
                k = k_desc.load([block_n_int, 0]).to(tl.float16)

            elif TYPE == 2:

                logical_kv_pages = k_idx // kv_page_size
                kv_page_offs = k_idx % kv_page_size

                phys_k_pages = tl.load(k_block_table_start + logical_kv_pages,mask=k_mask_1d,other=0,)
                k_ptrs = (k_ptr + (phys_k_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + offs_d[None, :])
                k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0,).to(tl.float16)

            qk = tl.dot(q, tl.trans(k))
            qk *= SM_SCALE
            qk = tl.where(mask, qk, float("-inf"))

            block_max = tl.max(qk, axis=1)
            new_max = tl.maximum(running_max, block_max,)

            alpha = tl.exp(running_max - new_max)
            qk = tl.exp(qk - new_max[:, None])

            running_sum *= alpha
            running_sum += tl.sum(qk, axis=1)
            running_max = new_max

            if TYPE == 0 or TYPE == 1:

                v = v_desc.load([block_n_int, 0]).to(tl.float32)

            elif TYPE == 2:

                phys_v_pages = tl.load(v_block_table_start + logical_kv_pages,mask=k_mask_1d,other=0,)
                v_ptrs = (v_ptr + (phys_v_pages[:, None] * kv_page_size + kv_page_offs[:, None]) * (KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + offs_d[None, :])
                v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0,).to(tl.float32)

            running_acc *= alpha[:, None]
            running_acc += tl.dot(qk, v)


        workspace = ((batch * Q_HEADS + q_head) * NUM_Q_BLOCKS + blk) * SPLIT_K + pid2
        
        tl.store(running_sum_ptr + workspace * BLOCK_M + tl.arange(0, BLOCK_M), running_sum,)
        tl.store(running_max_ptr + workspace * BLOCK_M + tl.arange(0, BLOCK_M), running_max,)
        tl.store(running_acc_ptr + workspace * BLOCK_M * HEAD_DIM + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + offs_d[None, :], running_acc,)

        blk += tl.num_programs(0)

@triton.jit
def _flash_attentionv2_merge_kernel(
    running_sum_ptr,
    running_max_ptr,
    running_acc_ptr,
    o_ptr,
    Q_HEADS,
    M,
    NUM_Q_BLOCKS,
    TYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    q_indptr=None,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    batch = pid1 // Q_HEADS
    q_head = pid1 % Q_HEADS

    if TYPE == 0 or TYPE == 2:
        seq_len_q = M

    else:
        q_start = tl.load(q_indptr + batch)
        q_end = tl.load(q_indptr + batch + 1)

        seq_len_q = q_end - q_start

    total_q_blocks = tl.cdiv(seq_len_q, BLOCK_M)

    if pid0 >= total_q_blocks:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    workspace = (
        ((batch * Q_HEADS + q_head) * NUM_Q_BLOCKS + pid0)
        * SPLIT_K
        + tl.arange(0, SPLIT_K)
    )

    running_max = tl.load(
        running_max_ptr
        + workspace[:, None] * BLOCK_M
        + offs_m[None, :]
    )

    running_sum = tl.load(
        running_sum_ptr
        + workspace[:, None] * BLOCK_M
        + offs_m[None, :]
    )

    running_acc = tl.load(
        running_acc_ptr
        + workspace[:, None, None] * BLOCK_M * HEAD_DIM
        + offs_m[None, :, None] * HEAD_DIM
        + offs_d[None, None, :]
    )

    global_max = tl.max(running_max, axis=0)

    alpha = tl.exp(running_max - global_max[None, :])

    total_sum = tl.sum(
        running_sum * alpha,
        axis=0,
    )

    total_acc = tl.sum(
        running_acc * alpha[:, :, None],
        axis=0,
    )

    output = total_acc / total_sum[:, None]

    blk_m = pid0 * BLOCK_M
    rows = blk_m + offs_m
    row_mask = rows < seq_len_q

    if TYPE == 0 or TYPE == 2:

        out_ptr = (
            o_ptr
            + (batch * Q_HEADS + q_head) * M * HEAD_DIM
        )

        tl.store(
            out_ptr
            + rows[:, None] * HEAD_DIM
            + offs_d[None, :],
            output,
            mask=row_mask[:, None],
        )

    else:

        out_ptr = (
            o_ptr
            + (q_start * Q_HEADS + q_head) * HEAD_DIM
        )

        tl.store(
            out_ptr
            + rows[:, None] * Q_HEADS * HEAD_DIM
            + offs_d[None, :],
            output,
            mask=row_mask[:, None],
        )

def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    q_indptr: torch.Tensor | None = None,
    kv_indptr: torch.Tensor | None = None,
    k_block_table: torch.Tensor | None = None,
    v_block_table: torch.Tensor | None = None,
    kv_page_size: int | None = None,
) -> torch.Tensor:

    assert q.dtype == k.dtype == v.dtype
    assert q.device == k.device == v.device
    assert q.is_cuda

    paged = k_block_table is not None
    ragged = (q_indptr is not None) and not paged

    if not ragged and not paged:

        assert q.ndim == 4
        assert k.ndim == 4
        assert v.ndim == 4

        batch, q_heads, M, head_dim = q.shape
        batch_k, kv_heads, N, head_dim_k = k.shape
        batch_v, kv_heads_v, N_v, head_dim_v = v.shape

        assert batch == batch_k == batch_v
        assert kv_heads == kv_heads_v
        assert N == N_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        out = torch.empty_like(q)
        mode = 0

    elif ragged:

        assert q.ndim == 3
        assert k.ndim == 3
        assert v.ndim == 3

        total_q, q_heads, head_dim = q.shape
        total_kv, kv_heads, head_dim_k = k.shape
        total_v, kv_heads_v, head_dim_v = v.shape

        assert total_kv == total_v
        assert kv_heads == kv_heads_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        assert q_indptr is not None
        assert kv_indptr is not None

        batch = q_indptr.numel() - 1

        M = total_q
        N = total_kv

        out = torch.empty_like(q)
        mode = 1

    else:

        assert q.ndim == 4
        assert k.ndim == 3
        assert v.ndim == 3

        batch, q_heads, M, head_dim = q.shape
        total_cache, kv_heads, head_dim_k = k.shape
        total_cache_v, kv_heads_v, head_dim_v = v.shape

        assert total_cache == total_cache_v
        assert kv_heads == kv_heads_v
        assert head_dim == head_dim_k == head_dim_v
        assert q_heads % kv_heads == 0

        assert kv_page_size is not None
        assert kv_indptr is not None
        assert k_block_table is not None
        assert v_block_table is not None

        out = torch.empty_like(q)
        mode = 2

        N = k_block_table.shape[1] * kv_page_size

    if N >= 4096:
        split_k = 8
    elif N >= 2048:
        split_k = 4
    elif N >= 1024:
        split_k = 2
    else:
        split_k = 1

    BLOCK_M = 64

    if mode == 1:
        q_lengths = q_indptr[1:] - q_indptr[:-1]
        max_q_len = int(q_lengths.max().item())
        num_q_blocks = triton.cdiv(max_q_len, BLOCK_M)
    else:
        num_q_blocks = triton.cdiv(M, BLOCK_M)

    running_sum = torch.empty(
        (batch, q_heads, num_q_blocks, split_k, BLOCK_M),
        device=q.device,
        dtype=torch.float32,
    )

    running_max = torch.empty_like(running_sum)

    running_acc = torch.empty(
        (batch, q_heads, num_q_blocks, split_k, BLOCK_M, head_dim),
        device=q.device,
        dtype=torch.float32,
    )

    sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    if mode == 1:
        launch_blocks = num_q_blocks
    else:
        launch_blocks = triton.cdiv(M, BLOCK_M)

    grid = lambda META: (
        min(
            launch_blocks,
            sms,
        ),
        batch * q_heads,
        split_k,
    )

    _flash_attentionv2_partial_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        KV_HEADS=kv_heads,
        Q_HEADS=q_heads,
        M=M,
        N=N,
        NUM_Q_BLOCKS=num_q_blocks,
        TYPE=mode,
        IS_CAUSAL=is_causal,
        SPLIT_K=split_k,
        SM_SCALE=head_dim ** -0.5,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        kv_page_size=kv_page_size if kv_page_size is not None else 1,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        k_block_table=k_block_table,
        v_block_table=v_block_table,
    )

    merge_grid = (
        num_q_blocks,
        batch * q_heads,
    )

    _flash_attentionv2_merge_kernel[merge_grid](
        running_sum_ptr=running_sum,
        running_max_ptr=running_max,
        running_acc_ptr=running_acc,
        o_ptr=out,
        Q_HEADS=q_heads,
        M=M,
        NUM_Q_BLOCKS=num_q_blocks,
        TYPE=mode,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        SPLIT_K=split_k,
        q_indptr=q_indptr,
    )

    return out