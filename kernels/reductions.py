import builtins

import torch
import triton
import triton.language as tl

from utils import validate_reduction

NUM_SMS = torch.cuda.get_device_properties(
    triton.runtime.driver.active.get_active_torch_device()
).multi_processor_count

SMALL_THRESHOLD = 1024
MEDIUM_THRESHOLD = 8192
BLOCK_SIZE = 1024
NUM_STAGES = 2
MAX_PROGRAMS_PER_ROW = 8
MEDIUM_ROW_THRESHOLD = NUM_SMS
NUM_WARPS = 8


@triton.jit
def _sum_kernel_small(x_ptr, out_ptr, x_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, x_sum)


@triton.jit
def _sum_kernel_medium(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_vec += x
    acc = tl.sum(sum_vec, axis=0)
    tl.store(out_ptr + row_id, acc)


@triton.jit
def _sum_kernel_large(
    x_ptr,
    out_buffer,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    sum_vec = tl.zeros([BLOCK_SIZE], tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_vec += x
    sum = tl.sum(sum_vec, axis=0)
    tl.store(out_buffer + pid, sum)


def sum(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the sum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _sum_kernel_small[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:
        _sum_kernel_medium[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES, num_warps=NUM_WARPS
        )
    else:
        programs_per_row = builtins.min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)

        _sum_kernel_large[(n_rows * programs_per_row,)](
            x,
            partial,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )
        partial = partial.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)
        _sum_kernel_small[(n_rows,)](
            partial, out, programs_per_row, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    return out.view(x.shape[:-1])


def mean(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the mean of over last dimension in the input tensor
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    out = sum(x, out=out)
    out.div_(x.shape[-1])
    return out


@triton.jit
def _max_kernel_small(x_ptr, out_ptr, x_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x_max = tl.max(x, axis=0)
    tl.store(out_ptr + pid, x_max)


@triton.jit
def _max_kernel_medium(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    max_vec = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        max_vec = tl.maximum(max_vec, x)
    tl.store(out_ptr + row_id, tl.max(max_vec, axis=0))


@triton.jit
def _max_kernel_large(
    x_ptr,
    out_buffer,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    max_vec = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        max_vec = tl.maximum(max_vec, x)
    tl.store(out_buffer + pid, tl.max(max_vec, axis=0))


def max(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the maximum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _max_kernel_small[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:
        _max_kernel_medium[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES, num_warps=NUM_WARPS
        )
    else:
        programs_per_row = builtins.min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)

        _max_kernel_large[(n_rows * programs_per_row,)](
            x,
            partial,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )
        partial = partial.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)
        _max_kernel_small[(n_rows,)](
            partial, out, programs_per_row, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    return out.view(x.shape[:-1]).to(x.dtype)


@triton.jit
def _argmax_kernel_small(x_ptr, out_ptr, x_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x_max = tl.argmax(x, axis=0)
    tl.store(out_ptr + pid, x_max.to(tl.int64))


@triton.jit
def _argmax_kernel_medium(
    x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    acc = tl.full((), float("-inf"), tl.float32)
    acc_idx = tl.zeros((), dtype=tl.int32)
    offset_range = tl.arange(0, BLOCK_SIZE)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

        block_max = tl.max(x, axis=0)
        block_idx = tl.argmax(x, axis=0)
        acc_idx = tl.where(acc < block_max, col_idx + block_idx, acc_idx)
        acc = tl.maximum(acc, block_max)
    tl.store(out_ptr + row_id, acc_idx.to(tl.int64))


@triton.jit
def _argmax_kernel_large(
    x_ptr,
    out_buffer_val,
    out_buffer_idx,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    acc = tl.full((), float("-inf"), tl.float32)
    acc_idx = tl.zeros((), dtype=tl.int32)
    offset_range = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

        block_max = tl.max(x, axis=0)
        block_idx = tl.argmax(x, axis=0)
        acc_idx = tl.where(acc < block_max, chunk * BLOCK_SIZE + block_idx, acc_idx)
        acc = tl.maximum(acc, block_max)
    tl.store(out_buffer_idx + pid, acc_idx.to(tl.int64))
    tl.store(out_buffer_val + pid, acc)


@triton.jit
def argmax_combine(val1, idx1, val2, idx2):
    take_second = (val2 > val1) | ((val2 == val1) & (idx2 < idx1))
    val = tl.where(take_second, val2, val1)
    idx = tl.where(take_second, idx2, idx1)
    return val, idx


@triton.jit
def _argmax_kernel_merge(
    partial_val_ptr,
    partial_idx_ptr,
    out_ptr,
    programs_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * programs_per_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < programs_per_row

    partial_val = tl.load(partial_val_ptr + row_start + offsets, mask=mask, other=float("inf"))
    partial_idx = tl.load(partial_idx_ptr + row_start + offsets, mask=mask, other=0).to(tl.int64)

    final_val, final_idx = tl.reduce((partial_val, partial_idx), axis=0, combine_fn=argmax_combine)

    tl.store(out_ptr + pid, final_idx)


def argmax(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the index of the maximum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a int64 tensor.
    """
    validate_reduction(x)
    assert out is None or out.dtype == torch.int64

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.int64, device=x.device) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _argmax_kernel_small[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:
        _argmax_kernel_medium[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES, num_warps=NUM_WARPS
        )
    else:
        programs_per_row = builtins.min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial_val = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        partial_idx = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.int64)
        _argmax_kernel_large[(n_rows * programs_per_row,)](
            x,
            partial_val,
            partial_idx,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

        partial_val = partial_val.reshape(n_rows, programs_per_row)
        partial_idx = partial_idx.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)

        merge_block_size = triton.next_power_of_2(programs_per_row)
        _argmax_kernel_merge[(n_rows,)](
            partial_val,
            partial_idx,
            out,
            programs_per_row,
            BLOCK_SIZE=merge_block_size,
            num_warps=NUM_WARPS,
        )
        out = out.view(-1)
    return out.view(x.shape[:-1]).to(torch.int64)


@triton.jit
def _min_kernel_small(x_ptr, out_ptr, x_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
    x_min = tl.min(x, axis=0)
    tl.store(out_ptr + pid, x_min)


@triton.jit
def _min_kernel_medium(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    min_vec = tl.full([BLOCK_SIZE], float("inf"), tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
        min_vec = tl.minimum(min_vec, x)
    tl.store(out_ptr + row_id, tl.min(min_vec, axis=0))


@triton.jit
def _min_kernel_large(
    x_ptr,
    out_buffer,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    min_vec = tl.full([BLOCK_SIZE], float("inf"), tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
        min_vec = tl.minimum(min_vec, x)
    partial_min = tl.min(min_vec, axis=0)
    tl.store(out_buffer + pid, partial_min)


def min(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the minimum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _min_kernel_small[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:
        _min_kernel_medium[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES, num_warps=NUM_WARPS
        )
    else:
        programs_per_row = builtins.min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)

        _min_kernel_large[(n_rows * programs_per_row,)](
            x,
            partial,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )
        partial = partial.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)
        _min_kernel_small[(n_rows,)](
            partial, out, programs_per_row, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    return out.view(x.shape[:-1]).to(x.dtype)


@triton.jit
def _argmin_kernel_small(x_ptr, out_ptr, x_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
    x_min = tl.argmin(x, axis=0)
    tl.store(out_ptr + pid, x_min.to(tl.int64))


@triton.jit
def _argmin_kernel_medium(
    x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    acc = tl.full((), float("inf"), tl.float32)
    acc_idx = tl.zeros((), dtype=tl.int32)
    offset_range = tl.arange(0, BLOCK_SIZE)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))

        block_min = tl.min(x, axis=0)
        block_idx = tl.argmin(x, axis=0)
        acc_idx = tl.where(acc > block_min, col_idx + block_idx, acc_idx)
        acc = tl.minimum(acc, block_min)
    tl.store(out_ptr + row_id, acc_idx.to(tl.int64))


@triton.jit
def _argmin_kernel_large(
    x_ptr,
    out_buffer_val,
    out_buffer_idx,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    acc = tl.full((), float("inf"), tl.float32)
    acc_idx = tl.zeros((), dtype=tl.int32)
    offset_range = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))

        block_min = tl.min(x, axis=0)
        block_idx = tl.argmin(x, axis=0)
        acc_idx = tl.where(acc > block_min, chunk * BLOCK_SIZE + block_idx, acc_idx)
        acc = tl.minimum(acc, block_min)
    tl.store(out_buffer_idx + pid, acc_idx.to(tl.int64))
    tl.store(out_buffer_val + pid, acc)


@triton.jit
def argmin_combine(val1, idx1, val2, idx2):
    take_second = (val2 < val1) | ((val2 == val1) & (idx2 < idx1))
    val = tl.where(take_second, val2, val1)
    idx = tl.where(take_second, idx2, idx1)
    return val, idx


@triton.jit
def _argmin_kernel_merge(
    partial_val_ptr,
    partial_idx_ptr,
    out_ptr,
    programs_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * programs_per_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < programs_per_row

    partial_val = tl.load(partial_val_ptr + row_start + offsets, mask=mask, other=float("inf"))
    partial_idx = tl.load(partial_idx_ptr + row_start + offsets, mask=mask, other=0).to(tl.int64)

    final_val, final_idx = tl.reduce((partial_val, partial_idx), axis=0, combine_fn=argmin_combine)

    tl.store(out_ptr + pid, final_idx)


def argmin(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the index of the minimum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a int64 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.int64, device=x.device) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _argmin_kernel_small[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=small_block_size, num_warps=NUM_WARPS
        )
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:
        _argmin_kernel_medium[(n_rows,)](
            x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES, num_warps=NUM_WARPS
        )
    else:
        programs_per_row = builtins.min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial_val = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        partial_idx = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.int64)
        _argmin_kernel_large[(n_rows * programs_per_row,)](
            x,
            partial_val,
            partial_idx,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

        partial_val = partial_val.reshape(n_rows, programs_per_row)
        partial_idx = partial_idx.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)

        merge_block_size = triton.next_power_of_2(programs_per_row)
        _argmin_kernel_merge[(n_rows,)](
            partial_val,
            partial_idx,
            out,
            programs_per_row,
            BLOCK_SIZE=merge_block_size,
            num_warps=NUM_WARPS,
        )
        out = out.view(-1)
    return out.view(x.shape[:-1]).to(torch.int64)
