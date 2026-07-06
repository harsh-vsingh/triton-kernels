import torch
import triton
import triton.language as tl
from utils import validate_reduction

# Reduction kernels use empirically determined dispatch heuristics.
# Small reductions are performed within a single program. Larger reductions
# stream through the row using one program when sufficient row-level
# parallelism exists to occupy the GPU. Multi-program reductions are used only
# when additional parallelism is required, followed by a second-stage
# reduction. These heuristics were selected from benchmarking rather than
# autotuning.

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

SMALL_THRESHOLD = 1024
MEDIUM_THRESHOLD = 8192
BLOCK_SIZE = 1024
NUM_STAGES = 2
MAX_PROGRAMS_PER_ROW = 8

@triton.jit
def _sum_kernel_small(
    x_ptr,
    out_ptr,
    x_stride,
    BLOCK_SIZE: tl.constexpr  
):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, x_sum)

@triton.jit
def _sum_kernel_medium(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    acc = tl.zeros((), dtype=tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + row_id, acc)

@triton.jit
def _sum_kernel_large(
    x_ptr,
    out_buffer,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    acc = tl.zeros((), dtype=tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out_buffer + pid, acc)

def sum(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _sum_kernel_small[(n_rows,)](x, out, n_cols, BLOCK_SIZE=small_block_size)
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= NUM_SMS:
        _sum_kernel_medium[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)
    else:
        programs_per_row = min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        
        _sum_kernel_large[(n_rows * programs_per_row,)](x, partial, n_cols, programs_per_row, chunks_per_row, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)
        partial = partial.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)
        _sum_kernel_small[(n_rows,)](partial, out, programs_per_row, BLOCK_SIZE=small_block_size)
    return out.view(x.shape[:-1])

    
def mean(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean of over last dimension in the input tensor
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    total_sum = sum(x)
    return total_sum / x.shape[-1]


@triton.jit
def _max_kernel_small(
    x_ptr,
    out_ptr,
    x_stride,
    BLOCK_SIZE: tl.constexpr  
):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x_max = tl.max(x, axis=0)
    tl.store(out_ptr + pid, x_max)

@triton.jit
def _max_kernel_medium(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    acc = tl.full((), float("-inf"), x_ptr.element_type())
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        acc = tl.maximum(acc, tl.max(x, axis=0))
    tl.store(out_ptr + row_id, acc)

@triton.jit
def _max_kernel_large(
    x_ptr,
    out_buffer,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    acc = tl.full((), float("-inf"), x_ptr.element_type())
    offset_range = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        offsets = block_start + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))    
        acc = tl.maximum(acc, tl.max(x, axis=0))
    tl.store(out_buffer + pid, acc)

def max(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the maximum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a float32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _max_kernel_small[(n_rows,)](x, out, n_cols, BLOCK_SIZE=small_block_size)
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= NUM_SMS:
        _max_kernel_medium[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)
    else:
        programs_per_row = min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        
        _max_kernel_large[(n_rows * programs_per_row,)](x, partial, n_cols, programs_per_row, chunks_per_row, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)
        partial = partial.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)
        _max_kernel_small[(n_rows,)](partial, out, programs_per_row, BLOCK_SIZE=small_block_size)
    return out.view(x.shape[:-1])


@triton.jit
def _argmax_kernel_small(
    x_ptr,
    out_ptr,
    x_stride,
    BLOCK_SIZE: tl.constexpr  
):
    pid = tl.program_id(0)
    block_start = pid * x_stride
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x_max = tl.argmax(x, axis=0)
    tl.store(out_ptr + pid, x_max)

@triton.jit
def _argmax_kernel_medium(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    row_id = tl.program_id(0)
    row_start = n_cols * row_id
    acc = tl.full((), float("-inf"), x_ptr.element_type())
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
    tl.store(out_ptr + row_id, acc_idx)

@triton.jit
def _argmax_kernel_large(
    x_ptr,
    out_buffer_val,
    out_buffer_idx,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols
    acc = tl.full((), float("-inf"), x_ptr.element_type())
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
    tl.store(out_buffer_idx + pid, acc_idx)
    tl.store(out_buffer_val + pid, acc)

def argmax(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the maximum over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    Returns a int32 tensor.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.int32, device=x.device)

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)
        _argmax_kernel_small[(n_rows,)](x, out, n_cols, BLOCK_SIZE=small_block_size)
    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= NUM_SMS:
        _argmax_kernel_medium[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)
    else:
        programs_per_row = min(triton.cdiv(n_cols, BLOCK_SIZE), MAX_PROGRAMS_PER_ROW)
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        partial_val = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        partial_idx = torch.empty(n_rows * programs_per_row, device=x.device, dtype=torch.int32)
        _argmax_kernel_large[(n_rows * programs_per_row,)](x, partial_val, partial_idx, n_cols, programs_per_row, chunks_per_row, BLOCK_SIZE=BLOCK_SIZE, num_stages=NUM_STAGES)

        partial_val = partial_val.reshape(n_rows, programs_per_row)
        partial_idx = partial_idx.reshape(n_rows, programs_per_row)
        small_block_size = triton.next_power_of_2(programs_per_row)

        winner = torch.empty((n_rows,), dtype=torch.int32, device=x.device)
        _argmax_kernel_small[(n_rows,)](partial_val, winner, programs_per_row, BLOCK_SIZE=small_block_size)
        out = partial_idx.gather(1, winner.unsqueeze(1)).squeeze(1)
    return out.view(x.shape[:-1])