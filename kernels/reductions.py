import torch
import triton
import triton.language as tl

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
    Computes the sum of all elements in the input tensor
    Assumes contiguous CUDA tensor of any shape and dtype.
    """
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be a contiguous CUDA tensor"

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    if n_cols <= 1024:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        _sum_kernel_small[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    elif n_cols <= 8192:
        _sum_kernel_medium[(n_rows,)](x, out, n_cols, BLOCK_SIZE=1024, num_stages=2)
    else:
        programs_per_row = min(triton.cdiv(n_cols, 1024), 32)
        chunks_per_row = triton.cdiv(n_cols, 1024)
        partial = torch.zeros(n_rows * programs_per_row, device=x.device, dtype=torch.float32)
        
        _sum_kernel_large[(n_rows * programs_per_row,)](
            x, partial, n_cols, programs_per_row, chunks_per_row,
            BLOCK_SIZE=1024, num_stages=2
        )
        partial = partial.reshape(n_rows, programs_per_row)
        BLOCK_SIZE = triton.next_power_of_2(programs_per_row)
        _sum_kernel_small[(n_rows,)](
            partial, out, programs_per_row, BLOCK_SIZE=BLOCK_SIZE
        )
    return out
