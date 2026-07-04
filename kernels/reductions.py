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
    offset = block_start + offset_range

    mask = offset_range < x_stride
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
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
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offset_range = tl.arange(0, BLOCK_SIZE)
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + row_id, acc)

@triton.jit
def _sum_kernel_large(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr 
):
    