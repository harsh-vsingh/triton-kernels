import triton
import triton.language as tl
import torch
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _softmax_kernel(input_ptr, 
                    output_ptr, 
                    row_stride, 
                    n_rows, 
                    n_cols, 
                    BLOCK_SIZE: tl.constexpr, 
                    num_stages: tl.constexpr
                    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        row_ptrs = row_start_ptr + offsets

        mask = offsets < n_cols
        row = tl.load(row_ptrs, mask=mask, other=float('-inf'))

        row_max = tl.max(row, axis=0)
        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        out = numerator / denominator

        out_start_ptr = output_ptr + row_idx * row_stride
        out_ptrs = out_start_ptr + offsets
        tl.store(out_ptrs, out, mask=mask)

def softmax(x:torch.tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )

    out = torch.empty_like(x)
    _softmax_kernel[grid](x, out, x.stride(0), n_rows, n_cols, BLOCK_SIZE, 2)
    return out
