import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _add_kernel(x0_ptr, x1_ptr, out0_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    y = tl.load(x1_ptr + offsets, mask=mask)
    
    output = x + y
    tl.store(out0_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE)
    return out