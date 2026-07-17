import torch
import triton
import triton.language as tl
from utils import pointwise_launch_config, validate_binary, validate_bias

@triton.jit
def _add_kernel(
    x0_ptr, 
    x1_ptr, 
    out0_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask = mask)
    y = tl.load(x1_ptr + offsets, mask = mask)
    
    output = x + y
    tl.store(out0_ptr + offsets, output, mask = mask)

def add(
    x: torch.Tensor, 
    y: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Elementwise add.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    validate_binary(x, y)
    n = x.numel()
    block_size, num_warps = pointwise_launch_config(n)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) if out is None else out   
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


@triton.jit
def _add_and_relu_kernel(
    x0_ptr, 
    x1_ptr, 
    out0_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask = mask)
    y = tl.load(x1_ptr + offsets, mask = mask)

    out = tl.maximum(x + y, 0.0)
    tl.store(out0_ptr + offsets, out, mask = mask)

def add_and_relu(
    x:torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Elementwise add and ReLU.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    validate_binary(x, y)
    n = x.numel()
    block_size, num_warps = pointwise_launch_config(n)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) if out is None else out
    _add_and_relu_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


@triton.jit
def _bias_and_gelu_kernel(
    x0_ptr,
    bias_ptr,
    out0_ptr,
    n_elements,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % hidden_dim, mask=mask)
    tmp = (x + bias).to(tl.float32)
    out = 0.5 * tmp * (1.0 + tl.extra.libdevice.tanh(0.7978845608028654 * (tmp + 0.044715 * tmp * tmp * tmp)))
    tl.store(out0_ptr + offsets, out, mask=mask)


def bias_and_gelu(
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Elementwise add bias and apply GELU activation.
    Assumes contiguous CUDA tensors of same dtype.
    """
    validate_bias(x, bias)
    n = x.numel()
    block_size, num_warps = pointwise_launch_config(n)
    hidden_dim = bias.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) if out is None else out
    _bias_and_gelu_kernel[grid](x, bias, out, n, hidden_dim, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


@triton.jit
def _silu_mult_kernel(
    x0_ptr,
    mul_ptr,
    out0_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask).to(tl.float32)
    mul = tl.load(mul_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid * mul 
    tl.store(out0_ptr + offsets, out, mask=mask)

def silu_mult(
    x: torch.Tensor,
    mul: torch.Tensor,
    out : torch.Tensor = None
) -> torch.Tensor:
    """
    Elementwise multiply with SiLU activation.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    validate_binary(x, mul)
    n = x.numel()
    block_size, num_warps = pointwise_launch_config(n)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) if out is None else out
    _silu_mult_kernel[grid](x, mul, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


@triton.jit
def _bias_silu_mult_kernel(
    x0_ptr,
    bias_ptr,
    mult_ptr,
    out0_ptr,
    n_elements,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % hidden_dim, mask=mask)
    mult = tl.load(mult_ptr + offsets, mask=mask)
    biased = (x + bias).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-biased))
    out = biased * sigmoid * mult
    tl.store(out0_ptr + offsets, out, mask=mask)


def bias_silu_mult(
    x: torch.Tensor,
    bias: torch.Tensor,
    mult: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Elementwise add bias, apply SiLU activation, and multiply with another tensor.
    Assumes contiguous CUDA tensors of same dtype.
    """
    validate_bias(x, bias)
    validate_binary(x, mult)
    n = x.numel()
    block_size, num_warps = pointwise_launch_config(n)
    hidden_dim = bias.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) if out is None else out
    _bias_silu_mult_kernel[grid](x, bias, mult, out, n, hidden_dim, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out