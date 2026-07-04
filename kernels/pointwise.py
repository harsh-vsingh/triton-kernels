import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _add_kernel(
    x0_ptr, 
    x1_ptr, 
    out0_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr, 
    num_stages: tl.constexpr
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
    BLOCK_SIZE=1024, 
    num_stages: int = 3
) -> torch.Tensor:
    """
    Elementwise add.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape."
    assert x.dtype == y.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device."
    assert x.device == DEVICE and y.device == DEVICE, "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and y.is_contiguous(), "Input tensors must be contiguous."

    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return out


@triton.jit
def _add_and_relu_kernel(x0_ptr, x1_ptr, out0_ptr, n_elements, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
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
    BLOCK_SIZE:int = 1024,
    num_stages:int = 3
) -> torch.Tensor:
    """
    Elementwise add and ReLU.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape."
    assert x.dtype == y.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device."
    assert x.device == DEVICE and y.device == DEVICE, "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and y.is_contiguous(), "Input tensors must be contiguous."

    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _add_and_relu_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return out

@triton.jit
def _bias_and_gelu_kernel(
    x0_ptr,
    bias_ptr,
    out0_ptr,
    n_elements,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % hidden_dim, mask=mask)
    tmp = x + bias
    out = 0.5 * tmp * (1.0 + tl.extra.libdevice.tanh(0.7978845608028654 * (tmp + 0.044715 * tmp * tmp * tmp)))
    tl.store(out0_ptr + offsets, out, mask=mask)


def bias_and_gelu(
    x: torch.Tensor,
    bias: torch.Tensor,
    BLOCK_SIZE: int = 1024,
    num_stages: int = 3
) -> torch.Tensor:
    """
    Elementwise add bias and apply GELU activation.
    Assumes contiguous CUDA tensors of same dtype.
    """
    assert (bias.shape == x.shape) or (bias.ndim == 1 and bias.numel() == x.shape[-1]), "bias must have shape x.shape or (hidden_dim,)"
    assert x.dtype == bias.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and bias.is_cuda, "Input tensors must be on CUDA device."
    assert x.device == DEVICE and bias.device == DEVICE, "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and bias.is_contiguous(), "Input tensors must be contiguous."

    n = x.numel()
    hidden_dim = bias.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _bias_and_gelu_kernel[grid](x, bias, out, n, hidden_dim, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return out


@triton.jit
def _silu_mult_kernel(
    x0_ptr,
    mul_ptr,
    out0_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    mul = tl.load(mul_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid * mul 
    tl.store(out0_ptr + offsets, out, mask=mask)

def silu_mult(
    x: torch.Tensor,
    mul: torch.Tensor,
    BLOCK_SIZE: int = 1024,
    num_stages: int = 3
) -> torch.Tensor:
    """
    Elementwise multiply with SiLU activation.
    Assumes contiguous CUDA tensors of same shape and dtype.
    """
    assert x.shape == mul.shape, "Input tensors must have the same shape."
    assert x.dtype == mul.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and mul.is_cuda, "Input tensors must be on CUDA device."
    assert x.device == DEVICE and mul.device == DEVICE, "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and mul.is_contiguous(), "Input tensors must be contiguous."

    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _silu_mult_kernel[grid](x, mul, out, n, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
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
    num_stages: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x0_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % hidden_dim, mask=mask)
    mult = tl.load(mult_ptr + offsets, mask=mask)
    biased = x + bias
    sigmoid = 1.0 / (1.0 + tl.exp(-biased))
    out = biased * sigmoid * mult
    tl.store(out0_ptr + offsets, out, mask=mask)


def bias_silu_mult(
    x: torch.Tensor,
    bias: torch.Tensor,
    mult: torch.Tensor,
    BLOCK_SIZE: int = 1024,
    num_stages: int = 3
) -> torch.Tensor:
    """
    Elementwise add bias, apply SiLU activation, and multiply with another tensor.
    Assumes contiguous CUDA tensors of same dtype.
    """
    assert (bias.shape == x.shape) or (bias.ndim == 1 and bias.numel() == x.shape[-1]), "bias must have shape x.shape or (hidden_dim,)"
    assert x.shape == mult.shape, "Input tensors must have the same shape."
    assert x.dtype == bias.dtype == mult.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and bias.is_cuda and mult.is_cuda, "Input tensors must be on CUDA device."
    assert x.device == DEVICE and bias.device == DEVICE and mult.device == DEVICE, "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and bias.is_contiguous() and mult.is_contiguous(), "Input tensors must be contiguous."

    n = x.numel()
    hidden_dim = bias.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    out = torch.empty_like(x)
    _bias_silu_mult_kernel[grid](x, bias, mult, out, n, hidden_dim, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return out