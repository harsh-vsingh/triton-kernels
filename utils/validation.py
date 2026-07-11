import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def validate_binary(
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    assert x.shape == y.shape, "Input tensors must have the same shape."
    assert x.dtype == y.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and y.is_cuda, "Input tensors must be CUDA tensors."
    assert x.device == DEVICE and y.device == DEVICE, \
        "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and y.is_contiguous(), \
        "Input tensors must be contiguous."


def validate_bias(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    assert (
        bias.shape == x.shape
        or (bias.ndim == 1 and bias.numel() == x.shape[-1])
    ), "bias must have shape x.shape or (hidden_dim,)"

    assert x.dtype == bias.dtype, "Input tensors must have the same dtype."
    assert x.is_cuda and bias.is_cuda, "Input tensors must be CUDA tensors."
    assert x.device == DEVICE and bias.device == DEVICE, \
        "Input tensors must be on the active Triton device."
    assert x.is_contiguous() and bias.is_contiguous(), \
        "Input tensors must be contiguous."


def validate_bias_binary(
    x: torch.Tensor,
    bias: torch.Tensor,
    other: torch.Tensor,
) -> None:
    validate_bias(x, bias)

    assert x.shape == other.shape, \
        "Input tensors must have the same shape."
    assert x.dtype == other.dtype, \
        "Input tensors must have the same dtype."
    assert other.is_cuda, "Input tensors must be CUDA tensors."
    assert other.device == DEVICE, \
        "Input tensors must be on the active Triton device."
    assert other.is_contiguous(), \
        "Input tensors must be contiguous."
    
def validate_reduction(x: torch.Tensor) -> None:
    assert x.is_cuda, "Input tensor must be a CUDA tensor."
    assert x.device == DEVICE, \
        "Input tensor must be on the active Triton device."
    assert x.ndim >= 1, \
        "Input tensor must have at least one dimension."
    assert x.is_contiguous(), \
        "Input tensor must be contiguous."
    assert x.numel() > 0, \
        "Input tensor must not be empty."
    assert x.shape[-1] > 0, \
        "Reduction dimension must be non-empty."
    assert x.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ), \
        "Input tensor must have a floating-point dtype."
    
def validate_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor | None = None,
) -> None:
    assert x.is_cuda, \
        "Input tensor must be a CUDA tensor."
    assert x.device == DEVICE, \
        "Input tensor must be on the active Triton device."
    assert x.ndim >= 1, \
        "Input tensor must have at least one dimension."
    assert x.is_contiguous(), \
        "Input tensor must be contiguous."
    assert x.numel() > 0, \
        "Input tensor must not be empty."
    assert x.shape[-1] > 0, \
        "Normalization dimension must be non-empty."
    assert x.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ), \
        "Input tensor must have a floating-point dtype."

    assert gamma.is_cuda, \
        "Gamma must be a CUDA tensor."
    assert gamma.device == x.device, \
        "Gamma must be on the same device as the input tensor."
    assert gamma.ndim == 1, \
        "Gamma must be a 1D tensor."
    assert gamma.is_contiguous(), \
        "Gamma must be contiguous."
    assert gamma.shape[0] == x.shape[-1], \
        "Gamma must have shape (x.shape[-1],)."
    assert gamma.dtype == x.dtype, \
        "Gamma must have the same dtype as the input tensor."

    assert beta.is_cuda, \
        "Beta must be a CUDA tensor."
    assert beta.device == x.device, \
        "Beta must be on the same device as the input tensor."
    assert beta.ndim == 1, \
        "Beta must be a 1D tensor."
    assert beta.is_contiguous(), \
        "Beta must be contiguous."
    assert beta.shape[0] == x.shape[-1], \
        "Beta must have shape (x.shape[-1],)."
    assert beta.dtype == x.dtype, \
        "Beta must have the same dtype as the input tensor."

    if out is not None:
        assert out.is_cuda, \
            "Output tensor must be a CUDA tensor."
        assert out.device == x.device, \
            "Output tensor must be on the same device as the input tensor."
        assert out.shape == x.shape, \
            "Output tensor must have the same shape as the input tensor."
        assert out.dtype == x.dtype, \
            "Output tensor must have the same dtype as the input tensor."
        assert out.is_contiguous(), \
            "Output tensor must be contiguous."
        
def validate_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    out: torch.Tensor | None = None,
) -> None:
    assert x.is_cuda, \
        "Input tensor must be a CUDA tensor."
    assert x.device == DEVICE, \
        "Input tensor must be on the active Triton device."
    assert x.ndim >= 1, \
        "Input tensor must have at least one dimension."
    assert x.is_contiguous(), \
        "Input tensor must be contiguous."
    assert x.numel() > 0, \
        "Input tensor must not be empty."
    assert x.shape[-1] > 0, \
        "Normalization dimension must be non-empty."
    assert x.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ), \
        "Input tensor must have a floating-point dtype."

    assert gamma.is_cuda, \
        "Gamma must be a CUDA tensor."
    assert gamma.device == x.device, \
        "Gamma must be on the same device as the input tensor."
    assert gamma.ndim == 1, \
        "Gamma must be a 1D tensor."
    assert gamma.is_contiguous(), \
        "Gamma must be contiguous."
    assert gamma.shape[0] == x.shape[-1], \
        "Gamma must have shape (x.shape[-1],)."
    assert gamma.dtype == x.dtype, \
        "Gamma must have the same dtype as the input tensor."

    if out is not None:
        assert out.is_cuda, \
            "Output tensor must be a CUDA tensor."
        assert out.device == x.device, \
            "Output tensor must be on the same device as the input tensor."
        assert out.shape == x.shape, \
            "Output tensor must have the same shape as the input tensor."
        assert out.dtype == x.dtype, \
            "Output tensor must have the same dtype as the input tensor."
        assert out.is_contiguous(), \
            "Output tensor must be contiguous."