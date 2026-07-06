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
    assert x.is_cuda, "Input tensor must be a CUDA tensor"
    assert x.ndim >= 1, "Input tensor must have at least one dimension"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.device == DEVICE, "Input tensor must be on the active Triton device"