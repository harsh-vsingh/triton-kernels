import pytest
import torch
import torch.nn.functional as F

from kernels import (
    add,
    add_and_relu,
    bias_and_gelu,
    silu_mult,
    bias_silu_mult,
)

DEVICE = "cuda"


POINTWISE_SHAPES = [
    (1024,),
    (4096,),
    (64, 128),
    (128, 512),
]

BROADCAST_SHAPES = [
    (8, 128),
    (32, 256),
    (64, 512),
]


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_add(shape):
    x = torch.randn(shape, device=DEVICE)
    y = torch.randn(shape, device=DEVICE)

    out = add(x, y)

    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_add_and_relu(shape):
    x = torch.randn(shape, device=DEVICE)
    y = torch.randn(shape, device=DEVICE)

    out = add_and_relu(x, y)

    torch.testing.assert_close(out, torch.relu(x + y))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_bias_and_gelu_elementwise(shape):
    x = torch.randn(shape, device=DEVICE)
    bias = torch.randn(shape, device=DEVICE)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
def test_bias_and_gelu_broadcast(shape):
    x = torch.randn(shape, device=DEVICE)
    bias = torch.randn(shape[-1], device=DEVICE)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_silu_mult(shape):
    x = torch.randn(shape, device=DEVICE)
    mult = torch.randn(shape, device=DEVICE)

    out = silu_mult(x, mult)

    ref = F.silu(x) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
def test_bias_silu_mult_elementwise(shape):
    x = torch.randn(shape, device=DEVICE)
    bias = torch.randn(shape, device=DEVICE)
    mult = torch.randn(shape, device=DEVICE)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
def test_bias_silu_mult_broadcast(shape):
    x = torch.randn(shape, device=DEVICE)
    bias = torch.randn(shape[-1], device=DEVICE)
    mult = torch.randn(shape, device=DEVICE)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)