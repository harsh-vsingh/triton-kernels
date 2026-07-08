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

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

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
@pytest.mark.parametrize("dtype", DTYPES)
def test_add(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = add(x, y)

    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_and_relu(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = add_and_relu(x, y)

    torch.testing.assert_close(out, torch.relu(x + y))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_elementwise(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_broadcast(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_mult(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = silu_mult(x, mult)

    ref = F.silu(x) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_elementwise(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_broadcast(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)