import pytest
import torch
import torch.nn.functional as F

from kernels import (
    add,
    add_and_relu,
    bias_and_gelu,
    bias_silu_mult,
    swiglu,
)

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

POINTWISE_SHAPES = [
    (1,),
    (7,),
    (1023,),
    (1024,),
    (1234,),
    (4096,),
    (17, 333),
    (31, 1025),
    (64, 128),
    (128, 512),
    (4, 8, 128),
    (2, 4, 16, 64),
    (3, 10000),
]

BROADCAST_SHAPES = [
    (8, 128),
    (32, 256),
    (64, 512),
    (4, 8, 128),
    (2, 4, 16, 64),
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
def test_add_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = add(x, y, out=out)

    assert ret.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = x + y
    ret = add(x, y, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_and_relu(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = add_and_relu(x, y)

    torch.testing.assert_close(out, torch.relu(x + y))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_and_relu_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = add_and_relu(x, y, out=out)

    assert ret.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.relu(x + y))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_and_relu_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = torch.relu(x + y)
    ret = add_and_relu(x, y, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_elementwise(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_elementwise_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = bias_and_gelu(x, bias, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_elementwise_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.gelu(x + bias, approximate="tanh")
    ret = bias_and_gelu(x, bias, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_broadcast(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = bias_and_gelu(x, bias)

    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_broadcast_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = bias_and_gelu(x, bias, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = F.gelu(x + bias, approximate="tanh")
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_and_gelu_broadcast_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    ref = F.gelu(x + bias, approximate="tanh")
    ret = bias_and_gelu(x, bias, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = swiglu(x, mult)

    ref = F.silu(x) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = swiglu(x, mult, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = F.silu(x) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.silu(x) * mult
    ret = swiglu(x, mult, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_elementwise(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_elementwise_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = bias_silu_mult(x, bias, mult, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_elementwise_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape, device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.silu(x + bias) * mult
    ret = bias_silu_mult(x, bias, mult, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_broadcast(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = bias_silu_mult(x, bias, mult)

    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_broadcast_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = bias_silu_mult(x, bias, mult, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = F.silu(x + bias) * mult
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_silu_mult_broadcast_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    mult = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.silu(x + bias) * mult
    ret = bias_silu_mult(x, bias, mult, out=x)

    assert ret.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)
