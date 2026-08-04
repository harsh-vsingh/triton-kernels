import pytest
import torch

from kernels import (
    argmax,
    argmin,
    max,
    mean,
    min,
    sum,
)

DEVICE = "cuda"

REDUCTION_SHAPES = [
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
    (32, 2048),
    (4, 8, 128),
    (2, 4, 16, 64),
    (4, 10000),
]

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sum(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = sum(x)
    ref = torch.sum(x, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=torch.float32)
    ret = sum(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.sum(x, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mean(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = mean(x)
    ref = torch.mean(x, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=torch.float32)
    ret = mean(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.mean(x, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_max(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = max(x)
    ref = torch.max(x, dim=-1).values

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=dtype)
    ret = max(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.max(x, dim=-1).values

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_min(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = min(x)
    ref = torch.min(x, dim=-1).values

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_min_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=dtype)
    ret = min(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.min(x, dim=-1).values

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = argmax(x)
    ref = torch.argmax(x, dim=-1)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=torch.int64)
    ret = argmax(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.argmax(x, dim=-1)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = argmin(x)
    ref = torch.argmin(x, dim=-1)

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty(x.shape[:-1], device=DEVICE, dtype=torch.int64)
    ret = argmin(x, out=out)

    assert ret.data_ptr() == out.data_ptr()
    ref = torch.argmin(x, dim=-1)

    torch.testing.assert_close(out, ref)
