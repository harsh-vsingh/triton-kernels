import pytest
import torch

from kernels import (
    sum,
    mean,
    max,
    min,
    argmax,
    argmin,
)

DEVICE = "cuda"

REDUCTION_SHAPES = [
    (1024,),
    (4096,),
    (64, 128),
    (128, 512),
    (32, 2048),
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

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mean(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = mean(x)
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
def test_min(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = min(x)
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
def test_argmin(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = argmin(x)
    ref = torch.argmin(x, dim=-1)

    torch.testing.assert_close(out, ref)