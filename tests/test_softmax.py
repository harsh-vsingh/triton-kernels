import pytest
import torch
import torch.nn.functional as F

from kernels import (
    softmax,
    log_softmax,
)

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

SHAPES = [
    # Small
    (1024,),
    (512,),
    (32, 1024),

    # Medium
    (4096,),
    (64, 2048),
    (128, 4096),
    (16, 1025),
    (8, 3072),

    # Large
    (1, 32768),
    (2, 32768),
    (4, 20000),
    (8, 32768),
    (1, 65536),
    (1, 131072),

    # Higher dimensional
    (8, 16, 1024),
    (4, 8, 32, 1024),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = softmax(x)
    ref = F.softmax(x, dim=-1)

    torch.testing.assert_close(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    returned = softmax(x, out=out)

    ref = F.softmax(x, dim=-1)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.softmax(x, dim=-1)

    returned = softmax(x, out=x)

    assert returned.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_softmax(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = log_softmax(x)
    ref = F.log_softmax(x, dim=-1)

    torch.testing.assert_close(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_softmax_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    returned = log_softmax(x, out=out)

    ref = F.log_softmax(x, dim=-1)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_softmax_inplace(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    ref = F.log_softmax(x, dim=-1)

    returned = log_softmax(x, out=x)

    assert returned.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)