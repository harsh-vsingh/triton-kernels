import pytest
import torch
import torch.nn.functional as F

from kernels import (
    layernorm,
    residual_layernorm,
    rms_norm,
)

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

NORM_SHAPES = [
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
]

EPS = 1e-5


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    beta = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = layernorm(x, gamma, beta, EPS)

    ref = F.layer_norm(
        x,
        (shape[-1],),
        weight=gamma,
        bias=beta,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    beta = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = layernorm(x, gamma, beta, EPS, out=out)

    assert ret.data_ptr() == out.data_ptr()

    ref = F.layer_norm(
        x,
        (shape[-1],),
        weight=gamma,
        bias=beta,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = rms_norm(x, gamma, EPS)

    ref = F.rms_norm(
        x,
        (shape[-1],),
        weight=gamma,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = rms_norm(x, gamma, EPS, out=out)

    assert ret.data_ptr() == out.data_ptr()

    ref = F.rms_norm(
        x,
        (shape[-1],),
        weight=gamma,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_residual(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = rms_norm(
        x,
        gamma,
        EPS,
        residual=residual,
    )

    ref = F.rms_norm(
        x + residual,
        (shape[-1],),
        weight=gamma,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_residual_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = rms_norm(
        x,
        gamma,
        EPS,
        residual=residual,
        out=out,
    )

    assert ret.data_ptr() == out.data_ptr()

    ref = F.rms_norm(
        x + residual,
        (shape[-1],),
        weight=gamma,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_residual_layernorm(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    beta = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = residual_layernorm(x, residual, gamma, beta, EPS)

    ref = F.layer_norm(
        x + residual,
        (shape[-1],),
        weight=gamma,
        bias=beta,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_residual_layernorm_out(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(shape, device=DEVICE, dtype=dtype)
    gamma = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    beta = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

    out = torch.empty_like(x)
    ret = residual_layernorm(
        x,
        residual,
        gamma,
        beta,
        EPS,
        out=out,
    )

    assert ret.data_ptr() == out.data_ptr()

    ref = F.layer_norm(
        x + residual,
        (shape[-1],),
        weight=gamma,
        bias=beta,
        eps=EPS,
    )

    torch.testing.assert_close(out, ref)
