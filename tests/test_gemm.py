import pytest
import torch

from kernels import gemm

torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

import math

BASE_TOL = {
    torch.float16:  dict(base_atol=2e-2, rtol=1e-2, eps=5e-3),
    torch.bfloat16: dict(base_atol=4e-2, rtol=2e-2, eps=1e-2),
    torch.float32:  dict(base_atol=2.5e-3, rtol=2e-2, eps=5e-3),
}

def gemm_tolerances(dtype, K):
    cfg = BASE_TOL[dtype]
    atol = cfg["base_atol"] * math.sqrt(max(K, 1)) + cfg["eps"]
    return dict(atol=atol, rtol=cfg["rtol"])

GEMM_SHAPES = [
    (1, 1, 1),
    (7, 13, 5),
    (32, 32, 32),
    (64, 64, 64),
    (127, 127, 127),
    (128, 128, 128),
    (129, 129, 129),
    (256, 256, 256),
    (511, 777, 333),
    (1000, 768, 512),
    (1024, 1024, 1024),
    (2048, 1024, 4096),
    (64, 8192, 64),
    (16, 16384, 16),
    (128, 32768, 128),
    (256, 16384, 256),
    (4096, 4096, 4096),
    (1025, 1023, 1027),
    (513, 4097, 769),
]


@pytest.mark.parametrize("shape", GEMM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm(shape, dtype):
    M, K, N = shape
    x = torch.randn((M, K), device=DEVICE, dtype=dtype)
    y = torch.randn((K, N), device=DEVICE, dtype=dtype)
    out = gemm(x, y)
    ref = torch.matmul(x, y).float()
    torch.testing.assert_close(out, ref, **gemm_tolerances(dtype, K))


@pytest.mark.parametrize("shape", GEMM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_out(shape, dtype):
    M, K, N = shape

    x = torch.randn((M, K), device=DEVICE, dtype=dtype)
    y = torch.randn((K, N), device=DEVICE, dtype=dtype)

    out = torch.empty((M, N), device=DEVICE, dtype=torch.float32)

    ret = gemm(x, y, out=out)

    assert ret.data_ptr() == out.data_ptr()

    ref = torch.matmul(x, y).float()

    torch.testing.assert_close(out, ref, **gemm_tolerances(dtype, K))