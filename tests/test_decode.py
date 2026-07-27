import pytest
import torch
import torch.nn.functional as F

from kernels import decode_attention

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
]

CONFIGS = [
    (4, 8, 8, 128, 64),   # MHA
    (4, 8, 2, 128, 64),   # GQA
    (4, 16, 1, 257, 64),  # MQA
]


def expand_gqa(k, v, q_heads):
    kv_heads = k.shape[1]
    if kv_heads == q_heads:
        return k, v

    repeat = q_heads // kv_heads
    return (
        k.repeat_interleave(repeat, dim=1),
        v.repeat_interleave(repeat, dim=1),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,QH,KVH,N,D", CONFIGS)
def test_decode_dense(
    dtype,
    B,
    QH,
    KVH,
    N,
    D,
):
    torch.manual_seed(0)

    q = torch.randn(
        (B, QH, D),
        device=DEVICE,
        dtype=dtype,
    )

    k = torch.randn(
        (B, KVH, N, D),
        device=DEVICE,
        dtype=dtype,
    )

    v = torch.randn(
        (B, KVH, N, D),
        device=DEVICE,
        dtype=dtype,
    )

    actual = decode_attention(
        q,
        k,
        v,
    )

    k_ref, v_ref = expand_gqa(k, v, QH)

    expected = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k_ref,
        v_ref,
        is_causal=False,
    ).squeeze(2)

    atol = rtol = 5e-3
    if dtype == torch.bfloat16:
        atol = rtol = 2e-2

    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )