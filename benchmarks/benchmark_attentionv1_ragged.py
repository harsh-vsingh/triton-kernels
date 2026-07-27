import pytest
import torch
import torch.nn.functional as F

from kernels import flash_attention_v1

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

CONFIGS = [

    # MHA
    (4, 8, 8, 64),

    # GQA
    (4, 16, 4, 64),

    # MQA
    (4, 16, 1, 64),
]


def expand_gqa(k, v, q_heads):

    if k.shape[0] == q_heads:
        return k, v

    groups = q_heads // k.shape[0]

    return (
        k.repeat_interleave(groups, dim=0),
        v.repeat_interleave(groups, dim=0),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("B,QH,KVH,D", CONFIGS)
def test_flash_attention_ragged(
    dtype,
    causal,
    cross,
    B,
    QH,
    KVH,
    D,
):

    torch.manual_seed(0)

    q_lengths = torch.randint(
        32,
        256,
        (B,),
    )

    if cross:
        kv_lengths = torch.randint(
            32,
            256,
            (B,),
        )
    else:
        kv_lengths = q_lengths.clone()

    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)

    q_indptr[1:] = torch.cumsum(q_lengths, 0)
    kv_indptr[1:] = torch.cumsum(kv_lengths, 0)

    q_indptr = q_indptr.cuda()
    kv_indptr = kv_indptr.cuda()

    total_q = int(q_lengths.sum())
    total_kv = int(kv_lengths.sum())

    q = torch.randn(
        (total_q, QH, D),
        device=DEVICE,
        dtype=dtype,
    )

    k = torch.randn(
        (total_kv, KVH, D),
        device=DEVICE,
        dtype=dtype,
    )

    v = torch.randn(
        (total_kv, KVH, D),
        device=DEVICE,
        dtype=dtype,
    )

    actual = flash_attention_v1(
        q,
        k,
        v,
        is_causal=causal,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
    )

    expected = []

    for b in range(B):

        qs = q_indptr[b].item()
        qe = q_indptr[b + 1].item()

        ks = kv_indptr[b].item()
        ke = kv_indptr[b + 1].item()

        q_i = q[qs:qe].permute(1, 0, 2).unsqueeze(0)
        k_i = k[ks:ke].permute(1, 0, 2).unsqueeze(0)
        v_i = v[ks:ke].permute(1, 0, 2).unsqueeze(0)

        k_i, v_i = expand_gqa(k_i[0], v_i[0], QH)

        k_i = k_i.unsqueeze(0)
        v_i = v_i.unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            is_causal=causal,
        )

        expected.append(
            out.squeeze(0).permute(1, 0, 2)
        )

    expected = torch.cat(expected, dim=0)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=5e-3,
        atol=5e-3,
    )