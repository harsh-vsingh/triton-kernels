@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("B,QH,KVH,D", [
    (4, 8, 8, 64),
    (4, 8, 2, 64),
    (4, 16, 1, 64),
])
def test_decode_ragged(
    dtype,
    cross,
    B,
    QH,
    KVH,
    D,
):
    torch.manual_seed(0)

    kv_lengths = torch.randint(
        32,
        256,
        (B,),
    )

    q_lengths = torch.ones(B, dtype=torch.int32)

    if not cross:
        kv_lengths = kv_lengths.clone()

    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)

    q_indptr[1:] = torch.arange(1, B + 1)
    kv_indptr[1:] = torch.cumsum(kv_lengths, 0)

    q_indptr = q_indptr.cuda()
    kv_indptr = kv_indptr.cuda()

    total_q = B
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

    actual = decode_attention(
        q,
        k,
        v,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
    )

    expected = []

    for b in range(B):

        ks = kv_indptr[b].item()
        ke = kv_indptr[b + 1].item()

        q_i = q[b].unsqueeze(0).unsqueeze(2)

        k_i = k[ks:ke].permute(1, 0, 2).unsqueeze(0)
        v_i = v[ks:ke].permute(1, 0, 2).unsqueeze(0)

        k_i, v_i = expand_gqa(k_i[0], v_i[0], QH)

        out = F.scaled_dot_product_attention(
            q_i,
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            is_causal=False,
        )

        expected.append(out.squeeze(0).squeeze(1))

    expected = torch.stack(expected)

    atol = rtol = 5e-3
    if dtype == torch.bfloat16:
        atol = rtol = 2e-2

    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )