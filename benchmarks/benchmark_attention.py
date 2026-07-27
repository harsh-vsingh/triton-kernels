import torch
import torch.nn.functional as F

from kernels import flash_attention

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def bytes_attention(q, k, v, out):
    return (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + out.numel() * out.element_size()
    )


def benchmark(q, k, v, causal):

    for _ in range(WARMUP):
        flash_attention(q, k, v, is_causal=causal)
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.cuda.synchronize()

    out = flash_attention(q, k, v, is_causal=causal)
    total_bytes = bytes_attention(q, k, v, out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        flash_attention(q, k, v, is_causal=causal)
    end.record()

    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / ITERS
    triton_bw = total_bytes / (triton_ms / 1000) / 1e9

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
        )
    end.record()

    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / ITERS
    torch_bw = total_bytes / (torch_ms / 1000) / 1e9

    print(
        f"Triton: {triton_bw:8.2f} GB/s"
        f" | {triton_ms:7.3f} ms"
        f" || Torch: {torch_bw:8.2f} GB/s"
        f" | {torch_ms:7.3f} ms"
        f" | {torch_ms / triton_ms:5.2f}x"
    )


def main():

    shapes = [
        # (B, QH, KVH, M, N, D)
        (1, 1, 1, 128, 128, 64),
        (1, 1, 1, 256, 256, 64),
        (1, 1, 1, 512, 512, 64),
        (1, 1, 1, 1024, 1024, 64),
        (1, 1, 1, 2048, 2048, 64),

        (1, 8, 8, 128, 128, 64),
        (1, 8, 8, 512, 512, 64),
        (1, 8, 8, 1024, 1024, 64),

        (1, 16, 16, 512, 512, 64),
        (1, 16, 16, 1024, 1024, 64),

        (2, 8, 8, 512, 512, 64),
        (4, 8, 8, 512, 512, 64),

        # Cross attention
        (1, 8, 8, 128, 512, 64),
        (1, 8, 8, 256, 1024, 64),

        # Decode
        (1, 8, 8, 1, 512, 64),
        (1, 8, 8, 1, 1024, 64),
        (1, 8, 8, 1, 2048, 64),
        (1, 8, 8, 1, 4096, 64),

        # GQA
        (1, 16, 4, 1024, 1024, 64),

        # MQA
        (1, 16, 1, 1024, 1024, 64),
    ]

    for dtype in DTYPES:

        print("\n" + "=" * 100)
        print(f"Dtype = {dtype}")
        print("=" * 100)

        for causal in [False, True]:

            print(f"\nCausal = {causal}")
            print("=" * 100)

            for B, QH, KVH, M, N, D in shapes:

                print(
                    f"\n(B={B}, QH={QH}, KVH={KVH}, M={M}, N={N}, D={D})"
                )
                print("-" * 100)

                q = torch.randn(
                    (B, QH, M, D),
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

                if KVH != QH:
                    groups = QH // KVH
                    k_ref = k.repeat_interleave(groups, dim=1)
                    v_ref = v.repeat_interleave(groups, dim=1)
                else:
                    k_ref = k
                    v_ref = v

                benchmark(q, k_ref, v_ref, causal)


if __name__ == "__main__":
    main()