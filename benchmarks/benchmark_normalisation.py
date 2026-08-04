import torch
import torch.nn.functional as F

from kernels import layernorm, residual_layernorm, rms_norm

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def bytes_layernorm(x, gamma, beta, out):
    return (
        x.numel() * x.element_size()
        + gamma.numel() * gamma.element_size()
        + beta.numel() * beta.element_size()
        + out.numel() * out.element_size()
    )


def bytes_rmsnorm(x, gamma, out):
    return (
        x.numel() * x.element_size()
        + gamma.numel() * gamma.element_size()
        + out.numel() * out.element_size()
    )


def bytes_residual_layernorm(x, residual, gamma, beta, out):
    return (
        x.numel() * x.element_size()
        + residual.numel() * residual.element_size()
        + gamma.numel() * gamma.element_size()
        + beta.numel() * beta.element_size()
        + out.numel() * out.element_size()
    )


def bytes_residual_rmsnorm(x, residual, gamma, out):
    return (
        x.numel() * x.element_size()
        + residual.numel() * residual.element_size()
        + gamma.numel() * gamma.element_size()
        + out.numel() * out.element_size()
    )


def benchmark(name, fn, ref_fn, args, bytes_fn):
    for _ in range(WARMUP):
        fn(*args)
        ref_fn(*args)

    torch.cuda.synchronize()

    out = fn(*args)
    total_bytes = bytes_fn(*args, out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / ITERS
    triton_bw = total_bytes / (triton_ms / 1000) / 1e9

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        ref_fn(*args)
    end.record()
    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / ITERS
    torch_bw = total_bytes / (torch_ms / 1000) / 1e9

    print(
        f"{name:12s}"
        f" Triton: {triton_bw:8.2f} GB/s"
        f" | Torch: {torch_bw:8.2f} GB/s"
        f" | {torch_ms / triton_ms:5.2f}x"
    )


def main():

    shapes = [
        (1024,),
        (4096,),
        (64, 128),
        (128, 512),
        (512, 1024),
        (2048, 2048),
        (4096, 4096),
        # Large kernel
        (1, 32768),
        (2, 32768),
        (8, 32768),
        (16, 32768),
        (1, 65536),
        (2, 65536),
        (8, 65536),
        (1, 131072),
    ]

    eps = 1e-5

    for dtype in DTYPES:
        print("\n" + "=" * 90)
        print(f"Dtype = {dtype}")
        print("=" * 90)

        for shape in shapes:
            print(f"\nShape = {shape}")
            print("-" * 90)

            x = torch.randn(shape, device=DEVICE, dtype=dtype)

            hidden_dim = shape[-1]
            gamma = torch.randn(hidden_dim, device=DEVICE, dtype=dtype)
            beta = torch.randn(hidden_dim, device=DEVICE, dtype=dtype)
            residual = torch.randn_like(x)

            benchmark(
                "layernorm",
                lambda x, g, b: layernorm(x, g, b, eps),
                lambda x, g, b: F.layer_norm(
                    x,
                    (hidden_dim,),
                    weight=g,
                    bias=b,
                    eps=eps,
                ),
                (x, gamma, beta),
                bytes_layernorm,
            )

            benchmark(
                "rmsnorm",
                lambda x, g: rms_norm(x, g, eps),
                lambda x, g: F.rms_norm(
                    x,
                    (hidden_dim,),
                    weight=g,
                    eps=eps,
                ),
                (x, gamma),
                bytes_rmsnorm,
            )

            benchmark(
                "residual rmsnorm",
                lambda x, r, g: rms_norm(x, g, eps, residual=r),
                lambda x, r, g: F.rms_norm(
                    x + r,
                    (hidden_dim,),
                    weight=g,
                    eps=eps,
                ),
                (x, residual, gamma),
                bytes_residual_rmsnorm,
            )

            benchmark(
                "res+layernorm",
                lambda x, r, g, b: residual_layernorm(x, r, g, b, eps),
                lambda x, r, g, b: F.layer_norm(
                    x + r,
                    (hidden_dim,),
                    weight=g,
                    bias=b,
                    eps=eps,
                ),
                (x, residual, gamma, beta),
                bytes_residual_layernorm,
            )


if __name__ == "__main__":
    main()
