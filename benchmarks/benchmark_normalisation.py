import torch
import torch.nn.functional as F

from kernels import layernorm

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def bytes_for(x, gamma, beta, out):
    read_bytes = (
        x.numel() * x.element_size()
        + gamma.numel() * gamma.element_size()
        + beta.numel() * beta.element_size()
    )
    write_bytes = out.numel() * out.element_size()
    return read_bytes + write_bytes


def benchmark(name, fn, ref_fn, args):
    for _ in range(WARMUP):
        fn(*args)
        ref_fn(*args)

    torch.cuda.synchronize()

    out = fn(*args)
    total_bytes = bytes_for(args[0], args[1], args[2], out)

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / ITERS
    triton_bw = total_bytes / (triton_ms / 1000) / 1e9

    # Torch
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
            )


if __name__ == "__main__":
    main()