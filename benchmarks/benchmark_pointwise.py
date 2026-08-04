import torch
import torch.nn.functional as F

from kernels import (
    add,
    add_and_relu,
    bias_and_gelu,
    bias_silu_mult,
    swiglu,
)

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def effective_bandwidth(tensors_read, tensor_write):
    read_bytes = sum(t.numel() * t.element_size() for t in tensors_read)
    write_bytes = tensor_write.numel() * tensor_write.element_size()
    return read_bytes + write_bytes


def benchmark(name, fn, ref_fn, args):
    # Warmup
    for _ in range(WARMUP):
        fn(*args)
        ref_fn(*args)

    torch.cuda.synchronize()

    out = fn(*args)
    total_bytes = effective_bandwidth(args, out)

    # ---------------- Triton ---------------- #

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()

    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / ITERS
    triton_s = triton_ms / 1000
    triton_bw = total_bytes / triton_s / 1e9

    # ---------------- Torch ---------------- #

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        ref_fn(*args)
    end.record()

    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / ITERS
    torch_s = torch_ms / 1000
    torch_bw = total_bytes / torch_s / 1e9

    speedup = torch_ms / triton_ms

    print(
        f"{name:18s}"
        f" Triton: {triton_bw:8.2f} GB/s ({triton_ms:7.4f} ms)"
        f" | Torch: {torch_bw:8.2f} GB/s ({torch_ms:7.4f} ms)"
        f" | Speedup: {speedup:5.2f}x"
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

    for dtype in DTYPES:
        print("\n" + "=" * 90)
        print(f"Dtype = {dtype}")
        print("=" * 90)

        for shape in shapes:
            print(f"\nShape = {shape}")
            print("-" * 90)

            x = torch.randn(shape, device=DEVICE, dtype=dtype)
            y = torch.randn(shape, device=DEVICE, dtype=dtype)
            mult = torch.randn(shape, device=DEVICE, dtype=dtype)

            if len(shape) == 1:
                bias = torch.randn(shape, device=DEVICE, dtype=dtype)
            else:
                bias = torch.randn(shape[-1], device=DEVICE, dtype=dtype)

            benchmark(
                "add",
                add,
                lambda a, b: a + b,
                (x, y),
            )

            benchmark(
                "add_relu",
                add_and_relu,
                lambda a, b: torch.relu(a + b),
                (x, y),
            )

            benchmark(
                "bias_gelu",
                bias_and_gelu,
                lambda a, b: F.gelu(a + b, approximate="tanh"),
                (x, bias),
            )

            benchmark(
                "swiglu",
                swiglu,
                lambda a, b: F.silu(a) * b,
                (x, mult),
            )

            benchmark(
                "bias_silu_mul",
                bias_silu_mult,
                lambda a, b, c: F.silu(a + b) * c,
                (x, bias, mult),
            )


if __name__ == "__main__":
    main()
