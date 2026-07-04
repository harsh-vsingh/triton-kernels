import torch
import torch.nn.functional as F

from kernels import (
    add,
    add_and_relu,
    bias_and_gelu,
    silu_mult,
    bias_silu_mult,
)

DEVICE = "cuda"

WARMUP = 20
ITERS = 100


def bytes_for(tensors_read, tensor_write):
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
    total_bytes = bytes_for(args, out)

    # Triton
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        fn(*args)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    elapsed_s = elapsed_ms / 1000

    triton_bw = total_bytes / elapsed_s / 1e9

    # Torch
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        ref_fn(*args)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERS
    elapsed_s = elapsed_ms / 1000

    torch_bw = total_bytes / elapsed_s / 1e9

    print(
        f"{name:18s}"
        f" Triton: {triton_bw:8.2f} GB/s"
        f" | Torch: {torch_bw:8.2f} GB/s"
    )


def main():
    shapes = [
        (1024,),
        (4096,),
        (64, 128),
        (128, 512),
        (512, 1024),
        (2048, 2048),
    ]

    for shape in shapes:
        print(f"\nShape = {shape}")
        print("-" * 70)

        x = torch.randn(shape, device=DEVICE)
        y = torch.randn(shape, device=DEVICE)
        mult = torch.randn(shape, device=DEVICE)

        # Bias for broadcast kernels
        if len(shape) == 1:
            bias = torch.randn(shape, device=DEVICE)
        else:
            bias = torch.randn(shape[-1], device=DEVICE)

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
            "silu_mul",
            silu_mult,
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