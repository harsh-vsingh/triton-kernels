import torch
import torch.nn.functional as F

from kernels import (
    log_softmax,
    softmax,
)

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def bytes_softmax(x, out):
    return x.numel() * x.element_size() + out.numel() * out.element_size()


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
        (1, 32768),
        (2, 32768),
        (8, 32768),
        (16, 32768),
        (1, 65536),
        (2, 65536),
        (8, 65536),
        (1, 131072),
    ]

    for dtype in DTYPES:
        print("\n" + "=" * 90)
        print(f"Dtype = {dtype}")
        print("=" * 90)

        for shape in shapes:
            print(f"\nShape = {shape}")
            print("-" * 90)

            x = torch.randn(shape, device=DEVICE, dtype=dtype)

            benchmark(
                "softmax",
                lambda x: softmax(x),
                lambda x: F.softmax(x, dim=-1),
                (x,),
                bytes_softmax,
            )

            benchmark(
                "log_softmax",
                lambda x: log_softmax(x),
                lambda x: F.log_softmax(x, dim=-1),
                (x,),
                bytes_softmax,
            )


if __name__ == "__main__":
    main()
