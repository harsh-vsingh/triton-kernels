import torch
import triton
from kernels.pointwise import _add_kernel

"""
Evaluate Triton autotuning for pointwise kernels.
Several BLOCK_SIZE and num_warps configurations were benchmarked over a wide
range of tensor sizes.

Conclusion
----------
Pointwise kernels are memory-bandwidth bound, and autotuning produced no
consistent performance improvement across tested GPUs. Differences between
configurations were generally within measurement noise while autotuning added
compile-time and tuning overhead.

The library therefore uses a fixed launch heuristic instead of runtime
autotuning.
"""

DEVICE = "cuda"
WARMUP = 20
ITERS = 100

CONFIGS = [
    (128, 2),
    (256, 2),
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
]


def benchmark_fn(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bandwidth(x, y, ms):
    bytes_moved = (
        x.numel() * x.element_size()
        + y.numel() * y.element_size()
        + x.numel() * x.element_size()
    )
    return bytes_moved / (ms / 1000) / 1e9


def run_config(x, y, block_size, num_warps):
    out = torch.empty_like(x)
    n = x.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return out


def main():
    print("=== Pointwise launch configuration experiment ===")

    sizes = [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        262144,
        1048576,
    ]

    for n in sizes:
        print(f"\nSize = {n}")
        print("-" * 45)

        x = torch.randn(n, device=DEVICE)
        y = torch.randn_like(x)

        best_bw = 0.0
        best_cfg = None

        for block_size, num_warps in CONFIGS:
            ms = benchmark_fn(
                run_config,
                (x, y, block_size, num_warps),
            )

            gbps = bandwidth(x, y, ms)

            print(
                f"BLOCK_SIZE={block_size:4d} "
                f"WARPS={num_warps:<2d} "
                f"{gbps:7.2f} GB/s"
            )

            if gbps > best_bw:
                best_bw = gbps
                best_cfg = (block_size, num_warps)

        print(
            f"BEST -> BLOCK_SIZE={best_cfg[0]} "
            f"WARPS={best_cfg[1]} "
            f"({best_bw:.2f} GB/s)"
        )


if __name__ == "__main__":
    main()