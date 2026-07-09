import torch
import triton

from kernels.normalization import (
    _layernorm_kernel_small,
    _layernorm_kernel_medium,
    BLOCK_SIZE,
    NUM_STAGES,
    NUM_WARPS,
)

DEVICE = "cuda"

WARMUP = 20
ITERS = 100
EPS = 1e-5

ROW_COUNTS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
]

COL_COUNTS = [
    64,
    128,
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    32768,
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


def bandwidth(x, gamma, beta, ms):
    total_bytes = (
        x.numel() * x.element_size()
        + gamma.numel() * gamma.element_size()
        + beta.numel() * beta.element_size()
        + x.numel() * x.element_size()
    )

    return total_bytes / (ms / 1000) / 1e9


def run_small(x, gamma, beta):
    out = torch.empty_like(x)

    small_block = triton.next_power_of_2(x.shape[-1])

    _layernorm_kernel_small[(x.shape[0],)](
        x,
        gamma,
        beta,
        EPS,
        x.shape[-1],
        out,
        BLOCK_SIZE=small_block,
        num_warps=NUM_WARPS,
    )

    return out


def run_medium(x, gamma, beta):
    out = torch.empty_like(x)

    _layernorm_kernel_medium[(x.shape[0],)](
        x,
        gamma,
        beta,
        EPS,
        x.shape[-1],
        out,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    return out


def main():
    props = torch.cuda.get_device_properties(0)

    print(f"GPU : {props.name}")
    print(f"SMs : {props.multi_processor_count}\n")

    print("=== LayerNorm Small vs Medium ===\n")

    print(
        f"{'Rows':>6}  "
        f"{'Cols':>7}  "
        f"{'Small':>10}  "
        f"{'Medium':>10}  "
        f"{'Winner':>8}"
    )

    print("-" * 60)

    for rows in ROW_COUNTS:
        print()

        for cols in COL_COUNTS:

            x = torch.randn(rows, cols, device=DEVICE)

            gamma = torch.randn(cols, device=DEVICE)
            beta = torch.randn(cols, device=DEVICE)

            small_ms = benchmark_fn(
                run_small,
                (x, gamma, beta),
            )

            medium_ms = benchmark_fn(
                run_medium,
                (x, gamma, beta),
            )

            small_bw = bandwidth(
                x,
                gamma,
                beta,
                small_ms,
            )

            medium_bw = bandwidth(
                x,
                gamma,
                beta,
                medium_ms,
            )

            winner = "small" if small_bw >= medium_bw else "medium"

            print(
                f"{rows:6d}  "
                f"{cols:7d}  "
                f"{small_bw:10.2f}  "
                f"{medium_bw:10.2f}  "
                f"{winner:>8}"
            )


if __name__ == "__main__":
    main()