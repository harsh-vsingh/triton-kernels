import torch
import triton

from kernels.reductions import _sum_kernel_medium, _sum_kernel_large, _sum_kernel_small

DEVICE = "cuda"
WARMUP = 20
ITERS = 100


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


def bw(x, ms):
    return x.numel() * x.element_size() / (ms / 1000) / 1e9


def run_medium(x, n_rows, n_cols):
    out = torch.empty(n_rows, dtype=torch.float32, device=DEVICE)

    _sum_kernel_medium[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=1024,
        num_stages=2,
    )
    return out


def run_large(x, n_rows, n_cols):
    programs_per_row = min(triton.cdiv(n_cols, 1024), 8)
    chunks_per_row = triton.cdiv(n_cols, 1024)

    partial = torch.empty(
        n_rows * programs_per_row,
        dtype=torch.float32,
        device=DEVICE,
    )

    _sum_kernel_large[(n_rows * programs_per_row,)](
        x,
        partial,
        n_cols,
        programs_per_row,
        chunks_per_row,
        BLOCK_SIZE=1024,
        num_stages=2,
    )

    out = torch.empty(n_rows, dtype=torch.float32, device=DEVICE)

    BLOCK_SIZE = triton.next_power_of_2(programs_per_row)

    _sum_kernel_small[(n_rows,)](
        partial,
        out,
        programs_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def main():
    print("=== Occupancy experiment ===\n")

    n_cols = 262144

    print(
        f"{'n_rows':>8}  "
        f"{'Medium':>10}  "
        f"{'Large':>10}  "
        f"{'Torch':>10}  "
        f"{'winner':>8}"
    )
    print("-" * 58)

    for n_rows in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x = torch.randn(n_rows, n_cols, device=DEVICE)

        medium_ms = benchmark_fn(run_medium, (x, n_rows, n_cols))
        large_ms = benchmark_fn(run_large, (x, n_rows, n_cols))
        torch_ms = benchmark_fn(lambda t: t.sum(dim=-1), (x,))

        medium_bw = bw(x, medium_ms)
        large_bw = bw(x, large_ms)
        torch_bw = bw(x, torch_ms)

        winner = "medium" if medium_bw > large_bw else "large"

        print(
            f"{n_rows:>8}  "
            f"{medium_bw:>9.2f}  "
            f"{large_bw:>9.2f}  "
            f"{torch_bw:>9.2f}  "
            f"{winner:>8}"
        )


if __name__ == "__main__":
    main()