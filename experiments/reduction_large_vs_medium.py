import torch
import triton

from kernels.reductions import _sum_kernel_medium, _sum_kernel_large

"""
Compare the streaming reduction kernel against the multi-program reduction.

The streaming kernel assigns one Triton program to each row and performs the
entire reduction within that program. The multi-program kernel assigns multiple
programs to each row, increasing parallelism but requiring a second reduction
over the partial sums.

This experiment varies only the number of rows while keeping the reduction
length fixed. When the number of independent row reductions is smaller than the
number of GPU Streaming Multiprocessors (SMs), the streaming kernel cannot fully
occupy the device and the multi-program implementation is advantageous. As the
number of rows approaches or exceeds the SM count, the streaming kernel naturally
saturates the GPU and consistently outperforms the multi-program approach by
avoiding the additional reduction stage.

Results shown below were obtained on a Tesla T4 (40 SMs), where the crossover
occurs at approximately 32–64 independent row reductions.
"""

DEVICE = "cuda"
WARMUP = 20
ITERS = 100

N_COLS = 262144
ROW_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

BLOCK_SIZE = 1024
PROGRAMS_PER_ROW = 8


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


def bandwidth(x, ms):
    return x.numel() * x.element_size() / (ms / 1000) / 1e9


def run_medium(x, n_rows, n_cols):
    out = torch.empty(n_rows, device=DEVICE, dtype=torch.float32)
    _sum_kernel_medium[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=1024,
        num_stages=2,
    )
    return out


def run_large(x, n_rows, n_cols):
    chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)

    partial = torch.empty(
        n_rows * PROGRAMS_PER_ROW,
        device=DEVICE,
        dtype=torch.float32,
    )

    _sum_kernel_large[(n_rows * PROGRAMS_PER_ROW,)](
        x,
        partial,
        n_cols,
        PROGRAMS_PER_ROW,
        chunks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=2,
    )

    partial = partial.view(n_rows, PROGRAMS_PER_ROW)

    out = torch.empty(n_rows, device=DEVICE, dtype=torch.float32)

    reduce_block = triton.next_power_of_2(PROGRAMS_PER_ROW)

    from kernels.reductions import _sum_kernel_small

    _sum_kernel_small[(n_rows,)](
        partial,
        out,
        PROGRAMS_PER_ROW,
        BLOCK_SIZE=reduce_block,
    )

    return out


def main():
    props = torch.cuda.get_device_properties(0)

    print(f"GPU : {props.name}")
    print(f"SMs : {props.multi_processor_count}")
    print(f"Reduction length : {N_COLS}\n")

    print("=== Occupancy experiment ===\n")
    print(
        f"{'Rows':>8}  "
        f"{'Programs':>10}  "
        f"{'Medium':>10}  "
        f"{'Large':>10}  "
        f"{'Torch':>10}  "
        f"{'Winner':>8}"
    )
    print("-" * 72)

    for n_rows in ROW_COUNTS:
        x = torch.randn(n_rows, N_COLS, device=DEVICE)

        medium_ms = benchmark_fn(run_medium, (x, n_rows, N_COLS))
        large_ms = benchmark_fn(run_large, (x, n_rows, N_COLS))
        torch_ms = benchmark_fn(lambda t: t.sum(dim=-1), (x,))

        medium_bw = bandwidth(x, medium_ms)
        large_bw = bandwidth(x, large_ms)
        torch_bw = bandwidth(x, torch_ms)

        winner = "medium" if medium_bw >= large_bw else "large"

        print(
            f"{n_rows:>8}  "
            f"{n_rows:>10}  "
            f"{medium_bw:>9.2f}  "
            f"{large_bw:>9.2f}  "
            f"{torch_bw:>9.2f}  "
            f"{winner:>8}"
        )


if __name__ == "__main__":
    main()