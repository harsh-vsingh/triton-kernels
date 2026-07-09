import torch
import triton

from kernels.normalization import (
    _layernorm_kernel_partial,
    _layernorm_kernel_merge,
    _layernorm_kernel_final,
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
]

COL_COUNTS = [
    32768,
    65536,
    131072,
    262144,
]

PROGRAMS = [
    1,
    2,
    4,
    8,
    16,
    32,
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
    bytes_processed = (
        x.numel() * x.element_size()
        + gamma.numel() * gamma.element_size()
        + beta.numel() * beta.element_size()
        + x.numel() * x.element_size()
    )

    return bytes_processed / (ms / 1000) / 1e9


def run_large(x, gamma, beta, programs_per_row):
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)

    partial_mean = torch.empty(
        n_rows * programs_per_row,
        device=DEVICE,
        dtype=torch.float32,
    )

    partial_m2 = torch.empty_like(partial_mean)
    partial_count = torch.empty_like(partial_mean)

    final_mean = torch.empty(
        n_rows,
        device=DEVICE,
        dtype=torch.float32,
    )

    final_inv_std = torch.empty_like(final_mean)

    out = torch.empty_like(x)

    _layernorm_kernel_partial[(n_rows * programs_per_row,)](
        x,
        partial_mean,
        partial_m2,
        partial_count,
        n_cols,
        programs_per_row,
        chunks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    merge_block = triton.next_power_of_2(programs_per_row)

    _layernorm_kernel_merge[(n_rows,)](
        partial_mean,
        partial_m2,
        partial_count,
        EPS,
        final_mean,
        final_inv_std,
        programs_per_row,
        BLOCK_SIZE=merge_block,
        num_warps=NUM_WARPS,
    )

    _layernorm_kernel_final[(n_rows * programs_per_row,)](
        x,
        gamma,
        beta,
        final_mean,
        final_inv_std,
        n_cols,
        out,
        programs_per_row,
        chunks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    return out


def main():
    props = torch.cuda.get_device_properties(0)

    print(f"GPU : {props.name}")
    print(f"SMs : {props.multi_processor_count}")
    print()

    print("=== Programs Per Row Sweep ===\n")

    print(
        f"{'Rows':>6} "
        f"{'Cols':>8} "
        f"{'PPR':>4} "
        f"{'GB/s':>10}"
    )

    print("-" * 38)

    for rows in ROW_COUNTS:
        print()

        for cols in COL_COUNTS:

            x = torch.randn(rows, cols, device=DEVICE)
            gamma = torch.randn(cols, device=DEVICE)
            beta = torch.randn(cols, device=DEVICE)

            best_bw = 0.0
            best_ppr = None

            for ppr in PROGRAMS:

                chunks = triton.cdiv(cols, BLOCK_SIZE)

                if ppr > chunks:
                    continue

                ms = benchmark_fn(
                    run_large,
                    (x, gamma, beta, ppr),
                )

                bw = bandwidth(
                    x,
                    gamma,
                    beta,
                    ms,
                )

                print(
                    f"{rows:6d} "
                    f"{cols:8d} "
                    f"{ppr:4d} "
                    f"{bw:10.2f}"
                )

                if bw > best_bw:
                    best_bw = bw
                    best_ppr = ppr

            print(
                f"{'':6} "
                f"{'':8} "
                f"{'Best':>4} "
                f"{best_ppr:>2d} ({best_bw:.2f} GB/s)"
            )


if __name__ == "__main__":
    main()