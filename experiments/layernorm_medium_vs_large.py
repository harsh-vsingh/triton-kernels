import torch
import torch.nn.functional as F
import triton

from kernels.normalization import (
    _layernorm_kernel_medium,
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

PROGRAMS_PER_ROW = 8

ROW_COUNTS = [
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    32,
    40,
    48,
    64,
    96,
    128,
    192,
    256,
]

COL_COUNTS = [
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
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


def run_large(x, gamma, beta):
    n_rows, n_cols = x.shape

    chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)

    partial_mean = torch.empty(
        n_rows * PROGRAMS_PER_ROW,
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

    _layernorm_kernel_partial[(n_rows * PROGRAMS_PER_ROW,)](
        x,
        partial_mean,
        partial_m2,
        partial_count,
        n_cols,
        PROGRAMS_PER_ROW,
        chunks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    merge_block = triton.next_power_of_2(PROGRAMS_PER_ROW)

    _layernorm_kernel_merge[(n_rows,)](
        partial_mean,
        partial_m2,
        partial_count,
        EPS,
        final_mean,
        final_inv_std,
        PROGRAMS_PER_ROW,
        BLOCK_SIZE=merge_block,
        num_warps=NUM_WARPS,
    )

    _layernorm_kernel_final[(n_rows * PROGRAMS_PER_ROW,)](
        x,
        gamma,
        beta,
        final_mean,
        final_inv_std,
        n_cols,
        out,
        PROGRAMS_PER_ROW,
        chunks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    return out


def run_torch(x, gamma, beta):
    return F.layer_norm(
        x,
        (x.shape[-1],),
        gamma,
        beta,
        EPS,
    )


def main():
    props = torch.cuda.get_device_properties(0)

    print(f"GPU : {props.name}")
    print(f"SMs : {props.multi_processor_count}\n")

    print("=== LayerNorm Medium vs Large ===\n")

    print(
        f"{'Rows':>6}  "
        f"{'Cols':>8}  "
        f"{'Medium':>10}  "
        f"{'Large':>10}  "
        f"{'Torch':>10}  "
        f"{'Ratio':>8}  "
        f"{'Winner':>8}"
    )

    print("-" * 76)

    for rows in ROW_COUNTS:
        print()

        for cols in COL_COUNTS:

            x = torch.randn(rows, cols, device=DEVICE)
            gamma = torch.randn(cols, device=DEVICE)
            beta = torch.randn(cols, device=DEVICE)

            medium_ms = benchmark_fn(
                run_medium,
                (x, gamma, beta),
            )

            large_ms = benchmark_fn(
                run_large,
                (x, gamma, beta),
            )

            torch_ms = benchmark_fn(
                run_torch,
                (x, gamma, beta),
            )

            medium_bw = bandwidth(
                x,
                gamma,
                beta,
                medium_ms,
            )

            large_bw = bandwidth(
                x,
                gamma,
                beta,
                large_ms,
            )

            torch_bw = bandwidth(
                x,
                gamma,
                beta,
                torch_ms,
            )

            if large_bw >= medium_bw:
                winner = "large"
                ratio = large_bw / medium_bw
            else:
                winner = "medium"
                ratio = medium_bw / large_bw

            print(
                f"{rows:6d}  "
                f"{cols:8d}  "
                f"{medium_bw:10.2f}  "
                f"{large_bw:10.2f}  "
                f"{torch_bw:10.2f}  "
                f"{ratio:8.2f}x  "
                f"{winner:>8}"
            )


if __name__ == "__main__":
    main()