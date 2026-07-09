from statistics import variance

import triton
import torch
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

from utils import reduction_launch_config
(
    SMALL_THRESHOLD,
    MEDIUM_THRESHOLD,
    BLOCK_SIZE,
    NUM_STAGES,
    MAX_PROGRAMS_PER_ROW,
    MEDIUM_ROW_THRESHOLD,
    NUM_WARPS,
) = reduction_launch_config()

@triton.jit
def _layernorm_kernel_small(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    eps,
    n_cols,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range
    mask = offset_range < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    x_minus_mean = x - mean
    variance = tl.sum(x_minus_mean * x_minus_mean, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)

    gamma = tl.load(gamma_ptr + offset_range, mask=mask).to(tl.float32)
    beta = tl.load(beta_ptr + offset_range, mask=mask).to(tl.float32)
    out = (x_minus_mean * inv_std) * gamma + beta
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def _layernorm_kernel_medium(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    eps,
    n_cols,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    
    sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_vec += chunk
    mean = tl.sum(sum_vec, axis=0) / n_cols

    sum_sq_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = chunk - mean
        sum_sq_vec += centered * centered
    variance = tl.sum(sum_sq_vec, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols
        chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offset_range + col_idx, mask=mask).to(tl.float32)
        gamma = tl.load(gamma_ptr + offset_range + col_idx, mask=mask).to(tl.float32)
        out = (chunk - mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def welford_combine(
    mean1, 
    m2_1, 
    count1, 
    mean2, 
    m2_2, 
    count2
):
    count = count1 + count2
    safe_count = tl.where(count > 0, count, 1.0)
    delta = mean2 - mean1
    mean = mean1 + delta * (count2 / safe_count)
    m2 = m2_1 + m2_2 + delta * delta * (count1 * count2 / safe_count)
    return mean, m2, count


@triton.jit
def _layernorm_kernel_partial(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_count_ptr,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    program_id = pid % programs_per_row
    row_start = row_id * n_cols

    running_mean = tl.zeros((), tl.float32)
    running_m2 = tl.zeros((), tl.float32)
    running_count = tl.zeros((), tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(program_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        mask = chunk * BLOCK_SIZE + offsets < n_cols

        x = tl.load(x_ptr + block_start + offsets, mask=mask, other=0.0,).to(tl.float32)
        chunk_mean = x
        chunk_m2 = tl.zeros([BLOCK_SIZE], tl.float32)
        chunk_count = tl.where(mask, 1.0, 0.0)

        chunk_mean, chunk_m2, chunk_count = tl.reduce(
            (chunk_mean, chunk_m2, chunk_count),
            axis=0,
            combine_fn=welford_combine,
        )

        running_mean, running_m2, running_count = welford_combine(
            running_mean,
            running_m2,
            running_count,
            chunk_mean,
            chunk_m2,
            chunk_count,
        )

    tl.store(partial_mean_ptr + pid, running_mean)
    tl.store(partial_m2_ptr + pid, running_m2)
    tl.store(partial_count_ptr + pid, running_count)   

@triton.jit 
def _layernorm_kernel_merge(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_count_ptr,
    eps,
    final_mean_ptr,
    final_inv_std_ptr,
    programs_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * programs_per_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < programs_per_row

    partial_mean = tl.load(partial_mean_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_m2 = tl.load(partial_m2_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_count = tl.load(partial_count_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    final_mean, final_m2, final_count = tl.reduce(
        (partial_mean, partial_m2, partial_count),
        axis=0,
        combine_fn=welford_combine,
    )
    safe_count = tl.where(final_count > 0, final_count, 1.0)
    variance = final_m2 / safe_count
    inv_std = tl.rsqrt(variance + eps)

    tl.store(final_mean_ptr + pid, final_mean)
    tl.store(final_inv_std_ptr + pid, inv_std)

@triton.jit
def _layernorm_kernel_final(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    final_mean_ptr,
    final_inv_std_ptr,
    n_cols,
    out_ptr,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    program_id = pid % programs_per_row
    row_start = row_id * n_cols

    offset_range = tl.arange(0, BLOCK_SIZE)
    mean = tl.load(final_mean_ptr + row_id).to(tl.float32)
    inv_std = tl.load(final_inv_std_ptr + row_id).to(tl.float32)

    for chunk in tl.range(program_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + chunk * BLOCK_SIZE + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offset_range + chunk * BLOCK_SIZE, mask=mask).to(tl.float32)
        gamma = tl.load(gamma_ptr + offset_range + chunk * BLOCK_SIZE, mask=mask).to(tl.float32)

        out = (x - mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)

def layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Layer Normalization over the last dimension.

    Assumes:
    - x is contiguous CUDA tensor
    - gamma and beta have shape (x.shape[-1],)
    - gamma and beta are contiguous CUDA tensors
    """

    # validate_layernorm(x, gamma, beta)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]

    out = torch.empty_like(x) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)

        _layernorm_kernel_small[(n_rows,)](
            x,
            gamma,
            beta,
            eps,
            n_cols,
            out,
            BLOCK_SIZE=small_block_size,
            num_warps=NUM_WARPS,
        )

    elif n_cols <= MEDIUM_THRESHOLD or n_rows >= MEDIUM_ROW_THRESHOLD:

        _layernorm_kernel_medium[(n_rows,)](
            x,
            gamma,
            beta,
            eps,
            n_cols,
            out,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    else:
        programs_per_row = min(
            triton.cdiv(n_cols, BLOCK_SIZE),
            MAX_PROGRAMS_PER_ROW,
        )
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)

        partial_mean = torch.empty(
            n_rows * programs_per_row,
            device=x.device,
            dtype=torch.float32,
        )

        partial_m2 = torch.empty_like(partial_mean)
        partial_count = torch.empty_like(partial_mean)

        final_mean = torch.empty(
            n_rows,
            device=x.device,
            dtype=torch.float32,
        )

        final_inv_std = torch.empty_like(final_mean)

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

        merge_block_size = triton.next_power_of_2(programs_per_row)

        _layernorm_kernel_merge[(n_rows,)](
            partial_mean,
            partial_m2,
            partial_count,
            eps,
            final_mean,
            final_inv_std,
            programs_per_row,
            BLOCK_SIZE=merge_block_size,
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