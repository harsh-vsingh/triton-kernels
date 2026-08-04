import torch
import triton
import triton.language as tl

from utils import validate_binary, validate_layernorm, validate_rmsnorm

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

SMALL_THRESHOLD = 1024
MEDIUM_ROW_THRESHOLD = NUM_SMS
MEDIUM_COL_THRESHOLD = 8192
BLOCK_SIZE = 1024
NUM_STAGES = 2
NUM_WARPS = 8


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
    x_minus_mean = tl.where(mask, x - mean, 0.0)
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

    running_mean = tl.zeros((), tl.float32)
    running_m2 = tl.zeros((), tl.float32)
    running_count = tl.zeros((), tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
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

    variance = running_m2 / running_count
    inv_std = tl.rsqrt(variance + eps)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + col_idx + offset_range,
            mask=mask,
        ).to(tl.float32)
        beta = tl.load(
            beta_ptr + col_idx + offset_range,
            mask=mask,
        ).to(tl.float32)
        out = (x - running_mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def welford_combine(mean1, m2_1, count1, mean2, m2_2, count2):
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
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    running_mean = tl.zeros((), tl.float32)
    running_m2 = tl.zeros((), tl.float32)
    running_count = tl.zeros((), tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        mask = chunk * BLOCK_SIZE + offsets < n_cols

        x = tl.load(
            x_ptr + block_start + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
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

    partial_mean = tl.load(partial_mean_ptr + row_start + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_m2 = tl.load(partial_m2_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_count = tl.load(partial_count_ptr + row_start + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

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
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    offset_range = tl.arange(0, BLOCK_SIZE)
    mean = tl.load(final_mean_ptr + row_id).to(tl.float32)
    inv_std = tl.load(final_inv_std_ptr + row_id).to(tl.float32)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
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

    validate_layernorm(x, gamma, beta, out)

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

    elif n_rows >= MEDIUM_ROW_THRESHOLD or n_cols <= MEDIUM_COL_THRESHOLD:
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
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        programs_per_row = min(chunks_per_row, max(1, NUM_SMS // n_rows))

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


@triton.jit
def _rms_kernel_small(
    x_ptr,
    eps,
    gamma_ptr,
    n_cols,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr = False,
    residual_ptr=0,
):
    pid = tl.program_id(0)
    block_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offset_range
    mask = offset_range < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x + residual

    sum_sq = tl.sum(x * x, axis=0)
    ms = sum_sq / n_cols + eps
    irms = tl.rsqrt(ms)

    gamma = tl.load(gamma_ptr + offset_range, mask=mask).to(tl.float32)
    out = irms * x * gamma
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _rms_kernel_medium(
    x_ptr,
    eps,
    gamma_ptr,
    n_cols,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr = False,
    residual_ptr=0,
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    sumsq_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offset = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            x = x + residual

        sumsq_vec += x * x

    sumsq = tl.sum(sumsq_vec, axis=0)
    ms = sumsq / n_cols + eps
    irms = tl.rsqrt(ms)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offset = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            x = x + residual
        gamma = tl.load(gamma_ptr + col_idx + offset_range, mask=mask, other=0.0).to(tl.float32)

        out = x * gamma * irms
        tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def _rms_kernel_partial(
    x_ptr,
    partial_sumsq_ptr,
    n_cols,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr = False,
    residual_ptr=0,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    running_sumsq_vec = tl.zeros([BLOCK_SIZE], tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        mask = chunk * BLOCK_SIZE + offsets < n_cols

        x = tl.load(x_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + block_start + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            x = x + residual
        running_sumsq_vec += x * x
    running_sumsq = tl.sum(running_sumsq_vec, axis=0)
    tl.store(partial_sumsq_ptr + pid, running_sumsq)


@triton.jit
def _rms_kernel_merge(
    partial_sumsq_ptr, eps, final_irms_ptr, programs_per_row, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * programs_per_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < programs_per_row

    partial_sumsq = tl.load(partial_sumsq_ptr + row_start + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    total_sumsq = tl.sum(partial_sumsq, axis=0)
    ms = total_sumsq / n_cols + eps
    irms = tl.rsqrt(ms)

    tl.store(final_irms_ptr + pid, irms)


@triton.jit
def _rms_kernel_final(
    x_ptr,
    gamma_ptr,
    final_irms_ptr,
    n_cols,
    out_ptr,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr = False,
    residual_ptr=0,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    offset_range = tl.arange(0, BLOCK_SIZE)
    irms = tl.load(final_irms_ptr + row_id).to(tl.float32)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + chunk * BLOCK_SIZE + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + chunk * BLOCK_SIZE + offset_range, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            x = x + residual

        out = x * gamma * irms
        tl.store(out_ptr + offsets, out, mask=mask)


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor = None,
    residual: torch.Tensor = None,
) -> torch.Tensor:
    """
    RMS Normalization over the last dimension.

    Assumes:
    - x is contiguous CUDA tensor
    - gamma has shape (x.shape[-1],)
    - gamma is contiguous CUDA tensor
    - if residual is not None, residual has the same shape as x
    """

    validate_rmsnorm(x, gamma, out, residual=residual)
    if residual is not None:
        HAS_RESIDUAL = True
    else:
        HAS_RESIDUAL = False

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]

    out = torch.empty_like(x) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)

        _rms_kernel_small[(n_rows,)](
            x,
            eps,
            gamma,
            n_cols,
            out,
            BLOCK_SIZE=small_block_size,
            num_warps=NUM_WARPS,
            HAS_RESIDUAL=HAS_RESIDUAL,
            residual_ptr=residual if HAS_RESIDUAL else 0,
        )

    elif n_rows >= MEDIUM_ROW_THRESHOLD or n_cols <= MEDIUM_COL_THRESHOLD:
        _rms_kernel_medium[(n_rows,)](
            x,
            eps,
            gamma,
            n_cols,
            out,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
            HAS_RESIDUAL=HAS_RESIDUAL,
            residual_ptr=residual if HAS_RESIDUAL else 0,
        )

    else:
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        programs_per_row = min(chunks_per_row, max(1, NUM_SMS // n_rows))

        partial_sumsq = torch.empty(
            n_rows * programs_per_row,
            device=x.device,
            dtype=torch.float32,
        )

        final_irms = torch.empty(
            n_rows,
            device=x.device,
            dtype=torch.float32,
        )

        _rms_kernel_partial[(n_rows * programs_per_row,)](
            x,
            partial_sumsq,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
            HAS_RESIDUAL=HAS_RESIDUAL,
            residual_ptr=residual if HAS_RESIDUAL else 0,
        )

        merge_block_size = triton.next_power_of_2(programs_per_row)

        _rms_kernel_merge[(n_rows,)](
            partial_sumsq,
            eps,
            final_irms,
            programs_per_row,
            n_cols,
            BLOCK_SIZE=merge_block_size,
            num_warps=NUM_WARPS,
        )

        _rms_kernel_final[(n_rows * programs_per_row,)](
            x,
            gamma,
            final_irms,
            n_cols,
            out,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
            HAS_RESIDUAL=HAS_RESIDUAL,
            residual_ptr=residual if HAS_RESIDUAL else 0,
        )

    return out


@triton.jit
def _residual_layernorm_kernel_small(
    x_ptr,
    residual_ptr,
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    z = (x + residual).to(tl.float32)
    mean = tl.sum(z, axis=0) / n_cols
    z_minus_mean = tl.where(mask, z - mean, 0.0)
    variance = tl.sum(z_minus_mean * z_minus_mean, axis=0) / n_cols
    inv_std = tl.rsqrt(variance + eps)

    gamma = tl.load(gamma_ptr + offset_range, mask=mask).to(tl.float32)
    beta = tl.load(beta_ptr + offset_range, mask=mask).to(tl.float32)
    out = (z_minus_mean * inv_std) * gamma + beta
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _residual_layernorm_kernel_medium(
    x_ptr,
    residual_ptr,
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

    running_mean = tl.zeros((), tl.float32)
    running_m2 = tl.zeros((), tl.float32)
    running_count = tl.zeros((), tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        z = (x + residual).to(tl.float32)
        chunk_mean = z
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

    variance = running_m2 / running_count
    inv_std = tl.rsqrt(variance + eps)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = col_idx + offset_range < n_cols

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        )
        z = (x + residual).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + col_idx + offset_range,
            mask=mask,
        ).to(tl.float32)
        beta = tl.load(
            beta_ptr + col_idx + offset_range,
            mask=mask,
        ).to(tl.float32)
        out = (z - running_mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def _residual_layernorm_kernel_partial(
    x_ptr,
    residual_ptr,
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
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    running_mean = tl.zeros((), tl.float32)
    running_m2 = tl.zeros((), tl.float32)
    running_count = tl.zeros((), tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        block_start = row_start + chunk * BLOCK_SIZE
        mask = chunk * BLOCK_SIZE + offsets < n_cols

        x = tl.load(
            x_ptr + block_start + offsets,
            mask=mask,
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + block_start + offsets,
            mask=mask,
            other=0.0,
        )
        z = (x + residual).to(tl.float32)
        chunk_mean = z
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
def _residual_layernorm_kernel_merge(
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

    partial_mean = tl.load(partial_mean_ptr + row_start + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_m2 = tl.load(partial_m2_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_count = tl.load(partial_count_ptr + row_start + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

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
def _residual_layernorm_kernel_final(
    x_ptr,
    residual_ptr,
    gamma_ptr,
    beta_ptr,
    final_mean_ptr,
    final_inv_std_ptr,
    n_cols,
    out_ptr,
    programs_per_row,
    chunks_per_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // programs_per_row
    chunk_id = pid % programs_per_row
    row_start = row_id * n_cols

    offset_range = tl.arange(0, BLOCK_SIZE)
    mean = tl.load(final_mean_ptr + row_id).to(tl.float32)
    inv_std = tl.load(final_inv_std_ptr + row_id).to(tl.float32)

    for chunk in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + chunk * BLOCK_SIZE + offset_range
        mask = chunk * BLOCK_SIZE + offset_range < n_cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        z = (x + residual).to(tl.float32)
        beta = tl.load(beta_ptr + offset_range + chunk * BLOCK_SIZE, mask=mask).to(tl.float32)
        gamma = tl.load(gamma_ptr + offset_range + chunk * BLOCK_SIZE, mask=mask).to(tl.float32)

        out = (z - mean) * inv_std * gamma + beta
        tl.store(out_ptr + offsets, out, mask=mask)


def residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Residual addition followed by Layer Normalization over the last dimension.

    Assumes:
    - x is contiguous CUDA tensor
    - gamma and beta have shape (x.shape[-1],)
    - gamma and beta are contiguous CUDA tensors
    - residual has the same shape as x
    """

    validate_layernorm(x, gamma, beta, out)
    validate_binary(x, residual)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]

    out = torch.empty_like(x) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)

        _residual_layernorm_kernel_small[(n_rows,)](
            x,
            residual,
            gamma,
            beta,
            eps,
            n_cols,
            out,
            BLOCK_SIZE=small_block_size,
            num_warps=NUM_WARPS,
        )

    elif n_rows >= MEDIUM_ROW_THRESHOLD or n_cols <= MEDIUM_COL_THRESHOLD:
        _residual_layernorm_kernel_medium[(n_rows,)](
            x,
            residual,
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
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        programs_per_row = min(triton.cdiv(chunks_per_row, 4), max(1, NUM_SMS // n_rows))

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

        _residual_layernorm_kernel_partial[(n_rows * programs_per_row,)](
            x,
            residual,
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

        _residual_layernorm_kernel_merge[(n_rows,)](
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

        _residual_layernorm_kernel_final[(n_rows * programs_per_row,)](
            x,
            residual,
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
