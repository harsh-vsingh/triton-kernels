import torch
import triton
import triton.language as tl

from utils import validate_reduction

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

SMALL_THRESHOLD = 1024
MEDIUM_ROW_THRESHOLD = NUM_SMS
MEDIUM_COL_THRESHOLD = 4096
BLOCK_SIZE = 1024
NUM_STAGES = 2
NUM_WARPS = 4


@triton.jit
def _softmax_kernel_small(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = row_start + offset_range
    mask = offset_range < n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = tl.exp(x - x_max)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _softmax_kernel_medium(
    x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    running_max = tl.full((), float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = offset_range + col_idx < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = offset_range + col_idx < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        x = tl.exp(x - running_max)
        x = x / running_sum
        tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _softmax_kernel_partial(
    x_ptr,
    running_sum_ptr,
    running_max_ptr,
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

    running_sum = tl.zeros((), tl.float32)
    running_max = tl.full((), float("-inf"), dtype=tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + col_idx * BLOCK_SIZE + offset_range
        mask = col_idx * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max
    tl.store(running_sum_ptr + pid, running_sum)
    tl.store(running_max_ptr + pid, running_max)


@triton.jit
def _softmax_kernel_merge(
    running_sum_ptr,
    running_max_ptr,
    max_ptr,
    sum_ptr,
    programs_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_range = tl.arange(0, BLOCK_SIZE)
    mask = offset_range < programs_per_row
    offsets = pid * programs_per_row + offset_range

    running_sum = tl.load(running_sum_ptr + offsets, mask=mask, other=0.0)
    running_max = tl.load(running_max_ptr + offsets, mask=mask, other=float("-inf"))

    max_val = tl.max(running_max, axis=0)
    sum_val = tl.sum(running_sum * tl.exp(running_max - max_val), axis=0)
    tl.store(max_ptr + pid, max_val)
    tl.store(sum_ptr + pid, sum_val)


@triton.jit
def _softmax_kernel_finalize(
    x_ptr,
    out_ptr,
    max_ptr,
    sum_ptr,
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

    max_val = tl.load(max_ptr + row_id)
    sum_val = tl.load(sum_ptr + row_id)

    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + col_idx * BLOCK_SIZE + offset_range
        mask = col_idx * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        x = tl.exp(x - max_val)
        x = x / sum_val
        tl.store(out_ptr + offsets, x, mask=mask)


def softmax(
    x: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the softmax over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]

    out = torch.empty_like(x) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)

        _softmax_kernel_small[(n_rows,)](
            x,
            out,
            n_cols,
            BLOCK_SIZE=small_block_size,
            num_warps=NUM_WARPS,
        )

    elif n_rows >= MEDIUM_ROW_THRESHOLD or n_cols <= MEDIUM_COL_THRESHOLD:
        _softmax_kernel_medium[(n_rows,)](
            x,
            out,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    else:
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        programs_per_row = min(chunks_per_row, max(1, NUM_SMS // n_rows))

        partial_sum = torch.empty(
            n_rows * programs_per_row,
            device=x.device,
            dtype=torch.float32,
        )

        partial_max = torch.empty_like(partial_sum)

        final_sum = torch.empty(
            n_rows,
            device=x.device,
            dtype=torch.float32,
        )

        final_max = torch.empty_like(final_sum)

        _softmax_kernel_partial[(n_rows * programs_per_row,)](
            x,
            partial_sum,
            partial_max,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

        merge_block_size = triton.next_power_of_2(programs_per_row)

        _softmax_kernel_merge[(n_rows,)](
            partial_sum,
            partial_max,
            final_max,
            final_sum,
            programs_per_row,
            BLOCK_SIZE=merge_block_size,
            num_warps=NUM_WARPS,
        )

        _softmax_kernel_finalize[(n_rows * programs_per_row,)](
            x,
            out,
            final_max,
            final_sum,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    return out


@triton.jit
def _logsoftmax_kernel_small(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    offsets = row_start + offset_range
    mask = offset_range < n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    log_denom = x_max + tl.log(tl.sum(tl.exp(x - x_max), axis=0))
    out = x - log_denom
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _logsoftmax_kernel_medium(
    x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offset_range = tl.arange(0, BLOCK_SIZE)
    running_max = tl.full((), float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)

    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = offset_range + col_idx < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max

    log_denom = running_max + tl.log(running_sum)
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        offsets = row_start + col_idx + offset_range
        mask = offset_range + col_idx < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        x = x - log_denom
        tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _logsoftmax_kernel_partial(
    x_ptr,
    running_sum_ptr,
    running_max_ptr,
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

    running_sum = tl.zeros((), tl.float32)
    running_max = tl.full((), float("-inf"), dtype=tl.float32)
    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + col_idx * BLOCK_SIZE + offset_range
        mask = col_idx * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max
    tl.store(running_sum_ptr + pid, running_sum)
    tl.store(running_max_ptr + pid, running_max)


@triton.jit
def _logsoftmax_kernel_merge(
    running_sum_ptr,
    running_max_ptr,
    log_denom_ptr,
    programs_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_range = tl.arange(0, BLOCK_SIZE)
    mask = offset_range < programs_per_row
    offsets = pid * programs_per_row + offset_range

    running_sum = tl.load(running_sum_ptr + offsets, mask=mask, other=0.0)
    running_max = tl.load(running_max_ptr + offsets, mask=mask, other=float("-inf"))

    max_val = tl.max(running_max, axis=0)
    sum_val = tl.sum(running_sum * tl.exp(running_max - max_val), axis=0)
    log_denom = max_val + tl.log(sum_val)
    tl.store(log_denom_ptr + pid, log_denom)


@triton.jit
def _logsoftmax_kernel_finalize(
    x_ptr,
    out_ptr,
    log_denom_ptr,
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

    log_denom = tl.load(log_denom_ptr + row_id)

    offset_range = tl.arange(0, BLOCK_SIZE)
    for col_idx in tl.range(chunk_id, chunks_per_row, programs_per_row, num_stages=num_stages):
        offsets = row_start + col_idx * BLOCK_SIZE + offset_range
        mask = col_idx * BLOCK_SIZE + offset_range < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        x = x - log_denom
        tl.store(out_ptr + offsets, x, mask=mask)


def log_softmax(
    x: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the log-softmax over the last dimension.
    Assumes contiguous CUDA tensor of any shape and dtype.
    """
    validate_reduction(x)

    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]

    out = torch.empty_like(x) if out is None else out

    if n_cols <= SMALL_THRESHOLD:
        small_block_size = triton.next_power_of_2(n_cols)

        _logsoftmax_kernel_small[(n_rows,)](
            x,
            out,
            n_cols,
            BLOCK_SIZE=small_block_size,
            num_warps=NUM_WARPS,
        )

    elif n_rows >= MEDIUM_ROW_THRESHOLD or n_cols <= MEDIUM_COL_THRESHOLD:
        _logsoftmax_kernel_medium[(n_rows,)](
            x,
            out,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    else:
        chunks_per_row = triton.cdiv(n_cols, BLOCK_SIZE)
        programs_per_row = min(chunks_per_row, max(1, NUM_SMS // n_rows))

        partial_sum = torch.empty(
            n_rows * programs_per_row,
            device=x.device,
            dtype=torch.float32,
        )

        partial_max = torch.empty_like(partial_sum)

        log_denom = torch.empty(
            n_rows,
            device=x.device,
            dtype=torch.float32,
        )

        _logsoftmax_kernel_partial[(n_rows * programs_per_row,)](
            x,
            partial_sum,
            partial_max,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

        merge_block_size = triton.next_power_of_2(programs_per_row)

        _logsoftmax_kernel_merge[(n_rows,)](
            partial_sum,
            partial_max,
            log_denom,
            programs_per_row,
            BLOCK_SIZE=merge_block_size,
            num_warps=NUM_WARPS,
        )

        _logsoftmax_kernel_finalize[(n_rows * programs_per_row,)](
            x,
            out,
            log_denom,
            n_cols,
            programs_per_row,
            chunks_per_row,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    return out
