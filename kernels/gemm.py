import torch
import triton
import triton.language as tl

from utils import validate_gemm

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 2}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 2}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 2}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 2}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 256, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _tiled_grouped_gemm_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    K,
    N,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)

    a_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=(M, K),
        strides=(K, 1),
        block_shape=(BM, BK),
    )

    b_desc = tl.make_tensor_descriptor(
        base=y_ptr,
        shape=(K, N),
        strides=(N, 1),
        block_shape=(BK, BN),
    )

    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BM, BN),
    )

    tiles_per_row = tl.cdiv(N, BN)
    tiles_per_group = GROUP_M * tiles_per_row
    group_id = pid // tiles_per_group
    group_start_row = group_id * GROUP_M
    group_size_row = tl.minimum(tl.cdiv(M, BM) - group_start_row, GROUP_M)
    pid_in_group = pid % tiles_per_group

    tile_row = group_start_row + (pid_in_group % group_size_row)
    tile_col = pid_in_group // group_size_row

    tile_m = tile_row * BM
    tile_n = tile_col * BN

    acc = tl.zeros((BM, BN), tl.float32)

    for tile_k in tl.range(0, K, BK, num_stages=num_stages):
        a = a_desc.load([tile_m, tile_k])
        b = b_desc.load([tile_k, tile_n])
        acc += tl.dot(a, b)

    out_desc.store([tile_m, tile_n], acc)


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 2}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 2}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 2}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 2}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 256, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _grouped_persistant_gemm_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    K,
    N,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    total_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=(M, K),
        strides=(K, 1),
        block_shape=(BM, BK),
    )

    b_desc = tl.make_tensor_descriptor(
        base=y_ptr,
        shape=(K, N),
        strides=(N, 1),
        block_shape=(BK, BN),
    )

    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BM, BN),
    )

    tile = pid
    tiles_per_group = GROUP_M * num_pid_n

    while tile < total_tiles:
        group_id = tile // tiles_per_group
        group_start_row = group_id * GROUP_M
        group_size_row = tl.minimum(num_pid_m - group_start_row, GROUP_M)
        pid_in_group = tile % tiles_per_group

        tile_row_idx = group_start_row + (pid_in_group % group_size_row)
        tile_col_idx = pid_in_group // group_size_row

        tile_m = tile_row_idx * BM
        tile_n = tile_col_idx * BN

        acc = tl.zeros((BM, BN), tl.float32)

        for tile_k in tl.range(0, K, BK, num_stages=num_stages):
            a = a_desc.load([tile_m, tile_k])
            b = b_desc.load([tile_k, tile_n])

            acc += tl.dot(a, b)

        out_desc.store([tile_m, tile_n], acc)

        tile += tl.num_programs(0)


_SPLITK_CONFIGS = [
    triton.Config({"BM": 16, "BN": 16, "BK": 64, "GROUP_M": 1}, num_warps=2, num_stages=2),
    triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_SPLITK_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _splitk_persistent_grouped_gemm(
    x_ptr,
    y_ptr,
    workspace_ptr,
    M,
    K,
    N,
    SPLIT_K,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    total_tiles = num_pid_m * num_pid_n
    programs = total_tiles * SPLIT_K

    tiles_per_group = GROUP_M * num_pid_n
    num_k_tiles = tl.cdiv(K, BK)
    base_tiles = num_k_tiles // SPLIT_K
    rem_tiles = num_k_tiles % SPLIT_K

    work_id = pid
    while work_id < programs:
        tile_id = work_id // SPLIT_K
        split_k_id = work_id % SPLIT_K

        group_id = tile_id // tiles_per_group
        group_start_row = group_id * GROUP_M
        group_size_row = tl.minimum(num_pid_m - group_start_row, GROUP_M)
        pid_in_group = tile_id % tiles_per_group

        tile_row_idx = group_start_row + (pid_in_group % group_size_row)
        tile_col_idx = pid_in_group // group_size_row

        tile_m = tile_row_idx * BM
        tile_n = tile_col_idx * BN

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        k_start = (split_k_id * base_tiles + tl.minimum(split_k_id, rem_tiles)) * BK
        k_end = ((split_k_id + 1) * base_tiles + tl.minimum(split_k_id + 1, rem_tiles)) * BK

        a_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(M, K),
            strides=(K, 1),
            offsets=(tile_m, k_start),
            block_shape=(BM, BK),
            order=(1, 0),
        )

        b_block = tl.make_block_ptr(
            base=y_ptr,
            shape=(K, N),
            strides=(N, 1),
            offsets=(k_start, tile_n),
            block_shape=(BK, BN),
            order=(1, 0),
        )

        for tile_k in tl.range(k_start, k_end, BK, num_stages=num_stages):
            a = tl.load(
                a_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            b = tl.load(
                b_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            acc += tl.dot(a, b)

            a_block = tl.advance(a_block, (0, BK))
            b_block = tl.advance(b_block, (BK, 0))

        workspace_block = tl.make_block_ptr(
            base=workspace_ptr + split_k_id * M * N,
            shape=(M, N),
            strides=(N, 1),
            offsets=(tile_m, tile_n),
            block_shape=(BM, BN),
            order=(1, 0),
        )

        tl.store(
            workspace_block,
            acc,
            boundary_check=(0, 1),
        )

        work_id += tl.num_programs(0)


@triton.jit
def splitk_merge_kernel(
    workspace_ptr,
    out_ptr,
    M,
    N,
    SPLIT_K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    total_tiles = num_pid_m * num_pid_n
    tiles_per_group = GROUP_M * num_pid_n

    tile_id = pid
    while tile_id < total_tiles:
        group_id = tile_id // tiles_per_group
        group_start_row = group_id * GROUP_M
        group_size_row = tl.minimum(num_pid_m - group_start_row, GROUP_M)
        pid_in_group = tile_id % tiles_per_group

        tile_row_idx = group_start_row + (pid_in_group % group_size_row)
        tile_col_idx = pid_in_group // group_size_row

        tile_m = tile_row_idx * BM
        tile_n = tile_col_idx * BN

        workspace_block = tl.make_block_ptr(
            base=workspace_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(tile_m, tile_n),
            block_shape=(BM, BN),
            order=(1, 0),
        )

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for split in tl.range(0, SPLIT_K):
            acc += tl.load(
                workspace_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            workspace_block = tl.advance(workspace_block, (M, 0))

        out_block = tl.make_block_ptr(
            base=out_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(tile_m, tile_n),
            block_shape=(BM, BN),
            order=(1, 0),
        )

        tl.store(
            out_block,
            acc.to(out_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tile_id += tl.num_programs(0)


def gemm(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Matrix multiplication.

    Assumes:
    - x and y are contiguous CUDA tensors
    - x.shape == (M, K)
    - y.shape == (K, N)
    """

    validate_gemm(x, y, out)

    M, K = x.shape
    N = y.shape[1]

    out = torch.empty((M, N), device=x.device, dtype=torch.float32) if out is None else out

    num_tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    SPLIT_K_TILE_THRESHOLD = max(4, NUM_SMS // 4)
    SPLIT_K_MIN_K = 8192

    LARGE_TILE_THRESHOLD = max(512, NUM_SMS * 25)

    if num_tiles <= SPLIT_K_TILE_THRESHOLD and K >= SPLIT_K_MIN_K:
        desired_split = max(1, K // 1024)
        max_split = max(1, NUM_SMS)
        split_k = min(desired_split, max_split)

        workspace = torch.empty(
            (split_k, M, N),
            device=x.device,
            dtype=torch.float32,
        )

        splitk_grid = lambda META: (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]) * META["SPLIT_K"],
            ),
        )

        _splitk_persistent_grouped_gemm[splitk_grid](
            x,
            y,
            workspace,
            M,
            K,
            N,
            split_k,
        )

        cfg = _splitk_persistent_grouped_gemm.best_config.kwargs

        merge_grid = lambda META: (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),
            ),
        )

        splitk_merge_kernel[merge_grid](
            workspace,
            out,
            M,
            N,
            SPLIT_K=split_k,
            BM=cfg["BM"],
            BN=cfg["BN"],
            GROUP_M=cfg["GROUP_M"],
            num_warps=4,
        )

    elif num_tiles >= LARGE_TILE_THRESHOLD:
        persistent_grid = lambda META: (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),
            ),
        )

        _grouped_persistant_gemm_kernel[persistent_grid](
            x,
            y,
            out,
            M,
            K,
            N,
        )

    else:
        tiled_grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)

        _tiled_grouped_gemm_kernel[tiled_grid](
            x,
            y,
            out,
            M,
            K,
            N,
        )

    return out
