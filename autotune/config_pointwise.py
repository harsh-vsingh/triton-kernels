import triton

POINTWISE_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE": 128},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 256},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 256},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 512},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 1024},
        num_warps=8,
        num_stages=2,
    ),
]


pointwise_autotune = triton.autotune(
    configs=POINTWISE_CONFIGS,
    key=["n_elements"],
)