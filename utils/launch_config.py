import triton


def pointwise_launch_config(n: int):
    block_size = min(1024, triton.next_power_of_2(n))

    if block_size <= 256:
        num_warps = 2
    elif block_size <= 512:
        num_warps = 4
    else:
        num_warps = 8

    return block_size, num_warps