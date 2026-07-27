import torch

from kernels import gemm

DEVICE = "cuda"

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

WARMUP = 20
ITERS = 100


def bytes_gemm(x, y, out):
    return (
        x.numel() * x.element_size()
        + y.numel() * y.element_size()
        + out.numel() * out.element_size()
    )


def tflops(M, K, N, ms):
    return 2 * M * K * N / (ms * 1e-3) / 1e12


def benchmark(x, y):
    M, K = x.shape
    N = y.shape[1]

    for _ in range(WARMUP):
        gemm(x, y)
        torch.matmul(x, y)

    torch.cuda.synchronize()

    out = gemm(x, y)
    total_bytes = bytes_gemm(x, y, out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    start.record()
    for _ in range(ITERS):
        gemm(x, y, out)
    end.record()

    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / ITERS
    triton_bw = total_bytes / (triton_ms / 1000) / 1e9
    triton_tflops = tflops(M, K, N, triton_ms)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        torch.matmul(x, y)
    end.record()

    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / ITERS
    torch_bw = total_bytes / (torch_ms / 1000) / 1e9
    torch_tflops = tflops(M, K, N, torch_ms)

    print(
        f"Triton: {triton_bw:8.2f} GB/s"
        f" | {triton_tflops:7.2f} TFLOPS"
        f" || Torch: {torch_bw:8.2f} GB/s"
        f" | {torch_tflops:7.2f} TFLOPS"
        f" | {torch_ms / triton_ms:5.2f}x"
    )


def main():

    shapes = [
        (1, 1, 1),
        (7, 13, 5),
        (31, 17, 23),
        (64, 64, 64),
        (127, 127, 127),
        (128, 128, 128),
        (129, 129, 129),
        (255, 255, 255),
        (256, 256, 256),
        (257, 257, 257),
        (511, 777, 333),
        (1000, 768, 512),
        (1025, 1023, 1027),
        (513, 4097, 769),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 512, 128),
        (8192, 1024, 256),
        (128, 512, 4096),
        (256, 1024, 8192),
        (64, 8192, 64),
        (16, 16384, 16),
        (128, 32768, 128),
        (256, 16384, 256),
        (4096, 256, 4096),
        (8192, 512, 8192),
        (2048, 1024, 4096),
        (4096, 4096, 1024),
        (1024, 4096, 4096),
    ]

    # shapes = [
    #     (256, 256, 256),
    #     (512, 512, 512),
    #     (768, 768, 768),
    #     (1024, 1024, 1024),
    #     (1280, 1280, 1280),
    #     (1536, 1536, 1536),
    #     (1792, 1792, 1792),
    #     (2048, 2048, 2048),
    #     (2560, 2560, 2560),
    #     (3072, 3072, 3072),
    #     (3584, 3584, 3584),
    #     (4096, 4096, 4096),
    #     (5120, 5120, 5120),
    #     (6144, 6144, 6144),
    #     (7168, 7168, 7168),
    #     (8192, 8192, 8192),
    #     (12000, 12000, 12000),
    #     (16000, 16000, 16000),
    # ]

    # shapes = [
    #     (16, 8192, 16),
    #     (16, 16384, 16),
    #     (16, 32768, 16),
    #     (16, 65536, 16),

    #     (32, 8192, 32),
    #     (32, 16384, 32),
    #     (32, 32768, 32),
    #     (32, 65536, 32),

    #     (64, 8192, 64),
    #     (64, 16384, 64),
    #     (64, 32768, 64),
    #     (64, 65536, 64),

    #     (128, 8192, 128),
    #     (128, 16384, 128),
    #     (128, 32768, 128),
    #     (128, 65536, 128),

    #     (256, 8192, 256),
    #     (256, 16384, 256),
    #     (256, 32768, 256),
    #     (256, 65536, 256),

    #     (512, 8192, 512),
    #     (512, 16384, 512),
    #     (512, 32768, 512),
    #     (512, 65536, 512),
    # ]

    for dtype in DTYPES:
        print("\n" + "=" * 100)
        print(f"Dtype = {dtype}")
        print("=" * 100)

        for M, K, N in shapes:
            print(f"\nShape = ({M}, {K}, {N})")
            print("-" * 100)

            x = torch.randn((M, K), device=DEVICE, dtype=dtype)
            y = torch.randn((K, N), device=DEVICE, dtype=dtype)

            benchmark(x, y)


if __name__ == "__main__":
    main()