# Inference Kernels

A GPU kernel library focused on inference workloads, built with Triton. The library provides
autotuned implementations of the primitives that dominate transformer inference: attention,
matrix multiplication, normalization, softmax, and elementwise operations. Each kernel is
correctness tested against a PyTorch reference implementation and benchmarked against PyTorch
eager execution.

Verified on Ampere GPUs, with Ampere as the primary target architecture. The kernels are expected
to run correctly on Ada Lovelace, Hopper, and Blackwell as well, though tuning has so far focused
on Ampere.

## Installation

```bash
git clone <repo-url>
cd triton-kernels
uv sync
```

This project uses `uv` for dependency management. Running `uv sync` creates a virtual environment
and installs all dependencies pinned in `uv.lock`.

## Usage

Kernels are exposed as ordinary functions that accept and return PyTorch tensors, with no custom
tensor types or additional setup required.

### General matrix multiplication

`gemm` selects automatically between a tiled autotuned kernel, a persistent grouped kernel, and a
persistent split K kernel depending on the shapes of the inputs.

```python
import torch
from kernels.gemm import gemm

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

out = gemm(x, y)
```

### Flash attention

`flash_attention_v2` supports three input layouts through a single entry point: standard dense
batched attention, ragged (variable length) sequences specified through `q_indptr` and
`kv_indptr`, and paged key value cache attention specified through `k_block_table` and
`v_block_table`.

```python
import torch
from kernels.attention import flash_attention_v2

batch, q_heads, kv_heads, seq_len, head_dim = 2, 32, 8, 2048, 128

q = torch.randn(batch, q_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

out = flash_attention_v2(q, k, v, is_causal=True)
```

Grouped query and multi query attention are supported whenever the number of query heads is a
multiple of the number of key/value heads. Exact shape requirements for the ragged and paged
modes are documented through the assertions in `kernels/attention.py`.

## Testing

Correctness is verified against PyTorch reference implementations for every kernel. The attention
suite, for example, checks every combination of layout (dense, ragged, paged), attention type
(self, cross), head topology (multi head, grouped query, multi query), causal masking, and dtype
(float16, bfloat16, float32) against exact PyTorch scaled dot product attention output, using
tolerances scaled to sequence length and dtype precision.

```bash
pytest tests/
pytest tests/test_attention.py -v
```

## Benchmarking

Each kernel has a corresponding benchmark script that compares its performance against PyTorch
eager execution across a range of shapes, sequence lengths, and configurations. Across the
library, kernels are generally competitive with PyTorch's built in implementations, matching or
exceeding them on many shapes and configurations while trailing on others. Speculative decode
attention, for instance, consistently outperforms PyTorch across the sequence lengths tested,
with the largest gains at shorter sequences.

Benchmarks were run on an RTX 3050 laptop GPU with 6GB of memory, a modest development GPU rather
than a data center accelerator, so absolute throughput numbers should be read with that in mind.

```bash
python benchmarks/benchmark_gemm.py
python benchmarks/attention.py
```

## Kernel list

### Pointwise
Add with ReLU, bias with GELU, SwiGLU, bias with SiLU and multiply.

### Reductions
Sum, max, mean, argmax.

### Normalization
LayerNorm, residual with RMSNorm, residual with LayerNorm.

### Softmax
Softmax, log softmax.

### Matrix multiplication
Tiled autotuned GEMM, persistent grouped GEMM, persistent grouped split K GEMM, quantized GEMM
variants including INT8, AWQ, and GPTQ.

### QKV projection
Tiled autotuned GEMM, persistent grouped GEMM, persistent grouped split K GEMM, quantized
variants including INT8, AWQ, and GPTQ.

### Attention
FlashAttention v1 and v2, supporting causal and non causal attention, multi head attention,
cross attention, grouped/multi query attention, and ragged sequences. FlashAttention v2
additionally supports paged key value caches. Decode attention with dense, ragged, and paged
variants. Speculative decode attention with dense, ragged, and paged variants.

### Parallel algorithms
Reduction, prefix sum, stream compaction, segmented scan, radix sort.

### Sampling
Temperature scaling, top k, top p, multinomial sampling, and a fused sampling kernel.

### Memory and layout
Copy, contiguous, transpose, type cast, permute, gather, scatter.

### Scaling
Per tensor, per channel, per token, and per group scaling.

### Quantization
FP16 to INT8, FP16 to INT4, INT8 to FP16, INT4 to FP16.

### Additional kernels
Rotary position embedding, fused QKV projection, INT8 GEMM, INT8 QKV projection.

## Project layout

The `kernels` directory contains all kernel implementations. The `tests` directory contains
correctness tests comparing each kernel against its PyTorch reference implementation. The
`benchmarks` directory contains performance benchmarks comparing each kernel against PyTorch
eager execution. The `utils` directory contains shared launch configuration and input validation
helpers used across the kernel implementations.

## Contributing

Issues and pull requests are welcome. Any new kernel should include a correctness test against a
PyTorch reference implementation and a benchmark script, following the conventions used in the
`tests` and `benchmarks` directories.