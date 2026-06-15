# Baseline Compression Methods

Clean implementation of five baseline compression methods for LLM training
checkpoint delta compression.

## Files

| File | Description |
|------|-------------|
| `benchmark_compression_methods.py` | End-to-end benchmark comparing all 6 methods on a checkpoint pair |
| `benchmark_hybrid_delta.py` | Detailed analysis of 3 transforms × 2 encodings per tensor |

## Methods

| Method | Delta? | Reference | Description |
|--------|--------|-----------|-------------|
| **hybrid** | ✓ | Ours | Adaptive: weight/exp_avg_sq → PCMap+FMDelta; exp_avg → ROTL1+FMDelta |
| **fmdelta** | ✓ | FM-Delta [14] | Uniform PCMap + FMDelta PCEncoder |
| **rangecode** | ✗ | — | Static Range Coding on raw float32 bytes (no delta) |
| **xor_zstd** | ✓ | ZipLLM [15] | XOR delta + Zstandard |
| **exponent_huffman** | ✗ | ZipNN [12] | Exponent-bit extraction + Huffman/ZSTD coding |
| **zstd** | ✗ | zstd [10] | Raw Zstandard on finetuned tensor (no delta) |

### References

- **[10]** Collet, Y. *Zstandard – Real-time data compression algorithm.* Facebook/Meta, 2016. https://facebook.github.io/zstd/
- **[12]** Ben-Nun, T., et al. *ZipNN: Lossless Compression for Neural Network Weights.* arXiv:2408.07429, 2024.
- **[14]** Chen, L., et al. *FM-Delta: Lossless Compression of Fine-Tuning Checkpoints via Float Monotone Mapping.* (Internal / preprint)
- **[15]** ZipLLM. *XOR-based delta compression for LLM training checkpoints.* (Internal / preprint)

## Prerequisites

```bash
export PYTHONPATH=/path/to/CodeCheckpoint-clean/src:$PYTHONPATH
export PYTHONPATH=/path/to/CodeCheckpoint-clean/CodeTensors/python/install:$PYTHONPATH
```

Required Python packages: `torch`, `numpy`.
Required C++ extensions: `tensor_compress`, `modelformat_encodings`, `fmdelta_encodings`.

## Usage

### `benchmark_compression_methods.py` — 6-method comparison

```bash
python benchmark_compression_methods.py \
    --base   /path/to/base_checkpoint.pt \
    --ft     /path/to/finetuned_checkpoint.pt \
    --output results.csv
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--base` | ✓ | Base checkpoint `.pt` path |
| `--ft` | ✓ | Fine-tuned checkpoint `.pt` path |
| `--output` | | Output CSV path (default: `benchmark_compression_methods.csv`) |
| `--methods` | | Comma-separated method list, e.g. `hybrid,fmdelta,rangecode` (default: all 6) |
| `--tensor-idx` | | Only test specific tensor indices, e.g. `0 1 4` (default: all) |
| `--iterations` | | Iterations per method for throughput averaging (default: 3) |

**Workflow:** loads both checkpoints → matches tensors by name/shape → runs all 6 methods
on each tensor pair → reports compression ratio, encode/decode throughput, and
bitwise-exact correctness.

### `benchmark_hybrid_delta.py` — transform × encoding analysis

```bash
python benchmark_hybrid_delta.py \
    --base /path/to/base.pt \
    --ft   /path/to/ft.pt
```

Compares Delta / Delta-rotl1 / Hybrid transforms with ByteCol-RC and FMD-style
encodings on every compressible tensor. Reports per-tensor and aggregate ratios.

## Output CSV Schema