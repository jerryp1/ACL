# Baseline Compression Methods

Clean implementations of baseline compression methods for LLM checkpoint
delta compression.

## Directory Layout

```
baseline/
├── README.md                    
└── src/
    ├── README.md                          
    ├── benchmark_compression_methods.py   
    └── benchmark_hybrid_delta.py          
```

## Quick Start

```bash
# 6-method end-to-end benchmark
./run.sh methods --base /path/to/base.pt --ft /path/to/ft.pt

# 3-transform × 2-encoding detailed analysis
./run.sh hybrid --base /path/to/base.pt --ft /path/to/ft.pt
```


## Methods & References

| Method | Reference | Description |
|--------|-----------|-------------|
| **hybrid** | Ours | Adaptive compression |
| **fmdelta** | FM-Delta | PCMap + range coding |
| **xor_zstd** | ZipLLM | XOR delta + Zstandard |
| **exponent_huffman** | ZipNN | Exponent-bit extraction + Huffman/ZSTD |
| **rangecode** | rangecoding | Static Range Coding |
| **zstd** | zstd | Raw Zstandard |

