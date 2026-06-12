# ALC: an adaptive lossless compression framework for LLM checkpoints.

## Environment

- Python ≥ 3.9, NumPy ≥ 1.24, PyTorch ≥ 2.0
- C++20 compiler (for building the extension)
- pybind11 (`pip install pybind11`)
- deepspeed (optional, for optimizer states)

```bash
pip install -r requirements.txt
```

## Build C++ Extension

Pre-built `.so` is included. To rebuild:

```bash
cd src/cpp
c++ -O3 -march=native -shared -std=c++20 -fPIC \
    $(python -m pybind11 --includes) \
    drotl1fmd_tans_cpp_codec.cpp \
    -o drotl1fmd_tans_cpp_codec$(python3-config --extension-suffix)
```

## Usage

```bash

# Compress .pt
./compress.sh --input /path/to/checkpoint-10/global_step10/mp_rank_00_model_states.pt \
              --base /path/to/checkpoint-0/global_step0/mp_rank_00_model_states.pt \
              --output_dir ./compressed
```


## Algorithm

`analyze_strategy()` samples ~64K elements, estimates entropy for three paths, and selects the minimum:

| Path | Transform  |
|------|-----------|
| **pcdelta** | `pcmap(ft) − pcmap(base)` → (s,k) symbols + mantissas | 
| **rotdelta** | `rotl1(ft) − rotl1(base)` → (s,k) symbols + mantissas |
| **I-Align** | `rotl1(ft)` → 4 byte columns → tANS each | 

All paths use C++ tANS entropy coding. 


