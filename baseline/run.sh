#!/bin/bash
# ============================================================
# Baseline Compression Benchmark — unified runner
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Prefer Python 3.10+ for from __future__ support; .so files compiled for cp311
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
for _py in python3.12 python3.11 python3.10; do
    if command -v "$_py" &>/dev/null; then
        PYTHON_BIN="$_py"; break
    fi
done

# Auto-add bundled .so libraries and src to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/lib:$SCRIPT_DIR/../src:$PYTHONPATH"

usage() {
    cat <<EOF
Usage:
  run methods --base /path/to/base.pt --ft /path/to/ft.pt [--output results.csv]
  run hybrid  --base /path/to/base.pt --ft /path/to/ft.pt [--tensor-idx 0 1 2]
  run help

Subcommands:
  methods   6-method end-to-end benchmark (hybrid, fmdelta, rangecode, ZipLLM, ZipNN, zstd)
  hybrid    3-transform × 2-encoding detailed analysis (Delta, Delta-rotl1, Hybrid × ByteCol/FMD)

Options:
  --base PATH        Base checkpoint .pt path (required)
  --ft PATH          Fine-tuned checkpoint .pt path (required)
  --output FILE      CSV output path [default: benchmark_compression_methods.csv]
  --methods LIST     Comma-separated method list (e.g. hybrid,fmdelta,rangecode)
  --tensor-idx N..   Tensor indices to test (0-based, optional)
  --iterations N     Iterations per method [default: 3]
EOF
    exit 1
}

[[ $# -lt 2 ]] && usage
MODE="$1"; shift

case "$MODE" in
    methods)
        exec $PYTHON_BIN "$SCRIPT_DIR/src/benchmark_compression_methods.py" "$@"
        ;;
    hybrid)
        exec $PYTHON_BIN "$SCRIPT_DIR/src/benchmark_hybrid_delta.py" "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        ;;
esac
