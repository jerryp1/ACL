#!/usr/bin/env python3
"""Baseline compression methods benchmark.

Usage:
    python benchmark_compression_methods.py --base BASE.pt --ft FT.pt --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import struct
import sys
import time
import types as _types
from typing import Dict, List, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

_SRC_DIR = os.path.join(_PROJECT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _so_dir in [
    os.path.join(_PROJECT_DIR, "CodeTensors", "python", "install"),
    os.path.join(_PROJECT_DIR, "CodeTensors", "cpp", "build", "Release", "src"),
]:
    if os.path.isdir(_so_dir) and _so_dir not in sys.path:
        sys.path.insert(0, _so_dir)

_INSTALLED_STUBS: set = set()

def _install_module_stubs(prefix: str) -> None:
    if prefix in _INSTALLED_STUBS:
        return

    class _StubModule(_types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__file__ = f"<stub:{name}>"

        def __getattr__(self, name):
            full = f"{self.__name__}.{name}"
            if full in sys.modules:
                return sys.modules[full]
            return type(name, (), {
                "__module__": self.__name__,
                "__init__": lambda self, *a, **kw: self.__dict__.update(kw),
                "__reduce__": lambda self: (dict, ()),
                "__reduce_ex__": lambda self, p: (dict, ()),
            })

        def __call__(self, *a, **kw):
            return {}

    class _StubFinder:
        def find_module(self, fullname, path=None):
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _StubModule(fullname)
            sys.modules[fullname] = mod
            return mod


def torch_load_checkpoint(path: str, max_retries: int = 5):
    import torch

    for _ in range(max_retries):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except (ModuleNotFoundError, ImportError) as exc:
            mod_str = str(exc)
            mod_name = mod_str.split("'")[1] if "'" in mod_str else mod_str.split()[-1]
            _install_module_stubs(mod_name.split(".")[0])
    raise RuntimeError(f"torch.load failed after {max_retries} stub retries for {path}")

for _stub_prefix in ("deepspeed", "apex", "flash_attn", "megatron"):
    _install_module_stubs(_stub_prefix)

import torch  # noqa: E402

SKIP_DTYPES = {
    torch.int32, torch.int64, torch.int16, torch.int8,
    torch.uint8, torch.bool,
}

import tensor_compress as tc          # noqa: E402
import modelformat_encodings as _mfe   # noqa: E402
import fmdelta_encodings as fmd        # noqa: E402

def method_drotl1fmd_encode(base_f32: np.ndarray, ft_f32: np.ndarray) -> bytes:
    return fmd.drotl1fmd_encode_f32(base_f32.tobytes(), ft_f32.tobytes())

def method_drotl1fmd_decode(encoded: bytes, base_f32: np.ndarray) -> np.ndarray:
    return np.frombuffer(fmd.drotl1fmd_decode_f32(encoded, base_f32.tobytes()), dtype=np.float32)

def method_fmdelta_encode(base_f32: np.ndarray, ft_f32: np.ndarray) -> bytes:
    return fmd.encode_f32(base_f32.tobytes(), ft_f32.tobytes())

def method_fmdelta_decode(encoded: bytes, base_f32: np.ndarray) -> np.ndarray:
    return np.frombuffer(fmd.decode_f32(encoded, base_f32.tobytes()), dtype=np.float32)

def method_zstd_encode(base_f32: np.ndarray, ft_f32: np.ndarray) -> bytes:
    return _mfe.compress("ZSTD", ft_f32.tobytes(), 1)

def method_zstd_decode(encoded: bytes, base_f32: np.ndarray) -> np.ndarray:
    return np.frombuffer(_mfe.decompress("ZSTD", encoded, len(base_f32) * 4), dtype=np.float32)

def method_xor_zstd_encode(base_f32: np.ndarray, ft_f32: np.ndarray) -> bytes:
    xor_delta = base_f32.view(np.uint32) ^ ft_f32.view(np.uint32)
    return _mfe.compress("ZSTD", xor_delta.tobytes(), 1)

def method_xor_zstd_decode(encoded: bytes, base_f32: np.ndarray) -> np.ndarray:
    xor_u32 = np.frombuffer(_mfe.decompress("ZSTD", encoded, len(base_f32) * 4), dtype=np.uint32)
    return (base_f32.view(np.uint32) ^ xor_u32).view(np.float32)

# =====================================================================
# RangeCode (static range coding on raw float32 bytes, no delta)
# =====================================================================

_RC_CHUNK_ELEMS = 50_000_000


_RC_CHUNK_ELEMS = 50_000_000

def method_rangecode_encode(base_f32: np.ndarray, ft_f32: np.ndarray) -> bytes:
    import struct as _struct

    ft_bytes = ft_f32.tobytes()
    n_total = len(ft_bytes) // 4
    if n_total == 0:
        return _struct.pack("<I", 0)

    chunks = []
    chunk_lens = []
    for start in range(0, n_total, _RC_CHUNK_ELEMS):
        end = min(start + _RC_CHUNK_ELEMS, n_total)
        chunk_data = ft_bytes[start * 4 : end * 4]
        enc = _mfe.tensor_encode(chunk_data, _mfe.Dtype.F32, "STATIC_RC_FLOAT32")
        chunks.append(enc)
        chunk_lens.append(len(enc))

    header = _struct.pack("<I", len(chunks)) + _struct.pack(f"<{len(chunks)}I", *chunk_lens)
    return header + b"".join(chunks)


def method_rangecode_decode(encoded: bytes, base_f32: np.ndarray) -> np.ndarray:
    import struct as _struct

    n_total = len(base_f32)
    if n_total == 0:
        return np.array([], dtype=np.float32)

    num_chunks = _struct.unpack("<I", encoded[:4])[0]
    chunk_lens = list(_struct.unpack(f"<{num_chunks}I", encoded[4 : 4 + 4 * num_chunks]))

    out_parts = []
    cursor = 4 + 4 * num_chunks
    elem_cursor = 0
    for i, clen in enumerate(chunk_lens):
        if i < num_chunks - 1:
            n_chunk_elems = _RC_CHUNK_ELEMS
        else:
            n_chunk_elems = n_total - elem_cursor
        chunk_enc = encoded[cursor : cursor + clen]
        dec = _mfe.tensor_decode(chunk_enc, _mfe.Dtype.F32, "STATIC_RC_FLOAT32", n_chunk_elems)
        out_parts.append(dec)
        cursor += clen
        elem_cursor += n_chunk_elems

    return np.frombuffer(b"".join(out_parts), dtype=np.float32)


def method_hybrid_encode(base_f32: np.ndarray, ft_f32: np.ndarray, tensor_category: str) -> bytes:
    if tensor_category in ("weight", "exp_avg_sq"):
        return method_fmdelta_encode(base_f32, ft_f32)
    return method_drotl1fmd_encode(base_f32, ft_f32)

def method_hybrid_decode(encoded: bytes, base_f32: np.ndarray, tensor_category: str) -> np.ndarray:
    if tensor_category in ("weight", "exp_avg_sq"):
        return method_fmdelta_decode(encoded, base_f32)
    return method_drotl1fmd_decode(encoded, base_f32)

COMPRESSION_METHODS = {
    "hybrid": {
        "name": "Hybrid",
        "encode": method_hybrid_encode,
        "decode": method_hybrid_decode,
        "requires_base": True,
        "needs_category": True,
    },
    "fmdelta": {
        "name": "PCMap+FMDelta",
        "encode": method_fmdelta_encode,
        "decode": method_fmdelta_decode,
        "requires_base": True,
        "needs_category": False,
    },
    "zstd": {
        "name": "zstd",
        "encode": method_zstd_encode,
        "decode": method_zstd_decode,
        "requires_base": False,
        "needs_category": False,
    },
    "xor_zstd": {
        "name": "ZipLLM (XOR+zstd)",
        "encode": method_xor_zstd_encode,
        "decode": method_xor_zstd_decode,
        "requires_base": True,
        "needs_category": False,
    },
    "rangecode": {
        "name": "RangeCode",
        "encode": method_rangecode_encode,
        "decode": method_rangecode_decode,
        "requires_base": False,
        "needs_category": False,
    },
}

def tensor_category(path_str: str) -> str:
    parts = path_str.split(".")
    for p in reversed(parts):
        if p == "exp_avg":
            return "exp_avg"
        if p == "exp_avg_sq":
            return "exp_avg_sq"
    return "weight"

def traverse_with_path(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from traverse_with_path(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from traverse_with_path(v, f"{prefix}.{i}" if prefix else str(i))

def benchmark_method(method_name, base_f32, ft_f32, num_iterations=3, tensor_category="weight"):
    if method_name not in COMPRESSION_METHODS:
        return {"error": f"Unknown method: {method_name}"}

    method = COMPRESSION_METHODS[method_name]
    encode_fn = method["encode"]
    decode_fn = method["decode"]
    needs_category = method.get("needs_category", False)

    orig_bytes = len(ft_f32) * 4
    results = {
        "method": method_name,
        "method_name": method["name"],
        "orig_bytes": orig_bytes,
        "tensor_category": tensor_category,
    }

    try:
        enc_times = []
        compressed_sizes = []
        encoded_data = None

        for _ in range(num_iterations):
            t0 = time.perf_counter()
            if needs_category:
                encoded = encode_fn(base_f32, ft_f32, tensor_category)
            else:
                encoded = encode_fn(base_f32, ft_f32)
            enc_times.append(time.perf_counter() - t0)
            compressed_sizes.append(len(encoded))
            if encoded_data is None:
                encoded_data = encoded

        dec_times = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            if needs_category:
                decoded = decode_fn(encoded_data, base_f32, tensor_category)
            else:
                decoded = decode_fn(encoded_data, base_f32)
            dec_times.append(time.perf_counter() - t0)

        enc_time_avg = np.mean(enc_times)
        dec_time_avg = np.mean(dec_times)
        compressed_size = compressed_sizes[0]
        is_correct = np.array_equal(decoded.view(np.uint32), ft_f32.view(np.uint32))

        results.update({
            "compressed_bytes": compressed_size,
            "ratio": compressed_size / orig_bytes,
            "enc_time": enc_time_avg,
            "dec_time": dec_time_avg,
            "enc_mbps": (orig_bytes / enc_time_avg / 1e6) if enc_time_avg > 0 else float('inf'),
            "dec_mbps": (orig_bytes / dec_time_avg / 1e6) if dec_time_avg > 0 else float('inf'),
            "is_correct": is_correct,
            "error": None,
        })

    except Exception as e:
        results.update({"error": str(e), "is_correct": False})

    return results

def main():
    parser = argparse.ArgumentParser(description="Baseline compression benchmark")
    parser.add_argument("--base", required=True, help="Base checkpoint .pt path")
    parser.add_argument("--ft", required=True, help="Fine-tuned checkpoint .pt path")
    parser.add_argument("--output", type=str, default="benchmark_compression_methods.csv")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list")
    parser.add_argument("--tensor-idx", type=int, nargs="*", default=None, help="Tensor indices to test (0-based)")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        invalid = [m for m in methods if m not in COMPRESSION_METHODS]
        if invalid:
            print(f"Unknown methods: {invalid}")
            print(f"Available: {', '.join(COMPRESSION_METHODS.keys())}")
            sys.exit(1)
    else:
        methods = list(COMPRESSION_METHODS.keys())
    
    print("=" * 120)
    print(f"  Baseline Compression Benchmark")
    print(f"  BASE: {args.base}")
    print(f"  FT:   {args.ft}")
    print(f"  Methods: {', '.join([COMPRESSION_METHODS[m]['name'] for m in methods])}")
    print(f"  Iterations: {args.iterations}")
    print("=" * 120)

    print("\nLoading BASE ...")
    t0 = time.perf_counter()
    base_ckpt = torch_load_checkpoint(args.base)
    base_tensors = list(traverse_with_path(base_ckpt))
    print(f"  {len(base_tensors)} tensors, loaded in {time.perf_counter()-t0:.1f}s")

    print("Loading FT ...")
    t0 = time.perf_counter()
    ft_ckpt = torch_load_checkpoint(args.ft)
    ft_tensors = list(traverse_with_path(ft_ckpt))
    print(f"  {len(ft_tensors)} tensors, loaded in {time.perf_counter()-t0:.1f}s")

    base_map = {p: t for p, t in base_tensors}
    compressible = []
    for path_str, ft_t in ft_tensors:
        if ft_t.dtype in SKIP_DTYPES or ft_t.numel() < 64:
            continue
        bt = base_map.get(path_str)
        if bt is not None and bt.dtype in SKIP_DTYPES:
            continue
        compressible.append((path_str, ft_t, bt))
    
    if args.tensor_idx:
        selected = [(i, compressible[i]) for i in args.tensor_idx if i < len(compressible)]
    else:
        selected = list(enumerate(compressible))
    
    print(f"\nCompressible tensors: {len(compressible)}, testing: {len(selected)}")

    csv_rows = []
    method_header = "  ".join(f"{m:>15s}" for m in methods)
    print(f"\n{'#':>3} {'Tensor':>45s} {'cat':>10s} {'numel':>12s} | {method_header}")
    print("-" * (80 + 20 * len(methods)))
    
    for idx, (path_str, ft_t, bt) in selected:
        cat = tensor_category(path_str)
        ft_f32 = ft_t.detach().float().numpy().flatten()
        base_f32 = bt.detach().float().numpy().flatten() if bt is not None else None
        
        short_name = path_str if len(path_str) <= 45 else "..." + path_str[-(45-3):]

        method_results = {}
        for method_name in methods:
            if base_f32 is None and COMPRESSION_METHODS[method_name]["requires_base"]:
                method_results[method_name] = {
                    "error": "No base tensor",
                    "is_correct": False,
                    "ratio": float('inf'),
                    "enc_mbps": 0,
                    "dec_mbps": 0,
                }
            else:
                result = benchmark_method(method_name, base_f32, ft_f32, args.iterations, cat)
                method_results[method_name] = result

        detail_parts = []
        for method_name in methods:
            res = method_results[method_name]
            if res.get("error"):
                detail_parts.append(f"{'ERROR':>15s}")
            elif not res.get("is_correct"):
                detail_parts.append(f"{'WRONG':>15s}")
            else:
                ratio = res["ratio"]
                enc_mbps = res["enc_mbps"]
                dec_mbps = res["dec_mbps"]
                detail_parts.append(f"{ratio:.3f} {enc_mbps:>5.0f}/{dec_mbps:>5.0f}")
        
        detail_str = "  ".join(detail_parts)
        print(f"{idx+1:>3} {short_name:>45s} {cat:>10s} {ft_t.numel():>12,d} | {detail_str}")

        for method_name in methods:
            res = method_results[method_name]
            csv_rows.append({
                "tensor_idx": idx + 1,
                "tensor_name": path_str,
                "category": cat,
                "numel": ft_t.numel(),
                "method": method_name,
                "method_display": COMPRESSION_METHODS[method_name]["name"],
                "compressed_bytes": res.get("compressed_bytes", 0),
                "ratio": res.get("ratio", float('inf')),
                "enc_time": res.get("enc_time", 0),
                "dec_time": res.get("dec_time", 0),
                "enc_mbps": res.get("enc_mbps", 0),
                "dec_mbps": res.get("dec_mbps", 0),
                "is_correct": res.get("is_correct", False),
                "error": res.get("error", ""),
            })
        
        gc.collect()

    print(f"\n{'=' * 120}")
    print("Summary (average over all tensors)")
    print(f"{'=' * 120}")
    print(f"{'Method':>20s} {'Avg Ratio':>10s} {'Avg Enc MB/s':>14s} {'Avg Dec MB/s':>14s} {'Correct':>8s}")
    print("-" * 70)
    
    for method_name in methods:
        method_rows = [r for r in csv_rows if r["method"] == method_name]
        if not method_rows:
            continue
        
        valid_rows = [r for r in method_rows if r["is_correct"]]
        if not valid_rows:
            print(f"{COMPRESSION_METHODS[method_name]['name']:>20s} {'N/A':>10s} {'N/A':>14s} {'N/A':>14s} {'0/0':>8s}")
            continue
        
        avg_ratio = np.mean([r["ratio"] for r in valid_rows])
        avg_enc = np.mean([r["enc_mbps"] for r in valid_rows])
        avg_dec = np.mean([r["dec_mbps"] for r in valid_rows])
        correct_count = len(valid_rows)
        total_count = len(method_rows)
        
        print(f"{COMPRESSION_METHODS[method_name]['name']:>20s} {avg_ratio:>10.4f} {avg_enc:>14.1f} {avg_dec:>14.1f} {correct_count}/{total_count:>4d}")
    
    large_rows = [r for r in csv_rows if r["numel"] > 10_000_000]
    if large_rows:
        print(f"\n{'=' * 120}")
        print("Large tensors (numel > 10M)")
        print(f"{'=' * 120}")
        print(f"{'Method':>20s} {'Avg Ratio':>10s} {'Avg Enc MB/s':>14s} {'Avg Dec MB/s':>14s}")
        print("-" * 62)
        
        for method_name in methods:
            method_rows = [r for r in large_rows if r["method"] == method_name]
            if not method_rows:
                continue
            
            valid_rows = [r for r in method_rows if r["is_correct"]]
            if not valid_rows:
                continue
            
            avg_ratio = np.mean([r["ratio"] for r in valid_rows])
            avg_enc = np.mean([r["enc_mbps"] for r in valid_rows])
            avg_dec = np.mean([r["dec_mbps"] for r in valid_rows])
            
            print(f"{COMPRESSION_METHODS[method_name]['name']:>20s} {avg_ratio:>10.4f} {avg_enc:>14.1f} {avg_dec:>14.1f}")
    
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[OUTPUT] Results written to: {args.output}")

    print(f"\n{'=' * 120}")
    print("Correctness check")
    print(f"{'=' * 120}")
    for method_name in methods:
        method_rows = [r for r in csv_rows if r["method"] == method_name]
        if not method_rows:
            continue
        
        correct_count = sum(1 for r in method_rows if r["is_correct"])
        total_count = len(method_rows)
        
        if correct_count == total_count:
            print(f"✓ {COMPRESSION_METHODS[method_name]['name']:>20s}: {correct_count}/{total_count} correct")
        else:
            print(f"✗ {COMPRESSION_METHODS[method_name]['name']:>20s}: {correct_count}/{total_count} correct")
            error_rows = [r for r in method_rows if not r["is_correct"]]
            for err_row in error_rows[:3]:
                print(f"    - {err_row['tensor_name']}: {err_row.get('error', 'Unknown error')}")
            if len(error_rows) > 3:
                print(f"    ... and {len(error_rows) - 3} more errors")

if __name__ == "__main__":
    main()
