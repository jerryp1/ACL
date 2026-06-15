#!/usr/bin/env python3
"""Hybrid compression benchmark comparing ALC paths vs baseline methods."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import types as _types

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

_SRC_DIR = os.path.join(_PROJECT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_CANDIDATE_SO_DIRS = [
    os.path.join(_PROJECT_DIR, "CodeTensors", "python", "install"),
    os.path.join(_PROJECT_DIR, "CodeTensors", "cpp", "build", "Release", "src"),
]
for _so_dir in _CANDIDATE_SO_DIRS:
    if os.path.isdir(_so_dir) and _so_dir not in sys.path:
        sys.path.insert(0, _so_dir)

# ---------------------------------------------------------------------------
# Stub loader – allows loading checkpoints saved with optional deps
# ---------------------------------------------------------------------------
_INSTALLED_STUBS: set = set()


_INSTALLED_STUBS: set = set()

def _install_module_stubs(prefix: str) -> None:


def torch_load_checkpoint(path: str, max_retries: int = 5):
    """Load a PyTorch checkpoint, auto-stubbing missing optional dependencies."""
    import torch  # noqa: delayed import

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

CHUNK_ELEMS = 16 * 1024 * 1024
ENCODING = "QS_RC_UINT8"

# =====================================================================
# Checkpoint format detection and tensor extraction
# (same as v2 script)
# =====================================================================
def detect_format(state_dict):
    if isinstance(state_dict, list):
        return "megatron"
    return "zero"

def extract_tensors_megatron(state_dict):
    results = []
    if not isinstance(state_dict, list):
        return extract_tensors_generic(state_dict)
    for group_idx, group in enumerate(state_dict):
        if group is None or not isinstance(group, dict):
            continue
        for bucket_key, bucket_val in group.items():
            if bucket_key == "buckets_coalesced" or not isinstance(bucket_val, dict):
                continue
            for dtype_key, tensor_dict in bucket_val.items():
                if not isinstance(tensor_dict, dict):
                    continue
                for tensor_name, tensor_val in tensor_dict.items():
                    if not torch.is_tensor(tensor_val):
                        continue
                    if tensor_val.dtype in SKIP_DTYPES or tensor_val.numel() < 64:
                        continue
                    path = f"group{group_idx}.bucket{bucket_key}.{tensor_name}"
                    category = _category_from_tensor_name(tensor_name)
                    results.append((path, tensor_val, category))
    return results

def extract_tensors_zero(state_dict):
    results = []
    def traverse(obj, prefix=""):
        if torch.is_tensor(obj):
            if obj.dtype not in SKIP_DTYPES and obj.numel() >= 64:
                results.append((prefix, obj, _category_from_path(prefix)))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                traverse(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                traverse(v, f"{prefix}.{i}" if prefix else str(i))
    traverse(state_dict)
    return results

def extract_tensors_generic(state_dict):
    return extract_tensors_zero(state_dict)

def _category_from_path(path_str):
    parts = path_str.split(".")
    for part in reversed(parts):
        if part == "exp_avg":
            return "exp_avg"
        if part == "exp_avg_sq":
            return "exp_avg_sq"
    return "weight"

def _category_from_tensor_name(name):
    if name == "exp_avg":
        return "exp_avg"
    if name == "exp_avg_sq":
        return "exp_avg_sq"
    return "weight"

# =====================================================================
# 1. Three data transforms
# =====================================================================

def transform_delta(base_f32, ft_f32):
    """pcmap delta (= FMDelta way): abs(pcmap(ft) - pcmap(base)), direction separate."""
    pm_b = tc.pcmap_f32(base_f32)
    pm_f = tc.pcmap_f32(ft_f32)
    abs_delta = np.where(pm_f >= pm_b, pm_f - pm_b, pm_b - pm_f).astype(np.uint32)
    direction = np.where(pm_f < pm_b, np.uint32(1), np.uint32(0))
    return abs_delta, direction

def transform_delta_rotl1(base_f32, ft_f32):
    """rotl1 delta: rotl1(ieee) then abs diff, direction separate."""
    ft_rot = tc.rotl1(ft_f32.view(np.uint32))
    base_rot = tc.rotl1(base_f32.view(np.uint32))
    abs_delta = np.where(ft_rot >= base_rot, ft_rot - base_rot, base_rot - ft_rot).astype(np.uint32)
    direction = np.where(ft_rot < base_rot, np.uint32(1), np.uint32(0))
    return abs_delta, direction

def transform_hybrid(base_f32, ft_f32):
    """Hybrid: different-sign elements use rotl1 delta, same-sign use pcmap delta.

    Returns:
        abs_delta: uint32 array
        direction: uint32 array (0=ft>=base in respective space, 1=ft<base)
        sign_flag: uint8 array (1=different sign, 0=same sign)
    """
    base_u32 = base_f32.view(np.uint32)
    ft_u32 = ft_f32.view(np.uint32)

    # Determine sign: bit31
    base_sign = (base_u32 >> 31) & np.uint32(1)
    ft_sign = (ft_u32 >> 31) & np.uint32(1)
    diff_sign_mask = base_sign != ft_sign  # True = different sign
    same_sign_mask = ~diff_sign_mask

    abs_delta = np.empty(len(base_f32), dtype=np.uint32)
    direction = np.empty(len(base_f32), dtype=np.uint32)

    # Same-sign elements: use pcmap delta
    if same_sign_mask.any():
        pm_b = tc.pcmap_f32(base_f32[same_sign_mask])
        pm_f = tc.pcmap_f32(ft_f32[same_sign_mask])
        abs_delta[same_sign_mask] = np.where(
            pm_f >= pm_b, pm_f - pm_b, pm_b - pm_f
        ).astype(np.uint32)
        direction[same_sign_mask] = np.where(
            pm_f < pm_b, np.uint32(1), np.uint32(0)
        )
        del pm_b, pm_f

    # Different-sign elements: use rotl1 delta
    if diff_sign_mask.any():
        ft_rot = tc.rotl1(ft_u32[diff_sign_mask])
        base_rot = tc.rotl1(base_u32[diff_sign_mask])
        abs_delta[diff_sign_mask] = np.where(
            ft_rot >= base_rot, ft_rot - base_rot, base_rot - ft_rot
        ).astype(np.uint32)
        direction[diff_sign_mask] = np.where(
            ft_rot < base_rot, np.uint32(1), np.uint32(0)
        )
        del ft_rot, base_rot

    sign_flag = diff_sign_mask.astype(np.uint8)
    return abs_delta, direction, sign_flag

# =====================================================================
# 2. Entropy & stats utilities
# =====================================================================

def shannon_entropy(counts_array):
    total = counts_array.sum()
    if total == 0:
        return 0.0
    probs = counts_array[counts_array > 0] / total
    ent = float(-np.sum(probs * np.log2(probs)))
    return max(ent, 0.0)

def byte_col_entropy(u32_arr):
    rot = tc.rotl1(u32_arr)
    byte_mat = rot.view(np.uint8).reshape(-1, 4)
    col_ent = []
    for c in range(4):
        counts = np.bincount(byte_mat[:, c], minlength=256)
        col_ent.append(shannon_entropy(counts))
    return col_ent

def nbits_distribution(u32_arr):
    counts = np.zeros(33, dtype=np.int64)
    zero_count = int((u32_arr == 0).sum())
    counts[0] = zero_count
    nonzero = u32_arr > 0
    if nonzero.any():
        nz_vals = u32_arr[nonzero].astype(np.float64)
        nbits_nz = (np.floor(np.log2(nz_vals)) + 1).astype(np.int32)
        nbits_nz = np.clip(nbits_nz, 1, 32)
        nz_counts = np.bincount(nbits_nz, minlength=33)
        counts += nz_counts[:33]
        del nz_vals, nbits_nz, nz_counts
    del nonzero
    total = len(u32_arr)
    pct = counts / total * 100
    return counts, pct

def compute_fmd_stats(abs_delta, direction):
    num_elements = len(abs_delta)
    level_counts = np.zeros(65, dtype=np.int64)
    total_remainder_bits = 0

    zero_mask = abs_delta == 0
    level_counts[32] = int(zero_mask.sum())
    nonzero_mask = ~zero_mask
    del zero_mask

    if nonzero_mask.any():
        nz_delta = abs_delta[nonzero_mask]
        nz_dir = direction[nonzero_mask]
        del nonzero_mask
        k = np.floor(np.log2(nz_delta.astype(np.float64))).astype(np.int32)
        k = np.clip(k, 0, 31)
        nz_levels = np.where(nz_dir == 0, 33 + k, 31 - k).astype(np.int32)
        nz_level_counts = np.bincount(nz_levels.clip(0, 64), minlength=65)
        level_counts += nz_level_counts[:65]
        total_remainder_bits = int(k.sum())
        del nz_delta, nz_dir, k, nz_levels, nz_level_counts

    level_ent = shannon_entropy(level_counts)
    avg_remainder_bits = total_remainder_bits / num_elements if num_elements > 0 else 0.0
    return level_counts, level_ent, avg_remainder_bits, total_remainder_bits

# =====================================================================
# 3. Compression
# =====================================================================

def bytecol_compress(u32_arr):
    total_size = 0
    total_enc_sec = 0.0
    for start in range(0, len(u32_arr), CHUNK_ELEMS):
        end = min(start + CHUNK_ELEMS, len(u32_arr))
        chunk = u32_arr[start:end]
        rot = tc.rotl1(chunk)
        byte_mat = rot.view(np.uint8).reshape(-1, 4)
        t0 = time.perf_counter()
        for col_idx in range(4):
            col_bytes = np.ascontiguousarray(byte_mat[:, col_idx]).tobytes()
            encoded = _mfe.tensor_encode(col_bytes, _mfe.U8, ENCODING)
            total_size += len(encoded)
        total_enc_sec += time.perf_counter() - t0
    return total_size, total_enc_sec

def compress_direction(direction):
    dir_bytes = direction.astype(np.uint8).tobytes()
    t0 = time.perf_counter()
    dir_encoded = _mfe.tensor_encode(dir_bytes, _mfe.U8, ENCODING)
    dir_sec = time.perf_counter() - t0
    return len(dir_encoded), dir_sec

def fmd_style_compress(abs_delta, direction, numel):
    """FMD-style: RC encode level sequence + remainder raw bits."""
    levels = np.full(numel, 32, dtype=np.uint8)
    total_remainder_bits = 0

    nonzero = abs_delta > 0
    if nonzero.any():
        nz = abs_delta[nonzero]
        nz_dir = direction[nonzero]
        k = np.floor(np.log2(nz.astype(np.float64))).astype(np.int32)
        k = np.clip(k, 0, 31)
        nz_levels = np.where(nz_dir == 0, 33 + k, 31 - k).astype(np.uint8)
        levels[nonzero] = nz_levels
        total_remainder_bits = int(k.sum())
        del nz, nz_dir, k, nz_levels
    del nonzero

    level_bytes = levels.tobytes()
    del levels
    t0 = time.perf_counter()
    level_encoded = _mfe.tensor_encode(level_bytes, _mfe.U8, ENCODING)
    level_sec = time.perf_counter() - t0
    level_sz = len(level_encoded)
    del level_bytes, level_encoded

    remainder_bytes = (total_remainder_bits + 7) // 8
    total_sz = level_sz + remainder_bytes
    return level_sz, remainder_bytes, total_sz, level_sec

def benchmark_fmdelta_original(ft_f32, base_f32):
    base_bytes = base_f32.tobytes()
    ft_bytes = ft_f32.tobytes()
    t0 = time.perf_counter()
    encoded = fmd.encode_f32(base_bytes, ft_bytes)
    enc_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    decoded = fmd.decode_f32(encoded, base_bytes)
    dec_sec = time.perf_counter() - t0
    rec_u32 = np.frombuffer(decoded, dtype=np.uint32)
    bitwise_ok = np.array_equal(rec_u32, ft_f32.view(np.uint32))
    return len(encoded), enc_sec, dec_sec, bitwise_ok

# =====================================================================
# Per-tensor analysis
# =====================================================================

def analyze_tensor(ft_f32, base_f32, idx, cat, short_name):
    numel = len(ft_f32)
    orig_bytes = numel * 4

    # Sign analysis
    base_u32 = base_f32.view(np.uint32)
    ft_u32 = ft_f32.view(np.uint32)
    base_sign = (base_u32 >> 31) & np.uint32(1)
    ft_sign = (ft_u32 >> 31) & np.uint32(1)
    diff_sign_count = int((base_sign != ft_sign).sum())
    same_sign_count = numel - diff_sign_count
    diff_sign_pct = diff_sign_count / numel * 100
    sign_flag_ent = shannon_entropy(np.array([same_sign_count, diff_sign_count], dtype=np.int64))

    print(f"\n{'='*120}")
    print(f"  [{idx}] {short_name}")
    print(f"      type: {cat}, elements: {numel:,}, original: {orig_bytes/1e6:.1f}MB")
    print(f"      sign flips: {diff_sign_count:,} ({diff_sign_pct:.1f}%), same sign: {same_sign_count:,} ({100-diff_sign_pct:.1f}%)")
    print(f"      sign_flag entropy: {sign_flag_ent:.3f} bit")
    print(f"{'='*120}", flush=True)

    transform_names = ["Delta", "Delta-rotl1", "Hybrid"]
    all_compress = {}

    for tname in transform_names:
        print(f"\n  ── {tname} ──", flush=True)

        # --- Transform ---
        sign_flag = None
        if tname == "Delta":
            abs_delta, direction = transform_delta(base_f32, ft_f32)
        elif tname == "Delta-rotl1":
            abs_delta, direction = transform_delta_rotl1(base_f32, ft_f32)
        else:
            abs_delta, direction, sign_flag = transform_hybrid(base_f32, ft_f32)

        # --- Stats ---
        col_ent = byte_col_entropy(abs_delta)
        col_avg = float(np.mean(col_ent))
        nbits_counts, nbits_pct = nbits_distribution(abs_delta)
        nbits_ent = shannon_entropy(nbits_counts)
        level_counts, level_ent, avg_rem, total_rem_bits = compute_fmd_stats(abs_delta, direction)
        dir_counts = np.bincount(direction.astype(np.int32), minlength=2)
        dir_ent = shannon_entropy(dir_counts)

        # Print stats
        print(f"    col_ent: [{col_ent[0]:.3f}, {col_ent[1]:.3f}, {col_ent[2]:.3f}, {col_ent[3]:.3f}]  avg={col_avg:.3f}")
        print(f"    nbits_ent={nbits_ent:.3f}  level_ent={level_ent:.3f}  avg_rem={avg_rem:.2f}  dir_ent={dir_ent:.3f}")

        grp0 = nbits_pct[0]
        grp1 = nbits_pct[1:9].sum()
        grp2 = nbits_pct[9:17].sum()
        grp3 = nbits_pct[17:25].sum()
        grp4 = nbits_pct[25:33].sum()
        top5_idx = np.argsort(nbits_counts)[::-1][:5]
        top5_str = ", ".join(f"nb={i}:{nbits_pct[i]:.1f}%" for i in top5_idx)
        print(f"    nbits: 0bit={grp0:.1f}% 1-8={grp1:.1f}% 9-16={grp2:.1f}% 17-24={grp3:.1f}% 25-32={grp4:.1f}%")
        print(f"    Top-5 nbits: {top5_str}")

        # --- Compression: ByteCol-RC ---
        bc_delta_sz, bc_delta_sec = bytecol_compress(abs_delta)
        dir_sz, dir_sec = compress_direction(direction)
        bc_total_sz = bc_delta_sz + dir_sz
        bc_total_sec = bc_delta_sec + dir_sec
        if sign_flag is not None:
            sf_sz, sf_sec = compress_direction(sign_flag.view(np.uint32) if sign_flag.dtype == np.uint32 else sign_flag.astype(np.uint32))
            # sign_flag is uint8, compress as uint8
            sf_bytes = sign_flag.tobytes()
            t0 = time.perf_counter()
            sf_encoded = _mfe.tensor_encode(sf_bytes, _mfe.U8, ENCODING)
            sf_sec2 = time.perf_counter() - t0
            sf_sz = len(sf_encoded)
            bc_total_sz += sf_sz
            bc_total_sec += sf_sec2
            del sf_bytes, sf_encoded
        bc_ratio = bc_total_sz / orig_bytes
        print(f"    ByteCol-RC: ratio={bc_ratio:.4f} ({bc_total_sz/1e6:.1f}MB, {bc_total_sec:.1f}s)", end="")
        if sign_flag is not None:
            print(f"  [delta={bc_delta_sz/1e6:.1f}MB + dir={dir_sz/1e6:.1f}MB + sf={sf_sz/1e6:.1f}MB]")
        else:
            print(f"  [delta={bc_delta_sz/1e6:.1f}MB + dir={dir_sz/1e6:.1f}MB]")

        # --- Compression: FMD-style ---
        lv_sz, rem_sz, fmd_total_sz, lv_sec = fmd_style_compress(abs_delta, direction, numel)
        fmd_total_sz += dir_sz  # direction still needed
        if sign_flag is not None:
            fmd_total_sz += sf_sz
        fmd_ratio = fmd_total_sz / orig_bytes
        print(f"    FMD-style:  ratio={fmd_ratio:.4f} ({fmd_total_sz/1e6:.1f}MB)", end="")
        if sign_flag is not None:
            print(f"  [lv={lv_sz/1e6:.1f}MB + rem={rem_sz/1e6:.1f}MB + dir={dir_sz/1e6:.1f}MB + sf={sf_sz/1e6:.1f}MB]")
        else:
            print(f"  [lv={lv_sz/1e6:.1f}MB + rem={rem_sz/1e6:.1f}MB + dir={dir_sz/1e6:.1f}MB]")

        sys.stdout.flush()

        all_compress[tname] = {
            "bc_ratio": bc_ratio, "bc_sz": bc_total_sz,
            "fmd_ratio": fmd_ratio, "fmd_sz": fmd_total_sz,
        }

        del abs_delta, direction, sign_flag
        gc.collect()

    # --- FMDelta C++ ---
    print(f"\n  ── FMDelta-C++ ──", flush=True)
    fm_sz, fm_enc_sec, fm_dec_sec, fm_ok = benchmark_fmdelta_original(ft_f32, base_f32)
    fm_ratio = fm_sz / orig_bytes
    print(f"    ratio={fm_ratio:.4f} ({fm_sz/1e6:.1f}MB, enc={fm_enc_sec:.1f}s, dec={fm_dec_sec:.1f}s, {'OK' if fm_ok else 'FAIL'})")
    all_compress["FMDelta-C++"] = {"ratio": fm_ratio, "sz": fm_sz}

    # --- Summary ---
    print(f"\n  -- Summary --")
    all_ratios = {}
    for tname in transform_names:
        short = tname.replace("Delta-rotl1", "Drotl1")
        all_ratios[f"{short}+BC"] = all_compress[tname]["bc_ratio"]
        all_ratios[f"{short}+FMD"] = all_compress[tname]["fmd_ratio"]
    all_ratios["FMDelta-C++"] = fm_ratio

    for rname in sorted(all_ratios, key=all_ratios.get):
        marker = " ★" if rname == min(all_ratios, key=all_ratios.get) else ""
        print(f"    {rname:>18s}: {all_ratios[rname]:.4f}{marker}")

    sys.stdout.flush()

    return {
        "idx": idx, "cat": cat, "numel": numel,
        "diff_sign_pct": diff_sign_pct,
        "compress": all_compress,
        "fm_ratio": fm_ratio,
    }

# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Hybrid delta benchmark: diff-sign rotl1 + same-sign pcmap")
    parser.add_argument("--base", required=True)
    parser.add_argument("--ft", required=True)
    parser.add_argument("--tensor-idx", type=int, nargs="*", default=None)
    args = parser.parse_args()

    print("=" * 120)
    print("  Hybrid Delta Benchmark")
    print("  3 transforms: Delta / Delta-rotl1 / Hybrid (diff-sign rotl1 + same-sign pcmap)")
    print("  2 encodings: ByteCol-RC / FMD-style")
    print(f"  BASE: {args.base}")
    print(f"  FT:   {args.ft}")
    print("=" * 120)

    # --- Load ---
    print("\nLoading BASE ...", flush=True)
    t0 = time.perf_counter()
    base_ckpt = torch_load_with_stubs(args.base)
    base_fmt = detect_format(base_ckpt)
    print(f"  Format: {base_fmt}, loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    if base_fmt == "megatron":
        base_tensors = extract_tensors_megatron(base_ckpt)
    else:
        base_tensors = extract_tensors_zero(base_ckpt)
    print(f"  Extracted {len(base_tensors)} compressible tensors", flush=True)

    print("Loading FT ...", flush=True)
    t0 = time.perf_counter()
    ft_ckpt = torch_load_with_stubs(args.ft)
    ft_fmt = detect_format(ft_ckpt)
    print(f"  Format: {ft_fmt}, loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    if ft_fmt == "megatron":
        ft_tensors = extract_tensors_megatron(ft_ckpt)
    else:
        ft_tensors = extract_tensors_zero(ft_ckpt)
    print(f"  Extracted {len(ft_tensors)} compressible tensors", flush=True)

    # --- Match ---
    base_map = {path: (tensor, cat) for path, tensor, cat in base_tensors}
    compressible = []
    for path_str, ft_tensor, ft_cat in ft_tensors:
        base_entry = base_map.get(path_str)
        if base_entry is None:
            continue
        base_tensor, base_cat = base_entry
        if ft_tensor.shape != base_tensor.shape:
            continue
        compressible.append((path_str, ft_tensor, base_tensor, ft_cat))

    if args.tensor_idx:
        selected = [(i, compressible[i]) for i in args.tensor_idx if i < len(compressible)]
    else:
        selected = list(enumerate(compressible))

    print(f"\nMatched tensor pairs: {len(compressible)}, testing: {len(selected)}", flush=True)

    del base_ckpt, ft_ckpt, base_tensors, ft_tensors, base_map
    gc.collect()

    all_results = []

    for idx, (path_str, ft_t, bt, cat) in selected:
        ft_f32 = ft_t.detach().float().numpy()
        base_f32 = bt.detach().float().numpy()
        short_name = path_str if len(path_str) <= 80 else "..." + path_str[-(80-3):]

        result = analyze_tensor(ft_f32, base_f32, idx, cat, short_name)
        all_results.append(result)

        del ft_f32, base_f32
        gc.collect()

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  Summary: Actual compression ratios")
    print(f"{'='*120}")

    methods = ["Delta+BC", "Drotl1+BC", "Hybrid+BC",
               "Delta+FMD", "Drotl1+FMD", "Hybrid+FMD", "FMDelta-C++"]

    print(f"  {'#':>3s} {'Type':>10s} {'Elements':>14s} {'DiffSign%':>9s} | " +
          " ".join(f"{m:>11s}" for m in methods) + f" | {'Best':>14s}")
    print(f"  {'-'*3}-{'-'*10}-{'-'*14}-{'-'*6}-+-" +
          "-".join(f"{'-'*11}" for _ in methods) + f"-+-{'-'*14}")

    total_orig = 0
    totals = {m: 0.0 for m in methods}

    for tr in all_results:
        orig = tr["numel"] * 4
        total_orig += orig
        cr = tr["compress"]

        ratios = {}
        for name in ["Delta", "Delta-rotl1", "Hybrid"]:
            short = name.replace("Delta-rotl1", "Drotl1")
            ratios[f"{short}+BC"] = cr[name]["bc_ratio"]
            ratios[f"{short}+FMD"] = cr[name]["fmd_ratio"]
        ratios["FMDelta-C++"] = tr["fm_ratio"]

        for m in methods:
            totals[m] += ratios[m] * orig

        best = min(ratios, key=ratios.get)
        ratio_strs = " ".join(f"{ratios[m]:>11.4f}" for m in methods)
        print(f"  {tr['idx']:>3d} {tr['cat']:>10s} {tr['numel']:>14,d} {tr['diff_sign_pct']:>5.1f}% | {ratio_strs} | {best:>14s}")

    if total_orig > 0:
        print(f"\n  Total: original {total_orig/1e9:.2f}GB")
        sorted_methods = sorted(methods, key=lambda m: totals[m])
        for m in sorted_methods:
            ratio = totals[m] / total_orig
            compressed_gb = totals[m] / 1e9
            marker = " ★" if m == sorted_methods[0] else ""
            print(f"    {m:>14s}: {compressed_gb:>6.2f}GB (ratio={ratio:.4f}){marker}")

    sys.stdout.flush()
    print("\n[DONE]", flush=True)

if __name__ == "__main__":
    main()
