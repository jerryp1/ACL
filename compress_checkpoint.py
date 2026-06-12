#!/usr/bin/env python3
"""Unified checkpoint compression tool.

Compatible with DeepSpeed and Megatron-LM checkpoint layouts.
Supports bf16, fp16, and fp32 data types.
Handles: model.safetensors, mp_rank_*_model_states.pt, bf16_zero_pp_rank_*_optim_states.pt

Usage:
  # Compress a single file (no delta, I-Align only)
  python compress_checkpoint.py --input model.safetensors --output compressed.bin

  # Delta compression between base and finetuned checkpoints
  python compress_checkpoint.py \\
      --base /path/to/base_model/ \\
      --finetuned /path/to/checkpoint-10/ \\
      --output_dir /path/to/output/

  # Auto-discover and compress all files in a directory
  python compress_checkpoint.py --input_dir /path/to/checkpoint-10/ --output_dir /path/to/output/

  # Compress specific file types only
  python compress_checkpoint.py --input_dir /path/to/ckpt/ --filter "*.safetensors" --output_dir out/
"""

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from checkpoint_io import read_checkpoint, discover_checkpoints, TensorDict
import tensor_compress as tc


def _serialize_chunks(all_chunks):
    """Serialize compress_tensor output to bytes for file storage."""
    import pickle
    return pickle.dumps(all_chunks, protocol=2)


def _deserialize_chunks(data):
    """Deserialize chunks from bytes."""
    import pickle
    return pickle.loads(data)


def compress_tensor_dict(ft_tensors, base_tensors=None, verbose=True, output_dir=None):
    """Compress all tensors in a TensorDict and optionally save to disk.

    Args:
        ft_tensors: TensorDict of finetuned tensors (float32 numpy arrays).
        base_tensors: Optional TensorDict of base tensors for delta compression.
        verbose: Print per-tensor results.
        output_dir: If set, save each compressed tensor as a .alc file.

    Returns:
        List of result dicts with compression stats.
    """
    results = []
    total_orig = 0
    total_comp = 0

    if verbose:
        header = "%-50s %-20s %8s %8s %6s %9s %9s %3s" % (
            "Tensor", "Shape", "Size(MB)", "Path", "Ratio",
            "Enc MB/s", "Dec MB/s", "OK")
        print(header)
        print("=" * 125)

    for name in ft_tensors:
        ft_arr = ft_tensors[name]
        shape = ft_tensors.metadata[name]["shape"]
        dtype_str = ft_tensors.metadata[name]["dtype"]
        n = len(ft_arr)

        if n < 100:
            continue

        orig_bytes = n * 4
        total_orig += orig_bytes

        base_arr = None
        if base_tensors is not None and name in base_tensors:
            base_arr = base_tensors[name]

        strategy = tc.analyze_strategy(ft_arr, base_arr)

        t0 = time.time()
        comp_sz, chunks, _ = tc.compress_tensor(ft_arr, strategy, base_arr)
        enc_time = time.time() - t0

        t0 = time.time()
        rec, _ = tc.decompress_tensor(chunks, strategy, n, base_arr)
        dec_time = time.time() - t0

        ok = np.array_equal(rec.view(np.uint32), ft_arr.view(np.uint32))
        ratio = comp_sz / float(orig_bytes)
        enc_mbs = orig_bytes / enc_time / 1e6 if enc_time > 0 else float("inf")
        dec_mbs = orig_bytes / dec_time / 1e6 if dec_time > 0 else float("inf")
        total_comp += comp_sz

        # Save compressed tensor to disk
        if output_dir is not None:
            safe_name = name.replace("/", "__").replace(".", "_")
            alc_path = os.path.join(output_dir, safe_name + ".alc")
            header_data = json.dumps({
                "name": name,
                "shape": list(shape),
                "dtype": dtype_str,
                "n_elems": n,
                "strategy": {
                    "path": strategy.path,
                    "encoding": strategy.encoding,
                    "ent_ialign": strategy.ent_ialign,
                    "ent_pcdelta": strategy.ent_pcdelta,
                    "ent_rotdelta": strategy.ent_rotdelta,
                },
            }).encode("utf-8")
            chunk_bytes = _serialize_chunks(chunks)
            with open(alc_path, "wb") as fh:
                fh.write(struct.pack("<I", len(header_data)))
                fh.write(header_data)
                fh.write(chunk_bytes)

        if verbose:
            shape_str = str(shape)
            size_mb = orig_bytes / 1e6
            status = "Y" if ok else "N"
            print("%-50s %-20s %8.2f %8s %6.4f %9.1f %9.1f %3s" % (
                name[:50], shape_str[:20], size_mb, strategy.path,
                ratio, enc_mbs, dec_mbs, status))

        results.append({
            "name": name,
            "shape": list(shape),
            "dtype": dtype_str,
            "original_bytes": orig_bytes,
            "compressed_bytes": comp_sz,
            "ratio": ratio,
            "path": strategy.path,
            "encode_mbps": enc_mbs,
            "decode_mbps": dec_mbs,
            "bitwise_exact": bool(ok),
        })

    if verbose:
        print("=" * 125)
        overall_ratio = total_comp / float(total_orig) if total_orig > 0 else 0
        all_ok = all(r["bitwise_exact"] for r in results)
        print("%-50s %-20s %8.2f %8s %6.4f" % (
            "TOTAL", "", total_orig / 1e6, "", overall_ratio))
        print("Tensors: %d, All bitwise-exact: %s" % (len(results), all_ok))

    return results, total_orig, total_comp


def find_matching_base(base_dir, ft_filename):
    """Find the matching base file for a given finetuned file.

    Handles common checkpoint layouts:
      - base/model.safetensors ↔ checkpoint-N/model.safetensors
      - base/mp_rank_00_model_states.pt ↔ ckpt/mp_rank_00_model_states.pt
    """
    if base_dir is None:
        return None

    # Direct match by filename
    candidate = os.path.join(base_dir, ft_filename)
    if os.path.exists(candidate):
        return candidate

    # Search recursively
    for root, dirs, files in os.walk(base_dir):
        if ft_filename in files:
            return os.path.join(root, ft_filename)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified checkpoint compression (DeepSpeed / Megatron / safetensors)")
    parser.add_argument("--input", type=str, help="Single input checkpoint file")
    parser.add_argument("--input_dir", type=str, help="Input directory to scan for checkpoints")
    parser.add_argument("--base", type=str, help="Base checkpoint file or directory (for delta)")
    parser.add_argument("--base_dir", type=str, help="Base checkpoint directory (for delta)")
    parser.add_argument("--output", type=str, help="Output file (single-file mode)")
    parser.add_argument("--output_dir", type=str, default="./compressed_output",
                        help="Output directory (default: ./compressed_output)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Glob filter for files (e.g., '*.safetensors')")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tensor output")
    parser.add_argument("--json_report", type=str, default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    verbose = not args.quiet

    # Determine input files
    input_files = []
    if args.input:
        input_files.append(args.input)
    elif args.input_dir:
        if args.filter:
            pattern = os.path.join(args.input_dir, "**", args.filter)
            input_files = sorted(glob.glob(pattern, recursive=True))
        else:
            discovered = discover_checkpoints(args.input_dir)
            input_files = [fp for fp, cat in discovered]
    else:
        parser.error("Must specify --input or --input_dir")

    if not input_files:
        print("No checkpoint files found.")
        sys.exit(1)

    # Determine base directory
    base_dir = args.base_dir or (os.path.dirname(args.base) if args.base else None)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    grand_orig = 0
    grand_comp = 0

    for filepath in input_files:
        filename = os.path.basename(filepath)
        if verbose:
            print("\n>>> %s" % filepath)

        t0 = time.time()
        ft_tensors = read_checkpoint(filepath)
        load_time = time.time() - t0
        if verbose:
            print("  Loaded %d tensors in %.1fs" % (len(ft_tensors), load_time))

        # Load base tensors if available
        base_tensors = None
        if args.base and os.path.isfile(args.base):
            base_tensors = read_checkpoint(args.base)
        elif base_dir:
            base_path = find_matching_base(base_dir, filename)
            if base_path and base_path != filepath:
                if verbose:
                    print("  Base: %s" % base_path)
                base_tensors = read_checkpoint(base_path)

        # Create per-file output subdirectory
        file_output_dir = os.path.join(args.output_dir, os.path.splitext(filename)[0])
        os.makedirs(file_output_dir, exist_ok=True)

        results, total_orig, total_comp = compress_tensor_dict(
            ft_tensors, base_tensors, verbose=verbose, output_dir=file_output_dir)

        all_results[filepath] = {
            "tensors": results,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_comp / float(total_orig) if total_orig > 0 else 0,
        }
        grand_orig += total_orig
        grand_comp += total_comp

    # Summary
    print("\n" + "=" * 80)
    print("GRAND TOTAL: %.2f MB → %.2f MB (ratio: %.4f)" % (
        grand_orig / 1e6, grand_comp / 1e6,
        grand_comp / float(grand_orig) if grand_orig > 0 else 0))
    print("Files processed: %d" % len(input_files))

    # Save JSON report
    if args.json_report:
        report = {
            "files": all_results,
            "grand_total_original_bytes": grand_orig,
            "grand_total_compressed_bytes": grand_comp,
            "grand_overall_ratio": grand_comp / float(grand_orig) if grand_orig > 0 else 0,
        }
        with open(args.json_report, "w") as fh:
            json.dump(report, fh, indent=2)
        print("Report saved to: %s" % args.json_report)


if __name__ == "__main__":
    main()
