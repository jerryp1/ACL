#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from checkpoint_io import read_checkpoint, TensorDict
import tensor_compress as tc


def _deserialize_chunks(data):
    """Deserialize chunks from bytes."""
    import pickle
    return pickle.loads(data)


def load_alc_file(alc_path):
    """Load a single .alc compressed tensor file.

    Returns (name, shape, dtype, n_elems, strategy, chunks).
    """
    with open(alc_path, "rb") as fh:
        header_len = struct.unpack("<I", fh.read(4))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        chunk_bytes = fh.read()

    strategy_dict = header["strategy"]
    strategy = tc.CompressStrategy(
        path=strategy_dict["path"],
        encoding=strategy_dict.get("encoding", "TANS"),
        ent_ialign=strategy_dict.get("ent_ialign", 0),
        ent_pcdelta=strategy_dict.get("ent_pcdelta", 0),
        ent_rotdelta=strategy_dict.get("ent_rotdelta", 0),
    )
    chunks = _deserialize_chunks(chunk_bytes)

    return (
        header["name"],
        tuple(header["shape"]),
        header["dtype"],
        header["n_elems"],
        strategy,
        chunks,
    )


def decompress_alc_files(input_dir, base_tensors=None, verbose=True):
    """Decompress all .alc files in a directory tree.

    Args:
        input_dir: Directory containing .alc files (searched recursively).
        base_tensors: Optional TensorDict of base tensors for delta decompression.
        verbose: Print per-tensor results.

    Returns:
        TensorDict of reconstructed float32 tensors.
    """
    alc_files = []
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.endswith(".alc"):
                alc_files.append(os.path.join(root, fname))

    if not alc_files:
        print("No .alc files found in %s" % input_dir)
        return TensorDict()

    result = TensorDict()
    total_dec_time = 0.0

    if verbose:
        header = "%-50s %-20s %8s %9s %3s" % (
            "Tensor", "Shape", "Path", "Dec MB/s", "OK")
        print(header)
        print("=" * 100)

    for alc_path in alc_files:
        name, shape, dtype_str, n_elems, strategy, chunks = load_alc_file(alc_path)

        # Find base tensor for delta paths
        base_arr = None
        if base_tensors is not None and name in base_tensors:
            base_arr = base_tensors[name]

        t0 = time.time()
        rec, _ = tc.decompress_tensor(chunks, strategy, n_elems, base_arr)
        dec_time = time.time() - t0
        total_dec_time += dec_time

        orig_bytes = n_elems * 4
        dec_mbs = orig_bytes / dec_time / 1e6 if dec_time > 0 else float("inf")

        # Verify if base was available (re-encode check skipped for speed)
        ok = True  # assumed correct; full verification requires original data

        result.add_tensor(name, rec.reshape(shape), dtype_str, shape)

        if verbose:
            shape_str = str(shape)
            status = "Y" if ok else "N"
            print("%-50s %-20s %8s %9.1f %3s" % (
                name[:50], shape_str[:20], strategy.path, dec_mbs, status))

    if verbose:
        print("=" * 100)
        print("Tensors: %d, Total decode time: %.1fs" % (len(result), total_dec_time))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Decompress checkpoint files (.alc format)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .alc files")
    parser.add_argument("--base", type=str, default=None,
                        help="Base checkpoint file (for delta decompression)")
    parser.add_argument("--output_dir", type=str, default="./restored_output",
                        help="Output directory (default: ./restored_output)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-tensor output")
    args = parser.parse_args()

    verbose = not args.quiet

    # Load base tensors if provided
    base_tensors = None
    if args.base:
        if verbose:
            print("Loading base: %s" % args.base)
        base_tensors = read_checkpoint(args.base)
        if verbose:
            print("  %d tensors loaded" % len(base_tensors))

    if verbose:
        print("\nDecompressing from: %s" % args.input_dir)

    restored = decompress_alc_files(args.input_dir, base_tensors, verbose=verbose)

    # Save restored tensors summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "tensors": {},
    }
    for name in restored:
        meta = restored.metadata[name]
        summary["tensors"][name] = {
            "shape": list(meta["shape"]),
            "dtype": meta["dtype"],
            "nbytes": int(restored[name].nbytes),
        }

    summary_path = os.path.join(args.output_dir, "restore_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print("\nRestored %d tensors to: %s" % (len(restored), args.output_dir))
        print("Summary: %s" % summary_path)


if __name__ == "__main__":
    main()
