import sys
import types
from unittest.mock import MagicMock

class _MegatronMockFinder:
    """自动 mock 所有 megatron.* 子模块，解决 pickle 反序列化时找不到模块的问题"""
    def find_module(self, fullname, path=None):
        if fullname == 'megatron' or fullname.startswith('megatron.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []
        mod.__package__ = fullname
        mod.__loader__ = self
        mod.__getattr__ = lambda n: MagicMock()
        sys.modules[fullname] = mod
        return mod

sys.meta_path.insert(0, _MegatronMockFinder())


import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any, Optional

import fmd
import torch
from tqdm import tqdm
import modelformat_encodings as mfe

import utils

CHUNK_SIZE_DEFAULT = (1 << 28) - 1

SKIP_DTYPES = {torch.uint8, torch.int8, torch.bool}
for _name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        SKIP_DTYPES.add(_dt)


def _str_bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def determine_compress_strategy(tensor_path: List[str]) -> str:
    name = str(tensor_path[-1]) if tensor_path else ""
    return "FM-Single" if ("exp_avg" in name and "exp_avg_sq" not in name) else "FM-Delta"


def _compress_fm_single(tensor: torch.Tensor, chunk_size: int) -> Tuple[bytes, List[int]]:
    flat = tensor.cpu().contiguous().view(-1)
    numel = flat.numel()
    chunks, lens = [], []
    for start in range(0, numel, chunk_size):
        data = flat[start: start + chunk_size].numpy().tobytes()
        enc = mfe.tensor_encode(data, mfe.F32, "STATIC_RC_FLOAT32")
        chunks.append(enc)
        lens.append(len(enc))
    return b"".join(chunks), lens


def _compress_fm_delta(
        base: torch.Tensor,
        ft: torch.Tensor,
        chunk_size: int
) -> Tuple[bytes, List[int], int]:
    if ft.numel() <= chunk_size:
        b = fmd.compress_param(base.contiguous(), ft.contiguous())
        return b, [len(b)], 0

    dim = max(range(ft.ndim), key=lambda d: ft.size(d))
    n = ft.size(dim)
    chunks, lens = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        bc = base.narrow(dim, start, end - start).contiguous()
        fc = ft.narrow(dim, start, end - start).contiguous()
        b = fmd.compress_param(bc, fc)
        chunks.append(b)
        lens.append(len(b))
    return b"".join(chunks), lens, dim


def fmd_compress_chunked(
        base_tensor: torch.Tensor,
        finetuned_tensor: torch.Tensor,
        tensor_path: Optional[List[str]] = None,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
) -> Tuple[bytes, List[int], str, int]:
    assert base_tensor.shape == finetuned_tensor.shape
    strategy = determine_compress_strategy(tensor_path or [])

    if strategy == "FM-Single":
        b, lens = _compress_fm_single(finetuned_tensor, chunk_size)
        return b, lens, "STATIC_RC_FLOAT32", 0
    else:
        b, lens, dim = _compress_fm_delta(base_tensor, finetuned_tensor, chunk_size)
        return b, lens, "FMD", dim


def _handle_tie_word_embeddings(pt, meta_dict, verbose=True):
    embed = next(
        (t for p, t in utils.traverse_with_path_array(pt)
         if p[-1] == "model.embed_tokens.weight"),
        None
    )
    if embed is None:
        if verbose:
            print("[TIE] model.embed_tokens.weight not found")
        return
    for tensor_path, _ in utils.traverse_with_path_array(pt):
        if tensor_path[-1] == "lm_head.weight":
            utils.set_value_by_path(pt, tensor_path, embed)
            path_str = ".".join(str(x) for x in tensor_path)
            if path_str in meta_dict:
                meta_dict[path_str]["tie_to"] = "model.embed_tokens.weight"
            if verbose:
                print("[TIE] Tied lm_head.weight → model.embed_tokens.weight")
            break


def compress_pt(
        base_pt: Dict[str, Any],
        finetuned_pt: Dict[str, Any],
        compressed_pt_name: str,
        tie_word_embeddings: bool = False,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        verbose: bool = True,
) -> Dict[str, Any]:
    meta_dict = {}
    tensors_info = utils.traverse_with_path_array(finetuned_pt)
    ok_count = skip_count = 0

    for tensor_path, tensor in tqdm(
            tensors_info,
            desc=f"compress {os.path.basename(compressed_pt_name)}",
            unit="tensor"
    ):
        path_str = ".".join(str(x) for x in tensor_path)
        base_meta = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}

        skip_reason = None
        if not torch.is_tensor(tensor):
            skip_reason = "not_tensor"
        elif tensor.ndim == 0:
            skip_reason = "scalar_tensor"
        elif tensor.dtype in SKIP_DTYPES:
            skip_reason = f"dtype_skip_{tensor.dtype}"
        else:
            try:
                base_tensor = utils.get_tensor_by_path_array(base_pt, tensor_path)
            except KeyError:
                base_tensor = None
            if base_tensor is None or tensor.shape != base_tensor.shape:
                if verbose:
                    print(f"[FALLBACK] {path_str}: base missing or shape mismatch → FM-Single")
                try:
                    compressed_bytes, chunk_lens = _compress_fm_single(tensor, chunk_size)
                    compressed_size = len(compressed_bytes)
                    tensor_data = torch.frombuffer(bytearray(compressed_bytes), dtype=torch.uint8).clone()
                    del compressed_bytes
                    utils.set_value_by_path(finetuned_pt, tensor_path, tensor_data)
                    compress_type = "FM-Single-Chunked" if len(chunk_lens) > 1 else "FM-Single"
                    meta_dict[path_str] = {
                        **base_meta,
                        "compress_type":   compress_type,
                        "strategy":        "FM-Single",
                        "encoding_type":   "STATIC_RC_FLOAT32",
                        "chunk":           {"dim": 0, "chunk_size": chunk_size, "chunk_lens": chunk_lens},
                        "original_size":   tensor.numel() * tensor.element_size(),
                        "compressed_size": compressed_size,
                        "fallback_reason": "base_missing_or_shape_mismatch",
                    }
                    ok_count += 1
                except Exception as e:
                    import traceback
                    print(f"[ERROR] {path_str} FM-Single fallback failed: {e}")
                    traceback.print_exc()
                    meta_dict[path_str] = {**base_meta, "compress_type": "ERROR", "error": str(e)}
                    skip_count += 1
                continue

        if skip_reason:
            if verbose:
                print(f"[SKIP] {path_str}: {skip_reason}")
            meta_dict[path_str] = {**base_meta, "compress_type": "RAW", "reason": skip_reason}
            skip_count += 1
            continue

        try:
            strategy = determine_compress_strategy(tensor_path)
            if verbose:
                print(f"[COMPRESS] {path_str}  shape={tuple(tensor.shape)}"
                      f"  dtype={tensor.dtype}  strategy={strategy}")

            compressed_bytes, chunk_lens, encoding_type, actual_dim = fmd_compress_chunked(
                base_tensor.contiguous(), tensor.contiguous(),
                tensor_path=tensor_path, chunk_size=chunk_size,
            )

            if verbose:
                ratio = len(compressed_bytes) / (tensor.numel() * tensor.element_size())
                print(f"  -> bytes={len(compressed_bytes)}  chunks={len(chunk_lens)}"
                      f"  encoding={encoding_type}  ratio={ratio:.2%}")

            compressed_size = len(compressed_bytes)
            tensor_data = torch.frombuffer(bytearray(compressed_bytes), dtype=torch.uint8).clone()
            del compressed_bytes
            utils.set_value_by_path(finetuned_pt, tensor_path, tensor_data)

            compress_type = f"{strategy}-Chunked" if len(chunk_lens) > 1 else strategy
            meta_dict[path_str] = {
                **base_meta,
                "compress_type":   compress_type,
                "strategy":        strategy,
                "encoding_type":   encoding_type,
                "chunk":           {"dim": actual_dim, "chunk_size": chunk_size, "chunk_lens": chunk_lens},
                "original_size":   tensor.numel() * tensor.element_size(),
                "compressed_size": compressed_size,
            }
            ok_count += 1

        except Exception as e:
            import traceback
            print(f"[ERROR] {path_str}: {e}")
            traceback.print_exc()
            meta_dict[path_str] = {**base_meta, "compress_type": "ERROR", "error": str(e)}
            skip_count += 1

    if tie_word_embeddings:
        _handle_tie_word_embeddings(finetuned_pt, meta_dict, verbose)

    if "args" in finetuned_pt and finetuned_pt["args"] is not None:
        _args = finetuned_pt["args"]
        for _attr in list(vars(_args).keys()):
            if isinstance(getattr(_args, _attr), MagicMock):
                setattr(_args, _attr, None)
    os.makedirs(os.path.dirname(os.path.abspath(compressed_pt_name)), exist_ok=True)

    torch.save(finetuned_pt, compressed_pt_name,
               _use_new_zipfile_serialization=True, pickle_protocol=5)
    meta_file = compressed_pt_name + ".json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {compressed_pt_name}")
    print(f"[SAVE] {meta_file}")
    print(f"[SUMMARY] total={len(tensors_info)}  ok={ok_count}  skipped={skip_count}")
    return meta_dict


import re

def _get_mp_rank(relpath: str) -> str:
    """
    从 relpath 提取 mp_rank 部分作为匹配 key
    iter_0000100/mp_rank_00_007_006 → mp_rank_00_007_006
    iter_0000002/mp_rank_00_007_006 → mp_rank_00_007_006
    """
    if not relpath:
        return ""
    parts = re.split(r"[/\\]", relpath)
    # 找到 mp_rank_XX_XXX_XXX 部分
    for part in parts:
        if part.startswith("mp_rank_"):
            return part
    # 没有 mp_rank，去掉 iter_XXXXXXX 前缀后返回剩余部分
    return re.sub(r"^iter_\d+[/\\]?", "", relpath)


def _compress_pt_pair(
        base_ckpt_path: str,
        finetuned_ckpt_path: str,
        pt_name: str,
        compressed_pt_file_prefix: str,
        tie_word_embeddings: bool,
        chunk_size: int,
        verbose: bool,
        device: str,
) -> Tuple[int, int]:
    base_pt_files = utils.find_pt_files_with_name_and_relpath(base_ckpt_path, pt_name)
    finetuned_pt_files = utils.find_pt_files_with_name_and_relpath(finetuned_ckpt_path, pt_name)

    if not base_pt_files:
        print(f"[SKIP] {pt_name} not found in base checkpoint")
        return 0, 0
    if not finetuned_pt_files:
        print(f"[SKIP] {pt_name} not found in finetuned checkpoint")
        return 0, 0

    # 用 mp_rank 作为 key 建立索引，方便 O(1) 查找
    base_by_mprank = {
        _get_mp_rank(f.get("relpath", "")): f
        for f in base_pt_files
    }

    if verbose:
        print(f"[INFO] base files:      {len(base_pt_files)}")
        print(f"[INFO] finetuned files: {len(finetuned_pt_files)}")
        print(f"[INFO] base mp_ranks:   {sorted(base_by_mprank.keys())[:4]} ...")

    total_orig = total_comp = 0

    for ft_file in finetuned_pt_files:
        ft_mp_rank = _get_mp_rank(ft_file.get("relpath", ""))

        # 用 mp_rank 找对应的 base
        base_file = base_by_mprank.get(ft_mp_rank)
        if base_file is None:
            print(f"[SKIP] no matching base for mp_rank={ft_mp_rank}")
            continue

        pt_filename       = ft_file["filename"]
        ft_relpath        = ft_file.get("relpath") or ""
        base_relpath      = base_file.get("relpath") or ""

        parent_dir        = os.path.dirname(finetuned_ckpt_path)
        ckpt_name         = os.path.basename(finetuned_ckpt_path)
        compressed_folder = os.path.join(parent_dir, f"compress_{ckpt_name}", ft_relpath)
        os.makedirs(compressed_folder, exist_ok=True)

        base_pt_path      = os.path.join(base_ckpt_path,      base_relpath, pt_filename)
        finetuned_pt_path = os.path.join(finetuned_ckpt_path, ft_relpath,   pt_filename)
        compressed_pt_path = os.path.join(compressed_folder, compressed_pt_file_prefix + pt_filename)

        missing = [p for p in [base_pt_path, finetuned_pt_path] if not os.path.isfile(p)]
        if missing:
            for p in missing:
                print(f"[SKIP] not found: {p}")
            continue

        orig_size = os.path.getsize(finetuned_pt_path)

        try:
            if verbose:
                print(f"\n[FILE] {pt_filename}  mp_rank={ft_mp_rank}")
                print(f"  base:      {base_pt_path}")
                print(f"  finetuned: {finetuned_pt_path}")
                print(f"  output:    {compressed_pt_path}")

            # ===== load =====
            t_load_start  = time.time()
            base_pt_data  = torch.load(base_pt_path,      map_location=device, weights_only=False)
            t_base_loaded = time.time()
            finetuned_pt_data = torch.load(finetuned_pt_path, map_location=device, weights_only=False)
            t_ft_loaded   = time.time()
            print(f"[TIME] load base_pt:      {t_base_loaded - t_load_start:.2f}s")
            print(f"[TIME] load finetuned_pt: {t_ft_loaded - t_base_loaded:.2f}s")
            print(f"[TIME] load total:        {t_ft_loaded - t_load_start:.2f}s")

            # ===== 压缩 =====
            t_compress_start = time.time()
            compress_pt(
                base_pt=base_pt_data,
                finetuned_pt=finetuned_pt_data,
                compressed_pt_name=compressed_pt_path,
                tie_word_embeddings=tie_word_embeddings,
                chunk_size=chunk_size,
                verbose=verbose,
            )
            t_compress_end = time.time()
            print(f"[TIME] compress:              {t_compress_end - t_compress_start:.2f}s")
            print(f"[TIME] total (load+compress): {t_compress_end - t_load_start:.2f}s")

            comp_size = os.path.getsize(compressed_pt_path)
            total_orig += orig_size
            total_comp += comp_size
            print(f"[OK] {pt_filename}  mp_rank={ft_mp_rank}  "
                  f"orig={orig_size / (1024**2):.1f}MB  "
                  f"comp={comp_size / (1024**2):.1f}MB  "
                  f"ratio={comp_size / orig_size:.2%}")

        except Exception as e:
            import traceback
            print(f"[ERROR] {pt_filename} mp_rank={ft_mp_rank}: {e}")
            traceback.print_exc()

    return total_orig, total_comp


def compress_checkpoints(
        ckpt_dir: str,
        pt_names: List[str],
        compressed_pt_file_prefix: str = "compressed_",
        tie_word_embeddings: bool = False,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        verbose: bool = True,
        device: str = "cpu",
) -> None:
    checkpoints = utils.get_sorted_checkpoints(ckpt_dir)

    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints to compress")
        return

    print(f"[INFO] Found {len(checkpoints)} checkpoints, {len(checkpoints) - 1} pair(s) to compress")

    total_orig = total_comp = 0
    t0 = time.time()

    for i in tqdm(range(len(checkpoints) - 1), desc="ckpt", unit="pair"):
        finetuned_ckpt_path = checkpoints[i]
        base_ckpt_path = checkpoints[i + 1]

        if verbose:
            print(f"\n[CKPT] base={os.path.basename(base_ckpt_path)}"
                  f"  finetuned={os.path.basename(finetuned_ckpt_path)}")

        for pt_name in pt_names:
            orig, comp = _compress_pt_pair(
                base_ckpt_path=base_ckpt_path,
                finetuned_ckpt_path=finetuned_ckpt_path,
                pt_name=pt_name,
                compressed_pt_file_prefix=compressed_pt_file_prefix,
                tie_word_embeddings=tie_word_embeddings,
                chunk_size=chunk_size,
                verbose=verbose,
                device=device,
            )
            total_orig += orig
            total_comp += comp

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"[FINAL] original={total_orig / (1024**3):.2f}GB  "
          f"compressed={total_comp / (1024**3):.2f}GB  "
          f"ratio={total_comp / total_orig:.2%}  time={elapsed:.1f}s")
    print("=" * 70)


def compress_two_checkpoints(
        base_ckpt_path: str,
        finetuned_ckpt_path: str,
        pt_names: List[str],
        compressed_pt_file_prefix: str = "compressed_",
        tie_word_embeddings: bool = False,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        verbose: bool = True,
        device: str = "cpu",
) -> None:
    """Compress two specified checkpoints."""
    
    if not os.path.isdir(base_ckpt_path):
        raise ValueError(f"base checkpoint 不存在或不是目录: {base_ckpt_path}")
    if not os.path.isdir(finetuned_ckpt_path):
        raise ValueError(f"finetuned checkpoint 不存在或不是目录: {finetuned_ckpt_path}")

    if verbose:
        print(f"[INFO] base:      {base_ckpt_path}")
        print(f"[INFO] finetuned: {finetuned_ckpt_path}")
        print(f"[INFO] pt_names:  {pt_names}")

    total_orig = total_comp = 0
    t0 = time.time()

    for pt_name in pt_names:
        orig, comp = _compress_pt_pair(
            base_ckpt_path=base_ckpt_path,
            finetuned_ckpt_path=finetuned_ckpt_path,
            pt_name=pt_name,
            compressed_pt_file_prefix=compressed_pt_file_prefix,
            tie_word_embeddings=tie_word_embeddings,
            chunk_size=chunk_size,
            verbose=verbose,
            device=device,
        )
        total_orig += orig
        total_comp += comp

    elapsed = time.time() - t0
    if total_orig > 0:
        print("=" * 70)
        print(f"[FINAL] original={total_orig / (1024**3):.2f}GB  "
              f"compressed={total_comp / (1024**3):.2f}GB  "
              f"ratio={total_comp / total_orig:.2%}  time={elapsed:.1f}s")
        print("=" * 70)
    else:
        print("[WARN] 没有处理任何文件，请检查路径和 pt_names")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint diff compressor")
    
    parser.add_argument("--ckpt_dir",    type=str, default=None,
                        help="扫描目录，自动配对相邻 checkpoint")
    
    parser.add_argument("--base_ckpt",       type=str, default=None,
                        help="base checkpoint 路径（旧的/参考的）")
    parser.add_argument("--finetuned_ckpt",  type=str, default=None,
                        help="finetuned checkpoint 路径（新的/要压缩的）")

    parser.add_argument("--pt_names",                  nargs="+",      required=True)
    parser.add_argument("--compressed_pt_file_prefix", type=str,       default="compressed_")
    parser.add_argument("--tie_word_embeddings",       type=_str_bool, default=False)
    parser.add_argument("--chunk_size",                type=int,       default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--verbose",                   type=_str_bool, default=True)
    parser.add_argument("--device",                    type=str,       default="cpu")
    
    args = parser.parse_args()

    use_dir  = args.ckpt_dir is not None
    use_pair = args.base_ckpt is not None and args.finetuned_ckpt is not None

    if use_dir and use_pair:
        parser.error("--ckpt_dir 和 --base_ckpt/--finetuned_ckpt 不能同时使用")
    if not use_dir and not use_pair:
        parser.error("必须指定 --ckpt_dir 或者同时指定 --base_ckpt 和 --finetuned_ckpt")

    common_kwargs = dict(
        pt_names=args.pt_names,
        compressed_pt_file_prefix=args.compressed_pt_file_prefix,
        tie_word_embeddings=args.tie_word_embeddings,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        device=args.device,
    )

    if use_pair:
        # 模式二：直接指定
        compress_two_checkpoints(
            base_ckpt_path=args.base_ckpt,
            finetuned_ckpt_path=args.finetuned_ckpt,
            **common_kwargs,
        )
    else:
        # 模式一：扫描目录
        if not os.path.isdir(args.ckpt_dir):
            raise ValueError(f"Invalid checkpoint directory: {args.ckpt_dir}")
        compress_checkpoints(
            ckpt_dir=args.ckpt_dir,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
