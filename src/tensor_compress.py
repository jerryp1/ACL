"""Lossless float32 tensor compression with entropy-driven hybrid path selection."""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_SRC, "cpp")
if _CPP_DIR not in sys.path:
    sys.path.insert(0, _CPP_DIR)

_CPP_TANS_AVAILABLE = False
try:
    import drotl1fmd_tans_cpp_codec as _cpp_tans
    _CPP_TANS_AVAILABLE = True
except ImportError:
    _cpp_tans = None

from tans_codec import tans_encode as _py_tans_encode, tans_decode as _py_tans_decode
from tans_codec import EncodedPayload as _TansPayload

try:
    import modelformat_encodings as _mfe
    _MFE_AVAILABLE = True
except ImportError:
    _mfe = None
    _MFE_AVAILABLE = False

CHUNK_ELEMS = 16 * 1024 * 1024  # 16M elements per chunk

TANS_CODECS = {"TANS"}
U8_CODECS = {"STATIC_RC_UINT8", "QS_RC_UINT8", "HUFFMAN_LUT", "DFLOAT11"}
GENERIC_CODECS = {"ZSTD", "LZ4", "ZLIB", "SNAPPY"}
ALL_ENCODINGS = sorted(TANS_CODECS | U8_CODECS | GENERIC_CODECS)

DEFAULT_ENCODING = "TANS"
DEFAULT_ENTROPY_THRESHOLD = 7.0
DEFAULT_SAMPLE = 65536
_ZSTD_LEVEL = 1


def pcmap_f32(arr: np.ndarray) -> np.ndarray:
    """IEEE 754 float32 bits → monotonic uint32 space (positive-continuous map)."""
    u = arr.view(np.uint32)
    sign = u >> np.uint32(31)
    return np.where(sign == 0, u + np.uint32(0x80000000), ~u).astype(np.uint32)

def pcmap_inverse_f32(r: np.ndarray) -> np.ndarray:
    """Monotonic uint32 space → IEEE 754 float32 bits."""
    r = np.asarray(r, dtype=np.uint32)
    mask = (-(r >> np.uint32(31))) >> np.uint32(1)
    return ~(r ^ mask)


def rotl1(u32: np.ndarray) -> np.ndarray:
    """Rotate left by 1 bit: bit31 (sign) → bit0.
    After rotl1, byte3 becomes pure exponent bits."""
    u = u32.view(np.uint32)
    return ((u << np.uint32(1)) | (u >> np.uint32(31))).astype(np.uint32)


def rotr1(u32: np.ndarray) -> np.ndarray:
    """Inverse of rotl1: bit0 → bit31."""
    u = u32.view(np.uint32)
    return ((u >> np.uint32(1)) | (u << np.uint32(31))).astype(np.uint32)


def pcmap_delta(base_f32: np.ndarray, ft_f32: np.ndarray) -> np.ndarray:
    """Compute signed pcmap delta: pcmap(ft) - pcmap(base) → view as uint32.
    Uses uint32 subtraction (mod 2^32) directly, avoiding int64 intermediate."""
    pm_b = pcmap_f32(base_f32)
    pm_f = pcmap_f32(ft_f32)
    # uint32 subtraction wraps mod 2^32, which when viewed as int32
    # gives the correct signed difference — equivalent to the int64 path.
    return (pm_f - pm_b)

def pcmap_delta_inverse(delta_u32: np.ndarray, base_f32: np.ndarray) -> np.ndarray:
    """Reconstruct ft IEEE 754 bits from delta and base.
    Returns uint32 array of IEEE 754 float32 bits.
    Uses uint32 addition (mod 2^32) directly, avoiding int64 intermediate."""
    pm_b = pcmap_f32(base_f32)
    pm_f_rec = pm_b + delta_u32
    return pcmap_inverse_f32(pm_f_rec)

def per_byte_entropy(u32_arr: np.ndarray) -> np.ndarray:
    """Shannon entropy for each of the 4 byte columns (byte0..byte3).
    Returns shape=(4,) array with values in [0, 8] bits.
"""
    raw = u32_arr.view(np.uint8)
    n = len(u32_arr)
    ents = np.zeros(4, dtype=np.float64)
    for b in range(4):
        col = raw[b::4]
        counts = np.bincount(col, minlength=256)
        p = counts[counts > 0] / n
        ents[b] = -np.sum(p * np.log2(p))
    return ents

def uint32_entropy(u32_arr: np.ndarray) -> float:
    """Shannon entropy over the full uint32 value space.
    Returns a scalar in [0, 32] bits."""
    n = len(u32_arr)
    if n == 0:
        return 0.0
    _, counts = np.unique(u32_arr.view(np.uint32), return_counts=True)
    p = counts / n
    return float(-np.sum(p * np.log2(p)))


PATH_IALIGN = "I-Align"      # rotl1(raw) → split 4 bytes → tANS per byte column
PATH_PCDELTA = "pcdelta"     # pcmap(ft) - pcmap(base) → (s,k) symbol + remainder → tANS
PATH_ROTDELTA = "rotdelta"   # rotl1(ft) - rotl1(base) → (s,k) symbol + remainder → tANS

@dataclass
class CompressStrategy:
    """Compression strategy for a single tensor."""
    path: str
    encoding: str
    ent_ialign: float
    ent_pcdelta: float
    ent_rotdelta: float

    @property
    def chosen_entropy(self) -> float:
        """Entropy of the chosen path (scalar)."""
        if self.path == PATH_IALIGN:
            return self.ent_ialign
        elif self.path == PATH_PCDELTA:
            return self.ent_pcdelta
        else:
            return self.ent_rotdelta

    def __repr__(self) -> str:
        return (f"CompressStrategy(path={self.path}, "
                f"encoding={self.encoding}, "
                f"ent_ialign={self.ent_ialign:.3f}, "
                f"ent_pcdelta={self.ent_pcdelta:.3f}, "
                f"ent_rotdelta={self.ent_rotdelta:.3f})")

def _entropy_byte_columns(u32_arr: np.ndarray) -> float:
    """Compute average byte-column Shannon entropy for I-Align path (global)."""
    rotated = rotl1(u32_arr.view(np.uint32))
    bytes_mat = rotated.view(np.uint8).reshape(-1, 4)
    n = len(rotated)
    total_entropy = 0.0
    for col in range(4):
        col_bytes = bytes_mat[:, col]
        _, counts = np.unique(col_bytes, return_counts=True)
        p = counts / n
        mask = p > 0
        total_entropy += float(-np.sum(p[mask] * np.log2(p[mask])))
    return max(total_entropy / 4.0, 0.0)

def _entropy_sk_symbolization(ref_u32: np.ndarray, tgt_u32: np.ndarray, period: int = 1024) -> float:
    """Compute seg1024 bound for (s,k) symbolization: H(65 symbols) + E[bsr].

    Uses uint32 wrapping subtraction to match the actual encoder behavior."""
    ref_u32 = ref_u32.ravel()
    tgt_u32 = tgt_u32.ravel()
    n = len(ref_u32)
    if n == 0:
        return 0.0
    # uint32 wrapping subtraction (matches encode_pc_symbol_f32 in C++)
    delta = tgt_u32 - ref_u32
    # Determine sign: delta > 0x80000000 means negative (wrapping)
    neg_mask = delta > np.uint32(0x80000000)
    abs_delta = np.where(neg_mask, -delta, delta).astype(np.uint32)
    symbols = np.full(n, _K_F32_BIAS, dtype=np.uint8)
    bsr_vals = np.full(n, 32, dtype=np.uint8)
    nonzero_mask = abs_delta > 0
    if nonzero_mask.any():
        nonzero_abs = abs_delta[nonzero_mask]
        bsr_nz = np.floor(np.log2(nonzero_abs.astype(np.float64))).astype(np.uint8)
        np.clip(bsr_nz, 0, 31, out=bsr_nz)
        bsr_vals[nonzero_mask] = bsr_nz
        pos_mask = ~neg_mask & nonzero_mask
        symbols[pos_mask] = _K_F32_BIAS + 1 + bsr_vals[pos_mask]
        symbols[neg_mask] = _K_F32_BIAS - 1 - bsr_vals[neg_mask]
    n_windows = (n + period - 1) // period
    pad_len = n_windows * period - n
    if pad_len > 0:
        symbols_padded = np.concatenate([symbols, np.full(pad_len, 65, dtype=np.uint8)])
        bsr_padded = np.concatenate([bsr_vals, np.full(pad_len, 33, dtype=np.uint8)])
    else:
        symbols_padded = symbols
        bsr_padded = bsr_vals
    sym_windows = symbols_padded.reshape(n_windows, period)
    bsr_windows = bsr_padded.reshape(n_windows, period)
    arange32 = np.arange(32, dtype=np.float64)
    total_bits = 0.0
    for w in range(n_windows):
        window_size = period if w < n_windows - 1 or pad_len == 0 else period - pad_len
        sym_counts = np.bincount(sym_windows[w], minlength=65)[:65]
        probs = sym_counts / window_size
        with np.errstate(divide='ignore', invalid='ignore'):
            log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        h_window = -np.sum(probs * log_probs)
        bsr_counts = np.bincount(bsr_windows[w], minlength=32)[:32]
        dist_bsr = bsr_counts / window_size
        e_window = dist_bsr @ arange32
        total_bits += window_size * (h_window + e_window)
    return total_bits / n

def analyze_strategy(
    ft_f32: np.ndarray,
    base_f32: Optional[np.ndarray] = None,
    sample: int = DEFAULT_SAMPLE,
    encoding: str = DEFAULT_ENCODING,
) -> CompressStrategy:
    """Analyze a tensor and determine the optimal compression path.

    Evaluates three candidate paths using seg1024-accurate entropy estimation:
      - I-Align:   rotl1(ft) → avg byte-column entropy × 4
      - pcdelta:   pcmap(ft) - pcmap(base) → H(65) + E[bsr]
      - rotdelta:  rotl1(ft) - rotl1(base) → H(65) + E[bsr]

    Args:
        ft_f32:   Finetuned tensor (flat float32 numpy array).
        base_f32: Base tensor (flat float32 numpy array). None → force I-Align.
        sample:   Max number of elements to sample for entropy estimation.
        encoding: Encoding algorithm name.

    Returns:
        CompressStrategy with selected path and all three entropy estimates.
    """
    n = len(ft_f32)
    # Sample at least 64K elements; for large tensors, sample 0.1% of total
    min_sample = max(DEFAULT_SAMPLE, max(n // 1000, 1))
    actual_sample = min(n, min_sample)

    # Sample indices (deterministic for reproducibility)
    rng = np.random.RandomState(0)
    idx = rng.choice(n, size=actual_sample, replace=False)
    ft_sample = ft_f32[idx]

    # I-Align entropy: rotl1(ft) → global byte-column entropy × 4
    ent_ialign = _entropy_byte_columns(ft_sample) * 4.0

    if base_f32 is not None:
        base_sample = base_f32[idx]

        # pcdelta: pcmap(ft) - pcmap(base)
        ref_pc = pcmap_f32(base_sample)
        tgt_pc = pcmap_f32(ft_sample)
        ent_pcdelta = _entropy_sk_symbolization(ref_pc, tgt_pc)

        # rotdelta: rotl1(ft) - rotl1(base)
        ref_rot = rotl1(base_sample.view(np.uint32))
        tgt_rot = rotl1(ft_sample.view(np.uint32))
        ent_rotdelta = _entropy_sk_symbolization(ref_rot, tgt_rot)
    else:
        # No base available: force I-Align
        ent_pcdelta = float('inf')
        ent_rotdelta = float('inf')

    # Select path with minimum entropy
    entropies = {
        PATH_IALIGN: ent_ialign,
        PATH_PCDELTA: ent_pcdelta,
        PATH_ROTDELTA: ent_rotdelta,
    }
    best_path = min(entropies, key=entropies.get)

    return CompressStrategy(
        path=best_path,
        encoding=encoding,
        ent_ialign=ent_ialign,
        ent_pcdelta=ent_pcdelta,
        ent_rotdelta=ent_rotdelta,
    )

# 3. Encoding / decoding dispatch
def check_encoding_available(encoding: str) -> None:
    """Raise RuntimeError if encoding is not available in the current environment."""
    if encoding in TANS_CODECS:
        return  # tANS is always available (pure Python)
    if encoding in U8_CODECS or encoding in GENERIC_CODECS:
        if not _MFE_AVAILABLE:
            raise RuntimeError(
                f"Encoding '{encoding}' requires the modelformat_encodings C++ extension, "
                "which is not installed. Use encoding='TANS' (no build required) or "
                "install the extension from CodeTensors/cpp/.")
        if encoding in GENERIC_CODECS and not _mfe.is_available(encoding):
            avail_cpp = [c for c in sorted(GENERIC_CODECS) if _mfe.is_available(c)]
            raise RuntimeError(
                f"Encoding '{encoding}' not compiled into this build. "
                f"Available C++ codecs: {sorted(U8_CODECS) + avail_cpp}")
        return
    raise RuntimeError(f"Unknown encoding '{encoding}'. Available: {ALL_ENCODINGS}")

_LEVEL_CODECS = {"ZSTD", "ZLIB"}  # C++ codecs that accept a compression level


def encode_byte_col(col_bytes: bytes, n_elems: int, encoding: str) -> bytes:
    if encoding in TANS_CODECS:
        if _CPP_TANS_AVAILABLE:
            try:
                return _cpp_tans.tans_encode(col_bytes)
            except (ValueError, RuntimeError):
                pass
        payload = _py_tans_encode(col_bytes)
        return payload.to_bytes()
    elif encoding in U8_CODECS:
        return _mfe.tensor_encode(col_bytes, _mfe.U8, encoding)
    elif encoding in _LEVEL_CODECS:
        return _mfe.compress(encoding, col_bytes, _ZSTD_LEVEL)
    else:
        return _mfe.compress(encoding, col_bytes)


def decode_byte_col(enc_bytes: bytes, n_elems: int, encoding: str) -> bytes:
    if encoding in TANS_CODECS:
        if enc_bytes[:4] == b'BTAN':
            return _cpp_tans.tans_decode(enc_bytes)
        payload = _TansPayload.from_bytes(enc_bytes)
        return _py_tans_decode(payload)
    elif encoding in U8_CODECS:
        return _mfe.tensor_decode(enc_bytes, _mfe.U8, encoding, n_elems)
    else:
        return _mfe.decompress(encoding, enc_bytes, n_elems)


# 4. (s, k) symbolization for pcdelta / rotdelta paths
_K_F32_BIAS = 32  # bias for signed level encoding (matches C++ kF32Bias)
_K_F32_SYMBOLS = 2 * _K_F32_BIAS + 1  # 65 symbols total

def _bsr32(value: np.ndarray) -> np.ndarray:
    """Bit-scan-right: position of highest set bit (0-indexed). Equivalent to floor(log2(x))."""
    # For uint32 array, use np.where to handle zeros
    result = np.zeros_like(value, dtype=np.uint32)
    nonzero = value > 0
    if np.any(nonzero):
        # Use log2 trick for vectorized bsr
        result[nonzero] = (np.log2(value[nonzero].astype(np.float64))).astype(np.uint32)
    return result

def encode_sk_symbol(reference_u32: np.ndarray, target_u32: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode (reference, target) pair into (symbol, mantissa, width) using C++ logic.

    Matches drotl1fmd_tans_cpp_codec.cpp::encode_pc_symbol_f32:
      - If ref < tgt:  delta = tgt - ref (positive), symbol = BIAS + 1 + bsr(delta)
      - If ref > tgt:  delta = ref - tgt (positive), symbol = BIAS - 1 - bsr(delta)
      - If ref == tgt: symbol = BIAS, mantissa = 0, width = 0

    Args:
        reference_u32: uint32 array (e.g., pcmap_f32(base) or rotl1(base))
        target_u32:    uint32 array (e.g., pcmap_f32(ft) or rotl1(ft))

    Returns:
        symbols: uint8 array of shape (n,), values in [0, 64]
        mantissas: uint32 array of shape (n,), the remainder bits
        widths: uint8 array of shape (n,), number of valid mantissa bits per element
    """
    n = len(reference_u32)
    symbols = np.empty(n, dtype=np.uint8)
    mantissas = np.empty(n, dtype=np.uint32)
    widths = np.empty(n, dtype=np.uint8)

    # Compare as unsigned uint32
    less_mask = reference_u32 < target_u32
    greater_mask = reference_u32 > target_u32
    equal_mask = reference_u32 == target_u32

    # Equal case
    symbols[equal_mask] = _K_F32_BIAS
    mantissas[equal_mask] = 0
    widths[equal_mask] = 0

    # Less case (target > reference): positive delta
    if np.any(less_mask):
        delta = target_u32[less_mask] - reference_u32[less_mask]  # uint32 subtraction
        levels = _bsr32(delta)
        symbols[less_mask] = (_K_F32_BIAS + 1 + levels).astype(np.uint8)
        mantissas[less_mask] = delta - (1 << levels)
        widths[less_mask] = levels.astype(np.uint8)

    # Greater case (reference > target): "negative" delta (magnitude stored)
    if np.any(greater_mask):
        delta = reference_u32[greater_mask] - target_u32[greater_mask]  # uint32 subtraction
        levels = _bsr32(delta)
        symbols[greater_mask] = (_K_F32_BIAS - 1 - levels).astype(np.uint8)
        mantissas[greater_mask] = delta - (1 << levels)
        widths[greater_mask] = levels.astype(np.uint8)

    return symbols, mantissas, widths

def decode_sk_symbol(
    symbols: np.ndarray,
    mantissas: np.ndarray,
    widths: np.ndarray,
    reference_u32: np.ndarray,
) -> np.ndarray:
    """Decode (symbol, mantissa, width) back to delta_u32, then reconstruct target.

    For pcdelta: reference_u32 = pcmap_f32(base), target = reference + signed_delta
    For rotdelta: reference_u32 = rotl1(base), target = reference + signed_delta

    Uses uint32 wrapping arithmetic throughout to match C++ behavior.

    Args:
        symbols: uint8 array from encode_sk_symbol
        mantissas: uint32 array from encode_sk_symbol
        widths: uint8 array from encode_sk_symbol
        reference_u32: the base/reference uint32 values

    Returns:
        target_u32: reconstructed target uint32 values
    """
    n = len(symbols)
    target_u32 = reference_u32.copy()

    # Zero case: symbol == BIAS → delta = 0 → target = reference
    # (already copied, no change needed)

    # Positive case (symbol > BIAS): target = reference + (1<<level + mantissa)
    pos_mask = symbols > _K_F32_BIAS
    if np.any(pos_mask):
        levels = (symbols[pos_mask].astype(np.uint32) - _K_F32_BIAS - 1)
        deltas = (1 << levels) + mantissas[pos_mask]
        target_u32[pos_mask] = reference_u32[pos_mask] + deltas

    # Negative case (symbol < BIAS): target = reference - (1<<level + mantissa)
    neg_mask = symbols < _K_F32_BIAS
    if np.any(neg_mask):
        levels = (_K_F32_BIAS - 1 - symbols[neg_mask].astype(np.uint32))
        magnitudes = (1 << levels) + mantissas[neg_mask]
        # uint32 subtraction wraps correctly
        target_u32[neg_mask] = reference_u32[neg_mask] - magnitudes

    return target_u32


# 5. Compress / decompress — three-path dispatcher
def compress_tensor(
    ft_f32: np.ndarray,
    strategy: CompressStrategy,
    base_f32: Optional[np.ndarray] = None,
) -> Tuple[int, list, float]:
    """Compress a float32 tensor according to the selected path.

    Args:
        ft_f32:   Flat float32 numpy array (finetuned tensor).
        strategy: CompressStrategy from analyze_strategy().
        base_f32: Flat float32 numpy array (base tensor). Required for delta paths.

    Returns:
        (total_compressed_bytes, all_chunk_parts, encode_seconds)
    """
    check_encoding_available(strategy.encoding)
    n = len(ft_f32)
    total_enc = 0
    t_enc = 0.0
    all_chunks = []

    for s in range(0, n, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n)
        ne = e - s

        t0 = time.perf_counter()
        chunk_data = _compress_chunk(ft_f32[s:e], base_f32[s:e] if base_f32 is not None else None, strategy, ne)
        t_enc += time.perf_counter() - t0

        chunk_sz = sum(len(part[0]) for part in chunk_data)
        total_enc += chunk_sz
        all_chunks.append(chunk_data)

    return total_enc, all_chunks, t_enc

def _compress_chunk(
    ft_chunk: np.ndarray,
    base_chunk: Optional[np.ndarray],
    strategy: CompressStrategy,
    ne: int,
) -> list:
    """Compress a single chunk according to the strategy path.

    Returns list of (encoded_bytes, metadata_dict) tuples.
    """
    if strategy.path == PATH_IALIGN:
        return _compress_ialign(ft_chunk, strategy.encoding, ne)
    elif strategy.path == PATH_PCDELTA:
        return _compress_pcdelta(ft_chunk, base_chunk, strategy.encoding, ne)
    elif strategy.path == PATH_ROTDELTA:
        return _compress_rotdelta(ft_chunk, base_chunk, strategy.encoding, ne)
    else:
        raise ValueError(f"Unknown path: {strategy.path}")

def _compress_ialign(ft_chunk: np.ndarray, encoding: str, ne: int) -> list:
    """I-Align: rotl1(raw ft) → split 4 bytes → tANS per column."""
    rot_u32 = rotl1(ft_chunk.view(np.uint32))
    byte_mat = rot_u32.view(np.uint8).reshape(-1, 4)

    parts = []
    for i in range(4):
        col = np.ascontiguousarray(byte_mat[:, i]).tobytes()
        enc = encode_byte_col(col, ne, encoding)
        parts.append((enc, {"col": i}))
    return parts

def _compress_pcdelta(ft_chunk: np.ndarray, base_chunk: np.ndarray, encoding: str, ne: int) -> list:
    """pcdelta: pcmap(ft) - pcmap(base) → (s,k) symbol + remainder."""
    ref_u32 = pcmap_f32(base_chunk)
    tgt_u32 = pcmap_f32(ft_chunk)
    symbols, mantissas, widths = encode_sk_symbol(ref_u32, tgt_u32)

    # Encode symbols with tANS
    sym_bytes = symbols.tobytes()
    sym_enc = encode_byte_col(sym_bytes, ne, encoding)

    # Pack mantissas: variable-width bits, pack into contiguous bitstream
    mantissa_stream = _pack_variable_bits(mantissas, widths)

    # Store widths as raw bytes (1 byte per element, max 32)
    widths_bytes = widths.tobytes()

    return [
        (sym_enc, {"type": "symbols"}),
        (mantissa_stream, {"type": "mantissa", "total_bits": len(mantissa_stream) * 8}),
        (widths_bytes, {"type": "widths"}),
    ]

def _compress_rotdelta(ft_chunk: np.ndarray, base_chunk: np.ndarray, encoding: str, ne: int) -> list:
    """rotdelta: rotl1(ft) - rotl1(base) → (s,k) symbol + remainder."""
    ref_u32 = rotl1(base_chunk.view(np.uint32))
    tgt_u32 = rotl1(ft_chunk.view(np.uint32))
    symbols, mantissas, widths = encode_sk_symbol(ref_u32, tgt_u32)

    sym_bytes = symbols.tobytes()
    sym_enc = encode_byte_col(sym_bytes, ne, encoding)

    mantissa_stream = _pack_variable_bits(mantissas, widths)
    widths_bytes = widths.tobytes()

    return [
        (sym_enc, {"type": "symbols"}),
        (mantissa_stream, {"type": "mantissa", "total_bits": len(mantissa_stream) * 8}),
        (widths_bytes, {"type": "widths"}),
    ]

def _pack_variable_bits(values: np.ndarray, widths: np.ndarray) -> bytes:
    """Pack variable-width integers into a contiguous bitstream (MSB first per value)."""
    # Simple implementation: accumulate bits into a bytearray
    bit_buffer = 0
    bits_in_buffer = 0
    output = bytearray()

    for val, w in zip(values, widths):
        w = int(w)
        if w == 0:
            continue
        # Append w bits of val (MSB first)
        mask = (1 << w) - 1
        val_bits = int(val) & mask
        bit_buffer = (bit_buffer << w) | val_bits
        bits_in_buffer += w
        while bits_in_buffer >= 8:
            bits_in_buffer -= 8
            output.append((bit_buffer >> bits_in_buffer) & 0xFF)
            bit_buffer &= (1 << bits_in_buffer) - 1

    # Flush remaining bits
    if bits_in_buffer > 0:
        output.append((bit_buffer << (8 - bits_in_buffer)) & 0xFF)

    return bytes(output)


# Try to import C++ rotdelta backend
_CPP_ROTDELTA_AVAILABLE = False
try:
    _cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")
    if _cpp_dir not in sys.path:
        sys.path.insert(0, _cpp_dir)
    import drotl1fmd_tans_cpp_codec as _cpp_rotdelta
    _CPP_ROTDELTA_AVAILABLE = True
except ImportError:
    _cpp_rotdelta = None

def compress_tensor(
    ft_f32: np.ndarray,
    strategy: CompressStrategy,
    base_f32: Optional[np.ndarray] = None,
) -> Tuple[int, list, float]:
    """Compress a float32 tensor according to the given strategy.

    Args:
        ft_f32:   Flat float32 numpy array (finetuned tensor).
        strategy: CompressStrategy from analyze_strategy().
        base_f32: Flat float32 numpy array (base tensor). Required for delta paths.

    Returns:
        (total_compressed_bytes, all_chunk_parts, encode_seconds)
        where all_chunk_parts is a list of chunk data for decompress_tensor.
    """
    # C++ fast paths for delta encoding
    if _CPP_ROTDELTA_AVAILABLE and base_f32 is not None:
        if strategy.path == PATH_ROTDELTA:
            t0 = time.perf_counter()
            encoded = _cpp_rotdelta.encode_drotl1fmd_tans_f32(
                base_f32.tobytes(), ft_f32.tobytes())
            t_enc = time.perf_counter() - t0
            all_chunks = [(encoded, {"cpp_delta": "rotdelta"})]
            return len(encoded), all_chunks, t_enc
        elif strategy.path == PATH_PCDELTA:
            t0 = time.perf_counter()
            encoded = _cpp_rotdelta.encode_pcdelta_tans_f32(
                base_f32.tobytes(), ft_f32.tobytes())
            t_enc = time.perf_counter() - t0
            all_chunks = [(encoded, {"cpp_delta": "pcdelta"})]
            return len(encoded), all_chunks, t_enc

    check_encoding_available(strategy.encoding)
    n = len(ft_f32)
    total_enc = 0
    t_enc = 0.0
    all_chunks = []

    for s in range(0, n, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n)
        ne = e - s

        t0 = time.perf_counter()
        chunk_data = _compress_chunk(ft_f32[s:e], base_f32[s:e] if base_f32 is not None else None, strategy, ne)
        t_enc += time.perf_counter() - t0

        chunk_sz = sum(len(part[0]) for part in chunk_data)
        total_enc += chunk_sz
        all_chunks.append(chunk_data)

    return total_enc, all_chunks, t_enc


def decompress_tensor(
    all_chunks: list,
    strategy: CompressStrategy,
    n_elems: int,
    base_f32: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Decompress and reconstruct the original float32 tensor.

    Args:
        all_chunks: Chunk parts list from compress_tensor().
        strategy:   CompressStrategy used for compression.
        n_elems:    Total number of float32 elements.
        base_f32:   Base tensor (required for delta paths).

    Returns:
        (reconstructed_f32_array, decode_seconds)
    """
    # C++ fast paths for delta decoding
    if (_CPP_ROTDELTA_AVAILABLE and len(all_chunks) == 1
            and isinstance(all_chunks[0], tuple) and len(all_chunks[0]) == 2
            and isinstance(all_chunks[0][1], dict)
            and base_f32 is not None):
        cpp_delta = all_chunks[0][1].get("cpp_delta")
        if cpp_delta:
            encoded_bytes = all_chunks[0][0]
            t0 = time.perf_counter()
            if cpp_delta == "rotdelta":
                dec_bytes = _cpp_rotdelta.decode_drotl1fmd_tans_f32(
                    encoded_bytes, base_f32.tobytes())
            else:
                dec_bytes = _cpp_rotdelta.decode_pcdelta_tans_f32(
                    encoded_bytes, base_f32.tobytes())
            t_dec = time.perf_counter() - t0
            result = np.frombuffer(dec_bytes, dtype=np.float32).copy()
            return result, t_dec

    result = np.empty(n_elems, dtype=np.float32)
    t_dec = 0.0
    chunk_idx = 0

    for s in range(0, n_elems, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n_elems)
        ne = e - s
        chunk_data = all_chunks[chunk_idx]
        chunk_idx += 1

        t0 = time.perf_counter()
        rec_u32 = _decompress_chunk(chunk_data, base_f32[s:e] if base_f32 is not None else None, strategy, ne)
        t_dec += time.perf_counter() - t0

        result[s:e] = rec_u32.view(np.float32)

    return result, t_dec

def _decompress_chunk(
    chunk_data: list,
    base_chunk: Optional[np.ndarray],
    strategy: CompressStrategy,
    ne: int,
) -> np.ndarray:
    """Decompress a single chunk."""
    if strategy.path == PATH_IALIGN:
        return _decompress_ialign(chunk_data, strategy.encoding, ne)
    elif strategy.path == PATH_PCDELTA:
        return _decompress_pcdelta(chunk_data, base_chunk, strategy.encoding, ne)
    elif strategy.path == PATH_ROTDELTA:
        return _decompress_rotdelta(chunk_data, base_chunk, strategy.encoding, ne)
    else:
        raise ValueError(f"Unknown path: {strategy.path}")

def _decompress_ialign(chunk_data: list, encoding: str, ne: int) -> np.ndarray:
    """I-Align: decode 4 byte columns → reassemble uint32 → rotr1."""
    dec_mat = np.empty((ne, 4), dtype=np.uint8)
    for part, meta in chunk_data:
        col = meta["col"]
        raw = decode_byte_col(part, ne, encoding)
        dec_mat[:, col] = np.frombuffer(raw, dtype=np.uint8)

    rec_rot = dec_mat.view(np.uint32).reshape(-1)
    return rotr1(rec_rot)

def _decompress_pcdelta(chunk_data: list, base_chunk: np.ndarray, encoding: str, ne: int) -> np.ndarray:
    """pcdelta: decode symbols + unpack mantissas → reconstruct target."""
    # Find parts by type
    sym_part = next(p for p, m in chunk_data if m["type"] == "symbols")
    mantissa_part = next(p for p, m in chunk_data if m["type"] == "mantissa")
    widths_part = next(p for p, m in chunk_data if m["type"] == "widths")

    # Decode symbols
    sym_raw = decode_byte_col(sym_part, ne, encoding)
    symbols = np.frombuffer(sym_raw, dtype=np.uint8)

    # Decode widths
    widths = np.frombuffer(widths_part, dtype=np.uint8)

    # Unpack mantissas
    mantissas = _unpack_variable_bits(mantissa_part, widths, ne)

    # Reconstruct delta and target (in pcmap space)
    ref_u32 = pcmap_f32(base_chunk)
    tgt_pcmap = decode_sk_symbol(symbols, mantissas, widths, ref_u32)

    # Inverse pcmap to get original float32 bits
    return pcmap_inverse_f32(tgt_pcmap)

def _decompress_rotdelta(chunk_data: list, base_chunk: np.ndarray, encoding: str, ne: int) -> np.ndarray:
    """rotdelta: decode symbols + unpack mantissas → reconstruct target."""
    sym_part = next(p for p, m in chunk_data if m["type"] == "symbols")
    mantissa_part = next(p for p, m in chunk_data if m["type"] == "mantissa")
    widths_part = next(p for p, m in chunk_data if m["type"] == "widths")

    sym_raw = decode_byte_col(sym_part, ne, encoding)
    symbols = np.frombuffer(sym_raw, dtype=np.uint8)

    widths = np.frombuffer(widths_part, dtype=np.uint8)
    mantissas = _unpack_variable_bits(mantissa_part, widths, ne)

    # Reconstruct in rotl1 space
    ref_u32 = rotl1(base_chunk.view(np.uint32))
    tgt_rotl1 = decode_sk_symbol(symbols, mantissas, widths, ref_u32)

    # Inverse rotl1 to get original float32 bits
    return rotr1(tgt_rotl1)

def _unpack_variable_bits(bitstream: bytes, widths: np.ndarray, n_elems: int) -> np.ndarray:
    """Unpack variable-width integers from a contiguous bitstream."""
    mantissas = np.zeros(n_elems, dtype=np.uint32)
    bit_pos = 0
    bits = bytearray(bitstream)

    for i, w in enumerate(widths):
        w = int(w)
        if w == 0:
            continue
        # Read w bits from bitstream (MSB first)
        val = 0
        for _ in range(w):
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)
            if byte_idx < len(bits):
                bit = (bits[byte_idx] >> bit_idx) & 1
                val = (val << 1) | bit
            bit_pos += 1
        mantissas[i] = val

    return mantissas


# 5. High-level convenience function
def compress_and_verify(
    ft_f32: np.ndarray,
    base_f32: Optional[np.ndarray] = None,
    sample: int = DEFAULT_SAMPLE,
    encoding: str = DEFAULT_ENCODING,
) -> dict:
    """One-stop: analyze → compress → decompress → bitwise verify.

    Returns dict with keys:
        strategy, compressed_size, ratio, bitwise_ok,
        enc_sec, dec_sec, ent_ialign, ent_pcdelta, ent_rotdelta
    """
    strategy = analyze_strategy(
        ft_f32, base_f32,
        sample=sample,
        encoding=encoding,
    )

    orig_bytes = len(ft_f32) * 4
    comp_sz, chunks, enc_sec = compress_tensor(ft_f32, strategy, base_f32)
    rec_f32, dec_sec = decompress_tensor(chunks, strategy, len(ft_f32), base_f32)

    # Bitwise verification
    bitwise_ok = np.array_equal(
        rec_f32.view(np.uint32),
        ft_f32.view(np.uint32),
    )

    return {
        "strategy": strategy,
        "compressed_size": comp_sz,
        "ratio": comp_sz / orig_bytes if orig_bytes > 0 else 0.0,
        "bitwise_ok": bitwise_ok,
        "enc_sec": enc_sec,
        "dec_sec": dec_sec,
        "ent_ialign": strategy.ent_ialign,
        "ent_pcdelta": strategy.ent_pcdelta,
        "ent_rotdelta": strategy.ent_rotdelta,
    }


# 6. Leading-Zero aware ByteCol (ByteCol-LZ)
# Instead of encoding all 4 byte columns independently, we first compute
# the number of valid (non-zero) bytes per element from the MSB side.
# This "nvalid" column (values 0-4) is encoded with RC, and then each
# byte column only stores bytes for elements where nvalid > col_index.
# This captures the same leading-zero information that FMDelta's PCEncoder
# exploits, but within the columnar framework for better throughput.
#
# After rotl1, byte layout is [byte0(LSB), byte1, byte2, byte3(MSB)].
# nvalid counts from MSB downward:
#   nvalid=0 → element is 0x00000000, store nothing
#   nvalid=1 → only byte0 is stored (byte1,2,3 are all 0)
#   nvalid=2 → byte0,byte1 stored
#   nvalid=3 → byte0,byte1,byte2 stored
#   nvalid=4 → all 4 bytes stored

def _compute_nvalid(rot_u32: np.ndarray) -> np.ndarray:
    """Compute number of valid bytes per element (0-4) from MSB side.
    nvalid = ceil(bit_length / 8), equivalently the index of the highest
    non-zero byte + 1."""
    nvalid = np.zeros(len(rot_u32), dtype=np.uint8)
    nvalid[rot_u32 > 0] = 1
    nvalid[rot_u32 > 0xFF] = 2
    nvalid[rot_u32 > 0xFFFF] = 3
    nvalid[rot_u32 > 0xFFFFFF] = 4
    return nvalid


def compress_tensor_lz(
    ft_f32: np.ndarray,
    strategy: CompressStrategy,
    base_f32: Optional[np.ndarray] = None,
    encoding: str = DEFAULT_ENCODING,
) -> Tuple[int, list, float]:
    """Leading-Zero aware ByteCol compression.

    Like compress_tensor but skips zero bytes from the MSB side per element.
    Encodes an extra 'nvalid' column (values 0-4) and only stores bytes
    for columns where the element has valid data.

    Args:
        ft_f32:   Flat float32 numpy array.
        strategy: CompressStrategy (use_delta and compress_mask are used).
        base_f32: Base tensor (required if use_delta).
        encoding: Encoding algorithm for byte columns.

    Returns:
        (total_compressed_bytes, all_chunk_parts, encode_seconds)
    """
    check_encoding_available(encoding)
    n = len(ft_f32)
    total_enc = 0
    t_enc = 0.0
    all_chunks = []

    for s in range(0, n, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n)
        ne = e - s

        rot_u32, _ = _prepare_rotl1_bytes(ft_f32, base_f32, strategy.use_delta, s, e)
        nvalid = _compute_nvalid(rot_u32)
        byte_mat = rot_u32.view(np.uint8).reshape(-1, 4)

        t0 = time.perf_counter()

        # Encode nvalid column (values 0-4, very low entropy)
        nvalid_enc = encode_byte_col(nvalid.tobytes(), ne, encoding)

        # Encode each byte column, but only for elements with nvalid > col_idx
        col_parts = []
        col_sizes = 0
        for col_idx in range(4):
            mask = nvalid > col_idx
            count = int(np.count_nonzero(mask))
            if count == 0:
                col_parts.append((b'', 0))
            else:
                col_bytes = np.ascontiguousarray(byte_mat[mask, col_idx]).tobytes()
                col_enc = encode_byte_col(col_bytes, count, encoding)
                col_parts.append((col_enc, count))
                col_sizes += len(col_enc)

        t_enc += time.perf_counter() - t0

        chunk_sz = len(nvalid_enc) + col_sizes
        total_enc += chunk_sz
        all_chunks.append((nvalid_enc, col_parts))

    return total_enc, all_chunks, t_enc


def decompress_tensor_lz(
    all_chunks: list,
    strategy: CompressStrategy,
    n_elems: int,
    base_f32: Optional[np.ndarray] = None,
    encoding: str = DEFAULT_ENCODING,
) -> Tuple[np.ndarray, float]:
    """Decompress a tensor compressed with compress_tensor_lz.

    Args:
        all_chunks: Chunk parts from compress_tensor_lz().
        strategy:   CompressStrategy used for compression.
        n_elems:    Total number of float32 elements.
        base_f32:   Base tensor (required if use_delta).
        encoding:   Encoding algorithm used during compression.

    Returns:
        (reconstructed_f32_array, decode_seconds)
    """
    result = np.empty(n_elems, dtype=np.float32)
    t_dec = 0.0
    chunk_idx = 0

    for s in range(0, n_elems, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n_elems)
        ne = e - s
        nvalid_enc, col_parts = all_chunks[chunk_idx]
        chunk_idx += 1

        t0 = time.perf_counter()

        # Decode nvalid column
        nvalid_raw = decode_byte_col(nvalid_enc, ne, encoding)
        nvalid = np.frombuffer(nvalid_raw, dtype=np.uint8)

        # Reconstruct byte matrix (all zeros initially)
        byte_mat = np.zeros((ne, 4), dtype=np.uint8)

        for col_idx in range(4):
            col_enc, count = col_parts[col_idx]
            if count == 0:
                continue
            col_raw = decode_byte_col(col_enc, count, encoding)
            col_bytes = np.frombuffer(col_raw, dtype=np.uint8)
            mask = nvalid > col_idx
            byte_mat[mask, col_idx] = col_bytes

        # rotr1 to undo rotl1
        rec_rot = byte_mat.view(np.uint32).reshape(-1)
        rec_u32 = rotr1(rec_rot)

        # Reconstruct float32
        if strategy.use_delta:
            ft_ieee = pcmap_delta_inverse(rec_u32, base_f32[s:e])
            result[s:e] = ft_ieee.view(np.float32) if ft_ieee.dtype == np.uint32 else ft_ieee
        else:
            result[s:e] = rec_u32.view(np.float32)

        t_dec += time.perf_counter() - t0

    return result, t_dec


# 7. Bit-level Leading-Zero aware ByteCol (ByteCol-LZB)
# Like ByteCol-LZ but with bit-level precision for the leading-zero count.
# Instead of nvalid (0-4 bytes), we encode nbits (0-32), the exact number
# of significant bits per element. This allows us to strip the known
# leading-1 bit from the top byte, reducing its entropy.
#
# This is analogous to FMDelta's PCEncoder which encodes bsr(delta) as
# the level symbol and then stores the remainder bits without the leading 1.

def _compute_nbits(rot_u32: np.ndarray) -> np.ndarray:
    """Compute number of significant bits per element (0-32).
    nbits = floor(log2(x)) + 1 for x > 0, 0 for x == 0."""
    nbits = np.zeros(len(rot_u32), dtype=np.uint8)
    nonzero_mask = rot_u32 > 0
    if np.any(nonzero_mask):
        nonzero_vals = rot_u32[nonzero_mask].astype(np.uint64)
        nbits[nonzero_mask] = np.floor(np.log2(nonzero_vals)).astype(np.uint8) + 1
    return nbits


def _strip_leading_one(byte_mat: np.ndarray, nbits: np.ndarray) -> np.ndarray:
    """Strip the leading 1 bit from the top byte of each element.
    Returns a modified copy of byte_mat.

    For element with nbits=b (b>0):
      top_col = (b-1) // 8
      bits_in_top = ((b-1) % 8) + 1
      Clear bit at position (bits_in_top - 1) in byte_mat[:, top_col].
    """
    result = byte_mat.copy()
    for b in range(1, 33):
        mask = nbits == b
        if not np.any(mask):
            continue
        top_col = (b - 1) // 8
        bits_in_top = ((b - 1) % 8) + 1
        clear_bit = np.uint8(1 << (bits_in_top - 1))
        result[mask, top_col] = result[mask, top_col] & np.uint8(~clear_bit & 0xFF)
    return result


def _restore_leading_one(byte_mat: np.ndarray, nbits: np.ndarray) -> np.ndarray:
    """Restore the leading 1 bit to the top byte of each element.
    Inverse of _strip_leading_one."""
    result = byte_mat.copy()
    for b in range(1, 33):
        mask = nbits == b
        if not np.any(mask):
            continue
        top_col = (b - 1) // 8
        bits_in_top = ((b - 1) % 8) + 1
        set_bit = np.uint8(1 << (bits_in_top - 1))
        result[mask, top_col] = result[mask, top_col] | set_bit
    return result


def compress_tensor_lzb(
    ft_f32: np.ndarray,
    strategy: CompressStrategy,
    base_f32: Optional[np.ndarray] = None,
    encoding: str = DEFAULT_ENCODING,
) -> Tuple[int, list, float]:
    """Bit-level Leading-Zero aware ByteCol compression.

    Like compress_tensor_lz but with bit-level precision:
    - Encodes nbits (0-32) instead of nvalid (0-4)
    - Strips the leading 1 bit from the top byte for better compression

    Args:
        ft_f32:   Flat float32 numpy array.
        strategy: CompressStrategy (use_delta is used).
        base_f32: Base tensor (required if use_delta).
        encoding: Encoding algorithm for byte columns.

    Returns:
        (total_compressed_bytes, all_chunk_parts, encode_seconds)
    """
    check_encoding_available(encoding)
    n = len(ft_f32)
    total_enc = 0
    t_enc = 0.0
    all_chunks = []

    for s in range(0, n, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n)
        ne = e - s

        rot_u32, _ = _prepare_rotl1_bytes(ft_f32, base_f32, strategy.use_delta, s, e)
        nbits = _compute_nbits(rot_u32)
        byte_mat = rot_u32.view(np.uint8).reshape(-1, 4)

        # Strip leading 1 bit from top byte
        byte_mat_stripped = _strip_leading_one(byte_mat, nbits)

        # Compute nvalid_bytes from nbits for column filtering
        nvalid = np.zeros(ne, dtype=np.uint8)
        nz = nbits > 0
        nvalid[nz] = ((nbits[nz].astype(np.int32) - 1) // 8 + 1).astype(np.uint8)

        t0 = time.perf_counter()

        # Encode nbits column (values 0-32, low entropy)
        nbits_enc = encode_byte_col(nbits.tobytes(), ne, encoding)

        # Encode each byte column, only for elements with nvalid > col_idx
        col_parts = []
        col_sizes = 0
        for col_idx in range(4):
            mask = nvalid > col_idx
            count = int(np.count_nonzero(mask))
            if count == 0:
                col_parts.append((b'', 0))
            else:
                col_bytes = np.ascontiguousarray(byte_mat_stripped[mask, col_idx]).tobytes()
                col_enc = encode_byte_col(col_bytes, count, encoding)
                col_parts.append((col_enc, count))
                col_sizes += len(col_enc)

        t_enc += time.perf_counter() - t0

        chunk_sz = len(nbits_enc) + col_sizes
        total_enc += chunk_sz
        all_chunks.append((nbits_enc, col_parts))

    return total_enc, all_chunks, t_enc


def decompress_tensor_lzb(
    all_chunks: list,
    strategy: CompressStrategy,
    n_elems: int,
    base_f32: Optional[np.ndarray] = None,
    encoding: str = DEFAULT_ENCODING,
) -> Tuple[np.ndarray, float]:
    """Decompress a tensor compressed with compress_tensor_lzb.

    Args:
        all_chunks: Chunk parts from compress_tensor_lzb().
        strategy:   CompressStrategy used for compression.
        n_elems:    Total number of float32 elements.
        base_f32:   Base tensor (required if use_delta).
        encoding:   Encoding algorithm used during compression.

    Returns:
        (reconstructed_f32_array, decode_seconds)
    """
    result = np.empty(n_elems, dtype=np.float32)
    t_dec = 0.0
    chunk_idx = 0

    for s in range(0, n_elems, CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, n_elems)
        ne = e - s
        nbits_enc, col_parts = all_chunks[chunk_idx]
        chunk_idx += 1

        t0 = time.perf_counter()

        # Decode nbits column
        nbits_raw = decode_byte_col(nbits_enc, ne, encoding)
        nbits = np.frombuffer(nbits_raw, dtype=np.uint8)

        # Compute nvalid from nbits
        nvalid = np.zeros(ne, dtype=np.uint8)
        nz = nbits > 0
        nvalid[nz] = ((nbits[nz].astype(np.int32) - 1) // 8 + 1).astype(np.uint8)

        # Reconstruct byte matrix (all zeros initially)
        byte_mat = np.zeros((ne, 4), dtype=np.uint8)

        for col_idx in range(4):
            col_enc, count = col_parts[col_idx]
            if count == 0:
                continue
            col_raw = decode_byte_col(col_enc, count, encoding)
            col_bytes = np.frombuffer(col_raw, dtype=np.uint8)
            mask = nvalid > col_idx
            byte_mat[mask, col_idx] = col_bytes

        # Restore leading 1 bit
        byte_mat = _restore_leading_one(byte_mat, nbits)

        # rotr1 to undo rotl1
        rec_rot = byte_mat.view(np.uint32).reshape(-1)
        rec_u32 = rotr1(rec_rot)

        # Reconstruct float32
        if strategy.use_delta:
            ft_ieee = pcmap_delta_inverse(rec_u32, base_f32[s:e])
            result[s:e] = ft_ieee.view(np.float32) if ft_ieee.dtype == np.uint32 else ft_ieee
        else:
            result[s:e] = rec_u32.view(np.float32)

        t_dec += time.perf_counter() - t0

    return result, t_dec
