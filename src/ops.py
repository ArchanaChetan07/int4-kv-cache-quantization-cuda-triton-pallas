"""
Dispatch layer for Flash Decoding INT4.

One entry point per op, four implementations behind it: the NumPy reference
(always available), the CUDA extension, the Triton quantizer and the Pallas
kernels. Auto-detection (backend=None) prefers CUDA and falls back to the
reference; an explicitly named backend is delivered or raises.
"""

import numpy as np
from typing import List, Optional, Tuple

from .quantize_int4_ref import quantize_int4_ref
from .flash_decode_ref import online_softmax_ref
from ._prepared import PreparedCall
from ._validate import normalize_attention_inputs

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from . import _C
    HAS_CUDA = _C is not None
except ImportError:
    HAS_CUDA = False
    _C = None

if not HAS_CUDA:
    # Opt-in JIT compile: FLASH_DECODE_JIT_CUDA=1 (needs GPU + nvcc)
    from ._jit import load_extension
    _C = load_extension("flash_decode_int4_C", "FLASH_DECODE_JIT_CUDA")
    HAS_CUDA = _C is not None

try:
    import jax  # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None

def _has_accelerator() -> bool:
    """True when jax sees a non-CPU device (TPU or supported GPU)."""
    if not HAS_JAX:
        return False
    try:
        return any(d.platform != "cpu" for d in jax.devices())
    except Exception:
        return False


def _has_tpu() -> bool:
    if not HAS_JAX:
        return False
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def _pallas_interpret(force: Optional[bool] = None) -> bool:
    """Whether the Pallas backend should run in interpret mode.

    Compiled only on TPU. The kernels accumulate across a sequential grid
    axis, which JAX's Pallas GPU backend runs in parallel -- measured racing
    on an H100 (see _pallas_platform). On GPU and CPU hosts the default is
    interpret mode: the kernel body runs as JAX ops (on the GPU where there is
    one) and is correct, but its timings are not kernel performance numbers.
    """
    if force is not None:
        return force
    return not _has_tpu()


def _use_cuda(force: Optional[bool] = None) -> bool:
    if force is not None:
        return force and HAS_CUDA
    return HAS_CUDA and HAS_TORCH


#: Backends implementing each op. Attention deliberately omits "triton": this
#: repository has a Triton quantizer but no Triton attention kernel.
QUANTIZE_BACKENDS = ("reference", "cuda", "triton", "pallas")
ATTENTION_BACKENDS = ("reference", "cuda", "pallas")


def _validate_backend(backend: Optional[str], supported: tuple, op: str) -> None:
    """Reject unknown or unimplemented backends instead of falling back.

    A silent fallback here is the worst possible failure mode for this project:
    asking for one backend and receiving another produces a benchmark that
    measures the wrong thing and reports it under the wrong name. Every caller
    that names a backend gets it or gets an exception.
    """
    if backend is None or backend in supported:
        return
    if backend == "triton" and op == "attention":
        raise ValueError(
            "no Triton attention kernel exists in this repository -- the Triton "
            "port covers the quantizer only. Use backend='cuda' or 'pallas' for "
            f"attention, or one of {QUANTIZE_BACKENDS} for quantize_int4()."
        )
    raise ValueError(
        f"unknown backend {backend!r} for {op}; supported: {supported}"
    )


def quantize_int4(
    kv: np.ndarray,
    use_cuda: Optional[bool] = None,
    backend: Optional[str] = None,
    interpret: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize KV to INT4 (per-channel asymmetric).

    Args:
        kv: 2D array [num_rows, num_channels]
        use_cuda: Force CUDA on/off (None = auto-detect). Legacy flag, kept so
            existing callers are unaffected.
        backend: Explicit backend: "cuda", "triton", "pallas" or "reference".
            Takes precedence over use_cuda when given.
        interpret: Pallas only -- run as ordinary JAX instead of lowering to
            Mosaic. Defaults to True when no accelerator is visible.

    Returns:
        (q, scale, zp) as NumPy arrays
    """
    _validate_backend(backend, QUANTIZE_BACKENDS, "quantize")

    if backend == "pallas":
        from .quantize_int4_pallas import quantize_int4_pallas
        return quantize_int4_pallas(kv, interpret=_pallas_interpret(interpret))

    if backend == "triton":
        from .quantize_int4_triton import quantize_int4_triton
        return quantize_int4_triton(kv)

    if backend == "reference":
        return quantize_int4_ref(kv, per_channel=True)

    if backend == "cuda" or (backend is None and _use_cuda(use_cuda)):
        if backend == "cuda" and not HAS_CUDA:
            raise RuntimeError(
                "CUDA backend requested but the extension is not built. Set "
                "FLASH_DECODE_JIT_CUDA=1 on a GPU host with nvcc."
            )
        if backend == "cuda" and kv.ndim != 2:
            raise ValueError(
                f"the CUDA quantizer takes a 2D [rows, channels] array, got "
                f"shape {kv.shape}. Use backend='reference' for higher-rank input."
            )
        if kv.ndim == 2 and HAS_CUDA:
            return prepare_quantize_int4_cuda(kv)()
        # backend == "cuda" with a bad shape or missing extension already raised
        # above; reaching here means auto-detect chose CUDA but the input is not
        # 2D, which legitimately falls back to the reference.

    return quantize_int4_ref(kv, per_channel=True)


#: The CUDA attention kernel keeps its output accumulator in 8 lane-register
#: slots per thread (MAX_DIM_FRAGS * WARP_SIZE in flash_decode_int4.cu).
CUDA_MAX_HEAD_DIM = 256


def _reference_attention(query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens):
    key_scale_zp = [(k_q_blocks[i], k_scales[i], k_zps[i])
                    for i in range(len(k_q_blocks))]
    output, _ = online_softmax_ref(query, key_scale_zp, v_blocks, block_lens)
    return output


def flash_decode(
    query: np.ndarray,
    k_q_blocks: List[np.ndarray],
    k_scales: List[np.ndarray],
    k_zps: List[np.ndarray],
    v_blocks: List[np.ndarray],
    block_lens: Optional[np.ndarray] = None,
    use_cuda: Optional[bool] = None,
    backend: Optional[str] = None,
    interpret: Optional[bool] = None,
) -> np.ndarray:
    """Fused online-softmax attention over INT4 quantized paged KV.

    Every query in the [batch, heads] grid attends over the SAME page stream.
    That is the kernels' interface: one launch serves many queries against one
    sequence's keys and values. Attention with distinct KV per head is one call
    per (sequence, head) stream.

    Args:
        query: [batch, heads, head_dim]
        k_q_blocks: List of INT4-quantized key pages [rows, head_dim], uint8
        k_scales: List of per-channel scales [head_dim]
        k_zps: List of per-channel zero-points [head_dim]
        v_blocks: List of value pages [rows, head_dim]
        block_lens: Valid rows per page; each must lie in [0, rows of that page].
            Defaults to full pages.
        use_cuda: Force CUDA on/off (None = auto-detect). Legacy flag.
        backend: Explicit backend: "cuda", "pallas" or "reference". A named
            backend is delivered or raises -- it is never substituted.
        interpret: Pallas only -- run as ordinary JAX instead of lowering.

    Returns:
        Attention output [batch, heads, head_dim], float32
    """
    _validate_backend(backend, ATTENTION_BACKENDS, "attention")
    query, block_lens = normalize_attention_inputs(
        query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens)

    num_blocks = len(k_q_blocks)
    if num_blocks == 0:
        # Attention over an empty cache is the all-pages-empty case: no mass
        # was accumulated, so every backend returns exact zeros.
        return np.zeros(query.shape, dtype=np.float32)

    if backend == "pallas":
        from .flash_decode_pallas import flash_decode_pallas
        return flash_decode_pallas(
            query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens,
            interpret=_pallas_interpret(interpret),
        )

    if backend == "reference":
        return _reference_attention(
            query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens)

    if backend == "cuda" and not HAS_CUDA:
        raise RuntimeError(
            "CUDA backend requested but the extension is not built. Set "
            "FLASH_DECODE_JIT_CUDA=1 on a GPU host with nvcc."
        )

    head_dim = query.shape[2]
    use_kernel = backend == "cuda" or (backend is None and _use_cuda(use_cuda))
    if use_kernel and head_dim > CUDA_MAX_HEAD_DIM:
        if backend == "cuda":
            raise ValueError(
                f"the CUDA attention kernel supports head_dim <= "
                f"{CUDA_MAX_HEAD_DIM}, got {head_dim}. Use backend='pallas' or "
                "'reference' for wider heads."
            )
        use_kernel = False   # auto-detect: fall back rather than overflow

    if use_kernel:
        return prepare_flash_decode_cuda(
            query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens)()

    return _reference_attention(
        query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens)


def _cuda_sync(_out) -> None:
    torch.cuda.synchronize()


def prepare_quantize_int4_cuda(kv: np.ndarray) -> PreparedCall:
    """Stage `kv` on the GPU once; run() is one kernel launch."""
    if not HAS_CUDA:
        raise RuntimeError("CUDA extension not built")
    kv_t = torch.from_numpy(np.ascontiguousarray(kv, dtype=np.float32)).cuda()
    return PreparedCall(
        run=lambda: _C.quantize_int4(kv_t),
        sync=_cuda_sync,
        to_numpy=lambda out: tuple(t.cpu().numpy() for t in out),
    )


def prepare_flash_decode_cuda(query, k_q_blocks, k_scales, k_zps, v_blocks,
                              block_lens=None) -> PreparedCall:
    """Validate, pad and stage inputs on the GPU once; run() is one launch.

    flash_decode(backend="cuda") is this plus a copy back to host, so the timed
    path and the tested path are the same code.
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA extension not built")
    query, block_lens = normalize_attention_inputs(
        query, k_q_blocks, k_scales, k_zps, v_blocks, block_lens)
    num_blocks = len(k_q_blocks)
    head_dim = query.shape[2]
    if num_blocks == 0:
        zeros = np.zeros(query.shape, dtype=np.float32)
        return PreparedCall(run=lambda: zeros)
    if head_dim > CUDA_MAX_HEAD_DIM:
        raise ValueError(
            f"the CUDA attention kernel supports head_dim <= {CUDA_MAX_HEAD_DIM}, "
            f"got {head_dim}")

    block_size = max(b.shape[0] for b in k_q_blocks)

    # Pad pages to a uniform size for the kernel's dense layout
    k_q_arr = np.zeros((num_blocks, block_size, head_dim), dtype=np.uint8)
    v_arr = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
    scale_arr = np.zeros((num_blocks, head_dim), dtype=np.float32)
    zp_arr = np.zeros((num_blocks, head_dim), dtype=np.float32)
    for i in range(num_blocks):
        n = k_q_blocks[i].shape[0]
        k_q_arr[i, :n] = k_q_blocks[i]
        v_arr[i, :n] = v_blocks[i]
        scale_arr[i] = k_scales[i]
        zp_arr[i] = k_zps[i]

    dev = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in
           (query.astype(np.float32), k_q_arr, scale_arr, zp_arr, v_arr, block_lens)]
    return PreparedCall(
        run=lambda: _C.flash_decode_int4(*dev),
        sync=_cuda_sync,
        to_numpy=lambda out: out.cpu().numpy(),
    )


def backend_status() -> dict:
    """Return current backend availability."""
    try:
        from .quantize_int4_triton import HAS_TRITON
    except Exception:   # a broken triton install can raise more than ImportError
        HAS_TRITON = False

    devices = []
    if HAS_JAX:
        try:
            devices = [d.platform for d in jax.devices()]
        except Exception:
            devices = []

    return {
        'has_torch': HAS_TORCH,
        'has_cuda': HAS_CUDA,
        'has_triton': HAS_TRITON,
        'has_jax': HAS_JAX,
        'jax_devices': devices,
        'pallas_interpret_default': _pallas_interpret(),
        'active_backend': 'cuda' if _use_cuda() else 'reference',
        'available_backends': (
            ['reference']
            + (['cuda'] if HAS_CUDA else [])
            + (['triton'] if HAS_TRITON else [])
            + (['pallas'] if HAS_JAX else [])
        ),
    }
