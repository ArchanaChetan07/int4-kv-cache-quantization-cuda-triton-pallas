"""Triton port of the per-channel asymmetric INT4 quantizer.

Same contract as quantize_int4_ref / the CUDA kernel:

    scale[c] = (max[c] - min[c]) / 15
    zp[c]    = -min[c] / scale[c]
    q        = clip(round_half_up((kv - min[c]) / scale[c]), 0, 15)

One Triton program per channel; each program strides over rows in BLOCK-sized
chunks: a min/max pass, then a quantize pass. Rounding is half-up
(floor(x + 0.5)) to match the CUDA kernel; the NumPy reference uses banker's
rounding, so parity tests allow a <=1-bin difference at exact .5 boundaries.

Runs on GPU via the Triton compiler, or on CPU via TRITON_INTERPRET=1 —
which is how CI validates these numerics without a GPU.
"""

from typing import Tuple

from ._prepared import PreparedCall

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:

    @triton.jit
    def _quantize_int4_kernel(
        kv_ptr,          # *f32 [n_rows, n_ch]
        q_ptr,           # *u8  [n_rows, n_ch]
        scale_ptr,       # *f32 [n_ch]
        zp_ptr,          # *f32 [n_ch]
        n_rows,
        n_ch,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        row = tl.arange(0, BLOCK)

        # Pass 1: per-channel min/max over all rows
        mn = float("inf")
        mx = float("-inf")
        for start in range(0, n_rows, BLOCK):
            offs = start + row
            mask = offs < n_rows
            x_min = tl.load(kv_ptr + offs * n_ch + c, mask=mask, other=float("inf"))
            x_max = tl.load(kv_ptr + offs * n_ch + c, mask=mask, other=float("-inf"))
            mn = tl.minimum(mn, tl.min(x_min, axis=0))
            mx = tl.maximum(mx, tl.max(x_max, axis=0))

        rng = tl.maximum(mx - mn, 1e-8)
        scale = rng / 15.0
        tl.store(scale_ptr + c, scale)
        tl.store(zp_ptr + c, -mn / scale)

        # Pass 2: quantize (half-up rounding; (x - mn)/scale >= 0 so
        # int truncation of value + 0.5 is floor)
        for start in range(0, n_rows, BLOCK):
            offs = start + row
            mask = offs < n_rows
            x = tl.load(kv_ptr + offs * n_ch + c, mask=mask, other=0.0)
            qf = (x - mn) / scale + 0.5
            qi = qf.to(tl.int32)
            qi = tl.minimum(tl.maximum(qi, 0), 15)
            tl.store(q_ptr + offs * n_ch + c, qi.to(tl.uint8), mask=mask)


def prepare_quantize_int4_triton(kv, block: int = 1024) -> PreparedCall:
    """Stage `kv` and the outputs on the device once; see quantize_int4_triton.

    The returned call's run() is a single kernel launch on device-resident
    tensors -- what a benchmark should time.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed")

    import os
    import numpy as np
    import torch

    if isinstance(kv, np.ndarray):
        kv_t = torch.from_numpy(np.ascontiguousarray(kv, dtype=np.float32))
    else:
        kv_t = kv.float().contiguous()

    if kv_t.dim() != 2:
        raise ValueError(f"expected [num_rows, num_channels], got {tuple(kv_t.shape)}")
    n_rows, n_ch = kv_t.shape
    if n_rows == 0 or n_ch == 0:
        raise ValueError(
            f"cannot quantize an empty array (shape {tuple(kv_t.shape)}): "
            "per-channel min/max over zero rows is undefined")

    # Prefer real GPU execution when present; interpreter mode
    # (TRITON_INTERPRET=1) runs on CPU tensors.
    if (torch.cuda.is_available() and kv_t.device.type == "cpu"
            and os.environ.get("TRITON_INTERPRET") != "1"):
        kv_t = kv_t.cuda()
    device = kv_t.device

    q = torch.empty((n_rows, n_ch), dtype=torch.uint8, device=device)
    scale = torch.empty(n_ch, dtype=torch.float32, device=device)
    zp = torch.empty(n_ch, dtype=torch.float32, device=device)

    def run():
        _quantize_int4_kernel[(n_ch,)](kv_t, q, scale, zp, n_rows, n_ch, BLOCK=block)
        return q, scale, zp

    def sync(_out):
        if device.type == "cuda":
            torch.cuda.synchronize()

    return PreparedCall(
        run=run,
        sync=sync,
        to_numpy=lambda out: tuple(t.cpu().numpy() for t in out),
    )


def quantize_int4_triton(kv) -> Tuple:
    """Quantize a 2D [num_rows, num_channels] tensor to INT4 via Triton.

    Accepts a NumPy array or torch tensor; returns
    (q uint8, scale f32, zp f32) as NumPy arrays.
    Requires triton (GPU, or CPU with TRITON_INTERPRET=1).
    """
    return prepare_quantize_int4_triton(kv)()
