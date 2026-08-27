"""Pallas port of the per-channel asymmetric INT4 quantizer.

Same contract as quantize_int4_ref / the CUDA kernel / the Triton kernel:

    scale[c] = (max[c] - min[c]) / 15
    zp[c]    = -min[c] / scale[c]
    q        = clip(round_half_up((kv - min[c]) / scale[c]), 0, 15)

Structural difference from the Triton port, and the reason this file does not
look like quantize_int4_triton.py:

    Triton runs ONE program per channel and loops over rows inside that program.
    Pallas TPU fully unrolls in-kernel loops at compile time, so a 4096-row loop
    would become 4096 unrolled bodies. The row loop therefore has to become a
    GRID DIMENSION, and the running min/max has to live in an output block that
    Pallas keeps resident across consecutive grid steps (index_map returns the
    same block every step), initialized on step 0 via pl.when.

Two kernels, not one:
    pass 1  _minmax_kernel  -- grid over row blocks, reduces to (1, n_ch)
                               min/max, and computes scale/zp in a last-step
                               epilogue (pl.when) so the whole quantization
                               statistic is produced on-device.
    pass 2  _quantize_kernel -- grid over row blocks, elementwise quantize.

Rounding is half-up (floor(x + 0.5)) to match the CUDA and Triton kernels; the
NumPy reference uses banker's rounding, so parity tests allow a <=1-bin
difference at exact .5 boundaries -- the identical tolerance contract that
tests/test_triton_quantize.py already enforces.

Runs on TPU/GPU via the Mosaic compiler, or on CPU via interpret=True -- which
is how CI validates these numerics without an accelerator, exactly mirroring
the TRITON_INTERPRET=1 job that validates the Triton port.
"""

from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only where jax is absent
    HAS_JAX = False
    jax = None
    jnp = None
    pl = None


#: Rows per grid step. On TPU the second-minor dimension wants to be a multiple
#: of 8; 256 keeps the tail masking cheap and the VMEM footprint small.
DEFAULT_BLOCK_ROWS = 256

#: INT4 asymmetric domain is [0, 15] -- 16 levels, 15 intervals.
_QMAX = 15.0

#: Matches the reference and CUDA kernels: never divide by a zero range.
_RANGE_EPS = 1e-8


if HAS_JAX:

    def _minmax_kernel(kv_ref, mn_ref, mx_ref, scale_ref, zp_ref, *,
                       n_rows: int, n_grid: int):
        """Pass 1: per-channel min/max over all rows, then scale/zp.

        kv_ref    (BLOCK_R, n_ch)  one row block
        mn_ref    (1, n_ch)        resident accumulator, same block every step
        mx_ref    (1, n_ch)        resident accumulator
        scale_ref (1, n_ch)        written only on the final grid step
        zp_ref    (1, n_ch)        written only on the final grid step
        """
        i = pl.program_id(0)

        @pl.when(i == 0)
        def _init():
            mn_ref[...] = jnp.full_like(mn_ref, jnp.inf)
            mx_ref[...] = jnp.full_like(mx_ref, -jnp.inf)

        x = kv_ref[...]

        # Tail masking. The last grid step reads past n_rows when n_rows is not
        # a multiple of BLOCK_R; those lanes must not participate in the
        # reduction. Two different neutrals are needed, so the mask is applied
        # twice rather than once to the source.
        row_id = i * x.shape[0] + jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
        valid = row_id < n_rows

        blk_min = jnp.min(jnp.where(valid, x, jnp.inf), axis=0, keepdims=True)
        blk_max = jnp.max(jnp.where(valid, x, -jnp.inf), axis=0, keepdims=True)

        mn_ref[...] = jnp.minimum(mn_ref[...], blk_min)
        mx_ref[...] = jnp.maximum(mx_ref[...], blk_max)

        # Epilogue on the final step: the accumulators are complete, so the
        # quantization statistics can be finished on-device. Triton computed
        # these inline because it held min/max in registers for the whole
        # channel; here the equivalent is a guarded last-step write.
        @pl.when(i == n_grid - 1)
        def _finish():
            rng = jnp.maximum(mx_ref[...] - mn_ref[...], _RANGE_EPS)
            scale = rng / _QMAX
            scale_ref[...] = scale
            zp_ref[...] = -mn_ref[...] / scale

    def _quantize_kernel(kv_ref, mn_ref, scale_ref, q_ref):
        """Pass 2: elementwise quantize with the statistics from pass 1.

        Uses (x - mn) / scale rather than x / scale + zp so the arithmetic is
        bit-for-bit the same expression the Triton and CUDA kernels evaluate.
        """
        x = kv_ref[...]
        mn = mn_ref[...]         # (1, n_ch), broadcasts over rows
        scale = scale_ref[...]   # (1, n_ch)

        qf = (x - mn) / scale + 0.5
        # (x - mn) / scale >= 0, so truncation toward zero == floor. This is
        # the same identity the Triton kernel relies on with .to(tl.int32).
        qi = jnp.floor(qf).astype(jnp.int32)
        qi = jnp.clip(qi, 0, 15)
        q_ref[...] = qi.astype(jnp.uint8)


def quantize_int4_pallas(
    kv,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a 2D [num_rows, num_channels] tensor to INT4 via Pallas.

    Args:
        kv: NumPy array or JAX array, shape [num_rows, num_channels].
        block_rows: Rows processed per grid step.
        interpret: Run the kernel as ordinary JAX on the host instead of
            lowering to Mosaic. This is the CPU/CI path, and also the control
            used to separate a port bug from a hardware precision floor --
            see docs/TRITON_TO_PALLAS.md.

    Returns:
        (q uint8, scale float32, zp float32) as NumPy arrays. scale and zp are
        1-D of length num_channels, matching quantize_int4_ref(per_channel=True)
        and quantize_int4_triton.
    """
    if not HAS_JAX:
        raise RuntimeError("jax is not installed")

    kv_j = jnp.asarray(np.ascontiguousarray(np.asarray(kv), dtype=np.float32))
    if kv_j.ndim != 2:
        raise ValueError(f"expected [num_rows, num_channels], got {kv_j.shape}")

    n_rows, n_ch = kv_j.shape
    block_r = min(block_rows, n_rows)
    n_grid = (n_rows + block_r - 1) // block_r

    stat_shape = jax.ShapeDtypeStruct((1, n_ch), jnp.float32)

    mn, mx, scale, zp = pl.pallas_call(
        # functools.partial would work too; a closure keeps the constants
        # visibly static, which is what Pallas requires of them.
        lambda kv_ref, mn_ref, mx_ref, s_ref, z_ref: _minmax_kernel(
            kv_ref, mn_ref, mx_ref, s_ref, z_ref, n_rows=n_rows, n_grid=n_grid
        ),
        grid=(n_grid,),
        in_specs=[pl.BlockSpec((block_r, n_ch), lambda i: (i, 0))],
        # Every grid step maps to block (0, 0): Pallas keeps it resident, so
        # these behave as accumulators across the sequential grid.
        out_specs=[
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
        ],
        out_shape=[stat_shape, stat_shape, stat_shape, stat_shape],
        interpret=interpret,
    )(kv_j)

    q = pl.pallas_call(
        _quantize_kernel,
        grid=(n_grid,),
        in_specs=[
            pl.BlockSpec((block_r, n_ch), lambda i: (i, 0)),
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
            pl.BlockSpec((1, n_ch), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_r, n_ch), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n_rows, n_ch), jnp.uint8),
        interpret=interpret,
    )(kv_j, mn, scale)

    return (
        np.asarray(q),
        np.asarray(scale).reshape(n_ch),
        np.asarray(zp).reshape(n_ch),
    )


def dequantize_int4_pallas(q, scale, zp) -> np.ndarray:
    """Dequantize with the same formula the reference uses: (q - zp) * scale.

    Provided so round-trip tests do not have to reach into the NumPy reference
    for the inverse of a Pallas-produced quantization.
    """
    q = np.asarray(q).astype(np.float32)
    scale = np.asarray(scale).astype(np.float32)
    zp = np.asarray(zp).astype(np.float32)
    return q * scale - zp * scale
