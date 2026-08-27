"""Pallas port of the fused INT4 flash-decoding attention kernel.

Same algorithm as csrc/flash_decode_int4.cu and flash_decode_ref.online_softmax_ref:
online softmax (Welford-style) over paged INT4 KV blocks, dequantizing keys
inside the kernel so the FP32 key matrix is never materialized.

Structural difference from the CUDA kernel, and the reason this file does not
look like flash_decode_int4.cu:

    The CUDA kernel's parallel decomposition is warp-level. Warps stride over
    sequence positions; each warp carries its own (m, l) online-softmax state
    plus an output accumulator distributed across lane registers; per-position
    dot products are reduced with __shfl_down_sync; warps merge at the end with
    a log-sum-exp combine.

    None of that vocabulary exists here. There are no warps, no lanes, no
    shuffles, and no manual shared-memory staging. The equivalent structure is:

        grid = (batch, heads, num_blocks)   -- blocks innermost

    with m/l/o as OUTPUT blocks whose index_map ignores the block axis, so
    Pallas keeps them resident across the whole sequence for a given (b, h).
    The grid runs sequentially in lexicographic order, which is what makes the
    accumulation safe; on CUDA the same guarantee had to be bought with
    __syncthreads and an explicit cross-warp merge.

Empty-page handling mirrors the CUDA kernel's explicit -inf guard. The naive
formulation computes exp(m_prev - m_new); when a page is empty AND nothing has
been seen yet, both are -inf and the subtraction is NaN. The CUDA kernel gates
on `s_m[w] != -CUDART_INF_F`; here the same defect is avoided by subtracting a
finite surrogate for m_new whenever m_new is -inf, which drives both the
correction factor and every probability to exactly zero.

Runs on TPU/GPU via Mosaic, or on CPU via interpret=True.
"""

from typing import List, Optional, Tuple

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


#: Reference normalizes by max(l, 1e-10); matched exactly so an all-empty
#: sequence produces zeros rather than NaN in both implementations.
_L_FLOOR = 1e-10


def _compiler_params(platform: str):
    """Declare which grid axes may run in parallel and which must not.

    THIS IS LOAD-BEARING FOR CORRECTNESS ON GPU, and it is invisible in
    interpret mode.

    The grid is (batch, heads, blocks). The block axis carries the online-
    softmax accumulators in resident output blocks, so it MUST be executed
    sequentially. On TPU that is free: the grid is always processed in
    lexicographic order. On GPU it is not -- Mosaic GPU partitions 'parallel'
    dimensions across CUDA thread blocks, which would turn the accumulator
    into a race.

    Returns None when the platform is unknown or the API is unavailable, in
    which case pallas_call is invoked without compiler params (correct on TPU,
    and the only option in interpret mode).
    """
    # (batch, heads) are genuinely independent; the block axis is not.
    try:
        if platform == "tpu":
            from jax.experimental.pallas import tpu as pltpu
            cls = getattr(pltpu, "CompilerParams", None) or getattr(
                pltpu, "TPUCompilerParams", None)
            # TPU vocabulary is parallel / arbitrary.
            return cls(dimension_semantics=("parallel", "parallel", "arbitrary"))
        if platform in ("cuda", "rocm", "gpu"):
            from jax.experimental.pallas import mosaic_gpu as plgpu
            cls = getattr(plgpu, "CompilerParams", None) or getattr(
                plgpu, "GPUCompilerParams", None)
            # GPU vocabulary is parallel / sequential.
            return cls(dimension_semantics=("parallel", "parallel", "sequential"))
    except Exception:
        return None
    return None


if HAS_JAX:

    def _flash_decode_kernel(
        q_ref, k_ref, s_ref, z_ref, v_ref, len_ref,   # inputs
        o_ref, m_ref, l_ref,                          # outputs (resident)
        *, n_blocks: int, use_dot: bool,
    ):
        """One (batch, head, kv-block) grid step.

        q_ref   (1, 1, D)  query for this (b, h)
        k_ref   (1, S, D)  INT4 keys for this page, one value per byte
        s_ref   (1, D)     per-channel scales for this page
        z_ref   (1, D)     per-channel zero-points for this page
        v_ref   (1, S, D)  FP32 values for this page
        len_ref (1,)       valid row count for this page
        o_ref   (1, 1, D)  running output accumulator, resident across pages
        m_ref   (1, 1, 1)  running max
        l_ref   (1, 1, 1)  running sum of exponentials
        """
        i = pl.program_id(2)

        @pl.when(i == 0)
        def _init():
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

        q = q_ref[...]                       # (1, 1, D)
        kq = k_ref[...].astype(jnp.float32)  # (1, S, D)
        scale = s_ref[...][:, None, :]       # (1, 1, D)
        zp = z_ref[...][:, None, :]          # (1, 1, D)
        v = v_ref[...]                       # (1, S, D)

        # Dequantize in-kernel: k = q * scale - zp * scale. Identical to the
        # reference formula and to the CUDA kernel's single-FMA form, where zp
        # was pre-multiplied by scale during shared-memory staging.
        k = kq * scale - zp * scale          # (1, S, D)

        if use_dot:
            # Routes through the MXU on TPU. Faster, but float32 inputs are
            # executed as multiple bfloat16 passes -- see the precision note in
            # docs/TRITON_TO_PALLAS.md before enabling this on TPU.
            logits = jax.lax.dot_general(
                q, k, (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )[:, 0, :]                        # (1, S)
        else:
            # Elementwise multiply + reduce keeps genuine float32 arithmetic on
            # every backend, at the cost of not using the matrix unit.
            logits = jnp.sum(q * k, axis=-1)  # (1, S)

        # Mask padded rows of this page. block_lens == 0 means an empty page,
        # which must contribute nothing at all.
        blen = len_ref[0]
        pos = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        valid = pos < blen
        logits = jnp.where(valid, logits, -jnp.inf)

        m_prev = m_ref[..., 0]                     # (1, 1)
        l_prev = l_ref[..., 0]
        o_prev = o_ref[...]                        # (1, 1, D)

        m_blk = jnp.max(logits, axis=-1, keepdims=True)    # (1, 1)
        m_new = jnp.maximum(m_prev, m_blk)

        # The NaN guard. When m_new is -inf (nothing live yet, empty page),
        # (-inf) - (-inf) is NaN. Subtracting a finite surrogate instead sends
        # every exp() to exactly 0, which is the arithmetically correct answer:
        # no mass has been accumulated.
        m_safe = jnp.where(jnp.isneginf(m_new), 0.0, m_new)

        corr = jnp.exp(m_prev - m_safe)                      # (1, 1)
        p = jnp.exp(logits - m_safe)                         # (1, S)

        l_new = l_prev * corr + jnp.sum(p, axis=-1, keepdims=True)
        o_new = o_prev * corr[..., None] + jnp.sum(p[..., None] * v, axis=1, keepdims=True)

        m_ref[...] = m_new[..., None]
        l_ref[...] = l_new[..., None]
        o_ref[...] = o_new

        # Final normalization in a last-step epilogue, so the kernel emits the
        # finished attention output rather than an unnormalized accumulator.
        @pl.when(i == n_blocks - 1)
        def _normalize():
            denom = jnp.maximum(l_ref[..., 0], _L_FLOOR)     # (1, 1)
            o_ref[...] = o_ref[...] / denom[..., None]


def flash_decode_pallas(
    query: np.ndarray,
    k_q_blocks: List[np.ndarray],
    k_scales: List[np.ndarray],
    k_zps: List[np.ndarray],
    v_blocks: List[np.ndarray],
    block_lens: Optional[np.ndarray] = None,
    interpret: bool = False,
    use_dot: bool = False,
) -> np.ndarray:
    """Fused online-softmax attention over INT4 paged KV, in Pallas.

    Argument shapes match ops.flash_decode / the CUDA path exactly so this can
    be dropped in as a third backend.

    Args:
        query: [batch, heads, head_dim] float32.
        k_q_blocks: list of [block_size, head_dim] uint8 INT4 key pages.
        k_scales: list of [head_dim] float32 per-channel scales.
        k_zps: list of [head_dim] float32 per-channel zero-points.
        v_blocks: list of [block_size, head_dim] float32 value pages.
        block_lens: [num_blocks] int32 valid row counts. Defaults to full pages.
        interpret: Run as ordinary JAX instead of lowering to Mosaic.
        use_dot: Use dot_general (MXU on TPU) instead of multiply-and-reduce.

    Returns:
        [batch, heads, head_dim] float32 attention output.
    """
    if not HAS_JAX:
        raise RuntimeError("jax is not installed")

    query = np.asarray(query, dtype=np.float32)
    if query.ndim != 3:
        raise ValueError(f"query must be [batch, heads, head_dim], got {query.shape}")

    batch, heads, head_dim = query.shape
    n_blocks = len(k_q_blocks)
    if n_blocks == 0:
        raise ValueError("at least one KV page is required")

    block_size = max(int(b.shape[0]) for b in k_q_blocks)

    if block_lens is None:
        block_lens = np.array([b.shape[0] for b in k_q_blocks], dtype=np.int32)
    block_lens = np.asarray(block_lens, dtype=np.int32)

    # Pad ragged pages into the dense [num_blocks, block_size, head_dim] layout
    # the CUDA path also uses. Real paged serving hands the kernel this layout
    # directly; the padding here exists only because the test fixtures build
    # Python lists.
    k_arr = np.zeros((n_blocks, block_size, head_dim), dtype=np.uint8)
    v_arr = np.zeros((n_blocks, block_size, head_dim), dtype=np.float32)
    s_arr = np.zeros((n_blocks, head_dim), dtype=np.float32)
    z_arr = np.zeros((n_blocks, head_dim), dtype=np.float32)
    for i in range(n_blocks):
        n = k_q_blocks[i].shape[0]
        k_arr[i, :n] = k_q_blocks[i]
        v_arr[i, :n] = v_blocks[i]
        s_arr[i] = k_scales[i]
        z_arr[i] = k_zps[i]

    # Grid-order semantics must be declared explicitly for any compiled
    # backend; see _compiler_params. Omitted in interpret mode, which executes
    # the grid sequentially by construction.
    platform = "" if interpret else jax.devices()[0].platform
    cparams = None if interpret else _compiler_params(platform)
    extra = {"compiler_params": cparams} if cparams is not None else {}

    out = pl.pallas_call(
        lambda *refs: _flash_decode_kernel(
            *refs, n_blocks=n_blocks, use_dot=use_dot
        ),
        grid=(batch, heads, n_blocks),
        **extra,
        in_specs=[
            pl.BlockSpec((1, 1, head_dim), lambda b, h, i: (b, h, 0)),      # query
            pl.BlockSpec((1, block_size, head_dim), lambda b, h, i: (i, 0, 0)),  # k_q
            pl.BlockSpec((1, head_dim), lambda b, h, i: (i, 0)),            # scale
            pl.BlockSpec((1, head_dim), lambda b, h, i: (i, 0)),            # zp
            pl.BlockSpec((1, block_size, head_dim), lambda b, h, i: (i, 0, 0)),  # value
            pl.BlockSpec((1,), lambda b, h, i: (i,)),                       # block_lens
        ],
        # The block axis is absent from every output index_map: that is what
        # makes these accumulators rather than per-step writes.
        out_specs=[
            pl.BlockSpec((1, 1, head_dim), lambda b, h, i: (b, h, 0)),
            pl.BlockSpec((1, 1, 1), lambda b, h, i: (b, h, 0)),
            pl.BlockSpec((1, 1, 1), lambda b, h, i: (b, h, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch, heads, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, 1), jnp.float32),
            jax.ShapeDtypeStruct((batch, heads, 1), jnp.float32),
        ],
        interpret=interpret,
    )(
        jnp.asarray(query),
        jnp.asarray(k_arr),
        jnp.asarray(s_arr),
        jnp.asarray(z_arr),
        jnp.asarray(v_arr),
        jnp.asarray(block_lens),
    )

    return np.asarray(out[0])
