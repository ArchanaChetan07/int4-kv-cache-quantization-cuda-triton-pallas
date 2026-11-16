"""Input validation shared by every attention backend.

Lives in its own module so the dispatch layer (ops.py) and the kernels that can
be called directly (flash_decode_pallas.py) enforce one contract without the
kernel module importing the dispatch layer.
"""

import numpy as np


def normalize_attention_inputs(query, k_q_blocks, k_scales, k_zps, v_blocks,
                               block_lens):
    """Validate paged-KV attention inputs once, identically for every backend.

    The backends used to disagree on malformed input: the NumPy reference
    silently truncated an over-long block length (slicing clamps), the Pallas
    kernel masked it, and the CUDA kernel read past the page. One validation
    path means one contract.

    Returns (query ndarray, block_lens int32 ndarray).
    """
    query = np.asarray(query)
    if query.ndim != 3:
        raise ValueError(
            f"query must be [batch, heads, head_dim], got shape {query.shape}")
    head_dim = query.shape[2]

    n = len(k_q_blocks)
    if not (len(k_scales) == len(k_zps) == len(v_blocks) == n):
        raise ValueError(
            "k_q_blocks, k_scales, k_zps and v_blocks must have the same "
            f"length, got {n}, {len(k_scales)}, {len(k_zps)}, {len(v_blocks)}")

    rows = np.empty(n, dtype=np.int64)
    for i in range(n):
        k_shape, v_shape = np.shape(k_q_blocks[i]), np.shape(v_blocks[i])
        if len(k_shape) != 2 or k_shape[1] != head_dim:
            raise ValueError(
                f"key page {i} must be [rows, {head_dim}], got shape {k_shape}")
        if v_shape != k_shape:
            raise ValueError(
                f"value page {i} has shape {v_shape}, expected {k_shape} to "
                "match its key page")
        if np.shape(k_scales[i]) != (head_dim,) or np.shape(k_zps[i]) != (head_dim,):
            raise ValueError(
                f"scale/zero-point for page {i} must have shape ({head_dim},), "
                f"got {np.shape(k_scales[i])} and {np.shape(k_zps[i])}")
        rows[i] = k_shape[0]

    if block_lens is None:
        lens = rows.copy()
    else:
        lens = np.asarray(block_lens).astype(np.int64).reshape(-1)
        if lens.shape[0] != n:
            raise ValueError(
                f"block_lens has {lens.shape[0]} entries for {n} pages")
        bad = np.nonzero((lens < 0) | (lens > rows))[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"block_lens[{i}] = {int(lens[i])} is outside [0, {int(rows[i])}], "
                "the number of rows actually present in that page")

    return query, lens.astype(np.int32)
