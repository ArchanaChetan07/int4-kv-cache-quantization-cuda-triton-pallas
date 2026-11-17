"""
vLLM integration shim for Flash Decoding with INT4 KV.

Provides a paged-KV-cache wrapper that quantizes blocks on write and
runs fused INT4 attention on decode. Designed to slot behind vLLM's
attention backend interface.

Storage format: keys are quantized to INT4 on write and stored NIBBLE-PACKED
(two values per byte) — the 4x-class compression is what actually sits in
memory, not an estimate. Blocks are unpacked to the kernels' one-value-per-byte
layout on read.

Usage (inside a vLLM fork or plugin):

    from flash_decode_int4.vllm_integration import INT4PagedKVCache

    cache = INT4PagedKVCache(num_blocks=4096, block_size=256, head_dim=128)
    cache.write_block(block_id, k_block, v_block)   # quantizes + packs K
    out = cache.decode_attention(query, block_table)  # fused INT4 attention
"""

import numpy as np
from typing import Dict, List

from .int4_pack import pack_int4, unpack_int4
from . import ops


class INT4PagedKVCache:
    """Paged KV cache with nibble-packed INT4 keys and FP16 values.

    Keys are quantized per-(block, channel) on write and stored packed
    (2 INT4 values/byte). Values stay FP16 — they contribute linearly to
    the output and are more error-sensitive.
    """

    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim

        self.k_packed: Dict[int, np.ndarray] = {}  # block_id -> uint8 [block_size, ceil(head_dim/2)]
        self.k_scale: Dict[int, np.ndarray] = {}   # block_id -> float32 [head_dim]
        self.k_zp: Dict[int, np.ndarray] = {}      # block_id -> float32 [head_dim]
        self.v: Dict[int, np.ndarray] = {}         # block_id -> float16 [block_size, head_dim]
        self.lens: Dict[int, int] = {}             # effective tokens per block

    def write_block(self, block_id: int, k_block: np.ndarray, v_block: np.ndarray) -> None:
        """Quantize, pack, and store one KV block."""
        assert k_block.shape == v_block.shape
        q, scale, zp = ops.quantize_int4(k_block.astype(np.float32))
        self.k_packed[block_id] = pack_int4(q)
        self.k_scale[block_id] = scale
        self.k_zp[block_id] = zp
        self.v[block_id] = v_block.astype(np.float16)
        self.lens[block_id] = k_block.shape[0]

    def free_block(self, block_id: int) -> None:
        for d in (self.k_packed, self.k_scale, self.k_zp, self.v, self.lens):
            d.pop(block_id, None)

    def decode_attention(
        self,
        query: np.ndarray,          # [batch, heads, head_dim]
        block_table: List[int],
    ) -> np.ndarray:
        """Fused online-softmax attention over the given blocks.

        Packed keys are unpacked to the kernels' 1-value/byte layout here;
        a packed-native kernel decode path is on the roadmap.
        """
        k_qs = [unpack_int4(self.k_packed[b], self.head_dim) for b in block_table]
        scales = [self.k_scale[b] for b in block_table]
        zps = [self.k_zp[b] for b in block_table]
        values = [self.v[b].astype(np.float32) for b in block_table]
        lens = np.array([self.lens[b] for b in block_table], dtype=np.int32)

        return ops.flash_decode(query, k_qs, scales, zps, values, lens)

    def memory_stats(self) -> dict:
        """Actual stored footprint vs FP16-equivalent — measured from the
        arrays in memory, not estimated."""
        n = len(self.k_packed)
        if n == 0:
            return {'blocks': 0, 'stored_mb': 0.0, 'fp16_equiv_mb': 0.0, 'ratio': 0.0}

        stored = sum(p.nbytes for p in self.k_packed.values()) \
            + sum(s.nbytes for s in self.k_scale.values()) \
            + sum(z.nbytes for z in self.k_zp.values()) \
            + sum(v.nbytes for v in self.v.values())

        fp16_equiv = sum(
            self.lens[b] * self.head_dim * 2 * 2  # K + V at 2 bytes each
            for b in self.k_packed
        )

        return {
            'blocks': n,
            'stored_mb': round(stored / 1048576, 3),
            'fp16_equiv_mb': round(fp16_equiv / 1048576, 3),
            'ratio': round(fp16_equiv / max(stored, 1), 2),
        }
