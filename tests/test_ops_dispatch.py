"""
Dispatch-layer tests for Flash Decoding INT4.

Verifies ops routes correctly; CUDA parity checks are skipped when
no compiled extension is present.
"""

import numpy as np
import pytest

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref
from src import ops


class TestDispatchReference:
    """Ops layer must produce reference results when reference is forced."""

    def test_quantize_dispatch(self):
        np.random.seed(1)
        kv = np.random.randn(256, 64).astype(np.float32)

        q, scale, zp = ops.quantize_int4(kv, use_cuda=False)
        q_ref, scale_ref, zp_ref = quantize_int4_ref(kv, per_channel=True)

        np.testing.assert_array_equal(q, q_ref)
        np.testing.assert_allclose(scale, scale_ref, rtol=1e-6)
        np.testing.assert_allclose(zp, zp_ref, rtol=1e-6)

    def test_flash_decode_dispatch(self):
        np.random.seed(2)
        batch, heads, dim = 2, 4, 32
        block_size, num_blocks = 128, 3

        query = np.random.randn(batch, heads, dim).astype(np.float32)
        k_qs, scales, zps, values = [], [], [], []
        for _ in range(num_blocks):
            k = np.random.randn(block_size, dim).astype(np.float32)
            v = np.random.randn(block_size, dim).astype(np.float32)
            q, s, z = quantize_int4_ref(k, per_channel=True)
            k_qs.append(q); scales.append(s); zps.append(z); values.append(v)

        out = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=False)

        key_scale_zp = list(zip(k_qs, scales, zps))
        lens = np.full(num_blocks, block_size, dtype=np.int32)
        out_ref, _ = online_softmax_ref(query, key_scale_zp, values, lens)

        np.testing.assert_allclose(out, out_ref, rtol=1e-5)

    def test_backend_status(self):
        status = ops.backend_status()
        assert status['active_backend'] in ('cuda', 'reference')


@pytest.mark.skipif(not ops.HAS_CUDA, reason="compiled CUDA extension not available")
class TestCUDAParity:
    """CUDA kernels must match reference (when compiled)."""

    def test_quantize_parity(self):
        np.random.seed(3)
        kv = np.random.randn(512, 128).astype(np.float32)

        q_cuda, scale_cuda, zp_cuda = ops.quantize_int4(kv, use_cuda=True)
        q_ref, scale_ref, zp_ref = quantize_int4_ref(kv, per_channel=True)

        # Rounding at bin edges may differ by 1; scales must match tightly
        np.testing.assert_allclose(scale_cuda, scale_ref, rtol=1e-4)
        assert np.mean(np.abs(q_cuda.astype(int) - q_ref.astype(int))) < 0.01

    def test_flash_decode_parity(self):
        np.random.seed(4)
        batch, heads, dim = 2, 4, 64
        block_size, num_blocks = 256, 4

        query = np.random.randn(batch, heads, dim).astype(np.float32)
        k_qs, scales, zps, values = [], [], [], []
        for _ in range(num_blocks):
            k = np.random.randn(block_size, dim).astype(np.float32)
            v = np.random.randn(block_size, dim).astype(np.float32)
            q, s, z = quantize_int4_ref(k, per_channel=True)
            k_qs.append(q); scales.append(s); zps.append(z); values.append(v)

        out_cuda = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=True)
        out_ref = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=False)

        np.testing.assert_allclose(out_cuda, out_ref, rtol=1e-3, atol=1e-4)


class TestBackendSelectionIsHonest:
    """A named backend is delivered or refused -- never silently substituted.

    Regression tests for a real defect: ops.flash_decode(backend="triton")
    returned NumPy reference results, byte-identical to backend="reference".
    Benchmarking that would have measured NumPy and reported it as Triton --
    precisely the failure the harness anti-cheat assertions exist to prevent,
    sitting one layer below them.
    """

    @staticmethod
    def _attention_args():
        rng = np.random.RandomState(0)
        query = rng.randn(1, 2, 16).astype(np.float32)
        k = rng.randn(32, 16).astype(np.float32)
        v = rng.randn(32, 16).astype(np.float32)
        q, s, z = quantize_int4_ref(k, per_channel=True)
        return (query, [q], [s], [z], [v], np.array([32], dtype=np.int32))

    def test_attention_rejects_triton_rather_than_faking_it(self):
        with pytest.raises(ValueError, match="no Triton attention kernel"):
            ops.flash_decode(*self._attention_args(), backend="triton")

    def test_attention_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown backend"):
            ops.flash_decode(*self._attention_args(), backend="nonsense")

    def test_quantize_rejects_unknown_backend(self):
        kv = np.random.RandomState(1).randn(32, 8).astype(np.float32)
        with pytest.raises(ValueError, match="unknown backend"):
            ops.quantize_int4(kv, backend="nonsense")

    def test_valid_backends_still_dispatch(self):
        """The guard must not break the paths that do exist."""
        kv = np.random.RandomState(2).randn(64, 16).astype(np.float32)
        q_ref, s_ref, _ = ops.quantize_int4(kv, backend="reference")
        assert q_ref.shape == kv.shape

        out_ref = ops.flash_decode(*self._attention_args(), backend="reference")
        assert np.isfinite(out_ref).all()

        # auto-detect (backend=None) must remain unaffected
        assert np.isfinite(ops.flash_decode(*self._attention_args())).all()

    def test_declared_backend_sets_match_reality(self):
        """Attention genuinely has no Triton kernel; quantize genuinely does."""
        assert "triton" in ops.QUANTIZE_BACKENDS
        assert "triton" not in ops.ATTENTION_BACKENDS
        for name in ops.ATTENTION_BACKENDS:
            assert name in ops.QUANTIZE_BACKENDS or name == "reference"


class TestQuantizerInputValidation:
    """Degenerate shapes must raise, not divide by zero."""

    def test_empty_array_rejected(self):
        jax = pytest.importorskip("jax", reason="jax not installed")
        from src.quantize_int4_pallas import quantize_int4_pallas
        with pytest.raises(ValueError, match="empty array"):
            quantize_int4_pallas(np.zeros((0, 8), np.float32), interpret=True)

    def test_zero_block_rows_rejected(self):
        jax = pytest.importorskip("jax", reason="jax not installed")
        from src.quantize_int4_pallas import quantize_int4_pallas
        kv = np.random.RandomState(3).randn(16, 4).astype(np.float32)
        with pytest.raises(ValueError, match="block_rows"):
            quantize_int4_pallas(kv, block_rows=0, interpret=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
