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


def _paged_problem(batch=2, heads=3, dim=32, n_blocks=3, rows=64, seed=0):
    """A small, valid INT4 paged-KV attention problem."""
    rng = np.random.RandomState(seed)
    query = (rng.randn(batch, heads, dim) / np.sqrt(dim)).astype(np.float32)
    k_q, k_s, k_z, vals = [], [], [], []
    for _ in range(n_blocks):
        k = rng.randn(rows, dim).astype(np.float32)
        v = rng.randn(rows, dim).astype(np.float32)
        q, s_, z = quantize_int4_ref(k, per_channel=True)
        k_q.append(q); k_s.append(s_); k_z.append(z); vals.append(v)
    return query, k_q, k_s, k_z, vals


def _available_attention_backends():
    out = ["reference"]
    if ops.HAS_CUDA:
        out.append("cuda")
    if ops.HAS_JAX:
        out.append("pallas")
    return out


@pytest.mark.parametrize("backend", _available_attention_backends())
class TestAttentionInputContract:
    """Malformed input raises the same error on every backend.

    Before this contract existed the backends disagreed: the NumPy reference
    silently truncated an over-long block length, the Pallas kernel masked it,
    and the CUDA kernel read past the end of the page.
    """

    def test_block_len_longer_than_page_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem(rows=64)
        with pytest.raises(ValueError, match="outside"):
            ops.flash_decode(query, k_q, k_s, k_z, vals,
                             np.array([64, 65, 64]), backend=backend)

    def test_negative_block_len_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        with pytest.raises(ValueError, match="outside"):
            ops.flash_decode(query, k_q, k_s, k_z, vals,
                             np.array([64, -1, 64]), backend=backend)

    def test_wrong_number_of_block_lens_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        with pytest.raises(ValueError, match="entries"):
            ops.flash_decode(query, k_q, k_s, k_z, vals,
                             np.array([64, 64]), backend=backend)

    def test_mismatched_page_lists_are_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        with pytest.raises(ValueError, match="same length"):
            ops.flash_decode(query, k_q, k_s, k_z, vals[:-1], backend=backend)

    def test_value_page_shape_mismatch_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        vals[1] = vals[1][:-3]
        with pytest.raises(ValueError, match="value page 1"):
            ops.flash_decode(query, k_q, k_s, k_z, vals, backend=backend)

    def test_key_page_wrong_head_dim_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem(dim=32)
        with pytest.raises(ValueError, match="key page"):
            ops.flash_decode(query[:, :, :16], k_q, k_s, k_z, vals, backend=backend)

    def test_non_3d_query_is_rejected(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        with pytest.raises(ValueError, match="query must be"):
            ops.flash_decode(query[0], k_q, k_s, k_z, vals, backend=backend)

    def test_empty_cache_returns_exact_zeros(self, backend):
        """Zero pages is the all-pages-empty case, on every backend."""
        query, _, _, _, _ = _paged_problem()
        out = ops.flash_decode(query, [], [], [], [], backend=backend)
        assert out.shape == query.shape
        np.testing.assert_array_equal(out, np.zeros_like(query))

    def test_block_lens_accepts_a_plain_list(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        as_list = ops.flash_decode(query, k_q, k_s, k_z, vals, [64, 10, 0],
                                   backend=backend)
        as_array = ops.flash_decode(query, k_q, k_s, k_z, vals,
                                    np.array([64, 10, 0], dtype=np.int64),
                                    backend=backend)
        np.testing.assert_array_equal(as_list, as_array)

    def test_valid_ragged_input_matches_reference(self, backend):
        query, k_q, k_s, k_z, vals = _paged_problem()
        lens = np.array([64, 0, 17], dtype=np.int32)
        got = ops.flash_decode(query, k_q, k_s, k_z, vals, lens, backend=backend)
        ref = ops.flash_decode(query, k_q, k_s, k_z, vals, lens, backend="reference")
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() < 1e-5


@pytest.mark.skipif(not ops.HAS_CUDA, reason="compiled CUDA extension not available")
class TestCUDAKernelBoundaries:
    """Shapes at and beyond the limits of the CUDA attention kernel.

    The kernel holds its output accumulator in 8 lane-register slots per thread,
    so it supports head_dim <= 256. A wider head used to index past that array:
    an out-of-bounds write in device code, with no check at any layer.
    """

    @pytest.mark.parametrize("dim", [1, 2, 31, 32, 33, 64, 127, 128, 129, 255, 256])
    def test_head_dim_boundaries_match_reference(self, dim):
        query, k_q, k_s, k_z, vals = _paged_problem(dim=dim, rows=96, seed=dim)
        lens = np.array([96, 40, 96], dtype=np.int32)
        got = ops.flash_decode(query, k_q, k_s, k_z, vals, lens, backend="cuda")
        ref = ops.flash_decode(query, k_q, k_s, k_z, vals, lens, backend="reference")
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() < 1e-4, f"head_dim {dim}"

    def test_head_dim_beyond_kernel_limit_is_rejected_not_overflowed(self):
        query, k_q, k_s, k_z, vals = _paged_problem(dim=320, rows=32)
        with pytest.raises(ValueError, match="head_dim"):
            ops.flash_decode(query, k_q, k_s, k_z, vals, backend="cuda")

    def test_auto_detect_falls_back_for_wide_heads(self):
        """backend=None may choose another implementation; it must not overflow."""
        query, k_q, k_s, k_z, vals = _paged_problem(dim=320, rows=32)
        got = ops.flash_decode(query, k_q, k_s, k_z, vals)
        ref = ops.flash_decode(query, k_q, k_s, k_z, vals, backend="reference")
        np.testing.assert_array_equal(got, ref)

    @pytest.mark.parametrize("rows", [1, 3, 4, 5, 127, 128, 129])
    def test_page_sizes_around_the_warp_count(self, rows):
        """Four warps stride over positions; sizes near multiples of 4 and 128
        exercise warps that see zero, one or uneven numbers of positions."""
        query, k_q, k_s, k_z, vals = _paged_problem(dim=64, rows=rows, seed=rows)
        got = ops.flash_decode(query, k_q, k_s, k_z, vals, backend="cuda")
        ref = ops.flash_decode(query, k_q, k_s, k_z, vals, backend="reference")
        assert np.abs(got - ref).max() < 1e-4, f"rows {rows}"

    def test_binding_rejects_mismatched_shapes(self):
        """Direct misuse of the extension raises instead of reading out of bounds."""
        import torch
        q = torch.zeros(1, 1, 64, device="cuda")
        k = torch.zeros(2, 16, 64, dtype=torch.uint8, device="cuda")
        v = torch.zeros(2, 16, 64, device="cuda")
        sc = torch.ones(2, 64, device="cuda")
        zp = torch.zeros(2, 64, device="cuda")
        lens = torch.full((2,), 16, dtype=torch.int32, device="cuda")

        ops._C.flash_decode_int4(q, k, sc, zp, v, lens)            # well-formed

        with pytest.raises(RuntimeError, match="k_scale"):
            ops._C.flash_decode_int4(q, k, sc[:1], zp, v, lens)
        with pytest.raises(RuntimeError, match="value"):
            ops._C.flash_decode_int4(q, k, sc, zp, v[:, :8], lens)
        with pytest.raises(RuntimeError, match="block_lens"):
            ops._C.flash_decode_int4(q, k, sc, zp, v, lens[:1])
        with pytest.raises(RuntimeError, match="int32"):
            ops._C.flash_decode_int4(q, k, sc, zp, v, lens.long())
        with pytest.raises(RuntimeError, match="head_dim"):
            wide = torch.zeros(1, 1, 320, device="cuda")
            ops._C.flash_decode_int4(
                wide, torch.zeros(1, 4, 320, dtype=torch.uint8, device="cuda"),
                torch.ones(1, 320, device="cuda"), torch.zeros(1, 320, device="cuda"),
                torch.zeros(1, 4, 320, device="cuda"),
                torch.full((1,), 4, dtype=torch.int32, device="cuda"))

    def test_binding_rejects_bad_quantizer_input(self):
        import torch
        with pytest.raises(RuntimeError, match="float32"):
            ops._C.quantize_int4(torch.zeros(4, 4, dtype=torch.float64, device="cuda"))
        with pytest.raises(RuntimeError, match="non-empty"):
            ops._C.quantize_int4(torch.zeros(0, 4, device="cuda"))
        with pytest.raises(RuntimeError, match="2D"):
            ops._C.quantize_int4(torch.zeros(4, device="cuda"))

    @pytest.mark.parametrize("shape", [(1, 1), (1, 64), (255, 3), (256, 3), (257, 3),
                                       (1030, 8), (4096, 128)])
    def test_quantizer_shapes_match_reference(self, shape):
        """Row counts around the 256-thread block; single-row and single-channel."""
        kv = np.random.RandomState(shape[0]).randn(*shape).astype(np.float32)
        q, scale, zp = ops.quantize_int4(kv, backend="cuda")
        q_r, scale_r, _ = quantize_int4_ref(kv, per_channel=True)
        np.testing.assert_allclose(scale, scale_r, rtol=1e-4)
        assert np.abs(q.astype(int) - q_r.astype(int)).max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
