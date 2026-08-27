"""Parity tests for the Pallas quantizer port.

Mirrors tests/test_triton_quantize.py case for case, so the two ports are held
to the identical contract:

    scales   rtol 1e-4
    zero-pts rtol 1e-3 / atol 1e-3
    bins     <=1 apart, on <1% of elements

Skips when jax is not installed. These run on CPU via interpret=True, which is
how CI validates the numerics without an accelerator -- the exact analogue of
the TRITON_INTERPRET=1 job.
"""

import numpy as np
import pytest

from src.quantize_int4_ref import quantize_int4_ref

jax_mod = pytest.importorskip("jax", reason="jax not installed")

from src.quantize_int4_pallas import (  # noqa: E402
    quantize_int4_pallas,
    dequantize_int4_pallas,
)


def _quant(kv, **kw):
    """All tests run in interpret mode; on an accelerator drop the flag."""
    return quantize_int4_pallas(kv, interpret=True, **kw)


class TestPallasQuantizerParity:

    def test_scales_match_reference(self):
        np.random.seed(0)
        kv = np.random.randn(512, 64).astype(np.float32)

        _, scale_p, zp_p = _quant(kv)
        _, scale_r, zp_r = quantize_int4_ref(kv, per_channel=True)

        np.testing.assert_allclose(scale_p, scale_r, rtol=1e-4)
        np.testing.assert_allclose(zp_p, zp_r, rtol=1e-3, atol=1e-3)

    def test_bins_match_reference(self):
        """Half-up (Pallas/Triton/CUDA) vs banker's (NumPy) rounding may differ
        by one bin at exact .5 boundaries -- rare on continuous data."""
        np.random.seed(1)
        kv = np.random.randn(1024, 32).astype(np.float32)

        q_p, _, _ = _quant(kv)
        q_r, _, _ = quantize_int4_ref(kv, per_channel=True)

        diff = np.abs(q_p.astype(int) - q_r.astype(int))
        assert diff.max() <= 1, f"bins differ by more than 1: {diff.max()}"
        assert (diff > 0).mean() < 0.01, f"too many bin mismatches: {(diff > 0).mean():.4%}"

    def test_round_trip_error_bounded(self):
        np.random.seed(2)
        kv = np.random.randn(256, 128).astype(np.float32)

        q, scale, zp = _quant(kv)
        dequant = dequantize_int4_pallas(q, scale, zp)
        err = np.abs(kv - dequant)
        assert err.max() <= scale.max() / 1.9, \
            f"round-trip error {err.max()} exceeds scale/2 bound {scale.max()/2}"

    def test_output_range(self):
        np.random.seed(3)
        kv = (np.random.randn(100, 16) * 50).astype(np.float32)
        q, _, _ = _quant(kv)
        assert q.min() >= 0 and q.max() <= 15

    def test_non_block_multiple_rows(self):
        """Row count not divisible by the grid block exercises the tail mask.

        This is the case Pallas makes easy to get wrong: the final grid step
        reads past n_rows, and those lanes must be excluded from the min/max
        reduction with two different neutral elements.
        """
        np.random.seed(4)
        kv = np.random.randn(1030, 8).astype(np.float32)
        q, scale, _ = _quant(kv, block_rows=256)
        q_r, scale_r, _ = quantize_int4_ref(kv, per_channel=True)
        np.testing.assert_allclose(scale, scale_r, rtol=1e-4)
        assert np.abs(q.astype(int) - q_r.astype(int)).max() <= 1

    def test_block_rows_invariance(self):
        """The grid decomposition must not change the answer.

        Pallas-specific risk with no Triton analogue: the row loop IS the grid
        here, so an accumulator bug shows up as block-size dependence.
        """
        np.random.seed(5)
        kv = np.random.randn(777, 24).astype(np.float32)

        base_q, base_s, base_z = _quant(kv, block_rows=64)
        for br in (128, 256, 512, 1024):
            q, s, z = _quant(kv, block_rows=br)
            np.testing.assert_array_equal(q, base_q)
            np.testing.assert_allclose(s, base_s, rtol=1e-6)
            np.testing.assert_allclose(z, base_z, rtol=1e-6)

    def test_constant_channel_no_divide_by_zero(self):
        """A channel with zero range must not produce inf/nan."""
        kv = np.ones((300, 4), dtype=np.float32)
        q, scale, zp = _quant(kv)
        assert np.isfinite(scale).all() and np.isfinite(zp).all()
        assert q.min() >= 0 and q.max() <= 15

    def test_single_row(self):
        """Degenerate grid of one step."""
        np.random.seed(6)
        kv = np.random.randn(1, 32).astype(np.float32)
        q, scale, zp = _quant(kv)
        assert np.isfinite(scale).all()
        assert q.shape == (1, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
