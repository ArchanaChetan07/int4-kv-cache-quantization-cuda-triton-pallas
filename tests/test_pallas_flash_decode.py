"""Parity tests for the Pallas fused flash-decoding attention port.

On the tolerance used here
--------------------------
The obvious gate -- "MAE against online_softmax_ref below some fixed epsilon" --
does not survive contact with float32. At head_dim 128 over ~1k positions the
NumPy reference's OWN error against a float64 evaluation is ~9e-7, which is the
expected sqrt(D) * eps * |o| accumulation floor. So a fixed 1e-8 gate is not
attainable at that shape by any correct implementation, and a loose fixed gate
would pass a genuinely wrong kernel.

These tests therefore use a float64 evaluation as the arbiter, and gate on the
float32 accumulation floor rather than on a fixed epsilon or on a head-to-head
comparison with the reference. See TestPallasAttentionAccuracy for why the
head-to-head form was tried first and had to be abandoned: it encodes an
assumption about XLA's reduction order that does not hold across versions.

Agreement with the float32 reference is still checked, but at the 1e-5 level
the repository's own multi-block consistency tests already use.
"""

import numpy as np
import pytest

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref

jax_mod = pytest.importorskip("jax", reason="jax not installed")

from src.flash_decode_pallas import flash_decode_pallas  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_case(batch=2, heads=4, dim=64, n_blocks=4, block_size=128,
              seed=0, lens=None):
    """Build a quantized paged-KV attention problem."""
    rng = np.random.RandomState(seed)
    query = rng.randn(batch, heads, dim).astype(np.float32)
    k_q, k_s, k_z, vals = [], [], [], []
    for _ in range(n_blocks):
        k = rng.randn(block_size, dim).astype(np.float32)
        v = rng.randn(block_size, dim).astype(np.float32)
        q_, s_, z_ = quantize_int4_ref(k, per_channel=True)
        k_q.append(q_); k_s.append(s_); k_z.append(z_); vals.append(v)
    if lens is None:
        lens = np.full(n_blocks, block_size, dtype=np.int32)
    return query, k_q, k_s, k_z, vals, np.asarray(lens, dtype=np.int32)


def f64_attention(query, k_q, k_s, k_z, vals, lens):
    """Float64 evaluation of the same mathematics -- the arbiter.

    Deliberately NOT an online softmax: it concatenates every live page and
    does one full-precision softmax, so it shares no accumulation strategy with
    either implementation under test.
    """
    batch, heads, dim = query.shape
    q64 = query.astype(np.float64)
    logits, values = [], []
    for i in range(len(k_q)):
        n = int(lens[i])
        if n == 0:
            continue
        s64 = k_s[i].astype(np.float64)
        z64 = k_z[i].astype(np.float64)
        k64 = k_q[i][:n].astype(np.float64) * s64 - z64 * s64
        logits.append(q64 @ k64.T)
        values.append(vals[i][:n].astype(np.float64))
    if not logits:
        return np.zeros((batch, heads, dim), dtype=np.float64)
    all_logits = np.concatenate(logits, axis=-1)
    all_values = np.concatenate(values, axis=0)
    m = all_logits.max(axis=-1, keepdims=True)
    p = np.exp(all_logits - m)
    return np.einsum('bhs,sd->bhd', p, all_values) / p.sum(axis=-1, keepdims=True)


def run_pallas(case, use_dot=False):
    query, k_q, k_s, k_z, vals, lens = case
    return flash_decode_pallas(query, k_q, k_s, k_z, vals, lens,
                               interpret=True, use_dot=use_dot)


def run_ref(case):
    query, k_q, k_s, k_z, vals, lens = case
    out, _ = online_softmax_ref(query, list(zip(k_q, k_s, k_z)), vals, lens)
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

class TestPallasAttentionParity:

    @pytest.mark.parametrize("use_dot", [False, True])
    def test_single_block(self, use_dot):
        case = make_case(batch=2, heads=4, dim=64, n_blocks=1, block_size=128, seed=0)
        got, ref = run_pallas(case, use_dot), run_ref(case)
        assert got.shape == ref.shape
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() < 1e-5

    @pytest.mark.parametrize("use_dot", [False, True])
    def test_multi_block(self, use_dot):
        case = make_case(batch=2, heads=8, dim=64, n_blocks=8, block_size=256, seed=1)
        got, ref = run_pallas(case, use_dot), run_ref(case)
        assert np.abs(got - ref).max() < 1e-4
        assert np.abs(got - ref).mean() < 1e-5

    def test_head_dim_128(self):
        case = make_case(batch=1, heads=4, dim=128, n_blocks=4, block_size=256, seed=2)
        got, ref = run_pallas(case), run_ref(case)
        assert np.abs(got - ref).mean() < 1e-5

    def test_variable_lengths(self):
        case = make_case(batch=2, heads=4, dim=64, n_blocks=4, block_size=128,
                         seed=3, lens=[128, 73, 5, 128])
        got, ref = run_pallas(case), run_ref(case)
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() < 1e-5

    def test_empty_page_in_middle(self):
        """The CUDA kernel needed an explicit -inf guard here; so does this."""
        case = make_case(batch=2, heads=4, dim=64, n_blocks=4, block_size=128,
                         seed=4, lens=[128, 0, 64, 128])
        got, ref = run_pallas(case), run_ref(case)
        assert not np.isnan(got).any(), "empty page produced NaN"
        assert np.abs(got - ref).max() < 1e-5

    def test_first_page_empty(self):
        """Worst case for the guard: m_prev and m_block are both -inf."""
        case = make_case(batch=1, heads=2, dim=64, n_blocks=3, block_size=64,
                         seed=6, lens=[0, 64, 64])
        got, ref = run_pallas(case), run_ref(case)
        assert not np.isnan(got).any()
        assert np.abs(got - ref).max() < 1e-5

    def test_all_pages_empty(self):
        """Degenerate sequence: must be exactly zeros, not NaN."""
        case = make_case(batch=1, heads=2, dim=64, n_blocks=3, block_size=64,
                         seed=5, lens=[0, 0, 0])
        got = run_pallas(case)
        assert not np.isnan(got).any()
        np.testing.assert_array_equal(got, np.zeros_like(got))


def float32_floor(truth, dim):
    """Expected float32 accumulation error for a length-`dim` reduction.

    sqrt(dim) * eps * |output| -- the standard random-walk estimate for
    round-off growth in a dot product of that length.
    """
    return float(np.sqrt(dim) * np.finfo(np.float32).eps * np.abs(truth).mean())


class TestPallasAttentionAccuracy:
    """The float64 arbiter. See the module docstring for why.

    An earlier version of this class asserted the Pallas kernel was at least as
    accurate as the NumPy reference. That gate passed on one machine and failed
    in CI, and the reason matters: the reference's error is identical on both
    (NumPy, deterministic), while the Pallas error moved 3.2x between jax/XLA
    versions because XLA is free to choose a different reduction order.

    Which of the two is more accurate is therefore NOT a property of either
    implementation -- it is a property of the XLA build. The gate below tests
    the thing that IS stable: both must sit within a small multiple of the
    float32 accumulation floor. A genuinely wrong kernel misses by orders of
    magnitude, not by 1.4x, so this still fails loudly for real defects.
    """

    #: Widest observed ratio to the floor across platforms is 1.39
    #: (Pallas, dim 64, Linux/modern XLA). 3x leaves headroom for a different
    #: reduction order without admitting an actually-broken kernel.
    FLOOR_TOLERANCE = 3.0

    @pytest.mark.parametrize("dim", [64, 128])
    def test_within_float32_noise_floor(self, dim):
        case = make_case(batch=1, heads=4, dim=dim, n_blocks=4,
                         block_size=256, seed=2)
        truth = f64_attention(*case)
        floor = float32_floor(truth, dim)

        err_pallas = np.abs(run_pallas(case, use_dot=False) - truth).mean()
        err_ref = np.abs(run_ref(case) - truth).mean()

        assert err_pallas <= floor * self.FLOOR_TOLERANCE, (
            f"Pallas error {err_pallas:.3e} exceeds {self.FLOOR_TOLERANCE}x the "
            f"float32 accumulation floor {floor:.3e} at head_dim {dim} -- that "
            f"is too large to be reduction-order noise"
        )
        # The reference is held to the same standard, so this test documents
        # the relationship rather than privileging either implementation.
        assert err_ref <= floor * self.FLOOR_TOLERANCE, (
            f"reference error {err_ref:.3e} exceeds the floor -- the assumption "
            f"in the module docstring has changed"
        )

    def test_reference_error_is_the_float32_floor(self):
        """Documents WHY a fixed 1e-8 gate is unattainable at this shape.

        If this ever fails, the accumulation floor assumption in the module
        docstring has changed and the tolerances above need revisiting.
        """
        case = make_case(batch=1, heads=4, dim=128, n_blocks=4,
                         block_size=256, seed=2)
        truth = f64_attention(*case)
        err_ref = np.abs(run_ref(case) - truth).mean()

        eps = np.finfo(np.float32).eps
        predicted_floor = np.sqrt(128) * eps * np.abs(truth).mean()
        assert err_ref > predicted_floor / 10, "reference is implausibly accurate"
        assert err_ref < predicted_floor * 10, "reference is worse than the f32 floor"


class TestPallasGridInvariance:

    def test_block_count_does_not_change_result(self):
        """Same total sequence, different page decomposition.

        The accumulators live in output blocks kept resident across the grid;
        if that residency were broken, splitting the sequence differently would
        change the answer.
        """
        rng = np.random.RandomState(11)
        dim, total = 64, 512
        query = rng.randn(1, 2, dim).astype(np.float32)
        keys = rng.randn(total, dim).astype(np.float32)
        vals = rng.randn(total, dim).astype(np.float32)

        outs = []
        for n_blocks in (1, 2, 4, 8):
            bs = total // n_blocks
            k_q, k_s, k_z, v_l = [], [], [], []
            for b in range(n_blocks):
                sl = slice(b * bs, (b + 1) * bs)
                # Quantize the WHOLE sequence once so pages share statistics;
                # otherwise per-page scales legitimately change the answer.
                q_, s_, z_ = quantize_int4_ref(keys, per_channel=True)
                k_q.append(q_[sl]); k_s.append(s_); k_z.append(z_)
                v_l.append(vals[sl])
            lens = np.full(n_blocks, bs, dtype=np.int32)
            outs.append(flash_decode_pallas(query, k_q, k_s, k_z, v_l, lens,
                                            interpret=True))

        for i in range(1, len(outs)):
            assert np.abs(outs[i] - outs[0]).max() < 1e-5, \
                f"page decomposition changed the result at split {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
