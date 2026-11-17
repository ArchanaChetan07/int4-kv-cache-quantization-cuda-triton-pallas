"""Tests for the real-model perplexity harness (scripts/validate_llama.py).

Regression guard for a harness that measured nothing: it computed the loss,
then quantized the KV cache after the fact, so "INT4" perplexity equalled FP16
perplexity bit-for-bit and the gate passed by construction. These tests run the
harness on a tiny randomly-initialised Llama -- no download, no GPU -- and
assert that INT4 keys actually reach attention.

Skips when torch or transformers is not installed.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from validate_llama import (  # noqa: E402
    fake_quantize_keys,
    make_key_quantizing_cache,
    perplexity,
    simulate_kv_quantization_impact,
)
from src.quantize_int4_ref import quantize_int4_ref  # noqa: E402


def tiny_llama(kv_heads=4, layers=2, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=kv_heads, max_position_embeddings=2048)
    return LlamaForCausalLM(cfg).eval()


def token_ids(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 512, (1, n), generator=g)


class TestINT4KeysReachAttention:

    def test_int4_keys_change_the_loss(self):
        """The defect this file exists for: INT4 must not equal FP16."""
        model, ids = tiny_llama(), token_ids(600)
        base = perplexity(model, ids, quantize_keys=False)
        int4 = perplexity(model, ids, quantize_keys=True)
        assert int4['ppl'] != base['ppl'], \
            "INT4-key perplexity identical to FP16: quantization never reached attention"
        assert int4['max_abs_key_change'] > 0

    def test_hook_fires_once_per_layer_per_window(self):
        model, ids = tiny_llama(layers=3), token_ids(600)
        r = perplexity(model, ids, quantize_keys=True, max_length=256, stride=128)
        assert r['windows'] > 1, "test must exercise multiple sliding windows"
        assert r['hook_calls'] == 3 * r['windows']

    def test_grouped_query_attention(self):
        """KV heads < query heads: keys are quantized per KV head, pre-repeat."""
        model, ids = tiny_llama(kv_heads=2), token_ids(400)
        base = perplexity(model, ids, quantize_keys=False)
        int4 = perplexity(model, ids, quantize_keys=True)
        assert int4['ppl'] != base['ppl']
        assert np.isfinite(int4['ppl'])

    def test_full_precision_path_is_untouched(self):
        """quantize_keys=False must be the model's ordinary perplexity."""
        model, ids = tiny_llama(), token_ids(300)
        with torch.no_grad():
            direct = float(model(ids, labels=ids).loss)
        r = perplexity(model, ids, quantize_keys=False, max_length=2048, stride=1024)
        assert abs(np.log(r['ppl']) - direct) < 1e-5


class TestNoSilentNoOp:

    def test_raises_when_attention_bypasses_the_cache(self):
        """A model that never calls Cache.update must not yield an 'INT4' number."""

        class BypassesCache(torch.nn.Module):
            def forward(self, ids, labels=None, use_cache=True, past_key_values=None):
                class Out:
                    loss = torch.tensor(1.0)
                return Out()

        with pytest.raises(RuntimeError, match="never fired"):
            perplexity(BypassesCache(), token_ids(64), quantize_keys=True)

    def test_raises_when_keys_are_left_unchanged(self, monkeypatch):
        import validate_llama
        monkeypatch.setattr(validate_llama, "fake_quantize_keys", lambda k, b: k)
        with pytest.raises(RuntimeError, match="unchanged"):
            perplexity(tiny_llama(), token_ids(200), quantize_keys=True)


class TestQuantizerMatchesKernelContract:

    def test_matches_reference_quantizer_per_block(self):
        """Same scheme as the oracle: per channel, per 256-token block.

        Rounding is half-up (as in the kernels) vs banker's in the NumPy
        reference, so bins may differ by one at exact .5 boundaries.
        """
        rng = np.random.RandomState(0)
        k = rng.randn(1, 2, 512, 32).astype(np.float32)
        dq = fake_quantize_keys(torch.from_numpy(k), block_size=256).numpy()

        for h in range(2):
            for blk in range(2):
                rows = k[0, h, blk * 256:(blk + 1) * 256]
                q, scale, zp = quantize_int4_ref(rows, per_channel=True)
                ref = q.astype(np.float32) * scale - zp * scale
                err = np.abs(dq[0, h, blk * 256:(blk + 1) * 256] - ref)
                assert (err <= scale * 1.0001 + 1e-6).all(), \
                    f"head {h} block {blk} differs by more than one bin"

    def test_partial_last_block_uses_only_its_own_rows(self):
        """Padding must not widen the last block's [min, max]."""
        rng = np.random.RandomState(1)
        k = rng.randn(1, 1, 300, 16).astype(np.float32)
        dq = fake_quantize_keys(torch.from_numpy(k), block_size=256).numpy()
        tail = k[0, 0, 256:]
        q, scale, zp = quantize_int4_ref(tail, per_channel=True)
        ref = q.astype(np.float32) * scale - zp * scale
        assert dq.shape == k.shape
        assert np.abs(dq[0, 0, 256:] - ref).max() <= scale.max() * 1.0001 + 1e-6

    def test_round_trip_error_bounded_by_half_a_step(self):
        rng = np.random.RandomState(2)
        k = torch.from_numpy(rng.randn(2, 3, 256, 64).astype(np.float32))
        dq = fake_quantize_keys(k, block_size=256)
        step = (k.amax(dim=2, keepdim=True) - k.amin(dim=2, keepdim=True)) / 15.0
        assert ((dq - k).abs() <= step / 2 + 1e-5).all()

    def test_preserves_dtype(self):
        k = torch.randn(1, 2, 100, 8, dtype=torch.float16)
        assert fake_quantize_keys(k).dtype == torch.float16


class TestSimulation:

    def test_simulation_reports_both_granularities(self):
        r = simulate_kv_quantization_impact(num_layers=2, seq_len=512)
        assert r['mean_kv_snr_db'] > r['mean_kv_snr_db_whole_sequence_scales'], \
            "per-block scales should beat one scale set per sequence"
        assert r['per_block_gain_db'] > 0


def test_cache_factory_returns_a_working_cache():
    stats = {'hook_calls': 0, 'max_abs_key_change': 0.0}
    cache = make_key_quantizing_cache(256, stats)
    k = torch.randn(1, 2, 10, 8)
    kq, v = cache.update(k, k.clone(), 0)
    assert stats['hook_calls'] == 1
    assert kq.shape == k.shape and v.shape == k.shape
