"""
Perplexity validation for INT4 KV quantization on real models.

Usage:
    python scripts/validate_llama.py --model meta-llama/Llama-2-7b-hf \
        --output results/perplexity_llama7b.json
    python scripts/validate_llama.py --model Qwen/Qwen2.5-7B      # ungated
    python scripts/validate_llama.py --simulate                    # no GPU, no weights

Requires (real-model mode): torch, transformers, and either `datasets` or a
local --text-file. Gate: perplexity delta must be < 0.3% for 7B.

How the INT4 measurement works -- and why an earlier version measured nothing
-----------------------------------------------------------------------------
The earlier harness ran a forward pass, read `out.loss`, and only THEN
quantized `out.past_key_values`. The loss had already been computed from the
full-precision keys, so the "INT4" perplexity was bit-identical to FP16 and the
gate passed by construction (on transformers 5.x the same code crashed instead,
because iterating the new Cache object yields None entries).

This version quantizes keys where attention actually reads them. Each
attention layer calls `past_key_values.update(k, v, layer_idx)` after applying
RoPE and attends over whatever `update` returns. Wrapping `update` on the cache
instance makes every attention layer consume INT4-quantized post-RoPE keys --
the same tensors a real INT4 KV cache would store. Values stay full precision,
matching the kernel design (keys only).

Quantization granularity matches the kernels: asymmetric INT4, per channel, per
KV head, over blocks of 256 token positions, with round-half-up.

Two guards make a silent no-op impossible: the run fails if the hook never
fired, or if it fired but left the keys unchanged. Both counts are written to
the results JSON, so the measurement carries its own evidence.

This is a fake-quantization measurement: it evaluates the accuracy cost of the
INT4 key scheme inside PyTorch attention. The CUDA/Triton/Pallas kernels'
numerical agreement with the same scheme is established separately by the
parity tests.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize_int4_ref import quantize_and_dequantize_ref

#: Paged KV block size; scales are per-(block, channel), as in the kernels.
BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# hardware-free SNR simulation
# ---------------------------------------------------------------------------

def simulate_kv_quantization_impact(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
) -> dict:
    """Simulate per-layer KV quantization error using realistic KV statistics.

    Real LLM KV activations are approximately Gaussian with per-channel
    structure. This simulation estimates the quantization SNR that drives
    perplexity impact, without needing model weights. It reports both the
    per-block scheme the kernels use and a single whole-sequence scale, so the
    per-block gain is reproducible rather than asserted.
    """
    rng = np.random.default_rng(seed)
    layer_snrs, layer_snrs_whole = [], []

    for _ in range(num_layers):
        # Per-channel std varies ~3x across channels in real models
        channel_std = rng.uniform(0.3, 1.0, size=head_dim).astype(np.float32)
        kv = rng.normal(0, 1, size=(seq_len, head_dim)).astype(np.float32) * channel_std
        signal = float(np.sum(kv.astype(np.float64) ** 2))

        # Per paged block (matches the kernel design)
        noise = 0.0
        for start in range(0, seq_len, BLOCK_SIZE):
            block = kv[start:start + BLOCK_SIZE]
            _, _, _, block_dq = quantize_and_dequantize_ref(block, per_channel=True)
            noise += float(np.sum((block - block_dq).astype(np.float64) ** 2))
        layer_snrs.append(10 * np.log10(signal / max(noise, 1e-12)))

        # One scale set over the whole sequence (the baseline per-block beats)
        _, _, _, whole_dq = quantize_and_dequantize_ref(kv, per_channel=True)
        noise_w = float(np.sum((kv - whole_dq).astype(np.float64) ** 2))
        layer_snrs_whole.append(10 * np.log10(signal / max(noise_w, 1e-12)))

    mean_snr = float(np.mean(layer_snrs))
    mean_snr_whole = float(np.mean(layer_snrs_whole))
    # Empirical mapping: SNR > 20 dB on KV typically yields < 0.3% PPL delta.
    # Gaussian synthetic data is a conservative WORST CASE -- real LLM KV has
    # heavy channel structure and quantizes tighter. The simulation gate is the
    # 0.5% fallback threshold; the hard 0.3% gate applies to real-model runs.
    est_ppl_delta_pct = max(0.01, 3.0 * 10 ** (-(mean_snr - 10) / 10))

    return {
        'mode': 'simulation',
        'num_layers': num_layers,
        'head_dim': head_dim,
        'seq_len': seq_len,
        'block_size': BLOCK_SIZE,
        'mean_kv_snr_db': round(mean_snr, 2),
        'min_layer_snr_db': round(float(min(layer_snrs)), 2),
        'mean_kv_snr_db_whole_sequence_scales': round(mean_snr_whole, 2),
        'per_block_gain_db': round(mean_snr - mean_snr_whole, 2),
        'estimated_ppl_delta_percent_worst_case': round(est_ppl_delta_pct, 4),
        'note': 'Gaussian worst-case bound; real KV quantizes tighter',
        'gate_threshold_percent': 0.5,
        'gate_pass': bool(est_ppl_delta_pct < 0.5),
    }


# ---------------------------------------------------------------------------
# fake quantization of post-RoPE keys
# ---------------------------------------------------------------------------

def fake_quantize_keys(k, block_size: int = BLOCK_SIZE):
    """INT4 quantize-dequantize, matching the kernels' scheme.

    Args:
        k: torch tensor [batch, kv_heads, seq, head_dim] -- post-RoPE keys, as
           handed to Cache.update().
    Returns:
        Tensor of the same shape and dtype holding dequantized INT4 keys.
        Asymmetric, per (batch, head, block of `block_size` positions, channel),
        round-half-up -- the same contract as the CUDA/Triton/Pallas kernels.
    """
    import torch

    dtype = k.dtype
    x = k.float()
    b, h, s, d = x.shape
    n_blk = (s + block_size - 1) // block_size
    pad = n_blk * block_size - s
    if pad:
        # Replicate the last real row: it cannot widen any block's [min, max].
        x = torch.cat([x, x[:, :, -1:, :].expand(b, h, pad, d)], dim=2)

    xb = x.reshape(b, h, n_blk, block_size, d)
    mn = xb.amin(dim=3, keepdim=True)
    mx = xb.amax(dim=3, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / 15.0
    q = torch.floor((xb - mn) / scale + 0.5).clamp_(0.0, 15.0)
    dq = (q * scale + mn).reshape(b, h, n_blk * block_size, d)[:, :, :s, :]
    return dq.to(dtype)


def make_key_quantizing_cache(block_size: int, stats: dict):
    """A DynamicCache whose update() returns INT4-quantized keys.

    Attention attends over update()'s return value, so wrapping it on the
    instance quantizes exactly the keys attention reads. `stats` records how
    often the hook fired and how much it changed the keys -- the evidence that
    the measurement measured something.
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    original_update = cache.update

    def update(key_states, value_states, layer_idx, *args, **kwargs):
        k, v = original_update(key_states, value_states, layer_idx, *args, **kwargs)
        kq = fake_quantize_keys(k, block_size)
        stats['hook_calls'] += 1
        change = float((kq.float() - k.float()).abs().max())
        stats['max_abs_key_change'] = max(stats['max_abs_key_change'], change)
        return kq, v

    cache.update = update
    return cache


def perplexity(model, input_ids, quantize_keys: bool, block_size: int = BLOCK_SIZE,
               max_length: int = 2048, stride: int = 1024) -> dict:
    """Sliding-window perplexity, optionally with INT4 keys inside attention.

    Args:
        model: a causal LM whose attention layers read keys via Cache.update().
        input_ids: LongTensor [1, n_tokens] on the model's device.
        quantize_keys: route every attention layer through INT4 keys.

    Returns:
        dict with 'ppl', 'tokens', 'windows' and, when quantizing, the hook
        evidence 'hook_calls' and 'max_abs_key_change'.

    Raises:
        RuntimeError if quantization was requested but did not happen -- the
        guard against reporting an FP16 number as an INT4 result.
    """
    import torch

    seq_len = input_ids.size(1)
    stats = {'hook_calls': 0, 'max_abs_key_change': 0.0}
    nll_sum, n_tokens, windows, prev_end = 0.0, 0, 0, 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        labels = ids.clone()
        labels[:, :-trg_len] = -100

        kwargs = {'labels': labels, 'use_cache': True}
        if quantize_keys:
            kwargs['past_key_values'] = make_key_quantizing_cache(block_size, stats)

        with torch.no_grad():
            out = model(ids, **kwargs)

        # out.loss is the mean over the scored (shifted) targets in this window
        scored = int((labels[:, 1:] != -100).sum())
        nll_sum += float(out.loss) * scored
        n_tokens += scored
        windows += 1
        prev_end = end
        if end == seq_len:
            break

    if quantize_keys:
        if stats['hook_calls'] == 0:
            raise RuntimeError(
                "INT4 key hook never fired: this model's attention does not read "
                "keys through Cache.update(), so no quantization took place. "
                "Refusing to report an FP16 perplexity as an INT4 result.")
        if stats['max_abs_key_change'] == 0.0:
            raise RuntimeError(
                "INT4 key hook fired but left every key unchanged; the "
                "measurement would be indistinguishable from FP16.")

    result = {'ppl': float(np.exp(nll_sum / max(n_tokens, 1))),
              'tokens': n_tokens, 'windows': windows}
    if quantize_keys:
        result.update(stats)
    return result


# ---------------------------------------------------------------------------
# real-model driver
# ---------------------------------------------------------------------------

def _load_text(dataset: str, dataset_config: str, text_file: str) -> str:
    if text_file:
        with open(text_file, encoding='utf-8') as f:
            return f.read()
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            f"`datasets` is required to download {dataset} ({e}). Install it, "
            "or pass --text-file with a local evaluation corpus.")
    data = load_dataset(dataset, dataset_config, split='test')
    return '\n\n'.join(data['text'])


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    # transformers >= 4.56 takes `dtype`; earlier releases take `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return model.to(device).eval()


def validate_with_model(model_name: str, dataset: str, dataset_config: str,
                        max_samples: int, text_file: str = None) -> dict:
    """Measure the perplexity delta from INT4 keys on a real model."""
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            f"Real-model validation requires torch and transformers: {e}. "
            "Run with --simulate for a hardware-free estimate.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_model(model_name, device)

    text = _load_text(dataset, dataset_config, text_file)[:100_000]
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    stride = 1024
    input_ids = input_ids[:, :min(input_ids.size(1), max_samples * stride)].to(device)

    base = perplexity(model, input_ids, quantize_keys=False, stride=stride)
    int4 = perplexity(model, input_ids, quantize_keys=True, stride=stride)
    delta_pct = (int4['ppl'] - base['ppl']) / base['ppl'] * 100

    return {
        'mode': 'real_model',
        'method': 'fake-quantized post-RoPE keys inside attention (Cache.update hook)',
        'model': model_name,
        'dataset': text_file or f'{dataset}/{dataset_config}',
        'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'tokens_scored': base['tokens'],
        'windows': base['windows'],
        'block_size': BLOCK_SIZE,
        'granularity': 'per (kv_head, 256-token block, channel), asymmetric, round-half-up',
        'quantized': 'keys only; values full precision',
        'ppl_fp16': round(base['ppl'], 4),
        'ppl_int4_keys': round(int4['ppl'], 4),
        'delta_percent': round(delta_pct, 4),
        'evidence': {
            'hook_calls': int4['hook_calls'],
            'expected_hook_calls': int4['windows'] * model.config.num_hidden_layers,
            'max_abs_key_change': int4['max_abs_key_change'],
        },
        'gate_threshold_percent': 0.3,
        'gate_pass': bool(delta_pct < 0.3),
    }


def main():
    parser = argparse.ArgumentParser(description='Validate INT4 KV perplexity impact')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--dataset', default='Salesforce/wikitext')
    parser.add_argument('--dataset-config', default='wikitext-2-raw-v1')
    parser.add_argument('--text-file', default=None,
                        help='evaluate on a local text file instead of downloading')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='number of 1024-token strides to evaluate')
    parser.add_argument('--simulate', action='store_true',
                        help='Run hardware-free SNR simulation instead of real model')
    parser.add_argument('--output', default='results/perplexity.json')
    args = parser.parse_args()

    if args.simulate:
        print("Running hardware-free KV quantization simulation...")
        result = simulate_kv_quantization_impact()
    else:
        print(f"Validating on real model: {args.model}")
        result = validate_with_model(args.model, args.dataset, args.dataset_config,
                                     args.max_samples, args.text_file)

    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {args.output}")

    if not result['gate_pass']:
        print(f"Perplexity gate NOT passed (threshold "
              f"{result['gate_threshold_percent']}%). This is a measured result, "
              "not a harness failure -- report it.")
        sys.exit(1)
    if args.simulate:
        print("Simulation gate passed (worst-case bound < 0.5% fallback threshold).")
    else:
        print("Gate passed: perplexity delta < 0.3%")


if __name__ == '__main__':
    main()
