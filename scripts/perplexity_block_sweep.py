"""Perplexity vs key-scale granularity: one FP16 baseline, INT4 keys at several
token-group sizes.

The group size is the paged-KV block size, which the CUDA, Triton and Pallas
kernels all take at runtime -- so choosing a smaller group is a configuration
change, not a code change. Smaller groups cost compression (more scale and
zero-point bytes per token); this measures what they buy.

    python scripts/perplexity_block_sweep.py                       # Llama-2-7b-hf
    python scripts/perplexity_block_sweep.py --model Qwen/Qwen2.5-7B

Needs a GPU with room for the model in fp16 (~14 GB for a 7B model). Uses the
same harness, evaluation slice and guards as scripts/validate_llama.py.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)

from validate_llama import _load_model, _load_text, perplexity  # noqa: E402


def key_compression(block_size: int, head_dim: int, tokens: int = 4096) -> float:
    """Keys only: FP16 bytes / (nibble-packed INT4 + per-block, per-channel
    float32 scale and zero-point)."""
    blocks = -(-tokens // block_size)
    fp16 = tokens * head_dim * 2
    int4 = tokens * head_dim // 2 + blocks * head_dim * 4 * 2
    return fp16 / int4


def main():
    ap = argparse.ArgumentParser(description="INT4-key perplexity vs token-group size")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--block-sizes", default="256,128,64,32",
                    help="comma-separated tokens per scale group")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="number of 1024-token strides to evaluate")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--text-file", default=None,
                    help="evaluate on a local text file instead of downloading")
    ap.add_argument("--output",
                    default=os.path.join(HERE, "results", "perplexity_block_sweep.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, device)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    text = _load_text(args.dataset, args.dataset_config, args.text_file)[:100_000]
    ids = tokenizer(text, return_tensors="pt").input_ids
    ids = ids[:, :min(ids.size(1), args.max_samples * 1024)].to(device)   # as validate_llama

    t0 = time.time()
    base = perplexity(model, ids, quantize_keys=False)
    print(f"fp16 baseline ppl {base['ppl']:.4f} over {base['tokens']} tokens", flush=True)

    rows = []
    for bs in (int(x) for x in args.block_sizes.split(",") if x.strip()):
        r = perplexity(model, ids, quantize_keys=True, block_size=bs)
        delta = (r["ppl"] - base["ppl"]) / base["ppl"] * 100
        rows.append({
            "block_size": bs,
            "ppl_int4_keys": round(r["ppl"], 4),
            "delta_percent": round(delta, 4),
            "key_compression_vs_fp16": round(key_compression(bs, head_dim), 2),
            "hook_calls": r["hook_calls"],
            "expected_hook_calls": r["windows"] * cfg.num_hidden_layers,
            "gate_0_3_pass": bool(delta < 0.3),
        })
        print(rows[-1], flush=True)

    out = {
        "model": args.model,
        "dataset": args.text_file or f"{args.dataset}/{args.dataset_config}",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "method": "fake-quantized post-RoPE keys inside attention (Cache.update hook); "
                  "values full precision; asymmetric, per (kv_head, block, channel), "
                  "round-half-up",
        "tokens_scored": base["tokens"],
        "ppl_fp16": round(base["ppl"], 4),
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
