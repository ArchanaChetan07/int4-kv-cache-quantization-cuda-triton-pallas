# Architecture: Flash Decoding with INT4 KV

## Component Map

```
┌──────────────────────────────────────────────────────────┐
│                 vLLM Attention Backend                   │
└───────────────┬──────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│           src/vllm_integration.py                        │
│  INT4PagedKVCache                                        │
│  - write_block(): quantize K on write, V stays FP16/32   │
│  - decode_attention(): fused INT4 attention              │
│  - memory_stats(): live compression ratio                │
└───────────────┬──────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│                  src/ops.py (dispatch)                   │
│  quantize_int4() / flash_decode() -> CUDA or reference   │
└──────┬──────────────────────────────────┬────────────────┘
       │                                  │
┌──────▼──────────────────┐   ┌───────────▼───────────────┐
│ quantize_int4_ref.py    │   │ csrc/flash_decode_int4.cu │
│ flash_decode_ref.py     │   │ - quantize_int4_kernel    │
│ (NumPy ground truth)    │◄─►│ - flash_decode_int4_kernel│
└─────────────────────────┘   └───────────────────────────┘
```

## Quantization scheme

Per-(block, channel) asymmetric INT4, applied to **keys only**:

```
scale[c] = (max[c] - min[c]) / 15        # over the block's rows
zp[c]    = -min[c] / scale[c]
q        = clip(round((k - min[c]) / scale[c]), 0, 15)
dequant  = q * scale[c] - zp[c] * scale[c]   # == q*scale + min
```

- **Why keys only:** logits pass through softmax (bounded sensitivity);
  values contribute linearly to the output and are more error-sensitive.
- **Why per-block scales:** a 256-token block has far tighter min/max
  than a 2048-token sequence → +1.72 dB SNR (measured by
  `scripts/validate_llama.py --simulate`).
- **Storage:** production packs two INT4 values per byte; the reference
  keeps one per byte for clarity. `memory_stats()` reports packed size.

## Online softmax (single pass over pages)

Per (batch, head), maintain running `(m, l, acc)`:

```
for each page:
    logit  = q · dequant(k_s)          (per position s)
    m_new  = max(m, logit)
    corr   = exp(m - m_new)
    p      = exp(logit - m_new)
    acc    = acc * corr + p * v_s
    l      = l * corr + p
    m      = m_new
output = acc / l
```

Multi-block result is bit-equivalent to single concatenated block
(verified by `test_attention_convergence_across_blocks`).

## CUDA kernel notes

| Kernel | Grid | Block | Shared mem |
|--------|------|-------|------------|
| quantize_int4 | (num_channels) | 256 | 2×256 f32 (min/max reduce) |
| flash_decode_int4 | (batch, heads) | 128 (4 warps) | q + scales + zp·scale + 4×head_dim merge area |

flash_decode_int4 is warp-parallel (flash-decoding style): warps stride
over positions with private online-softmax state and lane-register
accumulators; logits reduce via warp shuffles (no block syncs in the hot
loop); per-page scales stage in shared memory with zp pre-multiplied;
warps merge at the end with a log-sum-exp combine. Dequantization happens
in registers during the dot product — the FP32 key matrix is never
materialized.

Measured on NVIDIA T1000 (batch 8 × 32 heads × dim 128 × seq 2048), kernel
only, device-resident inputs: 2.24 ms median over 9 runs (2.13–2.68 ms; clocks
unlocked). Reproduce with `benchmarks/bench_three_backend.py`.

The kernel is **latency-bound**, not bandwidth-bound: every (batch, head)
program streams the same page sequence, unique-byte DRAM utilization is
0.3–0.4%, and streamed read throughput exceeds DRAM peak at short sequences
because L2 serves the shared stream. The per-position warp-shuffle reduction
and serial softmax update dominate; tiling several positions per warp is the
next optimization. (An earlier version of this document claimed the kernel ran
near the bandwidth roofline. The measurement above refutes that.)

Scope: all queries in the [batch, heads] grid attend over one page stream
(MQA-style). A vLLM backend additionally needs a per-sequence block table and
per-KV-head indexing.

## Validation gates

| Gate | Where | Threshold |
|------|-------|-----------|
| Round-trip error | test_quantization.py | ≤ scale/2 per value |
| Multi-block == concat | test_flash_decode.py | MAE < 1e-5 |
| INT4 vs FP32 attention | test_correctness.py | MAE < 0.15 (Gaussian worst case) |
| Scaled-query drift | test_vllm_integration.py | MAE < 0.05 |
| SNR simulation | scripts/validate_llama.py | < 0.5% est. PPL (fallback gate) |
| Real-model PPL | scripts/validate_llama.py | < 0.3% hard gate (INT4 keys fake-quantized inside attention) |

## Fallback path

If real-model perplexity delta exceeds 0.5%: switch to symmetric INT8
(2× compression instead of 4×) — same kernel structure, wider scales.
