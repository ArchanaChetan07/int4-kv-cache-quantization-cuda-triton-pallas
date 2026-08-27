# H100 runbook

Everything needed to turn ~$10 of rented Hopper into the results this repository
is currently missing. Read the pre-flight section **before** you start an
instance — the failures it prevents are the ones that cost money.

---

## What this run produces

| Deliverable | Status today | After this run |
|---|---|---|
| **D14** — three backends on identical silicon | never satisfiable (no shared device) | ✅ the project's original goal |
| Performance numbers | T1000 only, 160 GB/s | ✅ H100, ~3.35 TB/s HBM3 — MBU becomes meaningful |
| Pallas compiled by a real backend | never — interpret mode only | ✅ or a documented rejection, which is also a result |
| Perplexity gate | simulated | ✅ real weights |

**What it does *not* produce:** the three UNRESOLVED hypotheses in
[TRITON_TO_PALLAS.md](TRITON_TO_PALLAS.md) — missing `int4` dtype, nibble-packing
axis vs. `(8,128)` tiling, reduction direction — are Mosaic **TPU** properties.
An H100 cannot resolve them. That leg is free on Colab/Kaggle and is
complementary, not an alternative.

---

## Before you rent (all free, do it now)

1. **Request Llama-2 access.** `meta-llama/Llama-2-7b-hf` is gated; approval can
   take hours. Accept the licence at
   <https://huggingface.co/meta-llama/Llama-2-7b-hf>, then `huggingface-cli login`.
   Ungated fallbacks needing no approval: `Qwen/Qwen2.5-7B`,
   `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

2. **Pick a `-devel` image.** The driver alone cannot build the CUDA extension —
   you need `nvcc`. Known-good: RunPod *PyTorch 2.x / CUDA 12.x*, Lambda
   *Lambda Stack*, Vast `cuda:12.4-devel`. A runtime-only image costs you 20
   minutes installing a toolkit at $3/hour.

3. **Confirm it is really Hopper.** Mosaic GPU needs compute capability ≥ 9.0.
   An H100 is `sm_90`; an A100 is `sm_80` and **will not lower Pallas**. If the
   provider substitutes an A100, the CUDA/Triton/perplexity legs still work but
   the Pallas-on-GPU leg is dead — which is the single thing you are renting for.

---

## The run

```bash
git clone https://github.com/ArchanaChetan07/int4-kv-cache-quantization-cuda-triton-pallas.git
cd int4-kv-cache-quantization-cuda-triton-pallas

bash scripts/h100_bootstrap.sh          # installs, builds, then preflights
export FLASH_DECODE_JIT_CUDA=1          # REQUIRED for every later command
bash scripts/h100_run_all.sh            # or: --model Qwen/Qwen2.5-7B
```

`h100_bootstrap.sh` is idempotent — re-run it after any failure.

**`FLASH_DECODE_JIT_CUDA=1` is load-bearing.** Without it the dispatch layer
falls back to the NumPy reference *silently*, and you will benchmark NumPy on an
H100 and not notice.

### Step order, and why

`h100_run_all.sh` runs cheapest-and-most-certain first, and does **not** use
`set -e` — a failure in step 5 must not discard steps 1–4.

| Step | What | ~Time | Why here |
|---|---|---|---|
| 1 | Full test suite | 1 min | Never benchmark a kernel you have not validated |
| 2 | Pallas lowering probe | 2 min | Highest uncertainty — find out early |
| 3 | Quantizer benchmark (4 backends) | 5 min | The op that genuinely has three implementations |
| 4 | Attention benchmark, full sweep | 10 min | The headline numbers |
| 5 | Legacy latency bench | 2 min | Directly comparable to the committed T1000 figures |
| 6 | Perplexity gate | 20–40 min | Long pole, and the most likely to fail on auth |

Steps 1–5 are ~20 minutes. If you are watching the meter, you have the core
deliverable before step 6 even starts.

---

## Reading the lowering probe

This is the step with a real chance of failure, and each outcome means something
different:

| Verdict | Meaning | What to do |
|---|---|---|
| **LOWERED** | Compiled and matched interpret mode | Nothing. Hypotheses 8/9/13 stay TPU-only, everything else is now proven. |
| **REJECTED** | Mosaic GPU refused the kernel | **Keep the error text verbatim** — it is the finding. This is exactly the portability defect a lowering attempt exists to catch. |
| **MISMATCH** | Compiled but disagreed with interpret mode | **The interesting one.** Almost certainly a grid-order race: the accumulators assume the block axis runs sequentially. Check that `_compiler_params` in `flash_decode_pallas.py` actually applied — the API was renamed (`GPUCompilerParams` → `CompilerParams`) and the helper falls back to `None` silently if neither resolves. |

A `MISMATCH` with no compile error is *not* a numerics problem. Do not chase it
with tolerances.

---

## Before destroying the instance

```bash
tar czf h100-results.tar.gz results/
```

Nothing under `results/` can be regenerated without the GPU. Everything else in
this repo can. Copy it off the box first.

---

## Cost

| Item | Estimate |
|---|---|
| H100 80GB, on-demand | $2–3 / hour |
| Bootstrap + steps 1–5 | ~40 min |
| Step 6 (perplexity) | ~40 min |
| **Total** | **~1.5 h ≈ $3–5**, budget $15 for retries |

If the lowering probe returns REJECTED and you decide to restructure the kernel
for Mosaic GPU, that is a separate, longer session — do not try to debug it on
the meter. Capture the error, destroy the instance, fix it locally against
interpret mode, and rent again.
