#!/usr/bin/env bash
# The rented-GPU runbook, in dependency order.
#
# Ordered cheapest-and-most-certain first, so a failure late in the run does not
# cost you the results from earlier steps. Every step writes its artifact to
# results/ immediately rather than at the end.
#
#   export FLASH_DECODE_JIT_CUDA=1
#   bash scripts/h100_run_all.sh
#   bash scripts/h100_run_all.sh --model Qwen/Qwen2.5-7B     # ungated fallback
#
# Steps 1-4 are the core deliverable and take ~20 minutes.
# Step 5 (perplexity) is the long pole and is last on purpose.

set -uo pipefail   # NOT -e: a failing step must not abort the ones after it

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="meta-llama/Llama-2-7b-hf"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) shift ;;
  esac
done

export FLASH_DECODE_JIT_CUDA="${FLASH_DECODE_JIT_CUDA:-1}"
mkdir -p results

STEP=0
step() { STEP=$((STEP+1)); printf '\n\033[1m######## STEP %s: %s\033[0m\n' "$STEP" "$*"; }
note() { printf '\033[33m  >> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "Correctness first -- never benchmark a kernel you have not validated"
python -m pytest tests/ -q 2>&1 | tail -15
note "If the Pallas tests failed here they will fail on device too. Stop and read."

# ---------------------------------------------------------------------------
step "Does the Pallas port actually lower on this GPU?"
note "Highest-uncertainty step. REJECTED is a publishable result, not an outage."
python scripts/pallas_lowering_probe.py --json results/lowering_probe_h100.json 2>&1 | tail -40

# ---------------------------------------------------------------------------
step "Quantizer benchmark -- the op with three real backends"
note "reference / cuda / triton / pallas on identical seeded inputs"
python benchmarks/bench_three_backend.py \
    --op quantize \
    --backends reference,cuda,triton,pallas \
    --iters 20 \
    --out results/three_backend_quantize_h100.json 2>&1 | tail -40

# ---------------------------------------------------------------------------
step "Attention benchmark -- full anchor sweep"
note "cuda / pallas / reference. No Triton attention kernel exists in this repo."
python benchmarks/bench_three_backend.py \
    --op attention \
    --backends reference,cuda,pallas \
    --iters 10 \
    --out results/three_backend_attention_h100.json 2>&1 | tail -60

# ---------------------------------------------------------------------------
step "Legacy latency benchmark (comparable to the committed T1000 numbers)"
python benchmarks/bench_flash_decode.py 2>&1 | tail -15

# ---------------------------------------------------------------------------
step "Real-model perplexity gate -- the long pole"
note "model: $MODEL"
note "If this is gated and unapproved it fails FAST here, after the cheap wins."
python scripts/validate_llama.py \
    --model "$MODEL" \
    --output results/perplexity_h100.json 2>&1 | tail -25

# ---------------------------------------------------------------------------
printf '\n\033[1m######## ARTIFACTS\033[0m\n'
ls -la results/*h100*.json 2>/dev/null || echo "  (none written)"

cat <<'EOF'

Before you destroy the instance, copy results/ off the box:

    tar czf h100-results.tar.gz results/
    # then scp / runpodctl / the provider's file browser

Nothing in results/ is reproducible without the GPU. Everything else in this
repo is.
EOF
