#!/usr/bin/env bash
# Bootstrap a rented H100 host for the three-backend benchmark.
#
# Provider-agnostic. Assumes a CUDA 12.x image with the driver already present
# (RunPod "PyTorch 2.x / CUDA 12.x", Lambda "Lambda Stack", Vast "cuda:12.x-devel"
# all qualify). If nvcc is missing the script says so rather than guessing -- a
# driver-only image cannot build the CUDA extension.
#
#   bash scripts/h100_bootstrap.sh
#
# Idempotent: safe to re-run after a failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Host"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
           --format=csv,noheader || { echo "no nvidia-smi: not a GPU host"; exit 1; }

say "CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
  cat <<'EOF'
nvcc not found. The driver alone is not enough -- building the CUDA extension
needs the toolkit. Either pick a "-devel" image, or:

    conda install -y -c nvidia cuda-nvcc      # fastest inside an existing env
    # or: apt-get update && apt-get install -y cuda-toolkit-12-4

Then re-run this script.
EOF
  exit 1
fi
nvcc --version | tail -2

say "Python dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pytest matplotlib huggingface_hub

# torch: only install if absent or CPU-only, so we do not fight a preinstalled
# provider build that is already correct.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "torch already CUDA-enabled: $(python -c 'import torch;print(torch.__version__)')"
else
  echo "installing CUDA torch..."
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi

say "JAX with CUDA"
if python -c "import jax,sys; sys.exit(0 if any(d.platform!='cpu' for d in jax.devices()) else 1)" 2>/dev/null; then
  echo "jax already sees an accelerator: $(python -c 'import jax;print(jax.__version__)')"
else
  python -m pip install --quiet --upgrade "jax[cuda12]"
fi

say "Triton"
python -m pip install --quiet triton || echo "triton install failed - that leg will skip"

say "Build the CUDA extension"
FLASH_DECODE_JIT_CUDA=1 python -c "
from src import ops
print('CUDA extension loaded:', ops.HAS_CUDA)
" || echo "JIT build failed; try: FLASH_DECODE_FORCE_CUDA=1 pip install -e ."

say "Preflight"
python scripts/h100_preflight.py "$@"

cat <<'EOF'

Bootstrap complete. Next:

    bash scripts/h100_run_all.sh

Keep FLASH_DECODE_JIT_CUDA=1 exported for every subsequent command, or the
dispatch layer silently falls back to the NumPy reference and you will benchmark
the wrong thing.
EOF
