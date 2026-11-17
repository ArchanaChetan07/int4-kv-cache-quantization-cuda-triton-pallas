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

say "Disk"
# Measured on an H100 rental: CUDA-13 torch + deps alone is 5.3 GB, JAX with
# CUDA ~3 GB more, the Llama-2-7b weights 13.5 GB, plus build and pip caches.
# /dev/shm cannot stand in: Docker mounts it noexec, so compiled libraries
# fail to load from it ("failed to map segment from shared object"), and an
# unprivileged container cannot remount it. Refuse before installing anything.
FREE_GB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print int($4 / 1048576)}')
echo "free on $(df -P "$REPO_ROOT" | awk 'NR==2 {print $6}'): ${FREE_GB} GB"
if [ "$FREE_GB" -lt 40 ]; then
  cat <<EOF
Only ${FREE_GB} GB free. The run needs ~30 GB of executable disk (environment
~9 GB + model 13.5 GB + caches) and /dev/shm is noexec, so it cannot be used
for the environment. Destroy this instance and rent again with the disk slider
set to >= 60 GB -- the disk shown in a listing is the host's, not yours.
EOF
  exit 1
fi

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
# perplexity gate (scripts/validate_llama.py)
python -m pip install --quiet transformers datasets

# torch must be built for the SAME CUDA major version as the image's nvcc:
# torch.utils.cpp_extension refuses to JIT-build the extension on a major
# mismatch. Found on a vast.ai H100 whose devel image shipped nvcc 13.0, where
# a hardcoded cu124 torch would have made the CUDA kernel unbuildable.
NVCC_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
case "$NVCC_MAJOR" in
  13) TORCH_INDEX=https://download.pytorch.org/whl/cu130 ;;
  12) TORCH_INDEX=https://download.pytorch.org/whl/cu126 ;;
  *)  echo "unsupported nvcc major version: '$NVCC_MAJOR'"; exit 1 ;;
esac
TORCH_CUDA_MAJOR=$(python -c "import torch; print((torch.version.cuda or '').split('.')[0])" 2>/dev/null || true)
if [ "$TORCH_CUDA_MAJOR" = "$NVCC_MAJOR" ] &&    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "torch already built for CUDA $NVCC_MAJOR: $(python -c 'import torch;print(torch.__version__)')"
else
  echo "installing torch for CUDA $NVCC_MAJOR (nvcc $NVCC_MAJOR, existing torch CUDA: '${TORCH_CUDA_MAJOR:-none}')"
  python -m pip install --quiet --force-reinstall torch --index-url "$TORCH_INDEX"
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
