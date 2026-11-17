# Source me (bootstrap and run_all both do). Fixes found on a real H100 rental:
#
# 1. Some CUDA 13 images ship nvcc without the cuSPARSE / cuSOLVER headers that
#    PyTorch's own headers include (ATen/cuda/CUDAContextLight.h), so the
#    extension's JIT build dies with "cusparse.h: No such file or directory".
#    The pip CUDA wheels installed alongside torch carry those headers.
# 2. JAX preallocates most of GPU memory per process by default; in a process
#    that also runs torch that starves one of them.
NV_INC=$(python - <<'EOF'
import importlib.util, os
spec = importlib.util.find_spec("nvidia")
for base in (spec.submodule_search_locations if spec else []):
    for sub in ("cu13/include", "cu12/include"):
        path = os.path.join(base, sub)
        if os.path.exists(os.path.join(path, "cusparse.h")):
            print(path)
            raise SystemExit
EOF
)
if [ -n "$NV_INC" ] && [ ! -f /usr/local/cuda/include/cusparse.h ]; then
  # -isystem semantics: searched after the toolkit's own include directory.
  export C_INCLUDE_PATH="$NV_INC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$NV_INC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  echo "CUDA headers missing from the toolkit; using pip wheel headers: $NV_INC"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
