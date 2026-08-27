"""Fail-fast checks before spending money on a rented H100.

Run this FIRST. Every check here is cheap; every check here guards something
expensive. The point is that a missing Hugging Face token should cost you five
seconds, not forty minutes of a $3/hour GPU.

Exit code is 0 only if every REQUIRED check passes. Optional checks report but
do not fail the run, because some legs of the plan are legitimately unavailable
depending on what you are paying for.

    python scripts/h100_preflight.py
    python scripts/h100_preflight.py --model Qwen/Qwen2.5-7B
"""

import argparse
import os
import shutil
import subprocess
import sys

REQUIRED = []
OPTIONAL = []

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def check(name, required=True):
    def deco(fn):
        (REQUIRED if required else OPTIONAL).append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

@check("GPU present and identified")
def _gpu():
    if not shutil.which("nvidia-smi"):
        return FAIL, "nvidia-smi not found -- is this actually a GPU host?"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if not out:
        return FAIL, "nvidia-smi returned nothing"
    return OK, out.replace("\n", " | ")


@check("Compute capability >= 9.0 (Mosaic GPU needs Hopper)", required=False)
def _hopper():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().split("\n")[0]
    try:
        major = float(out)
    except ValueError:
        return WARN, f"could not parse compute_cap {out!r}"
    if major >= 9.0:
        return OK, f"sm_{out} -- Mosaic GPU is a supported target"
    return WARN, (f"sm_{out} < 9.0: Mosaic GPU will NOT lower here. CUDA, Triton "
                  f"and the Llama gate still work; the Pallas-on-GPU leg does not.")


@check("Disk space >= 60 GB free (model weights + wheels)", required=False)
def _disk():
    free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
    if free_gb >= 60:
        return OK, f"{free_gb:.0f} GB free"
    return WARN, f"only {free_gb:.0f} GB free -- a 7B model in fp16 needs ~15 GB"


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------

@check("nvcc available (needed to build the CUDA extension)")
def _nvcc():
    if not shutil.which("nvcc"):
        return FAIL, "nvcc not on PATH -- CUDA toolkit missing, not just the driver"
    v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    line = [l for l in v.split("\n") if "release" in l]
    return OK, (line[0].strip() if line else "present")


@check("PyTorch sees the GPU")
def _torch():
    try:
        import torch
    except ImportError:
        return FAIL, "torch not installed"
    if not torch.cuda.is_available():
        return FAIL, f"torch {torch.__version__} installed but CUDA unavailable"
    return OK, f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}"


@check("JAX sees the GPU")
def _jax():
    try:
        import jax
    except ImportError:
        return FAIL, "jax not installed -- pip install 'jax[cuda12]'"
    plats = {d.platform for d in jax.devices()}
    if plats == {"cpu"}:
        return FAIL, (f"jax {jax.__version__} sees CPU only. Pallas cannot lower "
                      f"on CPU, so the whole Pallas-on-GPU leg is dead here.")
    return OK, f"jax {jax.__version__} | devices={[d.device_kind for d in jax.devices()]}"


@check("Triton importable", required=False)
def _triton():
    try:
        import triton
        return OK, f"triton {triton.__version__}"
    except Exception as exc:
        return WARN, f"{type(exc).__name__}: the Triton quantizer leg will skip"


# ---------------------------------------------------------------------------
# this repository
# ---------------------------------------------------------------------------

@check("Repo imports and backends probe")
def _repo():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from src import ops
    except Exception as exc:
        return FAIL, f"cannot import src.ops: {exc!r}"
    st = ops.backend_status()
    return OK, f"available={st['available_backends']} jax_devices={st['jax_devices']}"


@check("CUDA extension built or buildable")
def _ext():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import ops
    if ops.HAS_CUDA:
        return OK, "extension loaded"
    return FAIL, ("extension not built. Run with FLASH_DECODE_JIT_CUDA=1, or "
                  "FLASH_DECODE_FORCE_CUDA=1 pip install -e .")


# ---------------------------------------------------------------------------
# the expensive leg: model weights
# ---------------------------------------------------------------------------

@check("Hugging Face auth + model access", required=False)
def _hf():
    model = os.environ.get("PREFLIGHT_MODEL", "meta-llama/Llama-2-7b-hf")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return WARN, "huggingface_hub not installed -- perplexity leg will skip"
    try:
        HfApi().model_info(model)
        return OK, f"{model} reachable"
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            return WARN, (
                f"{model} is GATED and you do not have access.\n"
                f"          Accept the licence at https://huggingface.co/{model}\n"
                f"          then `huggingface-cli login`. Approval can take hours --\n"
                f"          do this BEFORE renting. Ungated fallbacks that need no\n"
                f"          approval: Qwen/Qwen2.5-7B, TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return WARN, f"{type(exc).__name__}: {msg[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model id for the perplexity gate access check")
    args = ap.parse_args()
    if args.model:
        os.environ["PREFLIGHT_MODEL"] = args.model

    width = 52
    failures = 0

    for title, group in (("REQUIRED", REQUIRED), ("OPTIONAL", OPTIONAL)):
        print("=" * 74)
        print(title)
        print("=" * 74)
        for name, fn in group:
            try:
                status, detail = fn()
            except Exception as exc:
                status, detail = FAIL, f"check raised {type(exc).__name__}: {exc}"
            if status == FAIL and group is REQUIRED:
                failures += 1
            print(f"  [{status}] {name:<{width}} {detail}")
        print()

    if failures:
        print(f"{failures} REQUIRED check(s) failed. Fix these before running the "
              f"benchmark -- do not burn GPU hours debugging them interactively.")
        return 1
    print("All required checks passed. Safe to run scripts/h100_run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
