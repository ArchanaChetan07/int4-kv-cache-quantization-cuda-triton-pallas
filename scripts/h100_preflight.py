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


@check("GPU not occupied by another process")
def _gpu_free():
    """Runs before any torch/JAX check, which would create contexts of our own.

    Found on a real rental: the vast.ai template had started a vLLM server
    that held 73 GB of the H100's 80 GB, starving JAX and leaving no room for
    the model weights.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    others = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[0] == str(os.getpid()):
            continue
        mib = parts[1].split()[0] if parts[1] else ""
        # Windows (WDDM) lists every graphics client with memory "[N/A]"; only
        # a process holding real memory -- 1 GiB or more -- blocks the run.
        if mib.isdigit() and int(mib) >= 1024:
            others.append(line)
    if not others:
        return OK, "no other process holds >= 1 GiB of GPU memory"
    return FAIL, (f"{len(others)} other process(es) hold GPU memory: {others[0]}\n"
                  f"          A provider template may have started a server (e.g. vLLM).\n"
                  f"          Stop it (`supervisorctl stop vllm`), then kill any orphaned\n"
                  f"          child such as VLLM::EngineCore, and re-run preflight.")


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

@check("transformers importable (perplexity step)", required=False)
def _transformers():
    try:
        import transformers
        return OK, f"transformers {transformers.__version__}"
    except Exception as exc:
        return WARN, f"{type(exc).__name__}: the perplexity step will fail"


def _download_permission(model: str):
    """(ok, detail) for whether this token may DOWNLOAD the model's files.

    model_info() is not enough: for a gated repo it returns metadata to anyone,
    access granted or not, so checking it reports PASS and the run then fails
    at download -- on rented time. Verified 2026-11: model_info succeeds while
    resolving config.json returns 403.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if hasattr(api, "auth_check"):           # huggingface_hub >= 0.24
        api.auth_check(model)
    else:                                    # older: fetch one tiny file
        hf_hub_download(model, "config.json", force_download=True)
    return True, f"{model}: download permission confirmed"


def _error_chain(exc) -> str:
    parts, c = [], exc
    while c is not None and len(parts) < 5:
        parts.append(str(c))
        c = c.__cause__ or c.__context__
    return " | ".join(parts)


@check("Hugging Face download permission for the model", required=False)
def _hf():
    model = os.environ.get("PREFLIGHT_MODEL", "meta-llama/Llama-2-7b-hf")
    if model.startswith("meta-llama/Llama-2") and not model.endswith("-hf"):
        return WARN, (
            f"{model} is Meta's ORIGINAL checkpoint format (consolidated.*.pth),\n"
            f"          which transformers cannot load. Use {model}-hf instead.")
    import importlib.util
    if importlib.util.find_spec("huggingface_hub") is None:
        return WARN, "huggingface_hub not installed -- the perplexity step will fail"
    try:
        ok, detail = _download_permission(model)
        return OK, detail
    except Exception as exc:
        chain = _error_chain(exc)
        if "fine-grained token" in chain:
            return WARN, (
                f"{model}: 403 from TOKEN SCOPE, not from the model's gate. This is\n"
                f"          a fine-grained token without 'Read access to contents of all\n"
                f"          public gated repos you can access' -- it fails even after the\n"
                f"          licence is approved. Enable that permission at\n"
                f"          https://huggingface.co/settings/tokens, or log in with a classic\n"
                f"          Read token (`hf auth login`), then re-run preflight.")
        if any(t in chain for t in ("403", "401", "Gated", "gated", "restricted")):
            return WARN, (
                f"{model}: NO download permission (403). Access is pending or was\n"
                f"          never requested. Request it at https://huggingface.co/{model}\n"
                f"          (one approval covers the whole Llama 2 collection), then\n"
                f"          `hf auth login`. Do this BEFORE renting. Ungated\n"
                f"          fallback needing no approval: --model Qwen/Qwen2.5-7B")
        return WARN, f"{type(exc).__name__}: {chain[:160]}"


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
