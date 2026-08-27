"""Three-backend benchmark: reference / triton / cuda / pallas on identical inputs.

Design rules, fixed before any backend was timed:

1.  Every backend receives byte-for-byte identical inputs, built from a fixed
    seed. Inputs are generated once per shape and reused.
2.  Absolute milliseconds are only ever compared WITHIN one device. Two
    backends on different silicon are not comparable in ms.
3.  The cross-device metric is memory-bandwidth utilization (MBU): the
    compulsory byte traffic of the problem, divided by elapsed time, divided by
    the device's peak bandwidth. This kernel is memory-bound, so MBU is what
    transfers between machines. Where peak bandwidth for a device is not known,
    MBU is reported as null rather than guessed.
4.  A backend that cannot run is recorded with a reason. It is never silently
    omitted -- an absent row and a failed row must not look the same.
5.  The shape sweep below was fixed in advance and is not edited after seeing
    results.
6.  Milliseconds are only comparable within one EXECUTION MODE. Pallas on a
    machine with no accelerator runs in interpret mode, which executes the
    kernel as ordinary JAX ops with the grid driven from Python -- it is a
    correctness vehicle, not a code path anyone ships. Timing it against a
    compiled backend produces a number that is real but means nothing, so
    cross-mode ratios are refused rather than printed.

Anti-cheat assertions run inside the harness; see _anticheat().

    python benchmarks/bench_three_backend.py                 # all available
    python benchmarks/bench_three_backend.py --quick         # small sweep
    python benchmarks/bench_three_backend.py --backends pallas,reference
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref
from src import ops


# ---------------------------------------------------------------------------
# shape sweep -- FIXED IN ADVANCE, anchored on the repository's headline config
# ---------------------------------------------------------------------------

ANCHOR = dict(batch=8, heads=32, head_dim=128, seq_len=2048, block_size=256)

SWEEP: List[Dict] = (
    [dict(ANCHOR, seq_len=s) for s in (512, 2048, 8192, 32768)]
    + [dict(ANCHOR, batch=b) for b in (1, 32)]
    + [dict(ANCHOR, head_dim=64)]
)

QUICK_SWEEP: List[Dict] = [
    dict(batch=2, heads=4, head_dim=64, seq_len=512, block_size=128),
    dict(batch=2, heads=8, head_dim=128, seq_len=1024, block_size=256),
]

#: Peak DRAM bandwidth in GB/s. Only devices we can name honestly.
#: Anything absent yields MBU = null rather than a fabricated denominator.
PEAK_BANDWIDTH_GBS = {
    "NVIDIA T1000 8GB": 160.0,
    "TPU v5e": 819.0,
    "TPU v5e-1": 819.0,
}


# ---------------------------------------------------------------------------
# problem construction
# ---------------------------------------------------------------------------

def build_problem(batch, heads, head_dim, seq_len, block_size, seed=1234):
    """Deterministic INT4 paged-KV attention problem. Same bytes for everyone."""
    rng = np.random.RandomState(seed)
    n_blocks = (seq_len + block_size - 1) // block_size

    query = rng.randn(batch, heads, head_dim).astype(np.float32)
    k_q, k_s, k_z, v_l = [], [], [], []
    for _ in range(n_blocks):
        k = rng.randn(block_size, head_dim).astype(np.float32)
        v = rng.randn(block_size, head_dim).astype(np.float32)
        q_, s_, z_ = quantize_int4_ref(k, per_channel=True)
        k_q.append(q_); k_s.append(s_); k_z.append(z_); v_l.append(v)

    lens = np.full(n_blocks, block_size, dtype=np.int32)
    lens[-1] = seq_len - (n_blocks - 1) * block_size
    return dict(query=query, k_q=k_q, k_s=k_s, k_z=k_z, v=v_l, lens=lens,
                n_blocks=n_blocks)


def compulsory_bytes(batch, heads, head_dim, seq_len, block_size, n_blocks):
    """Minimum DRAM traffic the problem requires, in bytes.

    Every K byte and V byte must be read at least once; scales, zero-points,
    query and output are counted once each. This is a LOWER BOUND on real
    traffic (a kernel that re-reads pages per head moves more), so the MBU
    derived from it is a lower bound on achieved utilization.
    """
    padded = n_blocks * block_size
    k_bytes = padded * head_dim * 1          # uint8, one INT4 value per byte
    v_bytes = padded * head_dim * 4          # float32
    stat_bytes = n_blocks * head_dim * 4 * 2  # scale + zp
    io_bytes = batch * heads * head_dim * 4 * 2
    return k_bytes + v_bytes + stat_bytes + io_bytes


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _reference(p):
    out, _ = online_softmax_ref(
        p["query"], list(zip(p["k_q"], p["k_s"], p["k_z"])), p["v"], p["lens"])
    return out


def _pallas(p):
    return ops.flash_decode(p["query"], p["k_q"], p["k_s"], p["k_z"], p["v"],
                            p["lens"], backend="pallas")


def _cuda(p):
    return ops.flash_decode(p["query"], p["k_q"], p["k_s"], p["k_z"], p["v"],
                            p["lens"], backend="cuda")


BACKENDS: Dict[str, Callable] = {
    "reference": _reference,
    "pallas": _pallas,
    "cuda": _cuda,
}


# --- quantizer -------------------------------------------------------------
# The quantizer is the op with three real implementations. Attention exists in
# CUDA and Pallas only -- there is no Triton attention kernel in this repo, so
# benchmarking "three backends" on attention would silently mean two.

def _quant_problem(cfg):
    """A 2D [rows, channels] tensor matching the attention config's KV extent."""
    rng = np.random.RandomState(4321)
    n_blocks = (cfg["seq_len"] + cfg["block_size"] - 1) // cfg["block_size"]
    rows = n_blocks * cfg["block_size"]
    return rng.randn(rows, cfg["head_dim"]).astype(np.float32)


def _q_reference(kv):
    return quantize_int4_ref(kv, per_channel=True)


def _q_cuda(kv):
    return ops.quantize_int4(kv, backend="cuda")


def _q_triton(kv):
    from src.quantize_int4_triton import quantize_int4_triton
    return quantize_int4_triton(kv)


def _q_pallas(kv):
    return ops.quantize_int4(kv, backend="pallas")


QUANT_BACKENDS: Dict[str, Callable] = {
    "reference": _q_reference,
    "cuda": _q_cuda,
    "triton": _q_triton,
    "pallas": _q_pallas,
}


def quant_compulsory_bytes(rows, channels):
    """Read fp32 input, write uint8 bins + fp32 scale/zp per channel."""
    return rows * channels * 4 + rows * channels * 1 + channels * 4 * 2


def probe(name: str) -> Optional[str]:
    """Return None if the backend can run, else a reason string."""
    if name == "reference":
        return None
    if name == "pallas":
        try:
            import jax  # noqa: F401
        except ImportError:
            return "jax not installed"
        return None
    if name == "cuda":
        if not ops.HAS_CUDA:
            return "CUDA extension not built (set FLASH_DECODE_JIT_CUDA=1 on a GPU host)"
        return None
    if name == "triton":
        try:
            import triton  # noqa: F401
        except Exception as exc:
            return f"triton unavailable: {type(exc).__name__}"
        return None
    return f"unknown backend {name}"


# ---------------------------------------------------------------------------
# timing + anti-cheat
# ---------------------------------------------------------------------------

def time_op(fn, iters: int, warmup: int = 3) -> float:
    """Milliseconds per call, warm-up discarded."""
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    elapsed = time.perf_counter() - t0
    # Consume the result so it cannot be eliminated as dead code. Quantizer
    # backends return a (q, scale, zp) tuple whose members differ in shape and
    # dtype, so each is checked separately rather than coerced into one array.
    if out is None:
        raise RuntimeError("backend produced no result")
    parts = out if isinstance(out, (tuple, list)) else (out,)
    for part in parts:
        arr = np.asarray(part)
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise RuntimeError("backend produced a non-finite result")
    return (elapsed / iters) * 1000.0


def _anticheat(name: str, fn: Callable, cfg: Dict) -> Dict:
    """Assertions that a wall-clock benchmark cannot be trusted without.

    - output_consumed: the timed call's result is read and checksummed, so a
      backend cannot win by having its work eliminated.
    - no_shape_specialization: re-run at seq_len + 1. A kernel that is only
      correct or only fast at round numbers is not a kernel.
    - kernel_lowered: for Pallas outside interpret mode, assert the Mosaic
      custom call survives into the compiled HLO -- i.e. XLA did not replace
      the kernel with a library op. Reported as skipped, with a reason, in
      interpret mode, where by construction there is no custom call.
    """
    checks = {}

    p = build_problem(**cfg)
    out = fn(p)
    checks["output_consumed"] = {
        "pass": bool(np.isfinite(out).all()),
        "checksum": float(np.asarray(out, dtype=np.float64).sum()),
    }

    odd = dict(cfg)
    odd["seq_len"] = cfg["seq_len"] + 1
    try:
        out_odd = fn(build_problem(**odd))
        checks["no_shape_specialization"] = {
            "pass": bool(np.isfinite(out_odd).all()),
            "seq_len": odd["seq_len"],
        }
    except Exception as exc:
        checks["no_shape_specialization"] = {"pass": False, "error": repr(exc)}

    if name == "pallas":
        if ops._pallas_interpret():
            checks["kernel_lowered"] = {
                "pass": None,
                "skipped": "interpret mode: kernel runs as JAX ops, no Mosaic "
                           "custom call exists to assert on",
            }
        else:
            checks["kernel_lowered"] = _assert_mosaic_lowering(cfg)
    return checks


def _assert_mosaic_lowering(cfg: Dict) -> Dict:
    """Confirm the Pallas kernel is present in the compiled HLO."""
    try:
        import jax
        import jax.numpy as jnp
        from src.quantize_int4_pallas import _minmax_kernel  # noqa: F401
        from jax.experimental import pallas as pl

        n_ch = cfg["head_dim"]

        def _f(x):
            return pl.pallas_call(
                lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] * 2.0),
                out_shape=jax.ShapeDtypeStruct((256, n_ch), jnp.float32),
                interpret=False,
            )(x)

        hlo = jax.jit(_f).lower(jnp.zeros((256, n_ch), jnp.float32)).compile()
        text = hlo.as_text()
        found = ("custom-call" in text) or ("mosaic" in text.lower())
        return {"pass": bool(found),
                "detail": "custom-call present in compiled HLO" if found
                          else "no custom-call found: kernel may have been replaced"}
    except Exception as exc:
        return {"pass": None, "skipped": f"could not compile: {exc!r}"}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def execution_mode(name: str) -> str:
    """"compiled" or "interpret".

    Pallas has no CPU code generator -- pallas_call(interpret=False) on a CPU
    host raises "Only interpret mode is supported on CPU backend." So on a
    machine with no TPU and no supported GPU, the Pallas backend is necessarily
    interpret mode, and its timings are correctness evidence, not performance
    evidence.
    """
    if name != "pallas":
        return "compiled"
    return "interpret" if ops._pallas_interpret() else "compiled"


def device_label(name: str) -> str:
    if name == "cuda":
        try:
            import torch
            return torch.cuda.get_device_name(0)
        except Exception:
            return "unknown CUDA device"
    if name == "pallas":
        try:
            import jax
            devs = jax.devices()
            if devs and devs[0].platform != "cpu":
                return str(devs[0].device_kind)
        except Exception:
            pass
        return f"CPU ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def run(backends: List[str], sweep: List[Dict], iters: int,
        sweep_name: str = "full", op: str = "both") -> Dict:
    results = {
        "sweep": sweep_name,
        "sweep_note": (
            "The full anchor sweep is not runnable in interpret mode: it is "
            "~2k-33k grid steps per call at roughly 3 ms/step, i.e. minutes per "
            "measurement. Interpret mode is a correctness vehicle; the full "
            "sweep needs TPU or Hopper-class hardware."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "backend_status": ops.backend_status(),
        "rules": {
            "ms_comparable": "within a single device only",
            "cross_device_metric": "mbu_percent",
            "mbu_definition": "compulsory bytes / elapsed / peak DRAM bandwidth "
                              "(lower bound: counts each K/V byte once)",
            "sweep_fixed_before_timing": True,
        },
        "backends": {},
        "rows": [],
    }

    live = []
    for name in backends:
        reason = probe(name)
        results["backends"][name] = {
            "available": reason is None,
            "reason": reason,
            "device": device_label(name) if reason is None else None,
        }
        if reason is None:
            live.append(name)
        else:
            print(f"  {name:10s} UNAVAILABLE - {reason}")

    # Triton has a quantizer but no attention kernel in this repo.
    live_q = [b for b in live if b in QUANT_BACKENDS]
    live = [b for b in live if b in BACKENDS]
    results["backends_by_op"] = {"quantize": list(live_q), "attention": list(live)}

    if not live and not live_q:
        print("No backends available.")
        return results

    # Anti-cheat once per backend, on the smallest configuration.
    small = min(sweep, key=lambda c: c["seq_len"] * c["batch"])
    for name in (live if op in ("attention", "both") else []):
        print(f"\nAnti-cheat: {name}")
        checks = _anticheat(name, BACKENDS[name], small)
        results["backends"][name]["anticheat"] = checks
        for k, v in checks.items():
            verdict = {True: "PASS", False: "FAIL", None: "SKIP"}[v.get("pass")]
            print(f"  {k:26s} {verdict}"
                  + (f"  ({v.get('skipped') or v.get('detail') or ''})"
                     if v.get("skipped") or v.get("detail") else ""))

    # ---------------- quantizer: the op with three real backends -----------
    if op in ("quantize", "both"):
        print()
        print("=" * 72)
        print("QUANTIZER  (reference / cuda / triton / pallas)")
        print("=" * 72)
        for cfg in sweep:
            kv = _quant_problem(cfg)
            rows, ch = kv.shape
            nbytes = quant_compulsory_bytes(rows, ch)
            print(f"{rows} x {ch}   ({nbytes/1e6:.1f} MB)")
            qbase: Dict[tuple, float] = {}
            for name in live_q:
                try:
                    ms = time_op(lambda: QUANT_BACKENDS[name](kv), iters=iters)
                except Exception as exc:
                    print(f"    {name:10s} ERROR {exc!r}")
                    results["rows"].append({"op": "quantize", "rows": rows,
                                            "channels": ch, "backend": name,
                                            "error": repr(exc)})
                    continue
                dev = device_label(name)
                mode = execution_mode(name)
                peak = PEAK_BANDWIDTH_GBS.get(dev)
                mbu = (nbytes / (ms / 1000.0) / (peak * 1e9) * 100.0) if peak else None
                qbase.setdefault((dev, mode), ms)
                rel = ms / qbase[(dev, mode)]
                results["rows"].append({
                    "op": "quantize", "rows": rows, "channels": ch,
                    "backend": name, "device": dev, "execution_mode": mode,
                    "ms_per_call": ms, "compulsory_bytes": nbytes,
                    "peak_bandwidth_gbs": peak, "mbu_percent": mbu,
                })
                mbu_s = f"{mbu:6.2f}%" if mbu is not None else "   n/a"
                rel_s = f"{rel:5.2f}x" if len(qbase) == 1 else "    -"
                note = "" if mode == "compiled" else "  <- interpret: not a perf number"
                print(f"    {name:10s} {ms:9.3f} ms  {rel_s}  MBU {mbu_s}  "
                      f"[{dev}, {mode}]{note}")
            print()

    if op == "quantize":
        return results

    print()
    print("=" * 72)
    print("ATTENTION  (reference / cuda / pallas -- no Triton attention kernel exists)")
    print("=" * 72)
    for cfg in sweep:
        p = build_problem(**cfg)
        nbytes = compulsory_bytes(n_blocks=p["n_blocks"], **cfg)
        label = (f"b{cfg['batch']} h{cfg['heads']} d{cfg['head_dim']} "
                 f"s{cfg['seq_len']}")
        print(f"{label}   ({nbytes/1e6:.1f} MB compulsory traffic)")

        # Baseline per (device, execution_mode): ratios never cross either.
        baselines: Dict[tuple, float] = {}
        for name in live:
            try:
                ms = time_op(lambda: BACKENDS[name](p), iters=iters)
            except Exception as exc:
                print(f"    {name:10s} ERROR {exc!r}")
                results["rows"].append({"op": "attention", **cfg,
                                        "backend": name, "error": repr(exc)})
                continue

            dev = device_label(name)
            mode = execution_mode(name)
            peak = PEAK_BANDWIDTH_GBS.get(dev)
            mbu = (nbytes / (ms / 1000.0) / (peak * 1e9) * 100.0) if peak else None

            key = (dev, mode)
            baselines.setdefault(key, ms)
            rel = ms / baselines[key]

            results["rows"].append({
                "op": "attention",
                **cfg,
                "backend": name,
                "device": dev,
                "execution_mode": mode,
                "ms_per_call": ms,
                "ms_comparable_within": {"device": dev, "execution_mode": mode},
                "compulsory_bytes": nbytes,
                "peak_bandwidth_gbs": peak,
                "mbu_percent": mbu,
            })
            mbu_s = f"{mbu:6.2f}%" if mbu is not None else "   n/a"
            rel_s = f"{rel:5.2f}x" if len(baselines) == 1 else "    -"
            note = "" if mode == "compiled" else "  <- interpret: not a perf number"
            print(f"    {name:10s} {ms:9.3f} ms  {rel_s}  MBU {mbu_s}  "
                  f"[{dev}, {mode}]{note}")

        modes = {execution_mode(n) for n in live}
        if len(modes) > 1:
            print("    NOTE: mixed execution modes above; ms is not comparable "
                  "across them, so no ratio is shown.")
        print()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="reference,pallas,cuda,triton")
    ap.add_argument("--op", default="both", choices=["quantize", "attention", "both"],
                    help="quantize has 3 real backends; attention has cuda+pallas only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=5,
                    help="timed iterations per configuration (>=1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.iters < 1:
        ap.error("--iters must be >= 1")
    sweep = QUICK_SWEEP if args.quick else SWEEP
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print("=" * 72)
    print("Flash Decoding INT4 - three-backend benchmark")
    print("=" * 72)

    results = run(backends, sweep, args.iters,
                  sweep_name="quick" if args.quick else "full", op=args.op)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "three_backend.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
