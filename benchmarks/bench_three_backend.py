"""Three-backend benchmark: reference / triton / cuda / pallas on identical inputs.

Design rules, fixed before any backend was timed:

1.  Every backend receives byte-for-byte identical inputs, built from a fixed
    seed. Inputs are generated once per shape and reused.
2.  Only the kernel is timed. Inputs are validated, padded and copied to the
    device ONCE, outside the timed region (see src/_prepared.py); the loop
    launches back-to-back on device-resident data and synchronises once at the
    end. Timing the end-to-end call instead would mostly measure host-side
    padding and PCIe -- 26% of the CUDA row even on a T1000, and the large
    majority on an H100, where the kernel is ~20x faster but the bus is not.
3.  Absolute milliseconds are only ever compared WITHIN one device and one
    execution mode. Two chips are not comparable in ms; neither is an
    interpreter and a compiler.
4.  Bandwidth is reported two ways (see attention_bytes), and only one of
    them is called a utilization:
      mbu_unique_percent  -- each page byte counted once, against the
                             spec-sheet DRAM peak. A true utilization, and a
                             lower bound on DRAM traffic.
      streamed_read_gbps  -- every byte every (batch, head) program reads, per
                             second. NOT a utilization: all programs here share
                             one page stream, so L2 serves repeats and this can
                             exceed the DRAM peak (measured: 142% of peak at
                             s512 on a T1000). Rows where it does are flagged.
    Where a device's peak is unknown, the utilization is null, not guessed.
5.  A backend that cannot run is recorded with a reason, never silently
    omitted -- an absent row and a failed row must not look the same.
6.  The shape sweep below was fixed in advance and is not edited after seeing
    results.

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
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref
from src._prepared import PreparedCall
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

#: Spec-sheet peak DRAM bandwidth, GB/s, matched as case-insensitive
#: substrings of the device name, MOST SPECIFIC FIRST ("H100 PCIe" must win
#: over "H100"). Anything unmatched yields MBU = null rather than a guess.
PEAK_BANDWIDTH_GBS = [
    ("H100 NVL", 3900.0),
    ("H100 PCIe", 2000.0),
    ("H100", 3350.0),            # SXM5 80GB HBM3: "NVIDIA H100 80GB HBM3"
    ("A100-SXM4-80GB", 2039.0),
    ("A100 80GB PCIe", 1935.0),
    ("A100-SXM4-40GB", 1555.0),
    ("A100-PCIE-40GB", 1555.0),
    ("L40S", 864.0),
    ("T1000", 160.0),
    ("TPU v5 lite", 819.0),      # v5e as reported by jax device_kind
    ("TPU v5e", 819.0),
    ("TPU v6 lite", 1640.0),     # v6e (Trillium)
    ("TPU v5p", 2765.0),
    ("TPU v4", 1200.0),
]


def peak_bandwidth_gbs(device: str) -> Optional[float]:
    d = (device or "").lower()
    for needle, gbs in PEAK_BANDWIDTH_GBS:
        if needle.lower() in d:
            return gbs
    return None


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


def attention_bytes(batch, heads, head_dim, seq_len, block_size, n_blocks):
    """Bytes moved by one attention call, under two models.

    unique   -- each K/V page byte counted once, plus scales, query and output.
    streamed -- every (batch, head) program streams the full page sequence, as
                the CUDA and Pallas kernels do; equal to the traffic of a real
                decode step in which each head has its own KV of this length.
    """
    padded = n_blocks * block_size
    page_stream = (padded * head_dim * 1          # uint8 keys, one value/byte
                   + padded * head_dim * 4        # float32 values
                   + n_blocks * head_dim * 4 * 2)  # scale + zero-point
    io_bytes = batch * heads * head_dim * 4 * 2
    return {"unique": page_stream + io_bytes,
            "streamed": batch * heads * page_stream + io_bytes}


def _quant_problem(cfg):
    """A 2D [rows, channels] tensor matching the attention config's KV extent."""
    rng = np.random.RandomState(4321)
    n_blocks = (cfg["seq_len"] + cfg["block_size"] - 1) // cfg["block_size"]
    rows = n_blocks * cfg["block_size"]
    return rng.randn(rows, cfg["head_dim"]).astype(np.float32)


def quant_bytes(rows, channels):
    """Read fp32 input, write uint8 bins + fp32 scale/zp per channel."""
    return rows * channels * 4 + rows * channels * 1 + channels * 4 * 2


# ---------------------------------------------------------------------------
# backends
#
# Two views of each backend. The dispatch view (ops.*) is what users call and
# what the anti-cheat checks for correctness. The staged view (prepare_*) is
# what gets timed. timed_path_matches_dispatch asserts the two compute the same
# thing, so the timed path cannot quietly drift from the validated one.
# ---------------------------------------------------------------------------

def _reference(p):
    out, _ = online_softmax_ref(
        p["query"], list(zip(p["k_q"], p["k_s"], p["k_z"])), p["v"], p["lens"])
    return out


def _dispatch_attention(name):
    if name == "reference":
        return _reference
    return lambda p: ops.flash_decode(p["query"], p["k_q"], p["k_s"], p["k_z"],
                                      p["v"], p["lens"], backend=name)


def stage_attention(name: str, p: Dict) -> PreparedCall:
    if name == "reference":
        return PreparedCall(run=lambda: _reference(p))
    if name == "cuda":
        return ops.prepare_flash_decode_cuda(
            p["query"], p["k_q"], p["k_s"], p["k_z"], p["v"], p["lens"])
    if name == "pallas":
        from src.flash_decode_pallas import prepare_flash_decode_pallas
        return prepare_flash_decode_pallas(
            p["query"], p["k_q"], p["k_s"], p["k_z"], p["v"], p["lens"],
            interpret=ops._pallas_interpret())
    raise ValueError(f"no attention kernel for backend {name!r}")


def stage_quantize(name: str, kv: np.ndarray) -> PreparedCall:
    if name == "reference":
        return PreparedCall(run=lambda: quantize_int4_ref(kv, per_channel=True))
    if name == "cuda":
        return ops.prepare_quantize_int4_cuda(kv)
    if name == "triton":
        from src.quantize_int4_triton import prepare_quantize_int4_triton
        return prepare_quantize_int4_triton(kv)
    if name == "pallas":
        from src.quantize_int4_pallas import prepare_quantize_int4_pallas
        return prepare_quantize_int4_pallas(kv, interpret=ops._pallas_interpret())
    raise ValueError(f"no quantizer for backend {name!r}")


ATTENTION_NAMES = ("reference", "cuda", "pallas")       # no Triton attention kernel
QUANTIZE_NAMES = ("reference", "cuda", "triton", "pallas")


def probe(name: str) -> Optional[str]:
    """Return None if the backend can run, else a reason string."""
    if name == "reference":
        return None
    if name == "pallas":
        return None if ops.HAS_JAX else "jax not installed"
    if name == "cuda":
        if not ops.HAS_CUDA:
            return "CUDA extension not built (set FLASH_DECODE_JIT_CUDA=1 on a GPU host)"
        return None
    if name == "triton":
        try:
            from src.quantize_int4_triton import HAS_TRITON
        except Exception as exc:
            return f"triton unavailable: {type(exc).__name__}"
        return None if HAS_TRITON else "triton not installed"
    return f"unknown backend {name}"


# ---------------------------------------------------------------------------
# timing + anti-cheat
# ---------------------------------------------------------------------------

def _check_finite(result) -> None:
    if result is None:
        raise RuntimeError("backend produced no result")
    parts = result if isinstance(result, (tuple, list)) else (result,)
    for part in parts:
        arr = np.asarray(part)
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise RuntimeError("backend produced a non-finite result")


def time_op(call: PreparedCall, iters: int, warmup: int = 3) -> float:
    """Milliseconds per launch on device-resident inputs.

    Back-to-back launches with one device sync after the loop: the throughput a
    serving loop sees. The output of the last launch is copied to host and
    checked AFTER the clock stops, so the work cannot be optimised away and the
    copy is not billed to the kernel.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    for _ in range(warmup):
        call.sync(call.run())
    t0 = time.perf_counter()
    for _ in range(iters):
        out = call.run()
    call.sync(out)
    elapsed = time.perf_counter() - t0
    _check_finite(call.to_numpy(out))
    return (elapsed / iters) * 1000.0


def _anticheat(name: str, cfg: Dict) -> Dict:
    """Assertions a wall-clock benchmark cannot be trusted without.

    - output_consumed: the timed result is copied back and checksummed, so a
      backend cannot win by having its work eliminated.
    - timed_path_matches_dispatch: the staged (timed) path and the public
      ops.flash_decode path agree. Guards against timing a different
      computation from the one the tests validate.
    - no_shape_specialization: re-run at seq_len + 1. A kernel that is only
      correct or only fast at round numbers is not a kernel.
    - kernel_lowered: for Pallas outside interpret mode, the custom call must
      survive into the compiled HLO of the ACTUAL kernel -- XLA must not have
      replaced it with a library op. Skipped, with the reason, in interpret
      mode, where by construction there is no custom call.
    """
    checks = {}
    p = build_problem(**cfg)
    staged = stage_attention(name, p)
    timed_out = np.asarray(staged())
    checks["output_consumed"] = {
        "pass": bool(np.isfinite(timed_out).all()),
        "checksum": float(timed_out.astype(np.float64).sum()),
    }

    dispatch_out = np.asarray(_dispatch_attention(name)(p))
    diff = float(np.abs(timed_out - dispatch_out).max())
    checks["timed_path_matches_dispatch"] = {"pass": diff < 1e-5, "max_abs_diff": diff}

    odd = dict(cfg)
    odd["seq_len"] = cfg["seq_len"] + 1
    try:
        out_odd = np.asarray(stage_attention(name, build_problem(**odd))())
        checks["no_shape_specialization"] = {
            "pass": bool(np.isfinite(out_odd).all()), "seq_len": odd["seq_len"]}
    except Exception as exc:
        checks["no_shape_specialization"] = {"pass": False, "error": repr(exc)}

    if name == "pallas":
        if ops._pallas_interpret() or staged.lowered_text is None:
            checks["kernel_lowered"] = {
                "pass": None,
                "skipped": "interpret mode: kernel runs as JAX ops, no custom "
                           "call exists to assert on",
            }
        else:
            try:
                text = staged.lowered_text()
                found = "custom-call" in text or "custom_call" in text
                checks["kernel_lowered"] = {
                    "pass": bool(found),
                    "detail": "custom call present in the compiled HLO of the "
                              "attention kernel" if found else
                              "no custom call: the kernel may have been replaced"}
            except Exception as exc:
                checks["kernel_lowered"] = {"pass": False,
                                            "error": f"could not lower: {exc!r}"}
    return checks


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def execution_mode(name: str) -> str:
    """"compiled" or "interpret".

    Pallas has no CPU code generator -- pallas_call(interpret=False) on a CPU
    host raises "Only interpret mode is supported on CPU backend." -- and
    Triton under TRITON_INTERPRET=1 runs its interpreter. Timings in either
    mode are correctness evidence, not performance evidence.
    """
    if name == "pallas":
        return "interpret" if ops._pallas_interpret() else "compiled"
    if name == "triton" and os.environ.get("TRITON_INTERPRET") == "1":
        return "interpret"
    return "compiled"


def _cpu_label() -> str:
    return f"CPU ({platform.processor() or platform.machine()})"


def device_label(name: str) -> str:
    if name == "cuda" or (name == "triton" and execution_mode("triton") == "compiled"):
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "unknown CUDA device"
    if name == "pallas":
        try:
            import jax
            devs = jax.devices()
            if devs and devs[0].platform != "cpu":
                return str(devs[0].device_kind)
        except Exception:
            pass
    return _cpu_label()


def _row(op, extra, name, ms, byte_models):
    dev = device_label(name)
    peak = peak_bandwidth_gbs(dev)
    secs = ms / 1000.0
    row = {"op": op, **extra, "backend": name, "device": dev,
           "execution_mode": execution_mode(name), "ms_per_call": ms,
           "peak_bandwidth_gbs": peak}
    unique = byte_models["unique"]
    row["unique_bytes"] = unique
    row["mbu_unique_percent"] = unique / secs / (peak * 1e9) * 100.0 if peak else None
    if "streamed" in byte_models:
        gbps = byte_models["streamed"] / secs / 1e9
        row["streamed_bytes"] = byte_models["streamed"]
        row["streamed_read_gbps"] = gbps
        row["streamed_exceeds_dram_peak"] = (gbps > peak) if peak else None
    return row


def _print_row(row, baselines):
    key = (row["device"], row["execution_mode"])
    baselines.setdefault(key, row["ms_per_call"])
    rel = row["ms_per_call"] / baselines[key]
    rel_s = f"{rel:5.2f}x" if len(baselines) == 1 else "    -"
    mbu = row["mbu_unique_percent"]
    parts = [f"MBU(unique) {mbu:5.2f}%" if mbu is not None else "MBU(unique) n/a"]
    if "streamed_read_gbps" in row:
        flag = "  >DRAM peak: L2 reuse" if row["streamed_exceeds_dram_peak"] else ""
        parts.append(f"streamed {row['streamed_read_gbps']:7.1f} GB/s{flag}")
    note = "" if row["execution_mode"] == "compiled" else "  <- interpret: not a perf number"
    print(f"    {row['backend']:10s} {row['ms_per_call']:9.3f} ms  {rel_s}  "
          f"{'  '.join(parts)}  [{row['device']}, {row['execution_mode']}]{note}")


def run(backends: List[str], sweep: List[Dict], iters: int,
        sweep_name: str = "full", op: str = "both") -> Dict:
    results = {
        "sweep": sweep_name,
        "sweep_note": (
            "Pallas in interpret mode cannot run the full anchor sweep in useful "
            "time (thousands of grid steps per call, each a few ms); interpret "
            "mode is a correctness vehicle. The full sweep needs TPU or Hopper."
        ),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "backend_status": ops.backend_status(),
        "rules": {
            "timed_region": "kernel launches on device-resident inputs only; "
                            "staging and copies excluded (src/_prepared.py)",
            "ms_comparable": "within a single device and execution mode only",
            "cross_device_metric": "mbu_unique_percent (true utilization, "
                                   "lower bound on DRAM traffic)",
            "streamed_read_gbps": "throughput, not utilization; can exceed DRAM "
                                  "peak because programs share one page stream "
                                  "and L2 serves repeats",
            "peak_bandwidth": "vendor spec sheet",
            "sweep_fixed_before_timing": True,
        },
        "backends": {},
        "rows": [],
    }

    live = []
    for name in backends:
        reason = probe(name)
        results["backends"][name] = {
            "available": reason is None, "reason": reason,
            "device": device_label(name) if reason is None else None,
            "execution_mode": execution_mode(name) if reason is None else None,
        }
        if reason is None:
            live.append(name)
        else:
            print(f"  {name:10s} UNAVAILABLE - {reason}")

    live_q = [b for b in live if b in QUANTIZE_NAMES]
    live_a = [b for b in live if b in ATTENTION_NAMES]
    results["backends_by_op"] = {"quantize": live_q, "attention": live_a}
    if not live_q and not live_a:
        print("No backends available.")
        return results

    if op in ("attention", "both"):
        small = min(sweep, key=lambda c: c["seq_len"] * c["batch"])
        for name in live_a:
            print(f"\nAnti-cheat: {name}")
            # A backend that cannot even run (e.g. Mosaic GPU rejecting the
            # Pallas kernel) must not take the other backends' rows down.
            try:
                checks = _anticheat(name, small)
            except Exception as exc:
                checks = {"anticheat": {"pass": False, "error": repr(exc)[:500]}}
            results["backends"][name]["anticheat"] = checks
            for k, v in checks.items():
                verdict = {True: "PASS", False: "FAIL", None: "SKIP"}[v.get("pass")]
                detail = v.get("skipped") or v.get("detail") or v.get("error") or ""
                print(f"  {k:28s} {verdict}" + (f"  ({detail})" if detail else ""))

    if op in ("quantize", "both"):
        print("\n" + "=" * 72 + "\nQUANTIZER  (reference / cuda / triton / pallas)\n" + "=" * 72)
        for cfg in sweep:
            kv = _quant_problem(cfg)
            rows_, ch = kv.shape
            print(f"{rows_} x {ch}")
            baselines: Dict[tuple, float] = {}
            for name in live_q:
                try:
                    ms = time_op(stage_quantize(name, kv), iters=iters)
                except Exception as exc:
                    print(f"    {name:10s} ERROR {exc!r}")
                    results["rows"].append({"op": "quantize", "rows": rows_,
                                            "channels": ch, "backend": name,
                                            "error": repr(exc)})
                    continue
                row = _row("quantize", {"rows": rows_, "channels": ch}, name, ms,
                           {"unique": quant_bytes(rows_, ch)})
                results["rows"].append(row)
                _print_row(row, baselines)
            print()

    if op in ("attention", "both"):
        print("\n" + "=" * 72 + "\nATTENTION  (reference / cuda / pallas -- no Triton "
              "attention kernel exists)\n" + "=" * 72)
        for cfg in sweep:
            p = build_problem(**cfg)
            nbytes = attention_bytes(n_blocks=p["n_blocks"], **cfg)
            print(f"b{cfg['batch']} h{cfg['heads']} d{cfg['head_dim']} s{cfg['seq_len']}"
                  f"   (unique {nbytes['unique']/1e6:.1f} MB,"
                  f" streamed {nbytes['streamed']/1e6:.1f} MB)")
            baselines = {}
            for name in live_a:
                try:
                    ms = time_op(stage_attention(name, p), iters=iters)
                except Exception as exc:
                    print(f"    {name:10s} ERROR {exc!r}")
                    results["rows"].append({"op": "attention", **cfg,
                                            "backend": name, "error": repr(exc)})
                    continue
                row = _row("attention", dict(cfg), name, ms, nbytes)
                results["rows"].append(row)
                _print_row(row, baselines)
            if len({execution_mode(n) for n in live_a}) > 1:
                print("    NOTE: mixed execution modes; ms is not comparable across "
                      "them, so no ratio is shown.")
            print()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="reference,pallas,cuda,triton")
    ap.add_argument("--op", default="both", choices=["quantize", "attention", "both"],
                    help="quantize has 3 real backends; attention has cuda+pallas only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=20,
                    help="timed iterations per configuration (>=1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.iters < 1:
        ap.error("--iters must be >= 1")
    sweep = QUICK_SWEEP if args.quick else SWEEP
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print("=" * 72 + "\nFlash Decoding INT4 - three-backend benchmark\n" + "=" * 72)
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
