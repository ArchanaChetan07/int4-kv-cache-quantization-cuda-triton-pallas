"""Does the Pallas port actually lower on this accelerator?

Both Pallas kernels in this repository have only ever executed under
`interpret=True`, which runs them as ordinary JAX ops and enforces none of the
constraints a real backend imposes. This script is the lowering attempt.

It is deliberately a SEPARATE script from the benchmark. The lowering attempt is
the highest-uncertainty step in the whole plan, and if Mosaic GPU rejects these
kernels that must not take down the CUDA and Triton measurements running beside
it. A failure here is a publishable result, not an outage.

Each probe reports one of:

    LOWERED   compiled, and matched the interpret-mode result numerically
    MISMATCH  compiled, but disagreed with interpret mode -- the interesting
              failure, and the one grid-semantics bugs produce
    REJECTED  the compiler refused it; the error text IS the finding
    SKIPPED   no accelerator, or a dependency is missing

    python scripts/pallas_lowering_probe.py
    python scripts/pallas_lowering_probe.py --json results/lowering_probe.json
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref


def _device_info():
    try:
        import jax
        devs = jax.devices()
        return {"platform": devs[0].platform,
                "kind": devs[0].device_kind,
                "count": len(devs),
                "jax": jax.__version__}
    except Exception as exc:
        return {"error": repr(exc)}


def probe_quantizer():
    """Lower the two-pass quantizer and compare bins against interpret mode."""
    from src.quantize_int4_pallas import quantize_int4_pallas

    kv = np.random.RandomState(0).randn(1024, 128).astype(np.float32)

    q_i, s_i, z_i = quantize_int4_pallas(kv, interpret=True)
    try:
        q_c, s_c, z_c = quantize_int4_pallas(kv, interpret=False)
    except Exception as exc:
        return {"status": "REJECTED", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}

    bin_diff = int(np.abs(q_c.astype(int) - q_i.astype(int)).max())
    scale_close = bool(np.allclose(s_c, s_i, rtol=1e-4))

    # also check against the oracle, since interpret mode is not ground truth
    q_r, s_r, _ = quantize_int4_ref(kv, per_channel=True)
    oracle_diff = int(np.abs(q_c.astype(int) - q_r.astype(int)).max())
    oracle_rate = float((np.abs(q_c.astype(int) - q_r.astype(int)) > 0).mean())

    ok = bin_diff == 0 and scale_close
    return {
        "status": "LOWERED" if ok else "MISMATCH",
        "max_bin_diff_vs_interpret": bin_diff,
        "scales_match_interpret": scale_close,
        "max_bin_diff_vs_oracle": oracle_diff,
        "bin_disagreement_rate_vs_oracle": oracle_rate,
    }


def probe_attention():
    """Lower the fused attention kernel.

    This is the probe that matters. The accumulators live in resident output
    blocks whose index_map ignores the block axis, which is only correct if that
    axis executes sequentially. On GPU that requires the dimension_semantics
    declaration in flash_decode_pallas._compiler_params; if that declaration is
    wrong or ignored, this probe returns MISMATCH rather than REJECTED, and the
    error is a race rather than a compile failure.
    """
    from src.flash_decode_pallas import flash_decode_pallas

    rng = np.random.RandomState(7)
    batch, heads, dim, n_blocks, block = 2, 8, 128, 6, 256
    query = rng.randn(batch, heads, dim).astype(np.float32)
    k_q, k_s, k_z, v_l = [], [], [], []
    for _ in range(n_blocks):
        k = rng.randn(block, dim).astype(np.float32)
        v = rng.randn(block, dim).astype(np.float32)
        a, s, z = quantize_int4_ref(k, per_channel=True)
        k_q.append(a); k_s.append(s); k_z.append(z); v_l.append(v)
    lens = np.full(n_blocks, block, dtype=np.int32)
    lens[2] = 0          # empty page: exercises the -inf guard
    lens[-1] = block // 3  # ragged tail

    out_i = flash_decode_pallas(query, k_q, k_s, k_z, v_l, lens, interpret=True)
    try:
        out_c = flash_decode_pallas(query, k_q, k_s, k_z, v_l, lens, interpret=False)
    except Exception as exc:
        return {"status": "REJECTED", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}

    ref, _ = online_softmax_ref(query, list(zip(k_q, k_s, k_z)), v_l, lens)
    d_interp = float(np.abs(out_c - out_i).max())
    d_ref = float(np.abs(out_c - ref).mean())
    has_nan = bool(np.isnan(out_c).any())

    ok = (not has_nan) and d_interp < 1e-4
    return {
        "status": "LOWERED" if ok else "MISMATCH",
        "max_abs_diff_vs_interpret": d_interp,
        "mae_vs_numpy_reference": d_ref,
        "nan_present": has_nan,
        "note": ("a large diff vs interpret with no compile error is the "
                 "signature of a grid-order race, not a numerics issue")
        if not ok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    info = _device_info()
    print("=" * 74)
    print("Pallas lowering probe")
    print("=" * 74)
    print(f"  device: {info}")
    print()

    results = {"device": info, "probes": {}}

    if info.get("platform", "cpu") == "cpu":
        print("  SKIPPED - no accelerator visible. Pallas has no CPU code")
        print("            generator, so there is nothing to lower here.")
        results["probes"] = {"quantizer": {"status": "SKIPPED"},
                             "attention": {"status": "SKIPPED"}}
    else:
        for name, fn in (("quantizer", probe_quantizer), ("attention", probe_attention)):
            try:
                r = fn()
            except Exception as exc:
                r = {"status": "REJECTED", "error": repr(exc),
                     "traceback": traceback.format_exc()[-2000:]}
            results["probes"][name] = r
            print(f"  {name:10s} {r['status']}")
            for k, v in r.items():
                if k in ("status", "traceback") or v is None:
                    continue
                print(f"      {k}: {v}")
            if r.get("traceback"):
                print("      --- compiler said ---")
                for line in r["traceback"].strip().split("\n")[-12:]:
                    print(f"      {line}")
            print()

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  written to {args.json}")

    statuses = {p.get("status") for p in results["probes"].values()}
    # A REJECTED probe is a legitimate finding, so this script exits 0 unless
    # something went wrong with the probe itself.
    print("  verdict:", ", ".join(sorted(statuses)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
