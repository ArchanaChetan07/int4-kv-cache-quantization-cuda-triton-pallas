"""Regenerate the Pallas-port figures in docs/assets/.

Every value plotted here was measured by running this repository; nothing is
estimated or illustrative. Sources:

  accuracy   tests/test_pallas_flash_decode.py (float64 arbiter), run locally
             on jax 0.4.38 / Windows and in GitHub Actions on Linux with the
             current jax release. The dim-128 Linux figure is deliberately
             absent: that job asserted a bound rather than recording the value,
             so plotting a number for it would be fabrication.

  backends   docs/TRITON_TO_PALLAS.md 5.5 plus the JAX Pallas documentation
             (Mosaic GPU hardware floor, Triton backend deprecation).

  compression  bytes held by INT4PagedKVCache's actual arrays.
  snr          scripts/validate_llama.py --simulate.
  quantizer    results/three_backend_t1000.json (bench_three_backend.py).
  perplexity   results/perplexity_block_sweep_h100.json
               (scripts/perplexity_block_sweep.py).

    python scripts/make_pallas_charts.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "assets")

INK = "#14171F"
MUTED = "#6B7486"
RULE = "#DCE0E8"
TEAL = "#0B6E75"
AMBER = "#C2851B"
RED = "#B02A2A"
GREEN = "#2B7A3D"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


# ---------------------------------------------------------------------------
# 1. accuracy relative to the float32 accumulation floor
# ---------------------------------------------------------------------------

def accuracy_chart():
    """Measured MAE vs a float64 evaluation, expressed against sqrt(D)*eps*|o|.

    The point of the chart: every configuration sits below the gate, and the
    Windows/Linux gap for Pallas is XLA's reduction-order choice, not a defect.
    """
    labels = [
        "NumPy oracle\nhead_dim 64",
        "Pallas\nhead_dim 64\njax 0.4.38",
        "Pallas\nhead_dim 64\njax current",
        "NumPy oracle\nhead_dim 128",
        "Pallas\nhead_dim 128\njax 0.4.38",
    ]
    ratios = [0.40, 0.43, 1.39, 1.16, 0.32]
    colors = [MUTED, TEAL, AMBER, MUTED, TEAL]

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    bars = ax.bar(range(len(ratios)), ratios, color=colors, width=0.62,
                  edgecolor="white", linewidth=1.2, zorder=3)

    ax.axhline(1.0, color=INK, ls="--", lw=1.2, zorder=2)
    ax.text(len(ratios) - 0.42, 1.05, "float32 accumulation floor",
            ha="right", va="bottom", fontsize=8.5, color=INK, style="italic")
    ax.axhline(3.0, color=RED, ls=":", lw=1.4, zorder=2)
    ax.text(-0.45, 3.06, "test gate  (3x floor)",
            ha="left", va="bottom", fontsize=8.5, color=RED, weight="bold")

    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.07, f"{r:.2f}x",
                ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.2, color=INK)
    ax.set_ylabel("error / float32 accumulation floor", fontsize=9.5)
    ax.set_ylim(0, 4.25)
    ax.set_title(
        "Attention accuracy against a float64 arbiter\n"
        "every configuration is normal float32 behaviour; none is a defect",
        fontsize=11.5, color=INK, weight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    _style(ax)

    ax.legend(handles=[
        Patch(facecolor=MUTED, label="NumPy reference oracle"),
        Patch(facecolor=TEAL, label="Pallas, jax 0.4.38 (local)"),
        Patch(facecolor=AMBER, label="Pallas, current jax (CI, Linux)"),
    ], loc="upper center", ncol=3, frameon=False, fontsize=8.5,
       bbox_to_anchor=(0.5, 1.0), columnspacing=1.6)

    fig.tight_layout()
    p = os.path.join(OUT, "pallas_accuracy_floor.png")
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 2. backend x hardware availability
# ---------------------------------------------------------------------------

def backend_matrix():
    """Which kernel backend runs these kernels on which silicon.

    T1000 and H100 entries are measured (the H100 with JAX 0.12.0.1); Ampere
    entries follow the documentation; TPU is expected but untested. On GPU,
    compiled Pallas either is not reached (Mosaic GPU) or races (Triton
    backend) -- results/pallas_compiled_on_gpu_h100.json.
    """
    backends = ["CUDA\n(hand-written)", "Triton", "Pallas\nMosaic TPU",
                "Pallas\nMosaic GPU", "Pallas\nTriton backend"]
    hw = ["CPU\n(no accel)", "T1000\nTuring SM75", "Ampere\nGPU",
          "H100\nHopper", "TPU\nv5e"]

    #  2 runs                 3 expected, untested      1 interpret only
    #  0 not applicable      -1 deprecated (documented)
    # -2 compiles, races (measured)   -3 not reached in jax 0.12 (measured)
    M = np.array([
        [0, 2, 2, 2, 0],    # CUDA
        [0, 2, 2, 2, 0],    # Triton
        [1, 0, 0, 0, 3],    # Pallas Mosaic TPU
        [1, 0, 0, -3, 0],   # Pallas Mosaic GPU
        [1, 0, -1, -2, 0],  # Pallas Triton backend
    ])

    cmap = {2: GREEN, 3: "#9BC9C3", 1: AMBER, 0: "#E8E8EC",
            -1: RED, -2: RED, -3: RED}
    txt = {2: "runs", 3: "expected\n(untested)", 1: "interpret\nonly", 0: "—",
           -1: "deprecated", -2: "compiles,\nraces", -3: "not reached\n(jax 0.12)"}

    fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=160)
    for i in range(len(backends)):
        for j in range(len(hw)):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((j, len(backends) - 1 - i), 1, 1,
                                       facecolor=cmap[v], edgecolor="white", lw=2))
            ax.text(j + 0.5, len(backends) - 1 - i + 0.5, txt[v],
                    ha="center", va="center", fontsize=8.2,
                    color="white" if v in (2, -1, -2, -3) else INK,
                    weight="bold" if v in (2, -1, -2, -3) else "normal")

    ax.set_xlim(0, len(hw)); ax.set_ylim(0, len(backends))
    ax.set_xticks(np.arange(len(hw)) + 0.5)
    ax.set_xticklabels(hw, fontsize=8.6, color=INK)
    ax.set_yticks(np.arange(len(backends)) + 0.5)
    ax.set_yticklabels(backends[::-1], fontsize=8.6, color=INK)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Which backend runs these kernels on which silicon\n"
        "T1000 and H100 measured (jax 0.12.0.1); interpret mode is correct on any device",
        fontsize=11.5, color=INK, weight="bold", loc="left", pad=12)

    fig.tight_layout()
    p = os.path.join(OUT, "pallas_backend_matrix.png")
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 3. key-cache compression, measured from the cache's actual arrays
# ---------------------------------------------------------------------------

def compression_chart():
    """Bytes held by INT4PagedKVCache vs FP16, keys and whole KV.

    Replaces an earlier chart that reported 3.98x using ONE scale set for 4096
    tokens; the kernels use a scale set per 256-token block, which costs more.
    """
    import sys
    sys.path.insert(0, HERE)
    from src.vllm_integration import INT4PagedKVCache

    blocks, bs, d = 16, 256, 128
    cache = INT4PagedKVCache(num_blocks=blocks, block_size=bs, head_dim=d)
    rng = np.random.RandomState(0)
    for b in range(blocks):
        cache.write_block(b, rng.randn(bs, d).astype(np.float32),
                          rng.randn(bs, d).astype(np.float32))
    k_int4 = (sum(x.nbytes for x in cache.k_packed.values())
              + sum(x.nbytes for x in cache.k_scale.values())
              + sum(x.nbytes for x in cache.k_zp.values()))
    v_fp16 = sum(x.nbytes for x in cache.v.values())
    k_fp16 = blocks * bs * d * 2
    mb = 1024 * 1024

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=160)
    for ax, title, labels, vals, colors in (
        (axes[0], "Keys", ["FP16 keys", "INT4 keys\n+ per-block scales"],
         [k_fp16 / mb, k_int4 / mb], [MUTED, TEAL]),
        (axes[1], "Whole KV cache (values stay FP16)", ["FP16 K + V", "INT4 K + FP16 V"],
         [(k_fp16 + v_fp16) / mb, (k_int4 + v_fp16) / mb], [MUTED, TEAL]),
    ):
        bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f} MB",
                    ha="center", va="bottom", fontsize=9, color=INK, weight="bold")
        ratio = vals[0] / vals[1]
        ax.set_title(f"{title}: {ratio:.2f}x smaller", fontsize=10.5, color=INK,
                     weight="bold", loc="left")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
        _style(ax)
    axes[0].set_ylabel("MB held by the cache", fontsize=9.5)
    fig.suptitle(f"Measured from INT4PagedKVCache arrays: {blocks * bs} tokens x "
                 f"head_dim {d}", fontsize=9, color=MUTED, x=0.01, ha="left", y=0.995)
    fig.tight_layout()
    out = os.path.join(OUT, "memory_compression.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"(keys {k_fp16 / k_int4:.2f}x, whole KV "
          f"{(k_fp16 + v_fp16) / (k_int4 + v_fp16):.2f}x)")


# ---------------------------------------------------------------------------
# 4. per-block vs whole-sequence scales, from the simulation itself
# ---------------------------------------------------------------------------

def snr_chart():
    import sys
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    sys.path.insert(0, HERE)
    from validate_llama import simulate_kv_quantization_impact
    r = simulate_kv_quantization_impact()
    whole, block = r["mean_kv_snr_db_whole_sequence_scales"], r["mean_kv_snr_db"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=160)
    bars = ax.bar(["one scale set per\nwhole sequence (2048)",
                   "one scale set per\npaged block (256)"],
                  [whole, block], color=[MUTED, TEAL], width=0.5, zorder=3)
    for bar, v in zip(bars, (whole, block)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f} dB",
                ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")
    ax.set_ylim(min(whole, block) - 2.5, max(whole, block) + 1.2)
    ax.set_ylabel("KV quantization SNR (dB)", fontsize=9.5)
    ax.set_title(f"Per-block scales: +{block - whole:.2f} dB SNR\n"
                 f"Gaussian worst-case simulation, {r['num_layers']} layers "
                 f"(validate_llama.py --simulate)", fontsize=10.5, color=INK,
                 weight="bold", loc="left")
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(OUT, "snr_per_block.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"(+{block - whole:.2f} dB)")


# ---------------------------------------------------------------------------
# 5. CUDA vs Triton quantizer, same device, from the committed artifact
# ---------------------------------------------------------------------------

def t1000_quantizer_chart():
    import json
    path = os.path.join(HERE, "results", "three_backend_t1000.json")
    with open(path) as f:
        rows = [r for r in json.load(f)["rows"]
                if r.get("op") == "quantize" and "ms_per_call" in r
                and r["backend"] in ("cuda", "triton")]
    by_rows = {}
    for r in rows:
        if r["channels"] != 128:
            continue
        by_rows.setdefault(r["rows"], {}).setdefault(r["backend"], []).append(
            r["ms_per_call"])
    # Several sweep configs share one quantizer shape (batch does not affect
    # it); they are repeat measurements, so plot the median, not the last.
    sizes = sorted(k for k, v in by_rows.items() if {"cuda", "triton"} <= set(v))
    cuda = [float(np.median(by_rows[n]["cuda"])) for n in sizes]
    triton = [float(np.median(by_rows[n]["triton"])) for n in sizes]

    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=160)
    ax.bar(x - 0.19, cuda, 0.36, color=TEAL, label="CUDA (hand-written)", zorder=3)
    ax.bar(x + 0.19, triton, 0.36, color=AMBER, label="Triton", zorder=3)
    for i, (c, t) in enumerate(zip(cuda, triton)):
        label = (f"CUDA {t / c:.2f}x faster" if t > c
                 else f"Triton {c / t:.2f}x faster")
        ax.text(i, max(c, t) * 1.08, label, ha="center", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,} x 128" for n in sizes], fontsize=9, color=INK)
    ax.set_ylabel("ms per call (log scale)", fontsize=9.5)
    ax.set_title("INT4 quantizer, CUDA vs Triton on the same NVIDIA T1000\n"
                 "kernel only, device-resident inputs, median of repeats "
                 "(results/three_backend_t1000.json)",
                 fontsize=10.5, color=INK, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(OUT, "t1000_quantizer_cuda_vs_triton.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



# ---------------------------------------------------------------------------
# 6. H100 vs T1000, from the committed artifacts
# ---------------------------------------------------------------------------

def h100_chart():
    import json

    def load(name):
        with open(os.path.join(HERE, "results", name)) as f:
            return json.load(f)["rows"]

    def attn(rows):
        return {r["seq_len"]: r["ms_per_call"] for r in rows
                if r.get("op") == "attention" and r.get("backend") == "cuda"
                and "ms_per_call" in r and r["batch"] == 8 and r["heads"] == 32
                and r["head_dim"] == 128}

    t1000 = attn(load("three_backend_t1000.json"))
    h100 = attn(load("three_backend_attention_h100.json"))
    seqs = sorted(set(t1000) & set(h100))

    qrows = [r for r in load("three_backend_quantize_h100.json")
             if "ms_per_call" in r and r["channels"] == 128
             and r["backend"] in ("cuda", "triton")]
    by = {}
    for r in qrows:
        by.setdefault(r["rows"], {}).setdefault(r["backend"], []).append(r["ms_per_call"])
    sizes = sorted(k for k, v in by.items() if {"cuda", "triton"} <= set(v))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    a1.plot(seqs, [t1000[s] for s in seqs], "o-", color=MUTED, lw=2, label="NVIDIA T1000")
    a1.plot(seqs, [h100[s] for s in seqs], "o-", color=TEAL, lw=2, label="NVIDIA H100 SXM")
    for s in seqs:
        a1.annotate(f"{t1000[s] / h100[s]:.1f}x", (s, h100[s]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=TEAL)
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xticks(seqs); a1.set_xticklabels([str(s) for s in seqs])
    a1.set_xlabel("sequence length (batch 8, 32 heads, head_dim 128)", fontsize=9)
    a1.set_ylabel("ms per call, kernel only (log)", fontsize=9)
    a1.set_title("CUDA attention: T1000 vs H100", fontsize=10.5, color=INK,
                 weight="bold", loc="left")
    a1.legend(frameon=False, fontsize=8.5)
    a1.grid(color=RULE, lw=0.7, which="both")
    _style(a1)

    x = np.arange(len(sizes))
    cuda = [float(np.median(by[n]["cuda"])) for n in sizes]
    tri = [float(np.median(by[n]["triton"])) for n in sizes]
    a2.bar(x - 0.19, cuda, 0.36, color=TEAL, label="CUDA", zorder=3)
    a2.bar(x + 0.19, tri, 0.36, color=AMBER, label="Triton", zorder=3)
    a2.set_yscale("log")
    a2.set_xticks(x); a2.set_xticklabels([f"{n:,} x 128" for n in sizes], fontsize=8.5)
    a2.set_ylabel("ms per call (log)", fontsize=9)
    a2.set_title("INT4 quantizer on the H100: CUDA and Triton tie", fontsize=10.5,
                 color=INK, weight="bold", loc="left")
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    _style(a2)

    fig.suptitle("Kernel-only timings on device-resident inputs, every row verified "
                 "against the NumPy reference before timing", fontsize=8.5, color=MUTED,
                 x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    out = os.path.join(OUT, "h100_results.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



# ---------------------------------------------------------------------------
# 7. perplexity cost vs key compression, Llama-2-7B on the H100
# ---------------------------------------------------------------------------

def perplexity_tradeoff_chart():
    import json
    with open(os.path.join(HERE, "results", "perplexity_block_sweep_h100.json")) as f:
        d = json.load(f)
    rows = sorted(d["rows"], key=lambda r: r["key_compression_vs_fp16"])
    x = [r["key_compression_vs_fp16"] for r in rows]
    y = [r["delta_percent"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
    ax.axhline(0.3, color=RED, ls=":", lw=1.4, zorder=2)
    ax.text(x[0] - 0.05, 0.305, "quality target: +0.3% perplexity", color=RED,
            fontsize=8.5, weight="bold", va="bottom")
    ax.plot(x, y, "-", color=RULE, lw=2, zorder=2)
    for r in rows:
        ok = r["gate_0_3_pass"]
        ax.scatter(r["key_compression_vs_fp16"], r["delta_percent"], s=90, zorder=3,
                   color=TEAL if ok else RED, edgecolor="white", linewidth=1.2)
        ax.annotate(f"{r['block_size']} tokens/scale\n+{r['delta_percent']:.3f}%",
                    (r["key_compression_vs_fp16"], r["delta_percent"]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=8.2, color=INK)
    ax.set_xlabel("key-cache compression vs FP16 (x)", fontsize=9.5)
    ax.set_ylabel("perplexity increase (%)", fontsize=9.5)
    ax.set_ylim(0, max(y) * 1.35)
    ax.set_xlim(min(x) - 0.25, max(x) + 0.25)
    ax.set_title(f"INT4 keys on Llama-2-7B: quality vs compression\n"
                 f"WikiText-2, {d['tokens_scored']:,} tokens, H100 "
                 f"(results/perplexity_block_sweep_h100.json)",
                 fontsize=10.5, color=INK, weight="bold", loc="left")
    ax.grid(color=RULE, lw=0.7, zorder=0)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(OUT, "perplexity_tradeoff.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



# ---------------------------------------------------------------------------
# 8. social preview card (upload: repo Settings -> General -> Social preview)
# ---------------------------------------------------------------------------

def social_card():
    """1280x640 card shown when the repository link is shared. Every figure on
    it is a committed measurement (results/*_h100*.json)."""
    bg, fg, sub, accent = "#0E1116", "#F5F7FA", "#A8B2C4", "#4FD1D9"
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280); ax.set_ylim(0, 640); ax.axis("off")

    ax.text(72, 560, "LLM INFERENCE  ·  GPU KERNELS  ·  QUANTIZATION", color=accent,
            fontsize=15, weight="bold")
    ax.text(72, 470, "INT4 KV-Cache Quantization +", color=fg, fontsize=40, weight="bold")
    ax.text(72, 408, "Fused Flash-Decoding Attention", color=fg, fontsize=40, weight="bold")
    ax.text(72, 350, "One decode kernel, three backends:  CUDA  ·  Triton  ·  JAX/Pallas",
            color=sub, fontsize=21)

    stats = [("0.40 ms", "decode attention\non an NVIDIA H100"),
             ("+0.29%", "Llama-2-7B perplexity\nat 3.56x key compression"),
             ("154", "tests passing\non the H100")]
    for i, (big, label) in enumerate(stats):
        x = 72 + i * 395
        ax.add_patch(plt.Rectangle((x, 112), 360, 178, facecolor="#151A22",
                                   edgecolor="#2A323F", lw=1.5))
        ax.text(x + 26, 216, big, color=accent, fontsize=38, weight="bold")
        ax.text(x + 28, 140, label, color=sub, fontsize=14.5, linespacing=1.35)

    ax.text(72, 52, "github.com/ArchanaChetan07/int4-kv-cache-quantization-cuda-triton-pallas",
            color="#78839A", fontsize=13.5)
    out = os.path.join(OUT, "social_preview.png")
    fig.savefig(out, facecolor=bg, dpi=100)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    accuracy_chart()
    backend_matrix()
    compression_chart()
    snr_chart()
    t1000_quantizer_chart()
    h100_chart()
    perplexity_tradeoff_chart()
    social_card()
