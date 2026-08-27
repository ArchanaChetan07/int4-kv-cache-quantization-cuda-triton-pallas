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
    """Which kernel backend actually runs on which silicon.

    This is the constraint that reshaped the project: the local Turing GPU is
    below the floor of both live Pallas GPU backends.
    """
    backends = ["CUDA\n(hand-written)", "Triton", "Pallas\nMosaic TPU",
                "Pallas\nMosaic GPU", "Pallas\nTriton backend"]
    hw = ["CPU\n(no accel)", "T1000\nTuring SM75", "Ampere\nGPU",
          "Hopper\nGPU", "TPU\nv5e"]

    # 2 = runs, 1 = interpret/limited, 0 = unavailable, -1 = deprecated
    M = np.array([
        [0, 2, 2, 2, 0],   # CUDA
        [0, 2, 2, 2, 0],   # Triton
        [1, 0, 0, 0, 2],   # Pallas Mosaic TPU  (interpret on CPU)
        [1, 0, 0, 2, 0],   # Pallas Mosaic GPU
        [1, 0, -1, -1, 0],  # Pallas Triton backend (deprecated)
    ])

    cmap = {2: GREEN, 1: AMBER, 0: "#E8E8EC", -1: RED}
    txt = {2: "runs", 1: "interpret\nonly", 0: "—", -1: "deprecated"}

    fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=160)
    for i in range(len(backends)):
        for j in range(len(hw)):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((j, len(backends) - 1 - i), 1, 1,
                                       facecolor=cmap[v], edgecolor="white", lw=2))
            ax.text(j + 0.5, len(backends) - 1 - i + 0.5, txt[v],
                    ha="center", va="center", fontsize=8.2,
                    color="white" if v in (2, -1) else INK,
                    weight="bold" if v in (2, -1) else "normal")

    ax.set_xlim(0, len(hw)); ax.set_ylim(0, len(backends))
    ax.set_xticks(np.arange(len(hw)) + 0.5)
    ax.set_xticklabels(hw, fontsize=8.6, color=INK)
    ax.set_yticks(np.arange(len(backends)) + 0.5)
    ax.set_yticklabels(backends[::-1], fontsize=8.6, color=INK)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Backend x hardware availability\n"
        "the local Turing GPU is below the floor of both live Pallas GPU backends",
        fontsize=11.5, color=INK, weight="bold", loc="left", pad=12)

    fig.tight_layout()
    p = os.path.join(OUT, "pallas_backend_matrix.png")
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    accuracy_chart()
    backend_matrix()
