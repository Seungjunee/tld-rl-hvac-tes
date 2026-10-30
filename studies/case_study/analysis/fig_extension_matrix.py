"""Extension figure: the four original source policies deployed on four new targets.

Rows are sources, columns targets; each cell shows the five-seed mean of the
source seeds' final policies deployed zero-shot on the target
(<pair>__src<seed> exports, savings vs the target's RBC) and, below it, the
selected-source value (seed with the highest final training reward). Tick
labels carry each building's S*. Data: artifacts/case_study/results/x_matrix.json
(analysis/r1_x_matrix.py).

Run:  python -m studies.case_study.analysis.fig_extension_matrix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
XM = ROOT / "artifacts/case_study/results/x_matrix.json"
OUT = ROOT / "artifacts/case_study/figures/paper/fig_extension_matrix.pdf"
PREVIEW = ROOT / "artifacts/case_study/results/fig_extension_matrix_preview.png"
SOURCES = ["office", "lbuilding", "warehouse", "mediumoffice"]
TARGETS = ["restaurant", "stripmall", "retail", "fastfood"]
LABEL = {
    "office": "OF",
    "lbuilding": "LB",
    "warehouse": "WH",
    "mediumoffice": "MO",
    "restaurant": "RE",
    "stripmall": "SM",
    "retail": "RT",
    "fastfood": "QS",
}
VMAX = 8.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> int:
    x = json.loads(XM.read_text())
    pairs, sstar = x["pairs"], dict(x["s_star"])
    ns, nt = len(SOURCES), len(TARGETS)
    mean = np.full((ns, nt), np.nan)
    dep = np.full((ns, nt), np.nan)
    for i, s in enumerate(SOURCES):
        for j, t in enumerate(TARGETS):
            p = pairs[f"{s}->{t}"]
            assert p["n"] == 5, (s, t, p["n"])
            mean[i, j], dep[i, j] = p["mean"], p["deployed"]
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    im = ax.imshow(mean, cmap="RdBu", vmin=-VMAX, vmax=VMAX)
    for i in range(ns):
        for j in range(nt):
            c = "white" if abs(mean[i, j]) > 0.62 * VMAX else "black"
            ax.text(
                j,
                i - 0.12,
                f"{mean[i, j]:+.1f}".replace("-", "\u2212"),
                ha="center",
                va="center",
                fontsize=8,
                color=c,
            )
            ax.text(
                j,
                i + 0.24,
                f"({dep[i, j]:+.1f})".replace("-", "\u2212"),
                ha="center",
                va="center",
                fontsize=6.3,
                color=c,
            )
    ax.set_xticks(range(nt), [f"{LABEL[b]}\n{sstar[b]:.1f}%" for b in TARGETS])
    ax.set_yticks(range(ns), [f"{LABEL[b]}\n{sstar[b]:.1f}%" for b in SOURCES])
    ax.set_xlabel(r"New target (on-peak share $S^{\star}$)")
    ax.set_ylabel(r"Source ($S^{\star}$)")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Savings vs. target RBC [%]")
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(PREVIEW, dpi=200)
    print(f"wrote {OUT}")
    print(np.round(mean, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
