"""fig3_4b (V6' generation): source-domain learning curves, 4 buildings,
TLD-RL vs E2E stochastic training return (progress.csv mean_reward),
5-seed mean +/- 1 std, from the CURRENT exports (V6' 56-dim, cost-only reward).

Run:  python -m studies.case_study.analysis.fig34b_v6
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
EXPORTS = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
MAN = ROOT / "artifacts/case_study/figures/paper"
MAN.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 777, 2024, 2025]
BLDGS = [("office", "OF (5-zone)"), ("lbuilding", "LB (8-zone)"),
         ("warehouse", "WH (3-zone)"), ("mediumoffice", "MO (15-zone)")]
C_PROP, C_E2E = "#1f77b4", "#d62728"


def train_curve(case):
    series = []
    for s in SEEDS:
        p = EXPORTS / case / f"seed_{s}" / "progress.csv"
        if p.exists():
            rows = list(csv.DictReader(open(p)))
            series.append([float(r["mean_reward"]) for r in rows[:10]])
    n = min(len(x) for x in series)
    arr = np.array([x[:n] for x in series])
    return np.arange(1, n + 1), arr.mean(0), arr.std(0)


plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.3))
for ax, (b, title) in zip(axes, BLDGS):
    for case, color, lab in [
        (f"pretrain_{b}_smud_sac_plus", C_PROP, "TLD-RL (Proposed)"),
        (f"pretrain_{b}_smud_sac_e2e", C_E2E, "E2E (Scratch)"),
    ]:
        x, m, s = train_curve(case)
        ax.plot(x, m, color=color, label=lab, lw=1.8)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Training season")
    ax.set_xticks(range(1, 11, 2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
axes[0].set_ylabel("Season mean reward")
axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9)
fig.tight_layout()
for out in (OUT / "fig3_4b.pdf", MAN / "fig3_4b.pdf"):
    fig.savefig(out)
fig.savefig(OUT / "fig3_4b.png", dpi=160)
print(f"written fig3_4b.pdf -> {OUT} and {MAN}")
