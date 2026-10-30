"""Source-domain learning-curve figure (office): TLD-RL vs E2E, RBC reference.

Reads the per-seed progress.csv of the policies retrained this session and plots
the stochastic training return (mean +/- 1 std over seeds) per episode, with the
RBC mean reward as a dotted reference. Mirrors the paper's source learning
curve but from the current exports (the prior era was cleaned).

Usage: python -m studies.case_study.analysis.fig_learning_curve
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
EXPORTS = ROOT / "artifacts/case_study/exports"
SEEDS = (42, 123, 777, 2024, 2025)
C_PROP, C_E2E, C_RBC = "#1f77b4", "#d62728", "0.35"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def curve(case: str):
    series = []
    for s in SEEDS:
        p = EXPORTS / case / f"seed_{s}" / "progress.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        series.append([float(r["mean_reward"]) for r in rows[:10]])
    if not series:
        return None
    n = min(len(s) for s in series)
    arr = np.array([s[:n] for s in series])
    return np.arange(1, n + 1), arr.mean(0), arr.std(0), len(series)


def rbc_reward():
    p = EXPORTS / "rbc_office_smud_tou" / "progress.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return float(rows[-1]["mean_reward"])


def padding_pretrain_reward():
    """Padding's source warm-start = BC clone of RBC; mean final reward over seeds."""
    vals = []
    for s in SEEDS:
        p = EXPORTS / "bc_pretrain_office_smud_sacp" / f"seed_{s}" / "progress.csv"
        if p.exists():
            vals.append(float(list(csv.DictReader(open(p)))[-1]["mean_reward"]))
    return float(np.mean(vals)) if vals else None


def main():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for case, color, lab in [
        ("pretrain_office_smud_sac_plus", C_PROP, "TLD-RL (Proposed)"),
        ("pretrain_office_smud_sacp_e2e", C_E2E, "E2E (Scratch)"),
    ]:
        c = curve(case)
        if c is None:
            print(f"  (skip {case}: no progress.csv yet)")
            continue
        x, m, s, n = c
        ax.plot(x, m, color=color, lw=1.8, marker="o", ms=3, label=f"{lab} (n={n})")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)
    pad = padding_pretrain_reward()
    if pad is not None:
        ax.axhline(
            pad, color="#2ca02c", lw=1.4, ls="-.", label="Padding (BC warm-start)"
        )
    rbc = rbc_reward()
    if rbc is not None:
        ax.axhline(rbc, color=C_RBC, lw=1.3, ls=":", label="RBC")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode mean reward")
    ax.set_title("Office source-domain learning curve", fontsize=8.5)
    ax.set_xticks(range(1, 11))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curve_office.pdf")
    fig.savefig(OUT / "fig_learning_curve_office.png")
    plt.close(fig)
    print(f"written: {OUT}/fig_learning_curve_office.pdf")


if __name__ == "__main__":
    main()
