"""Adoption-cost figure (V6' generation): cumulative realized savings vs seasons
since deployment, per target building.

Lines per target panel:
  - TLD-RL zero-shot from each source (dashed, linear: deterministic frozen policy,
    realized == deterministic, slope = zs savings of the deployed/selected policy)
  - E2E (Scratch) online (training return incl. exploration), 5-seed mean
  - Padding online (fine-tuning return), pooled over the 3 sources, 5-seed mean
Cumulative units: % of one cooling-season RBC cost (sum of per-season savings %).
Data: artifacts/case_study/analysis_archive_2026-07/adoption_online_data.json (P4).

Run:  python -m studies.case_study.analysis.fig_adoption_v6
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import csv

REPO = Path(__file__).resolve().parents[3]
# artifacts live in the main checkout; worktrees only carry the repo files
_CANDS = [REPO, Path("/workspaces/sinergym")]
ROOT = next(p for p in _CANDS if (p / "artifacts/case_study/exports").exists())
MAN = REPO / "artifacts/case_study/figures/paper"
MAN.mkdir(parents=True, exist_ok=True)
ARC = ROOT / "artifacts/case_study/analysis_archive_2026-07"
EXP = ROOT / "artifacts/case_study/exports"
SEEDS = ["42", "123", "777", "2024", "2025"]
STEP = 900.0 / 3.6e6


def _rows(path):
    return [r for r in csv.DictReader(open(path)) if r.get("length(timesteps)")]


def _cost(r):
    return float(r["mean_reward_energy_term"]) * int(float(r["length(timesteps)"])) * STEP


def seed_cum_savings(case, rbc_season, n=10):
    """Per-seed cumulative online savings [% of one-season RBC cost]."""
    out = []
    for s in SEEDS:
        f = EXP / case / f"seed_{s}" / "progress.csv"
        if not f.exists():
            continue
        sav = [100 * (rbc_season - _cost(r)) / rbc_season for r in _rows(f)[:n]]
        out.append(np.cumsum(sav))
    return out
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
B = ["office", "lbuilding", "warehouse", "mediumoffice"]
YAXIS = {  # (ymin, ymax, tick step); frame ends on ticks, zero always ticked,
    "office": (-20, 100, 20),          # range sized to the mean+/-1 std bands
    "lbuilding": (-50, 75, 25),
    "warehouse": (-120, 80, 40),
    "mediumoffice": (-30, 45, 15),
}
LABEL = {"office": "OF (5-zone)", "lbuilding": "LB (8-zone)",
         "warehouse": "WH (3-zone)", "mediumoffice": "MO (15-zone)"}
AB = {"office": "OF", "lbuilding": "LB", "warehouse": "WH", "mediumoffice": "MO"}

d = json.load(open(ARC / "adoption_online_data.json"))
RBC_SEASON = {}
for b in B:
    r = _rows(EXP / f"rbc_{b}_smud_tou" / "progress.csv")[-1]
    RBC_SEASON[b] = _cost(r)

# final printed size (\textwidth = 7.1667 in): rcParams sizes are page sizes
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.2,
    "grid.linewidth": 0.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, axes = plt.subplots(1, 4, figsize=(7.1667, 2.3), sharex=True)
N = 10
x = np.arange(0, N + 1)
SRC_COLOR = {"office": "#1f77b4", "lbuilding": "#2ca02c",
             "warehouse": "#9467bd", "mediumoffice": "#17becf"}

handles = {}
for ax, tgt in zip(axes, B):
    e2e_cums = seed_cum_savings(f"pretrain_{tgt}_smud_sac_e2e", RBC_SEASON[tgt], N)
    arr = np.array([np.concatenate([[0], c]) for c in e2e_cums])
    m, sd = arr.mean(0), arr.std(0)
    (h,) = ax.plot(x[:arr.shape[1]], m, color="#d62728", lw=1.2)
    ax.fill_between(x[:arr.shape[1]], m - sd, m + sd, color="#d62728", alpha=0.15, lw=0)
    handles["E2E (Scratch), online"] = h
    pad_cums = []
    for src in B:
        if src == tgt:
            continue
        pad_cums += seed_cum_savings(f"transfer_{src}_to_{tgt}_smud_sacp_padrl",
                                     RBC_SEASON[tgt], N)
    kmin = min(len(c) for c in pad_cums)
    arr = np.array([np.concatenate([[0], c[:kmin]]) for c in pad_cums])
    m, sd = arr.mean(0), arr.std(0)
    (h,) = ax.plot(x[:arr.shape[1]], m, color="#ff7f0e", lw=1.2)
    ax.fill_between(x[:arr.shape[1]], m - sd, m + sd, color="#ff7f0e", alpha=0.15, lw=0)
    handles["Padding (fine-tune), online"] = h
    for p in sorted(d):
        if p.endswith(f"_to_{tgt}") and "zs_sel" in d[p]:
            src = p.split("_to_")[0]
            (h,) = ax.plot(x, d[p]["zs_sel"] * x, ls="--", lw=0.95,
                           color=SRC_COLOR[src])
            handles[f"TLD-RL zero-shot from {AB[src]}"] = h
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(LABEL[tgt], fontsize=9)
    ax.set_xlabel("Seasons since deployment")
    ax.set_xlim(0, N)
    ax.set_xticks(range(0, N + 1, 2))
    lo, hi, step = YAXIS[tgt]
    ax.set_ylim(lo, hi)
    ax.set_yticks(range(lo, hi + 1, step))
axes[0].set_ylabel("Cumulative savings vs. RBC\n[one season's RBC cost = 100]")
order = ["E2E (Scratch), online", "Padding (fine-tune), online"] + [
    f"TLD-RL zero-shot from {AB[b]}" for b in B]
order = [k for k in order if k in handles]
fig.legend([handles[k] for k in order], order, ncol=3, fontsize=7.5,
           loc="lower center", frameon=False, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig(MAN / "fig_adoption.pdf")
fig.savefig(OUT / "fig_adoption_cumulative.png", dpi=160)
print(f"written {MAN}/fig_adoption.pdf")
for tgt in B:
    e2e = d[tgt]["e2e_online"][:N]
    cum = np.cumsum(e2e)
    zs = [d[p]["zs_sel"] for p in d if p.endswith(f"_to_{tgt}") and "zs_sel" in d[p]]
    best = max(zs); worst = min(zs)
    k_best = next((k + 1 for k in range(N) if cum[k] > best * (k + 1)), None)
    k_worst = next((k + 1 for k in range(N) if cum[k] > worst * (k + 1)), None)
    print(f"  {tgt:13s} cumulative break-even vs best-zs: "
          f"{k_best or '>10'} szn | vs worst-zs: {k_worst or '>10'} szn "
          f"(E2E cum@10 {cum[-1]:+.1f} vs best-zs cum@10 {best*10:+.1f})")
