"""Fig. 7 (fig_sstar.pdf): zero-shot savings against the target's on-peak share S*.

The controlled office schedule-shift sweep: nine variants of the office
(delta = -3..+5 h), each with its RBC season, a predictor refit on that season,
and the five frozen office source policies deployed zero-shot. Points are
five-seed means +- s.d. (ddof 1) with the 45 per-seed values overlaid; the
dashed line is the least-squares fit over the variant means.

Data: artifacts/case_study/results/sstar_sweep/ written by
analysis/r1_sstar_sweep.py.

Run:  python -m studies.case_study.analysis.fig_sstar
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/paper"
DELTAS = (-3, -2, -1, 0, 1, 2, 3, 4, 5)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#2166ac"

# Data from r1_sstar_sweep.py: RBC priced per step and S* on the logged hour
# labels, as in Tables 1 and 3; predictors refit on an RBC season.
SWEEP = ROOT / "artifacts/case_study/results/sstar_sweep"
rbc_t2 = {
    str(d): json.load(open(SWEEP / f"rbc_shift{d}.json")) for d in DELTAS
}
per_seed = {}
for p in SWEEP.glob("shift*_seed*.json"):
    r = json.load(open(p))
    per_seed.setdefault(r["delta"], []).append(r["savings_pct"])
assert set(per_seed) == set(DELTAS), sorted(per_seed)
assert all(len(v) == 5 for v in per_seed.values()), {k: len(v) for k, v in per_seed.items()}

xs = np.array([rbc_t2[str(d)]["onpeak_share_pct"] for d in DELTAS])
ys = np.array([np.mean(per_seed[d]) for d in DELTAS])
es = np.array([np.std(per_seed[d], ddof=1) for d in DELTAS])

A = np.column_stack([np.ones_like(xs), xs])
c, *_ = np.linalg.lstsq(A, ys, rcond=None)
r2 = 1 - ((ys - A @ c) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

fig, ax = plt.subplots(figsize=(3.5, 1.9))
# OLS line clipped to the data range (no extrapolation implication)
ax.plot(
    [xs.min(), xs.max()],
    [c[0] + c[1] * xs.min(), c[0] + c[1] * xs.max()],
    "--",
    color="#888888",
    lw=1.0,
    zorder=1.5,
)
# faint raw cloud: all 45 seed evaluations at their true x (never jittered --
# x is the measured S*). Shows the honest per-run spread behind the means.
for d, x in zip(DELTAS, xs):
    ax.plot(
        [x] * len(per_seed[d]),
        per_seed[d],
        "o",
        ms=2.0,
        color=BLUE,
        alpha=0.28,
        mew=0,
        zorder=2,
    )
ax.errorbar(
    xs,
    ys,
    yerr=es,
    fmt="o",
    ms=4.5,
    color=BLUE,
    ecolor=BLUE,
    elinewidth=0.9,
    capsize=2.5,
    markeredgecolor="white",
    markeredgewidth=0.5,
    zorder=4,
)
# top-center: the top-left slot is crossed by the source-S* dotted line
ax.annotate(
    f"$R^2 = {r2:.2f}$ (variant means)",
    xy=(0.5, 0.86),
    xycoords="axes fraction",
    ha="center",
)
# name the two shift clusters whose S* saturates (the point of the extension);
# the left label spans -3..0 h because the base variant sits in that cluster
ax.text(
    np.mean(xs[:3]),
    9.62,
    "$\\Delta = -3$ to $-1$ h",
    fontsize=7,
    color="0.40",
    ha="center",
    va="bottom",
)
ax.text(
    np.mean(xs[-3:]) - 0.5,
    6.82,
    "$\\Delta = +3$ to $+5$ h",
    fontsize=7,
    color="0.40",
    ha="center",
    va="top",
)
# the five deployed policies' own training point: office S* = 15.05 (Table 1;
# prints as 15.1); the delta-0 variant measures the same.
SSTAR_SRC = 15.05
ax.axvline(SSTAR_SRC, color="0.5", lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.text(
    SSTAR_SRC + 0.12,
    5.55,
    "source $S^{\\star}$",
    fontsize=7,
    color="0.40",
    ha="left",
    va="bottom",
)
# ticks bracket the data (13.20-21.07) so no point sits outside the
# labeled range; frame margins stay symmetric around the end ticks
ax.set_xlim(12.9, 22.0)
ax.set_xticks([13, 15, 17, 19, 21])
# frame ends on ticks and the ticks bracket the raw cloud (5.51-10.61),
# matching the x treatment and the Fig 6/8 house convention
ax.set_ylim(5, 11)
ax.set_yticks([5, 6, 7, 8, 9, 10, 11])
ax.set_xlabel(r"Target $S^{\star}$ (on-peak load share) [%]")
ax.set_ylabel("Zero-shot savings [%]")
fig.tight_layout(pad=0.35)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_sstar.pdf")
print(f"wrote {OUT / 'fig_sstar.pdf'} (9-variant sweep, raw cloud overlay)")
print(f"  R2={r2:.3f} slope={c[1]:+.3f}")
for d, x, y, e in zip(DELTAS, xs, ys, es):
    print(f"  d={d:+d}  S*={x:5.2f}  {y:5.2f} ± {e:4.2f}")
