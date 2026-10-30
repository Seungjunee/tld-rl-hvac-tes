"""Tier-2 figures: causal savings-vs-M curve + the perturbed load profiles.

Reads tier2_sweep.json (frozen-policy savings per Delta) and tier2_rbc_metrics.json
(M + daily load profile per Delta), plus the Tier-1 real-building anchors from
shiftability_metrics.json / _final_tables.json.

fig_tier2_savings_vs_M : frozen office policy savings vs shiftability M across the
   controlled timing sweep, with real OF/LB/WH for context.
fig_tier2_load_profiles: how the timing shift moves cooling load into 17-20h.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SMUD = [0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / sst if sst else float("nan")
    return c[1], c[0], r2, float(np.corrcoef(x, y)[0, 1])


def savings_vs_M():
    sweep = json.load(open(OUT / "tier2_sweep.json"))
    pts = sorted(sweep.values(), key=lambda v: v["M3_oracle_sched_pct"])
    M = np.array([p["M3_oracle_sched_pct"] for p in pts])
    S = np.array([p["savings_pct"] for p in pts])
    dl = np.array([p["delta"] for p in pts])

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    slope, b, r2, r = ols(M, S)
    xs = np.linspace(M.min() - 0.5, M.max() + 0.5, 50)
    ax.plot(
        xs,
        b + slope * xs,
        color="0.45",
        lw=1.0,
        ls="--",
        zorder=1,
        label=f"OLS fit ($R^2$={r2:.2f}, r={r:+.2f})",
    )
    sc = ax.scatter(
        M, S, c=dl, cmap="viridis", s=70, edgecolor="k", linewidth=0.5, zorder=3
    )
    for p in pts:
        ax.annotate(
            f"$\\Delta$={p['delta']:+d}h",
            (p["M3_oracle_sched_pct"], p["savings_pct"]),
            fontsize=6.2,
            xytext=(4, -1),
            textcoords="offset points",
        )
    # real-building context anchors (Tier-1 M + on-site TLD-RL savings)
    try:
        sm = json.load(open(OUT / "shiftability_metrics.json"))["per_building"]
        ft = json.load(
            open(
                ROOT / "artifacts/case_study/figures/"
                "metric_a_5seed_2026-06-14/_final_tables.json"
            )
        )["source"]
        for b_, mark in [("warehouse", "WH"), ("lbuilding", "LB"), ("office", "OF")]:
            xm = sm[b_]["metric3_oracle"]["savings_schedule_pct"]
            ym = ft[b_]["prop"]["sav"]
            ax.scatter(
                xm,
                ym,
                marker="*",
                s=130,
                color="#d62728",
                edgecolor="k",
                linewidth=0.4,
                zorder=4,
            )
            ax.annotate(
                mark,
                (xm, ym),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
                color="#d62728",
                weight="bold",
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  (anchor overlay skipped: {exc})")

    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("Shiftability $M$ (oracle potential) [%]")
    ax.set_ylabel("Frozen-policy savings vs RBC [%]")
    ax.set_title(
        "Causal timing sweep: savings track $M$\n" "(office, only gain timing varied)",
        fontsize=8,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("timing shift $\\Delta$ [h]", fontsize=7)
    h, _lb = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    h.append(
        Line2D(
            [],
            [],
            marker="*",
            color="#d62728",
            ls="",
            ms=9,
            label="real building (on-site)",
        )
    )
    ax.legend(handles=h, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tier2_savings_vs_M.pdf")
    fig.savefig(OUT / "fig_tier2_savings_vs_M.png")
    plt.close(fig)
    print(
        f"  savings range {S.min():+.1f}..{S.max():+.1f}%, "
        f"M range {M.min():.1f}..{M.max():.1f}%, fit R2={r2:.2f}"
    )


def load_profiles():
    rbc = json.load(open(OUT / "tier2_rbc_metrics.json"))
    deltas = sorted(rbc, key=int)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for h in range(24):
        c = {0.3765: "#ffcccc", 0.2139: "#fff0cc"}.get(SMUD[h])
        if c:
            ax.axvspan(h, h + 1, color=c, alpha=0.55, lw=0, zorder=0)
    cmap = plt.get_cmap("viridis")
    for d in deltas:
        prof = np.array(rbc[d]["daily_load_profile_kwh"])
        if prof.size != 24:  # stored as full hourly series -> collapse to daily mean
            ndays = prof.size // 24
            prof = prof[: ndays * 24].reshape(ndays, 24).mean(axis=0)
        frac = (int(d) + 2) / 5.0
        ax.plot(
            np.arange(24) + 0.5,
            prof,
            color=cmap(frac),
            lw=1.6,
            marker="o",
            ms=2,
            label=f"$\\Delta$={int(d):+d}h (M={rbc[d]['M3_oracle_sched_pct']:.1f}%)",
        )
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 17, 20, 24])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cooling load [kWh-th/h]")
    ax.set_title("Timing shift moves load into 17-20h on-peak", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tier2_load_profiles.pdf")
    fig.savefig(OUT / "fig_tier2_load_profiles.png")
    plt.close(fig)


def main():
    load_profiles()
    if (OUT / "tier2_sweep.json").exists():
        savings_vs_M()
    else:
        print("  (tier2_sweep.json not found; run tier2_run.py first)")


if __name__ == "__main__":
    main()
