"""Tier-2 MAIN result (paper-grade): on-site shiftability sweep with seed bands.

Controlled ablation: deploy the frozen baseline-office policy (each of the 5
trained seeds) on office timing-variants whose shiftability M is swept by load
timing only, refitting the predictor per variant. Everything except the gain
timing is fixed (envelope, equipment, COP, climate, policy architecture). If the
realized savings rise monotonically with M, shiftability M causally drives savings.

This is the clean Tier-2 result (the source-side sweep is a noise-limited
complement, since office M is timing-bounded to 14.7-17.7%).

Modes:
  --mode eval --delta D --seed S   deploy baseline-office seed S on variant D
  --mode plot                      aggregate mean+/-std, regenerate figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
PRED = ROOT / "artifacts/case_study/predictors"
RESULTS = OUT / "tier2_onsite_seeds"
SEEDS = (42, 123, 777, 2024, 2025)
DELTAS = (0, 1, 2, 3)


def _predictor_for(delta: int) -> Path:
    return (
        (PRED / "office_5zone_lstm.pt")
        if delta == 0
        else (PRED / f"office_shift{delta}_lstm.pt")
    )


def eval_one(delta: int, seed: int, registry):
    from studies.case_study.analysis.tier2_run import eval_policy

    policy = (
        ROOT / "artifacts/case_study/exports/"
        f"pretrain_office_smud_sac_plus/seed_{seed}/model.zip"
    )
    if not policy.exists():
        raise FileNotFoundError(policy)
    ev = eval_policy(delta, policy, _predictor_for(delta), registry)
    rbc = json.load(open(OUT / "tier2_rbc_metrics.json"))[str(delta)]
    sav = 100 * (rbc["cost_rbc_per_day"] - ev["cost_per_day"]) / rbc["cost_rbc_per_day"]
    rec = {
        "delta": delta,
        "seed": seed,
        "M3_oracle_sched_pct": rbc["M3_oracle_sched_pct"],
        "cost_policy_per_day": ev["cost_per_day"],
        "savings_pct": sav,
        "viol_pct": ev["viol_pct"],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"shift{delta}_seed{seed}.json").write_text(json.dumps(rec))
    print(
        f"  on-site d={delta} seed={seed}: savings={sav:+.2f}% M={rec['M3_oracle_sched_pct']:.1f}"
    )
    return rec


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    recs = [json.load(open(p)) for p in RESULTS.glob("shift*_seed*.json")]
    by_d = {}
    for r in recs:
        by_d.setdefault(r["delta"], []).append(r)
    rows = []
    for d in sorted(by_d):
        sav = [r["savings_pct"] for r in by_d[d]]
        rows.append(
            {
                "delta": d,
                "M_src": by_d[d][0]["M3_oracle_sched_pct"],
                "savings_mean": float(np.mean(sav)),
                "savings_std": float(np.std(sav)),
                "n": len(sav),
            }
        )
    (OUT / "tier2_onsite_sweep.json").write_text(json.dumps(rows, indent=2))

    M = np.array([r["M_src"] for r in rows])
    S = np.array([r["savings_mean"] for r in rows])
    E = np.array([r["savings_std"] for r in rows])
    A = np.column_stack([np.ones_like(M), M])
    c, *_ = np.linalg.lstsq(A, S, rcond=None)
    r2 = 1 - ((S - A @ c) ** 2).sum() / ((S - S.mean()) ** 2).sum()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = np.linspace(M.min() - 0.4, M.max() + 0.4, 50)
    ax.plot(
        xs,
        c[0] + c[1] * xs,
        color="0.45",
        ls="--",
        lw=1.0,
        label=f"OLS fit ($R^2$={r2:.2f})",
    )
    ax.errorbar(
        M,
        S,
        yerr=E,
        fmt="o",
        color="#1f77b4",
        capsize=3,
        ms=7,
        mec="k",
        lw=1.6,
        zorder=4,
        label="frozen policy on variant (5 seeds)",
    )
    for r in rows:
        ax.annotate(
            f"$\\Delta$={r['delta']:+d}h",
            (r["M_src"], r["savings_mean"]),
            fontsize=7,
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Building shiftability $S^{*}$ (on-peak load share) [%]")
    ax.set_ylabel("Realized savings vs RBC [%]")
    ax.set_title(
        "Shiftability $S^{*}$ causally drives savings\n"
        "(office, only load timing varied; policy fixed)",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tier2_onsite_sweep.pdf")
    fig.savefig(OUT / "fig_tier2_onsite_sweep.png")
    plt.close(fig)
    print("=== on-site sweep (5-seed) ===")
    for r in rows:
        print(
            f"  d={r['delta']:+d} M={r['M_src']:.1f}: "
            f"{r['savings_mean']:+.2f}±{r['savings_std']:.2f} (n={r['n']})"
        )
    print(f"  OLS R^2={r2:.3f}; written fig_tier2_onsite_sweep.pdf")
    _ = Line2D  # keep import referenced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "plot"], required=True)
    ap.add_argument("--delta", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.mode == "plot":
        plot()
        return
    from studies.case_study.core.registry import load_registry

    registry = load_registry(str(ROOT / "studies/case_study/config"))
    eval_one(args.delta, args.seed, registry)


if __name__ == "__main__":
    main()
