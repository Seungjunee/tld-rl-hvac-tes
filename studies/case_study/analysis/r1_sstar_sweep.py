"""Controlled S* sweep (Fig. 7 data) under the paper's zero-shot protocol.

Compared with the earlier rig (tier2_rbc_metrics.py + tier2_onsite_seeds.py),
(a) the RBC side is priced and windowed with each step's own hour label, as the
policy side, and (b) the variant predictors are fit on an RBC-operated season,
as every other target predictor, instead of fixed-setpoint (constant 5 C) logs.

Each variant (delta = -3..+5 h) is handled the way every other target in the
paper is handled:
  1. run the RBC dataset case (build_office_observation_dataset_rbc) on the
     variant -> C_RBC from progress.csv (Table 3 convention) and S* from the
     logged hour labels (Table 1 convention; reproduces 26.37 $/day and 15.05 %
     on the base office);
  2. refit the LSTM predictor on that RBC season;
  3. deploy the five frozen office source policies (final model.zip) zero-shot
     with the refit predictor and price every step with its own tariff.

Outputs go to new locations only (nothing canonical is overwritten):
  exports/tier2r1_datasets_rbc/shift<d>/, predictors/office_shift<d>_lstm_rbclog.pt,
  results/sstar_sweep/{rbc_shift<d>.json, shift<d>_seed<s>.json, summary.json}

Run:
  python -m studies.case_study.analysis.r1_sstar_sweep --delta 0      # one variant
  python -m studies.case_study.analysis.r1_sstar_sweep --aggregate    # summary
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from studies.case_study.analysis import tier2_run as T

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
DS = EXP / "tier2r1_datasets_rbc"
PRED = ROOT / "artifacts/case_study/predictors"
OUT = ROOT / "artifacts/case_study/results/sstar_sweep"
SEEDS = (42, 123, 777, 2024, 2025)
DELTAS = (-3, -2, -1, 0, 1, 2, 3, 4, 5)
DAYS = 122
SF = 900 / 3.6e6


def rbc_dataset(delta: int, registry) -> Path:
    from studies.case_study.core.registry import resolve_case
    from studies.case_study.core.runner import run_case

    dest = DS / f"shift{delta}"
    if (dest / "observations_processed.csv").exists():
        return dest
    case = deepcopy(resolve_case("build_office_observation_dataset_rbc", registry))
    case.id = f"tier2r1_dataset_rbc_shift{delta}"
    T._override_env(case, delta)
    case.export = dict(case.export)
    case.export["destination"] = str(dest)
    run_case(case, registry)
    if not (dest / "observations_processed.csv").exists():
        raise FileNotFoundError(dest)
    return dest


def rbc_metrics(dest: Path) -> dict:
    r = pd.read_csv(dest / "progress.csv").iloc[-1]
    cost = float(r["mean_reward_energy_term"]) * int(r["length(timesteps)"]) * SF
    df = pd.read_csv(dest / "observations.csv")
    cr = df["CoolingRate"].to_numpy()
    h = df["hour"].astype(int).to_numpy() % 24
    return {
        "cost_rbc_per_day": cost / DAYS,
        "onpeak_share_pct": float(100.0 * cr[np.isin(h, [17, 18, 19])].sum() / cr.sum()),
        "steps": int(r["length(timesteps)"]),
    }


def refit(delta: int, dest: Path) -> Path:
    from studies.case_study.cli.train_predictor import _run_building

    pt = PRED / f"office_shift{delta}_lstm_rbclog.pt"
    if pt.exists():
        return pt

    class A:  # same settings as tier2_run.refit_predictor and the paper predictors
        data = str(dest / "observations_processed.csv")
        output = str(pt)
        plot_dir = str(OUT / "predictor_plots")
        epochs = 50
        lr = 1e-3
        hidden = 128
        layers = 2
        seq_len = 24
        horizon = 12
        batch_size = 64
        dropout = 0.2
        patience = 10
        evaluate_only = False
        seed = 42

    _run_building("office_5zone", A())
    return pt


def run_delta(delta: int) -> None:
    from studies.case_study.core.registry import load_registry

    OUT.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    dest = rbc_dataset(delta, registry)
    m = rbc_metrics(dest)
    m["delta"] = delta
    (OUT / f"rbc_shift{delta}.json").write_text(json.dumps(m, indent=2))
    pt = refit(delta, dest)
    for s in SEEDS:
        f = OUT / f"shift{delta}_seed{s}.json"
        if f.exists():
            continue
        zip_ = EXP / "pretrain_office_smud_sac_plus" / f"seed_{s}" / "model.zip"
        r = T.eval_policy(delta, zip_, pt, registry)
        r.update(
            delta=delta,
            seed=s,
            predictor=str(pt.relative_to(ROOT)),
            cost_rbc_per_day=m["cost_rbc_per_day"],
            onpeak_share_pct=m["onpeak_share_pct"],
            savings_pct=100.0
            * (m["cost_rbc_per_day"] - r["cost_per_day"])
            / m["cost_rbc_per_day"],
        )
        f.write_text(json.dumps(r, indent=2))
        print(f"delta={delta} seed={s} savings={r['savings_pct']:+.2f}%")


def aggregate() -> dict:
    rows = []
    for d in DELTAS:
        m = json.loads((OUT / f"rbc_shift{d}.json").read_text())
        sv = [
            json.loads((OUT / f"shift{d}_seed{s}.json").read_text())["savings_pct"]
            for s in SEEDS
        ]
        rows.append(
            {
                "delta": d,
                "s_star": m["onpeak_share_pct"],
                "cost_rbc_per_day": m["cost_rbc_per_day"],
                "savings_mean": float(np.mean(sv)),
                "savings_sd": float(np.std(sv, ddof=1)),
                "per_seed": sv,
            }
        )
    x = np.array([r["s_star"] for r in rows])
    y = np.array([r["savings_mean"] for r in rows])
    slope, icpt = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
    lo = min(rows, key=lambda r: r["s_star"])
    hi = max(rows, key=lambda r: r["s_star"])
    gain = [h - l for h, l in zip(hi["per_seed"], lo["per_seed"])]
    res = {
        "rows": rows,
        "fit": {"slope": float(slope), "intercept": float(icpt), "r2": float(r2)},
        "lowest_to_highest": {
            "mean_pp": float(np.mean(gain)),
            "positive_seeds": int(sum(g > 0 for g in gain)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    for r in rows:
        print(
            f"delta {r['delta']:+d}: S* {r['s_star']:5.2f}%  C_RBC {r['cost_rbc_per_day']:.2f}"
            f"  savings {r['savings_mean']:+.2f} +- {r['savings_sd']:.2f}"
        )
    print(f"R2 {r2:.3f} slope {slope:.3f}; lowest->highest {res['lowest_to_highest']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=int)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    elif a.delta is not None:
        run_delta(a.delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
