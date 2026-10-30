"""Comfort-weighted reward: on-site table (Table A.2).

- Runs: pretrain_<b>_smud_sac_plus__r1cw{1..4} (penalty = rho x cost term over
  the RBC season, rho = 0.01 / 0.1 / 1 / 10); rho = 0 = the cost-only runs
  pretrain_<b>_smud_sac_plus, same seeds (default: the paper's five seeds).
- best = the evaluation season with the highest mean_reward (the Table 3 code
  path in extract_paper_tables.py, i.e. the training objective); final = the
  tenth evaluation. Savings vs each building's RBC, cost =
  mean_reward_energy_term x steps x 900 / 3.6e6. +- = ddof 1 over seeds.
- Violation time: logged comfort_violation_time(%) of the same evaluation
  season (all season steps), and the same count over occupied weekday steps.
- Realized rho = comfort term / energy term of the final evaluation season.
- Weekday mean tank setpoint of the best and final evaluation seasons, read
  from the run workspaces when they are still present (rho > 0 runs only).

Usage:
    python -m studies.case_study.analysis.r1_comfort_c2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
WS = ROOT / "artifacts/case_study/workspaces"
MAN = ROOT / "artifacts/case_study/manifests"
CFG = ROOT / "studies/case_study/config"
OUT = ROOT / "artifacts/case_study/results"

BUILDINGS = ["office", "lbuilding", "warehouse", "mediumoffice"]
DEFAULT_SEEDS = [42, 123, 777, 2024, 2025]
SEEDS = list(DEFAULT_SEEDS)
RHO = {1: 0.01, 2: 0.1, 3: 1.0, 4: 10.0}
SF = 900.0 / 3.6e6


def _occupied_weekday_steps(key: str) -> int:
    kw = yaml.safe_load((CFG / "recipes.yaml").read_text())["reward_profiles"][
        f"{key}_reward_smud"
    ]["kwargs"]
    start, end = kw["occupied_hours"]
    days = [dt.date(1991, 6, 1) + dt.timedelta(days=i) for i in range(122)]
    weekdays = sum(1 for d in days if d.weekday() not in set(kw["exclude_days"]))
    return weekdays * 4 * (end - start)


def _rbc_cost(key: str) -> float:
    r = pd.read_csv(EXP / f"rbc_{key}_smud_tou" / "progress.csv").iloc[-1]
    return float(r["mean_reward_energy_term"]) * int(r["length(timesteps)"]) * SF


def _eval_rows(case: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(EXP / case / f"seed_{seed}" / "evaluation_metrics.csv")


def _weekday_setpoint(case: str, seed: int, eval_num: int) -> float | None:
    """Weekday mean setpoint (C) of evaluation season `eval_num` (1-based)."""
    root = WS / case
    if not root.exists():
        return None
    for run in root.iterdir():
        man = MAN / case / f"{run.name}.json"
        if not man.exists():
            continue
        if json.loads(man.read_text()).get("metadata", {}).get("seed") != seed:
            continue
        ev = next(run.glob("*_EVAL_*-res1"), None)
        mon = ev / f"episode-{eval_num}" / "monitor" if ev else None
        if mon is None or not (mon / "simulated_actions.csv").exists():
            return None
        a = pd.read_csv(mon / "simulated_actions.csv").iloc[:, 0].to_numpy(float)
        o = pd.read_csv(mon / "observations.csv")
        wk = o["is_weekend"].to_numpy()[: len(a)] < 0.5
        return float(a[wk].mean())
    return None


def run_stats(case: str, key: str, with_setpoint: bool) -> dict:
    rbc = _rbc_cost(key)
    occ = _occupied_weekday_steps(key)
    per = {}
    for s in SEEDS:
        e = _eval_rows(case, s)
        steps = e["length(timesteps)"].astype(float)
        sav = 100.0 * (rbc - e["mean_reward_energy_term"] * steps * SF) / rbc
        ib = int(e["mean_reward"].astype(float).idxmax())
        il = len(e) - 1
        viol = e["comfort_violation_time(%)"].astype(float)
        rec = {
            "best_eval": ib + 1,
            "best": float(sav[ib]),
            "final": float(sav[il]),
            "viol_best": float(viol[ib]),
            "viol_final": float(viol[il]),
            "viol_best_occ": float(viol[ib] * steps[ib] / occ),
            "viol_final_occ": float(viol[il] * steps[il] / occ),
            "rho_final": float(
                e["mean_reward_comfort_term"][il] / e["mean_reward_energy_term"][il]
            ),
        }
        if with_setpoint:
            rec["weekday_setpoint_best"] = _weekday_setpoint(case, s, ib + 1)
            rec["weekday_setpoint_final"] = _weekday_setpoint(case, s, il + 1)
        per[str(s)] = rec

    def agg(k):
        v = [per[str(s)][k] for s in SEEDS if per[str(s)].get(k) is not None]
        if not v:
            return None
        return {
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }

    keys = [k for k in per[str(SEEDS[0])] if k != "best_eval"]
    return {"per_seed": per, **{k: agg(k) for k in keys}}


def main() -> int:
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    SEEDS = list(ap.parse_args().seeds)
    res = {"seeds": SEEDS, "rho": RHO, "buildings": {}}
    for b in BUILDINGS:
        base = run_stats(f"pretrain_{b}_smud_sac_plus", b, with_setpoint=False)
        rows = {"0": base}
        for k, rho in RHO.items():
            r = run_stats(f"pretrain_{b}_smud_sac_plus__r1cw{k}", b, with_setpoint=True)
            r["delta_vs_cost_only"] = {
                "best_savings_pp": r["best"]["mean"] - base["best"]["mean"],
                "viol_pp": r["viol_best"]["mean"] - base["viol_best"]["mean"],
            }
            rows[str(rho)] = r
        res["buildings"][b] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"c2_onsite_{len(SEEDS)}seeds.json"
    path.write_text(json.dumps(res, indent=2))

    print(
        f"{'bldg':<13}{'rho':>6}{'best':>14}{'final':>14}{'viol%':>8}{'occ%':>7}{'rho_fin':>9}{'sp_wkd':>8}"
    )
    for b, rows in res["buildings"].items():
        for rho, r in rows.items():
            sp = r.get("weekday_setpoint_final")
            print(
                f"{b:<13}{rho:>6}{r['best']['mean']:8.2f}+-{r['best']['sd']:4.2f}"
                f"{r['final']['mean']:8.2f}+-{r['final']['sd']:4.2f}{r['viol_best']['mean']:8.2f}"
                f"{r['viol_best_occ']['mean']:7.2f}{r['rho_final']['mean']:9.3g}"
                f"{(sp['mean'] if sp else float('nan')):8.2f}"
            )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
