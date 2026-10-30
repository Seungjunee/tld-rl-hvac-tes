"""Inclusion checks (a)-(d) and S* for an added building (further buildings).

Applied before any transfer run. The four original buildings are computed with
the same code as the reference for each check.

(a) The RBC season terminates normally: success manifest, full season
    (11,711 steps), EnergyPlus "Completed Successfully" with 0 severe errors.
(b) Design-day chiller PLR inside the paper buildings' guardrail. Design day
    = the season's peak-load step: PLR = max CoolingRate (15-min, RBC season)
    / q_rated. Guardrail [0.80, 1.05], the span of the three paper buildings
    sized to their load (OF 0.99, WH 1.03, MO 0.83); LB (0.52) is the
    documented oversized exception and does not widen it.
(c) RBC violation time < 5 % of all season steps (Table 3 metric). The share
    of occupied weekday steps is reported alongside.
(d) Predictor horizon-mean nRMSE < 5 % of rated capacity on the held-out
    last 20 % (as in analysis/plot_predictor.py), for both roles: source
    (fixed-setpoint log) and zero-shot target (RBC log).
    Occupied windows for (c) come from each reward profile; daylight-saving
    models carry theirs in the standard-time label.
S* = 100 x on-peak (17-20 h) share of the season cooling load, from the
    RBC operation log the predictor is fitted on (paper eq. for S*).

Usage:
    python -m studies.case_study.analysis.r1_inclusion --candidate restaurant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from studies.case_study.core.load_predictor import (
    BUILDING_INPUT_VARIABLES,
    LoadDataset,
    LoadPredictor,
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
PRED = ROOT / "artifacts/case_study/predictors"
CFG = ROOT / "studies/case_study/config"
OUT = ROOT / "artifacts/case_study/results"

SEASON_STEPS = 11711
PLR_GUARD = (0.80, 1.05)
VIOL_MAX = 5.0
NRMSE_MAX = 5.0
ONPEAK = (17, 18, 19)

PAPER = ["office", "lbuilding", "warehouse", "mediumoffice"]
REG = {
    "office": "office_5zone",
    "lbuilding": "office_lbuilding_8zone",
    "warehouse": "warehouse_3zone",
    "mediumoffice": "mediumoffice_15zone",
    "restaurant": "restaurant_3zone",
    "stripmall": "stripmall_10zone",
    "retail": "retail_4zone",
    "fastfood": "fastfood_2zone",
}


def _hour(df: pd.DataFrame) -> np.ndarray:
    if "hour" in df.columns:
        return df["hour"].to_numpy().astype(int) % 24
    ang = np.arctan2(df["hour_sin"], df["hour_cos"])
    return np.rint((ang % (2 * np.pi)) / (2 * np.pi) * 24).astype(int) % 24


def _q_rated_kw(key: str) -> float:
    b = yaml.safe_load((CFG / "buildings.yaml").read_text())["buildings"]
    return float(b[REG[key]]["physical"]["q_rated"]) / 1000.0


def _reward_kwargs(key: str) -> dict:
    r = yaml.safe_load((CFG / "recipes.yaml").read_text())["reward_profiles"]
    return r[f"{key}_reward_smud"]["kwargs"]


def _eplus_summary(manifest: dict) -> str | None:
    sel = manifest.get("resolved_episode_selectors", {}).get("last")
    if not sel:
        return None
    err = Path(sel["source_path"]) / "output" / "eplusout.err"
    if not err.exists():
        return None
    lines = [
        ln for ln in err.read_text(errors="ignore").splitlines() if "Completed" in ln
    ]
    return lines[-1].strip() if lines else "no completion line"


def rule_a(key: str) -> dict:
    d = EXP / f"rbc_{key}_smud_tou"
    manifest = json.loads((d / "run_manifest.json").read_text())
    last = pd.read_csv(d / "progress.csv").iloc[-1]
    steps = int(last["length(timesteps)"])
    eplus = _eplus_summary(manifest)
    ok = manifest.get("status") == "success" and steps == SEASON_STEPS
    if eplus is not None:
        ok = ok and "Completed Successfully" in eplus and " 0 Severe" in eplus
    return {
        "status": manifest.get("status"),
        "git_sha": manifest.get("git_sha"),
        "steps": steps,
        "eplus": eplus,
        "pass": ok,
    }


def rule_b(key: str) -> dict:
    o = pd.read_csv(EXP / f"rbc_{key}_smud_tou" / "observations.csv")
    q = _q_rated_kw(key)
    cr = o["CoolingRate"].to_numpy()
    wk = o["is_weekend"].to_numpy() > 0.5
    peak_kw = float(cr.max()) / 1000.0
    plr = peak_kw / q
    return {
        "q_rated_kw": q,
        "peak_load_kw": peak_kw,
        "peak_row": int(cr.argmax()),
        "plr_peak": plr,
        "guardrail": list(PLR_GUARD),
        "weekend_load_share": float(cr[wk].sum() / cr.sum()),
        "pass": PLR_GUARD[0] <= plr <= PLR_GUARD[1],
    }


def rule_c(key: str) -> dict:
    d = EXP / f"rbc_{key}_smud_tou"
    full = float(pd.read_csv(d / "progress.csv").iloc[-1]["comfort_violation_time(%)"])
    o = pd.read_csv(d / "observations.csv").iloc[1:]  # row 0 = reset
    info = pd.read_csv(d / "infos.csv").iloc[1:]
    kw = _reward_kwargs(key)
    start, end = kw["occupied_hours"]
    h = _hour(o)
    occupied = (o["is_weekend"].to_numpy() < 0.5) & (h >= start) & (h < end)
    viol = info["total_temperature_violation"].to_numpy() > 0
    return {
        "viol_full_pct": full,
        "viol_steps": int(viol.sum()),
        "occupied_weekday_steps": int(occupied.sum()),
        "viol_occupied_pct": 100.0 * viol.sum() / occupied.sum(),
        "viol_outside_occupied": int((viol & ~occupied).sum()),
        "band": kw.get("zone_comfort_ranges") or kw["range_comfort_summer"],
        "occupied_hours": [start, end],
        "label_window_viol_full_pct": _label_window_viol(key, kw),
        "pass": full < VIOL_MAX,
    }


def _label_window_viol(key: str, kw: dict) -> float | None:
    """Candidates whose reward window was DST-aligned ([s-1, e-1) in the
    label): violation time on the unaligned label window [s, e), i.e. the
    original schedule hours used as labels, from the RBC log's raw zone
    temperatures. None for the paper buildings (windows never shifted)."""
    if key in PAPER or list(kw["occupied_hours"]) == [0, 24]:
        return None
    df = pd.read_csv(EXP / f"build_{key}_observation_dataset_rbc" / "observations.csv")
    df = df.iloc[1:]
    start, end = kw["occupied_hours"]
    lo, hi = kw["range_comfort_summer"]
    h = _hour(df)
    import datetime as dt

    if "month" in df.columns:
        wd = np.array(
            [
                dt.date(1991, int(m), int(d)).weekday()
                for m, d in zip(df["month"], df["day_of_month"])
            ]
        )
    else:  # datetime-encoded log (MO-shaped export): weekend flag only
        wd = np.where(df["is_weekend"].to_numpy() > 0.5, 6, 0)
    occ = (h >= start + 1) & (h < end + 1) & ~np.isin(wd, list(kw["exclude_days"]))
    temps = np.column_stack([df[v].to_numpy() for v in kw["temperature_variables"]])
    zr = kw.get("zone_comfort_ranges")
    if zr:
        lo = np.array([zr[v][0] for v in kw["temperature_variables"]])
        hi = np.array([zr[v][1] for v in kw["temperature_variables"]])
    viol = ((temps < lo) | (temps > hi)).any(axis=1) & occ
    return float(100.0 * viol.sum() / len(df))


def _nrmse(key: str, predictor: str, dataset: str) -> dict:
    reg = REG[key]
    ck = torch.load(PRED / predictor, map_location="cpu", weights_only=False)
    H, S = int(ck["horizon"]), int(ck["seq_len"])
    ds = LoadDataset(
        EXP / dataset / "observations_processed.csv",
        BUILDING_INPUT_VARIABLES[reg],
        S,
        H,
    )
    m = LoadPredictor(
        len(BUILDING_INPUT_VARIABLES[reg]), ck["hidden_dim"], ck["num_layers"], H
    )
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n = len(ds)
    P, A = [], []
    with torch.no_grad():
        for i in range(int(n * 0.8), n):
            x, y = ds[i]
            P.append(m(x.unsqueeze(0)).squeeze(0).numpy())
            A.append(y.numpy())
    P = np.clip(np.array(P) * ds.target_std + ds.target_mean, 0, None)
    A = np.array(A) * ds.target_std + ds.target_mean
    qr_w = _q_rated_kw(key) * 1000.0
    per_h = [
        float(np.sqrt(np.mean((P[:, h] - A[:, h]) ** 2)) / qr_w * 100) for h in range(H)
    ]
    r2 = 1 - ((P - A) ** 2).sum() / ((A - A.mean()) ** 2).sum()
    return {
        "predictor": predictor,
        "dataset": dataset,
        "horizon": H,
        "nrmse_pct": float(np.mean(per_h)),
        "nrmse_h1_pct": per_h[0],
        "nrmse_hH_pct": per_h[-1],
        "r2": float(r2),
    }


def rule_d(key: str) -> dict:
    """Both predictor roles: canonical (source role, fixed-setpoint log; the
    paper Fig. 4 predictors) and rbclog (zero-shot target role)."""
    reg = REG[key]
    out = {
        "canonical": _nrmse(key, f"{reg}_lstm.pt", f"build_{key}_observation_dataset"),
        "rbclog": _nrmse(
            key, f"{reg}_lstm_rbclog.pt", f"build_{key}_observation_dataset_rbc"
        ),
    }
    out["pass"] = all(out[r]["nrmse_pct"] < NRMSE_MAX for r in ("canonical", "rbclog"))
    return out


def s_star(key: str) -> dict:
    df = pd.read_csv(EXP / f"build_{key}_observation_dataset_rbc" / "observations.csv")
    cr = df["CoolingRate"].to_numpy()
    on = np.isin(_hour(df), ONPEAK)
    return {"s_star_pct": float(100.0 * cr[on].sum() / cr.sum()), "log": "rbc dataset"}


def evaluate(key: str, paper: bool) -> dict:
    res = {
        "a": rule_a(key),
        "b": rule_b(key),
        "c": rule_c(key),
        "d": rule_d(key),
    }
    res["s_star"] = s_star(key)
    res["include"] = all(res[r]["pass"] for r in "abcd")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--candidate", required=True, choices=[k for k in REG if k not in PAPER]
    )
    args = ap.parse_args(argv)

    report = {
        "rules": {
            "plr_guardrail": list(PLR_GUARD),
            "viol_max_pct": VIOL_MAX,
            "nrmse_max_pct": NRMSE_MAX,
        }
    }
    for key in PAPER + [args.candidate]:
        report[key] = evaluate(key, paper=key in PAPER)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"x1_inclusion_{args.candidate}.json"
    path.write_text(json.dumps(report, indent=2))

    print(
        f"{'bldg':<13}{'(a)':>5} {'PLR':>6}{'(b)':>5} {'viol%':>7}{'occ%':>7}{'(c)':>5} "
        f"{'nRMSErb':>8}{'(d)':>5} {'S*':>6}  include"
    )
    for key in PAPER + [args.candidate]:
        r = report[key]
        print(
            f"{key:<13}{'ok' if r['a']['pass'] else 'FAIL':>5} {r['b']['plr_peak']:6.3f}"
            f"{'ok' if r['b']['pass'] else 'FAIL':>5} {r['c']['viol_full_pct']:7.2f}"
            f"{r['c']['viol_occupied_pct']:7.2f}{'ok' if r['c']['pass'] else 'FAIL':>5} "
            f"{r['d']['rbclog']['nrmse_pct']:8.2f}{'ok' if r['d']['pass'] else 'FAIL':>5} "
            f"{r['s_star']['s_star_pct']:6.2f}  {r['include']}"
        )
    for key in PAPER + [args.candidate]:
        c = report[key]["d"]["canonical"]
        print(
            f"  {key} canonical (source-role) predictor: nRMSE {c['nrmse_pct']:.2f}% R2 {c['r2']:.3f}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
