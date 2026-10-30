"""Padding with frozen features (method `padding_frozen`) against Padding fine-tuning
(Table 4 Padding columns).

- Cases: transfer_<s>_to_<t>_smud_sacp_padff (12 pairs of the four original
  buildings), each the twin of the Table 4 Padding fine-tuning case
  ..._padrl; five FT seeds from the best padded source seed.
- best: each run's best of its ten seasonal evaluations (highest mean_reward),
  five-seed mean +- sd (ddof 1); the Table 4 fine-tuning convention.
  final: the tenth evaluation.
- first season (online): training season 1 (progress.csv row 1), the cost paid
  while the policy learns.
- per-season evaluation: season 0 is the deterministic zero-shot run of the
  shared starting policy (<padrl>__pad0best), seasons 1-10 the evaluations.
- Violation time: the logged comfort_violation_time(%) (occupied weekday
  violation steps over all control steps); max |run - target RBC| reported.
- Classes: similar-load (no warehouse), to WH, from WH, all 12.

Gate (--gate --seeds ...): success manifest, ten evaluations and ten training
seasons, finite rewards, the sanity criteria of _flags, and the freeze check
(frozen layers unchanged, every trainable tensor updated, target first layer
equal to the online one within 1e-5, source model and initial state equal to
the padrl twin's).

Usage:
    python -m studies.case_study.analysis.r1_padff --gate --seeds 42
    python -m studies.case_study.analysis.r1_padff
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/results"
SEEDS = [42, 123, 777, 2024, 2025]
SF = 900.0 / 3.6e6
MAX_SAVINGS, MAX_DVIOL = 30.0, 1.0
SEASON_STEPS = 11711
PAPER = ["office", "lbuilding", "warehouse", "mediumoffice"]
PAIRS = [(s, t) for s in PAPER for t in PAPER if s != t]
LABEL = {"office": "OF", "lbuilding": "LB", "warehouse": "WH", "mediumoffice": "MO"}
SEASONS = 10
TARGET_TOL = 1e-5


def _cost(row) -> float:
    return float(row["mean_reward_energy_term"]) * float(row["length(timesteps)"]) * SF


def _rbc(b: str) -> tuple[float, float]:
    r = pd.read_csv(EXP / f"rbc_{b}_smud_tou" / "progress.csv").iloc[-1]
    return _cost(r), float(r["comfort_violation_time(%)"])


def _msd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "n": len(v),
        "per_seed": v,
    }


def _flags(rows: pd.DataFrame, rbc) -> list[str]:
    """Sanity flags: savings > 30 % vs RBC, violation time more than 1 pp away
    from the RBC's, or a season that is not 11,711 steps long."""
    cost = (
        rows["mean_reward_energy_term"].astype(float)
        * rows["length(timesteps)"].astype(float)
        * SF
    )
    sav = 100.0 * (rbc[0] - cost) / rbc[0]
    dv = (rows["comfort_violation_time(%)"].astype(float) - rbc[1]).abs()
    out = []
    if sav.max() > MAX_SAVINGS:
        out.append(f"savings {sav.max():.2f} %")
    if dv.max() > MAX_DVIOL:
        out.append(f"|violation - RBC| {dv.max():.2f} pp")
    short = rows["length(timesteps)"].astype(float) != SEASON_STEPS
    if short.any():
        out.append(f"{int(short.sum())} season(s) not {SEASON_STEPS} steps")
    return out


def _case(s: str, t: str, arm: str) -> str:
    return f"transfer_{s}_to_{t}_smud_sacp_{arm}"


def _class(s: str, t: str) -> str:
    if s == "warehouse":
        return "from_wh"
    if t == "warehouse":
        return "to_wh"
    return "similar"


def _sav(row, rbc: float) -> float:
    return 100.0 * (rbc - _cost(row)) / rbc


def run_metrics(case: str, t: str, seed: int) -> dict | None:
    d = EXP / case / f"seed_{seed}"
    ev, pr = d / "evaluation_metrics.csv", d / "progress.csv"
    if not ev.exists() or not pr.exists():
        return None
    rbc, rv = _rbc(t)
    e, p = pd.read_csv(ev), pd.read_csv(pr)
    ib = int(e["mean_reward"].astype(float).idxmax())
    viol = e["comfort_violation_time(%)"].astype(float)
    return {
        "best": _sav(e.iloc[ib], rbc),
        "best_season": int(e.iloc[ib]["evaluation_num"]),
        "final": _sav(e.iloc[-1], rbc),
        "first_online": _sav(p.iloc[0], rbc),
        "eval_by_season": [_sav(r, rbc) for _, r in e.iterrows()],
        "online_by_season": [_sav(r, rbc) for _, r in p.iterrows()],
        "viol_best": float(viol.iloc[ib]),
        "max_abs_dviol": float((viol - rv).abs().max()),
    }


def season0(s: str, t: str) -> float | None:
    p = EXP / f"{_case(s, t, 'padrl')}__pad0best" / "progress.csv"
    if not p.exists():
        return None
    return _sav(pd.read_csv(p).iloc[-1], _rbc(t)[0])


def arm_pair(s: str, t: str, arm: str, seeds) -> dict:
    runs = {k: run_metrics(_case(s, t, arm), t, k) for k in seeds}
    runs = {k: v for k, v in runs.items() if v is not None}
    out = {"n": len(runs), "by_seed": {str(k): v for k, v in runs.items()}}
    for key in ("best", "final", "first_online"):
        out[key] = _msd([v[key] for v in runs.values()])
    if runs:
        out["eval_mean_by_season"] = [
            float(np.mean([v["eval_by_season"][i] for v in runs.values()]))
            for i in range(SEASONS)
        ]
        out["max_abs_dviol"] = max(v["max_abs_dviol"] for v in runs.values())
    return out


# --------------------------------------------------------------------- gate
def _source_sha(path: str) -> str:
    from stable_baselines3 import SAC

    from studies.case_study.methods.padding_transfer import _state_sha256

    return _state_sha256(SAC.load(path).policy)


def gate(seeds) -> int:
    issues, sha_cache = [], {}
    for s, t in PAIRS:
        case = _case(s, t, "padff")
        twin = json.loads(
            (EXP / _case(s, t, "padrl") / "seed_42" / "run_manifest.json").read_text()
        )
        for k in seeds:
            d = EXP / case / f"seed_{k}"
            tag = f"{LABEL[s]}->{LABEL[t]} seed {k}"
            man = d / "run_manifest.json"
            if not man.exists():
                issues.append(f"{tag}: no manifest")
                continue
            m = json.loads(man.read_text())
            md = m.get("metadata", {})
            if m.get("status") != "success":
                issues.append(f"{tag}: status {m.get('status')}")
            e = pd.read_csv(d / "evaluation_metrics.csv")
            p = pd.read_csv(d / "progress.csv")
            if len(e) != SEASONS or len(p) < SEASONS:
                issues.append(f"{tag}: {len(e)} evaluations, {len(p)} seasons")
            vals = p[[c for c in p.columns if c.startswith("mean_reward")]]
            if not all(math.isfinite(float(x)) for x in vals.to_numpy().ravel()):
                issues.append(f"{tag}: non-finite reward")
            for f in _flags(e, _rbc(t)):
                issues.append(f"{tag}: anomaly {f}")
            if md.get("method") != "padding_frozen" or not md.get(
                "freeze_first_layer"
            ):
                issues.append(f"{tag}: not a frozen run ({md.get('method')})")
            chk = md.get("freeze_check") or {}
            if chk.get("frozen_max_abs_change") != 0.0:
                issues.append(f"{tag}: frozen layers moved {chk}")
            if chk.get("trainable_tensors_changed") != chk.get(
                "trainable_tensors_total"
            ):
                issues.append(f"{tag}: trainable tensors not all updated {chk}")
            if not chk.get("target_first_layer_max_abs_dev", 1.0) <= TARGET_TOL:
                issues.append(f"{tag}: target first layer off {chk}")
            src = md.get("source_model")
            if src != twin["source_bundle"]:
                issues.append(f"{tag}: source {src} != padrl {twin['source_bundle']}")
            elif md.get("initial_state_sha256") != sha_cache.setdefault(
                src, _source_sha(src)
            ):
                issues.append(f"{tag}: initial state differs from the source zip")
            print(
                f"{tag}: frozen change {chk.get('frozen_max_abs_change')}, "
                f"trainable {chk.get('trainable_tensors_changed')}/"
                f"{chk.get('trainable_tensors_total')} "
                f"(min max|d| {chk.get('trainable_min_max_abs_change', float('nan')):.2e}), "
                f"target dev {chk.get('target_first_layer_max_abs_dev', float('nan')):.1e}, "
                f"source {Path(src or '').parent.name}"
            )
    for x in issues:
        print("ISSUE", x)
    print(f"gate: {len(PAIRS) * len(seeds)} runs, {len(issues)} issues")
    return 1 if issues else 0


# ------------------------------------------------------------------- report
def _fmt(m):
    return "n/a" if not m else f"{m['mean']:+.1f} ± {m['sd']:.1f}"


def report(seeds) -> dict:
    res = {"seeds": list(seeds), "pairs": {}}
    for s, t in PAIRS:
        res["pairs"][f"{s}->{t}"] = {
            "class": _class(s, t),
            "season0_zero_shot": season0(s, t),
            "padrl": arm_pair(s, t, "padrl", seeds),
            "padff": arm_pair(s, t, "padff", seeds),
        }
    classes = {
        "similar": [k for k, v in res["pairs"].items() if v["class"] == "similar"],
        "to_wh": [k for k, v in res["pairs"].items() if v["class"] == "to_wh"],
        "from_wh": [k for k, v in res["pairs"].items() if v["class"] == "from_wh"],
        "all12": list(res["pairs"]),
    }
    res["class_means"] = {}
    for c, keys in classes.items():
        row = {}
        for arm in ("padrl", "padff"):
            for key in ("best", "final", "first_online"):
                v = [res["pairs"][k][arm][key] for k in keys]
                if all(v):
                    row[f"{arm}_{key}"] = float(np.mean([x["mean"] for x in v]))
        row["season0"] = float(
            np.mean([res["pairs"][k]["season0_zero_shot"] for k in keys])
        )
        res["class_means"][c] = row

    print("| pair | season 0 | Padding FT best | frozen best | frozen final "
          "| frozen first season | max|dviol| |")
    for k, v in res["pairs"].items():
        s, t = k.split("->")
        ff = v["padff"]
        print(
            f"| {LABEL[s]}->{LABEL[t]} | {v['season0_zero_shot']:+.1f} "
            f"| {_fmt(v['padrl']['best'])} | {_fmt(ff['best'])} "
            f"| {_fmt(ff['final'])} | {_fmt(ff['first_online'])} "
            f"| {ff.get('max_abs_dviol', float('nan')):.2f} |"
        )
    for c, row in res["class_means"].items():
        print(c, {k: round(x, 2) for k, x in row.items()})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args(argv)
    if args.gate:
        return gate(args.seeds)
    res = report(args.seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "padff.json" if args.seeds == SEEDS else "padff_pilot.json"
    (OUT / name).write_text(json.dumps(res, indent=2))
    print("wrote", OUT / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
