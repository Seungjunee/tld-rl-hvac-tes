"""Tier-2 capstone: causal timing sweep.

Holds the frozen shared office policy fixed and sweeps a building's shiftability
M by a controlled internal-gain timing shift (tier2_variants.py). For each Delta:
  1. build a variant observation dataset (reuses build_office_observation_dataset
     on the variant env)               -> observations_processed.csv
  2. refit the LSTM load predictor on that dataset (TLD-RL Stage-3 refit)
  3. deploy the FROZEN office policy zero-shot on the variant (variant predictor)
     and deterministically evaluate     -> cost, savings vs the variant's RBC

Everything except the gain timing is held fixed (same envelope, equipment, COP,
climate, policy, predictor architecture, normalization). If savings rise
monotonically with M, source/target shiftability -- not zone count -- causally
drives transfer, confirming Tier-1.

Outputs tier2_sweep.json + fig_tier2_savings_vs_M(.pdf/.png).

Usage:
  python -m studies.case_study.analysis.tier2_run --deltas 0 1 2 3 \
      [--policy-case pretrain_office_smud_sac_plus] [--eval-only]
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "artifacts/case_study/tier2_variants"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
PRED_DIR = ROOT / "artifacts/case_study/predictors"
DS_DIR = ROOT / "artifacts/case_study/exports/tier2_datasets"
DAYS = 122


def _env_id(delta: int) -> str:
    return f"Eplus-5zoneOFshift{delta}-mixed-continuous-v1"


def _override_env(case, delta: int):
    """Point a resolved case at the variant building env (keep office physical
    params + wrappers; only the epJSON/registration changes)."""
    import sinergym

    yaml_path = CFG / f"5ZoneOF_shift{delta}.yaml"
    sinergym.register_envs_from_yaml(str(yaml_path))
    case.env = dict(case.env)
    case.env["register_yaml"] = (
        f"artifacts/case_study/tier2_variants/5ZoneOF_shift{delta}.yaml"
    )
    case.env["id"] = _env_id(delta)
    case.env["name"] = _env_id(delta)
    return case


def prepare_dataset(delta: int, registry) -> Path:
    from studies.case_study.core.registry import resolve_case
    from studies.case_study.core.runner import run_case

    out_csv = DS_DIR / f"shift{delta}" / "observations_processed.csv"
    if out_csv.exists():
        print(f"  [dataset] delta={delta} cached -> {out_csv}")
        return out_csv
    case = deepcopy(resolve_case("build_office_observation_dataset", registry))
    case.id = f"tier2_dataset_shift{delta}"
    _override_env(case, delta)
    case.export = dict(case.export)
    case.export["destination"] = str(DS_DIR / f"shift{delta}")
    print(f"  [dataset] delta={delta} building variant dataset ...")
    run_case(case, registry)
    if not out_csv.exists():
        raise FileNotFoundError(f"expected {out_csv} after dataset run")
    return out_csv


def refit_predictor(delta: int, csv_path: Path) -> Path:
    from studies.case_study.cli.train_predictor import _run_building

    pt = PRED_DIR / f"office_shift{delta}_lstm.pt"
    if pt.exists():
        print(f"  [predictor] delta={delta} cached -> {pt}")
        return pt

    class A:  # minimal args namespace for _run_building
        data = str(csv_path)
        output = str(pt)
        plot_dir = str(OUT / "tier2_predictor_plots")
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
        seed = 42  # train_predictor CLI default (arg added after this rig)

    print(f"  [predictor] delta={delta} refitting LSTM ...")
    _run_building("office_5zone", A())
    return pt


def best_policy_path(policy_case: str) -> Path:
    """Highest final-mean-reward seed of the trained source policy."""
    import csv as _csv

    base = ROOT / f"artifacts/case_study/exports/{policy_case}"
    best, bv = None, -1e18
    for seed_dir in sorted(base.glob("seed_*")):
        prog = seed_dir / "progress.csv"
        mp = seed_dir / "model.zip"
        if not prog.exists() or not mp.exists():
            continue
        rows = list(_csv.DictReader(open(prog)))
        v = float(rows[-1]["mean_reward"])
        if v > bv:
            bv, best = v, mp
    if best is None:
        # legacy single-run export
        mp = base / "model.zip"
        if mp.exists():
            best = mp
    if best is None:
        raise FileNotFoundError(f"no trained policy under {base}")
    print(f"  [policy] using {best} (final mean_reward {bv:.3f})")
    return best


def eval_policy(delta: int, policy_zip: Path, predictor_pt: Path, registry) -> dict:
    from stable_baselines3 import SAC

    from studies.case_study.core.env_builder import build_environment
    from studies.case_study.core.registry import resolve_case
    from studies.case_study.core.types import RunContext

    case = deepcopy(resolve_case("pretrain_office_smud_sac_plus", registry))
    case.id = f"tier2_eval_shift{delta}"
    _override_env(case, delta)
    case.wrappers = deepcopy(case.wrappers)
    case.wrappers["load_predictor"] = dict(case.wrappers["load_predictor"])
    case.wrappers["load_predictor"]["model_path"] = str(predictor_pt.relative_to(ROOT))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ctx = RunContext(
        case_id=case.id,
        run_id=f"tier2_{ts}",
        started_at=ts,
        workspace_root=ROOT / "artifacts/case_study/workspaces/tier2_eval",
        export_root=ROOT / "artifacts/case_study/exports/tier2_eval",
        figure_root=ROOT / "artifacts/case_study/figures",
        manifest_root=ROOT / "artifacts/case_study/manifests",
        third_party_root=ROOT / "third_party",
        project_root=ROOT,
    )
    env = build_environment(case, ctx, mode="train")
    model = SAC.load(str(policy_zip), env=env)
    obs, _ = env.reset()
    cost, viol, n, done = 0.0, 0, 0, False
    while not done:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _reward, term, trunc, info = env.step(action)
        p, pr = info.get("total_power_demand"), info.get("current_price")
        if p is not None and pr is not None:
            cost += float(p) * 900 / 3.6e6 * float(pr)
        if float(info.get("total_temperature_violation", 0.0)) > 0:
            viol += 1
        n += 1
        done = term or trunc
    env.close()
    return {"cost_per_day": cost / DAYS, "viol_pct": 100 * viol / max(n, 1), "steps": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--policy-case", default="pretrain_office_smud_sac_plus")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="assume datasets+predictors exist; only run policy eval",
    )
    args = ap.parse_args()

    from studies.case_study.core.registry import load_registry

    registry = load_registry(str(ROOT / "studies/case_study/config"))
    DS_DIR.mkdir(parents=True, exist_ok=True)

    rbc = json.load(open(OUT / "tier2_rbc_metrics.json"))
    policy = best_policy_path(args.policy_case)

    results = {}
    for d in args.deltas:
        print(f"\n===== Delta = {d} =====")
        if (
            d == 0
            and (PRED_DIR / "office_5zone_lstm.pt").exists()
            and not (PRED_DIR / "office_shift0_lstm.pt").exists()
        ):
            pt = PRED_DIR / "office_5zone_lstm.pt"  # delta 0 == base office
            print(f"  [predictor] delta=0 reuse base office predictor")
        else:
            csv_path = prepare_dataset(d, registry)
            pt = refit_predictor(d, csv_path)
        ev = eval_policy(d, policy, pt, registry)
        m = rbc[str(d)]
        sav = 100 * (m["cost_rbc_per_day"] - ev["cost_per_day"]) / m["cost_rbc_per_day"]
        results[str(d)] = {
            "delta": d,
            "M3_oracle_sched_pct": m["M3_oracle_sched_pct"],
            "onpeak_share_pct": m["onpeak_share_pct"],
            "pws": m["pws"],
            "cost_rbc_per_day": m["cost_rbc_per_day"],
            "cost_policy_per_day": ev["cost_per_day"],
            "savings_pct": sav,
            "viol_pct": ev["viol_pct"],
        }
        print(
            f"  RESULT delta={d}: M3={m['M3_oracle_sched_pct']:.1f}% "
            f"savings={sav:+.1f}% (C_RBC ${m['cost_rbc_per_day']:.2f} -> "
            f"C_pol ${ev['cost_per_day']:.2f})"
        )

    path = OUT / "tier2_sweep.json"
    if path.exists():
        prev = json.load(open(path))
        prev.update(results)
        results = prev
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
