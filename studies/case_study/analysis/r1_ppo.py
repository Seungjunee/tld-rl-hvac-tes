"""PPO replication: on-site and zero-shot results (Table 6).

- Arms: TLD-RL-PPO = pretrain_<b>_smud_sac_plus__r1ppo, Padding-PPO =
  pretrain_<b>_smud_sacp_e2e__r1ppo, seeds 42/123/777/2024/2025. The SAC
  sources of the same states are computed with the same code as the reference.
- On-site: best = the evaluation season with the highest mean_reward (Table 3
  code path); final = the tenth evaluation. Savings vs the building's RBC,
  cost = mean_reward_energy_term x steps x 900 / 3.6e6; +- = ddof 1 over
  seeds. Violation time: logged value (all season steps) and the same count
  over occupied weekday steps.
- Zero-shot: each source seed's final model on the target (<case>__src<seed>,
  scripts/run_r1_zs.py); TLD-RL-PPO on the target rbclog predictor,
  Padding-PPO on the padded state, both frozen. Selected (deployed) = the
  source seed with the highest final training reward (last complete-season
  progress.csv row, the paper's k*), chosen per arm.
- Similar-load pairs = the six directed pairs among OF, LB and MO.
- Per arm and building, the savings of each evaluation season (five-seed mean)
  and the best-season indices are also written.

Usage:
    python -m studies.case_study.analysis.r1_ppo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from studies.case_study.analysis.r1_comfort_c2 import _occupied_weekday_steps

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/results"
SF = 900.0 / 3.6e6
SEEDS = [42, 123, 777, 2024, 2025]
BUILDINGS = ["office", "lbuilding", "warehouse", "mediumoffice"]
LABEL = {"office": "OF", "lbuilding": "LB", "warehouse": "WH", "mediumoffice": "MO"}
SIMILAR = ["office", "lbuilding", "mediumoffice"]
SEASON_STEPS = 11711
ARMS = {
    "tld_ppo": "pretrain_{b}_smud_sac_plus__r1ppo",
    "pad_ppo": "pretrain_{b}_smud_sacp_e2e__r1ppo",
    "tld_sac": "pretrain_{b}_smud_sac_plus",
    "pad_sac": "pretrain_{b}_smud_sacp_e2e",
}
ZS = {
    "tld_ppo": "transfer_{s}_to_{t}_smud_sacp_zs__r1ppo__src{k}",
    "pad_ppo": "transfer_{s}_to_{t}_smud_sacp_padzs__r1ppo__src{k}",
    "tld_sac": "transfer_{s}_to_{t}_smud_sacp_zs__src{k}",
}
ZS_SOURCE = {"tld_ppo": "tld_ppo", "pad_ppo": "pad_ppo", "tld_sac": "tld_sac"}


def _cost(row) -> float:
    return float(row["mean_reward_energy_term"]) * float(row["length(timesteps)"]) * SF


def _rbc_cost(b: str) -> float:
    return _cost(pd.read_csv(EXP / f"rbc_{b}_smud_tou" / "progress.csv").iloc[-1])


def _msd(v: list[float]) -> dict | None:
    if not v:
        return None
    return {
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "n": len(v),
    }


def onsite(arm: str, b: str) -> dict | None:
    rbc, occ = _rbc_cost(b), _occupied_weekday_steps(b)
    per, curves = {}, []
    for s in SEEDS:
        f = EXP / ARMS[arm].format(b=b) / f"seed_{s}" / "evaluation_metrics.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f)
        sav = [100.0 * (rbc - _cost(r)) / rbc for _, r in e.iterrows()]
        ib, il = int(e["mean_reward"].astype(float).idxmax()), len(e) - 1
        viol = e["comfort_violation_time(%)"].astype(float)
        steps = e["length(timesteps)"].astype(float)
        per[str(s)] = {
            "n_evals": len(e),
            "best_eval": ib + 1,
            "best": sav[ib],
            "final": sav[il],
            "viol_best": float(viol[ib]),
            "viol_final": float(viol[il]),
            "viol_best_occ": float(viol[ib] * steps[ib] / occ),
            "viol_final_occ": float(viol[il] * steps[il] / occ),
        }
        curves.append(sav)
    if not per:
        return None
    n = min(len(c) for c in curves)
    return {
        "per_seed": per,
        **{
            k: _msd([p[k] for p in per.values()])
            for k in (
                "best",
                "final",
                "viol_best",
                "viol_final",
                "viol_best_occ",
                "viol_final_occ",
            )
        },
        "eval_curve_mean": [float(np.mean([c[i] for c in curves])) for i in range(n)],
        "best_eval_indices": [p["best_eval"] for p in per.values()],
    }


def deployed_seed(arm: str, b: str) -> int | None:
    """k*: highest final training reward = the last COMPLETE training season.

    PPO's 58 rollouts overshoot ten seasons by 1,674 steps, so its progress.csv
    ends with a partial 11th-season row, which is skipped (SAC runs have ten
    full rows).
    """
    best, arg = -np.inf, None
    for s in SEEDS:
        p = EXP / ARMS[arm].format(b=b) / f"seed_{s}" / "progress.csv"
        if p.exists():
            prog = pd.read_csv(p)
            full = prog[prog["length(timesteps)"].astype(int) == SEASON_STEPS]
            v = float(full.iloc[-1]["mean_reward"])
            if v > best:
                best, arg = v, s
    return arg


def zero_shot(arm: str) -> dict:
    res = {}
    for s in BUILDINGS:
        k_star = deployed_seed(ZS_SOURCE[arm], s)
        for t in BUILDINGS:
            if s == t:
                continue
            rbc = _rbc_cost(t)
            per = {}
            for k in SEEDS:
                p = EXP / ZS[arm].format(s=s, t=t, k=k) / "progress.csv"
                if p.exists():
                    r = pd.read_csv(p).iloc[-1]
                    per[k] = (
                        100.0 * (rbc - _cost(r)) / rbc,
                        float(r["comfort_violation_time(%)"]),
                    )
            res[f"{s}->{t}"] = {
                "per_seed": {str(k): v[0] for k, v in per.items()},
                "viol_per_seed": {str(k): v[1] for k, v in per.items()},
                **(_msd([v[0] for v in per.values()]) or {"mean": None, "n": 0}),
                "deployed_seed": k_star,
                "deployed": per[k_star][0] if k_star in per else None,
                "similar_load": s in SIMILAR and t in SIMILAR,
            }
    sim = [v for v in res.values() if v["similar_load"] and v["n"] == len(SEEDS)]
    summary = {
        "similar_complete_pairs": len(sim),
        "similar_mean": float(np.mean([v["mean"] for v in sim])) if sim else None,
        "similar_positive_pairs": sum(v["mean"] > 0 for v in sim),
        "similar_deployed_mean": (
            float(np.mean([v["deployed"] for v in sim])) if sim else None
        ),
        "all12_mean": (
            float(np.mean([v["mean"] for v in res.values() if v["n"] == len(SEEDS)]))
            if any(v["n"] == len(SEEDS) for v in res.values())
            else None
        ),
    }
    return {"pairs": res, "summary": summary}


def main() -> int:
    res = {"onsite": {}, "zero_shot": {}}
    for arm in ARMS:
        res["onsite"][arm] = {b: onsite(arm, b) for b in BUILDINGS}
    for arm in ZS:
        res["zero_shot"][arm] = zero_shot(arm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p_ppo.json").write_text(json.dumps(res, indent=2))

    def fmt(x):
        return f"{x['mean']:+.2f}+-{x['sd']:.2f}(n{x['n']})" if x else "-"

    for b in BUILDINGS:
        print(
            f"{LABEL[b]} "
            + " | ".join(
                f"{arm} best {fmt(res['onsite'][arm][b] and res['onsite'][arm][b]['best'])}"
                f" final {fmt(res['onsite'][arm][b] and res['onsite'][arm][b]['final'])}"
                for arm in ARMS
            )
        )
    for arm in ZS:
        print(arm, "zs", res["zero_shot"][arm]["summary"])
    print(f"wrote {OUT / 'p_ppo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
