"""Tier-2 (policy-independent half): RBC baseline cost + shiftability M +
load profile for each office timing-variant. Mirrors the Tier-1 metric pipeline
(lp_benchmark oracle + on-peak share / PWS) but on the synthetic variants.

Runs one RBC TOU episode per Delta on the raw variant env; caches results so the
policy-eval half (tier2_run.py) can reuse them. No policy or predictor needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import sinergym  # noqa: F401
from studies.case_study.analysis import lp_benchmark as lpb

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "artifacts/case_study/tier2_variants"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SMUD = np.array([0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4)
DAYS = 122
Q_RATED = 80.0  # kW (office chiller, unchanged across timing variants)
EMAX = 290.1  # kWh-th (50 m3 x 5 K span)
COP = 3.2
CHILLER_ON_W = 100.0


def rbc_episode(env_id: str) -> dict:
    """Run RBC TOU; aggregate 15-min stream to hourly load/elec/etc."""
    env = gym.make(env_id)
    ov = list(env.get_wrapper_attr("observation_variables"))
    idx = {
        k: ov.index(k)
        for k in [
            "CoolingRate",
            "ChillerElectricity",
            "EvapCoolingRate",
            "hour",
            "EIRfT",
        ]
    }
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        h = int(round(float(obs[idx["hour"]]))) % 24
        setpoint = 5.0 if h < 12 else 10.0
        obs, _, term, trunc, _ = env.step(np.array([setpoint], dtype=np.float32))
        done = term or trunc
        rows.append(
            [
                float(obs[idx["CoolingRate"]]),
                float(obs[idx["ChillerElectricity"]]),
                float(obs[idx["EvapCoolingRate"]]),
                h,
                float(obs[idx["EIRfT"]]),
            ]
        )
    env.close()
    arr = np.array(rows)  # cols: cool, elec, evap, hour, eirft  (15-min)
    nh = len(arr) // 4
    d = np.zeros(nh)  # building load kWh-th/h
    q_rbc = np.zeros(nh)  # chiller thermal kWh/h
    u_rbc = np.zeros(nh)  # chiller on-fraction
    lam = np.zeros(nh)
    cost_elec = 0.0
    eirft_sum, eirft_n = 0.0, 0
    for t in range(nh):
        chunk = arr[t * 4 : (t + 1) * 4]
        d[t] = chunk[:, 0].mean() / 1000.0
        q_rbc[t] = chunk[:, 2].mean() / 1000.0
        u_rbc[t] = (chunk[:, 1] > CHILLER_ON_W).mean()
        h = int(chunk[0, 3]) % 24
        lam[t] = SMUD[h]
        # 15-min electric cost: W * 900s / 3.6e6 J/kWh * price
        cost_elec += float((chunk[:, 1] * 900 / 3.6e6 * SMUD[h]).sum())
        on = chunk[:, 1] > CHILLER_ON_W
        if on.any():
            eirft_sum += float(chunk[on, 4].sum())
            eirft_n += int(on.sum())
    eirft_mean = eirft_sum / eirft_n if eirft_n else 1.0
    return {
        "d": d,
        "q_rbc": q_rbc,
        "u_rbc": u_rbc,
        "lam": lam,
        "cost_rbc": cost_elec,
        "eirft_mean": eirft_mean,
    }


def metrics_for(delta: int) -> dict:
    sinergym.register_envs_from_yaml(str(CFG / f"5ZoneOF_shift{delta}.yaml"))
    env_id = f"Eplus-5zoneOFshift{delta}-mixed-continuous-v1"
    print(f"=== RBC + metrics: delta={delta} ===")
    ep = rbc_episode(env_id)
    d, lam = ep["d"], ep["lam"]
    total = float(d.sum())
    # oracle SCHED savings (reuse base-office chiller curve; only load timing varies)
    a, b, _ = lpb.affine_fit("office", ep["eirft_mean"])
    on_mask = ep["u_rbc"] > 0.25
    q_on = float(np.median(ep["q_rbc"][on_mask] / ep["u_rbc"][on_mask]))
    c_sched, _, _ = lpb.solve_milp_schedule(d, lam, q_on, EMAX, a, b)
    c_rbc_thermal = float(np.sum(lam * (a * ep["u_rbc"] + b * ep["q_rbc"])))
    sched_sav = 100 * (c_rbc_thermal - c_sched) / c_rbc_thermal
    onpeak = float(d[np.isin(lam, [0.3765])].sum() / total)
    pws = float(np.sum(lam * d) / (lam.mean() * total))
    ndays = len(d) // 24
    prof = d[: ndays * 24].reshape(ndays, 24).mean(axis=0)  # 24-h daily mean kWh-th
    out = {
        "delta": delta,
        "cost_rbc_per_day": ep["cost_rbc"] / DAYS,
        "M3_oracle_sched_pct": sched_sav,
        "onpeak_share_pct": 100 * onpeak,
        "pws": pws,
        "total_load_kwh": total,
        "daily_load_profile_kwh": prof.tolist(),
    }
    print(
        f"  C_RBC ${out['cost_rbc_per_day']:.2f}/day | M3(oracle SCHED) "
        f"{sched_sav:.1f}% | on-peak {100*onpeak:.1f}% | PWS {pws:.3f}"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    res = {}
    for d in args.deltas:
        res[str(d)] = metrics_for(d)
    path = OUT / "tier2_rbc_metrics.json"
    # merge with any existing (so partial reruns accumulate)
    if path.exists():
        prev = json.load(open(path))
        prev.update(res)
        res = prev
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
