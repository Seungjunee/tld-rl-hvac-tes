"""De-risk check (mandatory before the full Tier-2 sweep): confirm the timing
shift produces a real 17-20h cooling-load bump in EnergyPlus, not just at the
schedule level. Runs the raw variant env under the RBC TOU setpoint policy and
reports the on-peak load share + daily load profile per Delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import sinergym  # noqa: F401  (registers envs / API)

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "artifacts/case_study/tier2_variants"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SMUD = [0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4


def rbc_episode(env_id: str) -> dict:
    env = gym.make(env_id)
    obs_vars = list(env.get_wrapper_attr("observation_variables"))
    ci = obs_vars.index("CoolingRate")
    hi = obs_vars.index("hour")
    ei = obs_vars.index("ChillerElectricity")
    obs, _ = env.reset()
    d_by_h = {h: [0.0, 0] for h in range(24)}
    p_by_h = {h: 0.0 for h in range(24)}
    done = False
    while not done:
        h = int(round(float(obs[hi]))) % 24
        setpoint = 5.0 if h < 12 else 10.0  # RBC: charge off-peak, hold otherwise
        obs, _, term, trunc, _ = env.step(np.array([setpoint], dtype=np.float32))
        done = term or trunc
        d_by_h[h][0] += float(obs[ci])
        d_by_h[h][1] += 1
        p_by_h[h] += float(obs[ei])
    env.close()
    prof = np.array([d_by_h[h][0] / max(d_by_h[h][1], 1) for h in range(24)])  # W
    total = prof.sum()
    onpeak = prof[17:20].sum() / total if total else 0.0
    shape = prof / total if total else prof
    pws = float(np.sum(np.array(SMUD) * prof) / (np.mean(SMUD) * total)) if total else 0
    return {
        "profile_W": prof.tolist(),
        "onpeak_share": onpeak,
        "pws": pws,
        "total_W": float(total),
        "shape": shape.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 3])
    args = ap.parse_args()
    res = {}
    for d in args.deltas:
        sinergym.register_envs_from_yaml(str(CFG / f"5ZoneOF_shift{d}.yaml"))
        env_id = f"Eplus-5zoneOFshift{d}-mixed-continuous-v1"
        print(f"=== running RBC on delta={d} ({env_id}) ===")
        r = rbc_episode(env_id)
        res[d] = r
        print(
            f"  on-peak(17-20) load share: {100*r['onpeak_share']:.1f}%  "
            f"PWS: {r['pws']:.3f}  total(meanW): {r['total_W']:.0f}"
        )
    (OUT / "tier2_load_verify.json").write_text(json.dumps(res, indent=2))
    print("\n=== daily cooling-load profile (mean W/h), by hour ===")
    print("hour " + " ".join(f"d{d:>6}" for d in args.deltas))
    for h in range(24):
        tag = "*" if 17 <= h < 20 else " "
        print(
            f"{h:>2}{tag}  "
            + " ".join(f"{res[d]['profile_W'][h]:>7.0f}" for d in args.deltas)
        )
    print(f"\nwritten: {OUT}/tier2_load_verify.json")


if __name__ == "__main__":
    main()
