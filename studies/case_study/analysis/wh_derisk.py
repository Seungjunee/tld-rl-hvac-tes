"""Feasibility de-risk: can the WAREHOUSE cooling load be pushed into the 17-20h
on-peak window by shifting its operating schedules?

The warehouse load is largely envelope/infiltration driven (weather-locked to
mid-afternoon), so this is NOT assumed -- it is measured. We roll every
schedulable load driver (occupancy, lights, equipment, infiltration door-opening)
later by Delta hours and run RBC, then compare the on-peak load share to baseline.

If the share rises substantially, a warehouse-source timing sweep is feasible
(the positive "rescue the warehouse" story). If it barely moves, the warehouse
load is intrinsically envelope-locked and we fall back to office.

Usage: python -m studies.case_study.analysis.wh_derisk --deltas 0 3
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

import sinergym  # noqa: F401
from studies.case_study.analysis.tier2_variants import roll_weekday_block

ROOT = Path(__file__).resolve().parents[3]
BLD = ROOT / "sinergym/data/buildings"
CFG = ROOT / "sinergym/data/default_configuration"
VARIANT_CFG = ROOT / "artifacts/case_study/tier2_variants"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SMUD = np.array([0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4)

# every schedulable load driver in the warehouse (gains + infiltration)
WH_LOAD_SCHEDULES = [
    "BLDG_OCC_SCH",
    "BLDG_LIGHT_STORAGE_SCH",
    "BLDG_LIGHT_OFFICE_SCH",
    "Bulk Storage Plug Schedule",
    "Office_Plug_SCH",
    "Bulk Storage Infil_Base Schedule",
    "Fine Sorage Infil Schedule",
    "INFIL_Door_Opening_SCH",
    "Office Infil Schedule",
]


def make_wh_variant(delta: int):
    ep = json.load(open(BLD / "Warehouse_final.epJSON"))
    sc = ep["Schedule:Compact"]
    shifted = []
    for name in WH_LOAD_SCHEDULES:
        if name not in sc:
            continue
        try:
            sc[name] = roll_weekday_block(copy.deepcopy(sc[name]), delta)
            shifted.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"   (skip {name}: {exc})")
    out_ep = BLD / f"WH_shift{delta}.epJSON"
    json.dump(ep, open(out_ep, "w"))
    cfg = yaml.safe_load(open(CFG / "Warehouse.yaml"))
    cfg["id_base"] = f"WHshift{delta}"
    cfg["building_file"] = f"WH_shift{delta}.epJSON"
    VARIANT_CFG.mkdir(parents=True, exist_ok=True)
    out_yaml = VARIANT_CFG / f"WH_shift{delta}.yaml"
    yaml.safe_dump(cfg, open(out_yaml, "w"), sort_keys=False)
    print(f"  delta={delta}: shifted {len(shifted)} schedules -> {out_ep.name}")
    return out_yaml


def rbc_onpeak(env_id: str) -> dict:
    env = gym.make(env_id)
    ov = list(env.get_wrapper_attr("observation_variables"))
    ci, hi = ov.index("CoolingRate"), ov.index("hour")
    obs, _ = env.reset()
    by_h = {h: [0.0, 0] for h in range(24)}
    done = False
    while not done:
        h = int(round(float(obs[hi]))) % 24
        obs, _, term, trunc, _ = env.step(
            np.array([5.0 if h < 12 else 10.0], dtype=np.float32)
        )
        done = term or trunc
        by_h[h][0] += float(obs[ci])
        by_h[h][1] += 1
    env.close()
    prof = np.array([by_h[h][0] / max(by_h[h][1], 1) for h in range(24)])
    total = prof.sum()
    onpeak = float(prof[17:20].sum() / total) if total else 0.0
    pws = float(np.sum(SMUD * prof) / (SMUD.mean() * total)) if total else 0.0
    return {"onpeak_share": onpeak, "pws": pws, "profile_W": prof.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 3])
    args = ap.parse_args()
    res = {}
    for d in args.deltas:
        yml = make_wh_variant(d)
        sinergym.register_envs_from_yaml(str(yml))
        env_id = f"Eplus-WHshift{d}-mixed-continuous-v1"
        print(f"  running RBC on {env_id} ...")
        res[str(d)] = rbc_onpeak(env_id)
        print(
            f"    delta={d}: on-peak share {100*res[str(d)]['onpeak_share']:.1f}%  "
            f"PWS {res[str(d)]['pws']:.3f}"
        )
    (OUT / "wh_derisk.json").write_text(json.dumps(res, indent=2))
    print("\n=== VERDICT ===")
    base = res.get("0", {}).get("onpeak_share", 0) * 100
    hi = max(res[k]["onpeak_share"] for k in res) * 100
    print(
        f"  on-peak share: baseline {base:.1f}% -> max {hi:.1f}%  "
        f"(delta {hi - base:+.1f} pts)"
    )
    print(
        "  WAREHOUSE FEASIBLE for source sweep"
        if hi - base > 4
        else "  WAREHOUSE WEAK (envelope-locked) -> fall back to office"
    )
    # hourly profile dump for the extremes
    print("\nhour " + " ".join(f"d{d:>5}" for d in args.deltas))
    for h in range(24):
        tag = "*" if 17 <= h < 20 else " "
        print(
            f"{h:>2}{tag} "
            + " ".join(f"{res[str(d)]['profile_W'][h]:>6.0f}" for d in args.deltas)
        )


if __name__ == "__main__":
    main()
