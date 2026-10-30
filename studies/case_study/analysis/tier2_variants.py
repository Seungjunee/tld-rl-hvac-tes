"""Tier-2 controlled timing-shift ablation: build office load-timing variants.

Lever (controlled ablation, ONLY timing): roll the internal-gain weekday
schedules (occupancy, equipment, lights) later by Delta hours. np.roll preserves
the daily integral exactly -- envelope, area, thermal mass, equipment capacity,
climate, and total internal-gain energy are all unchanged; only the hour-of-day
of the gains moves. Shifting the 08-17h gain block later pushes cooling demand
into the 17-20h on-peak window, raising shiftability M.

HVAC availability (ON 06-22h) and the cooling-setpoint window (06-22h) already
span every shifted block, so cooling is enabled across the moved demand; leaving
them fixed keeps the ablation clean (only demand timing varies).

For each Delta it writes:
  sinergym/data/buildings/5ZoneOF_shift{D}.epJSON         (variant building)
  sinergym/data/default_configuration/5ZoneOF_shift{D}.yaml (env registration)

Usage:
  python -m studies.case_study.analysis.tier2_variants --deltas 0 1 2 3
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BLD = ROOT / "sinergym/data/buildings"
CFG = ROOT / "sinergym/data/default_configuration"
# Variant registration yamls live OUTSIDE default_configuration so sinergym's
# import-time auto-scan does not register the ablation envs globally; the Tier-2
# scripts register them explicitly by path. The epJSONs stay in buildings/
# (where building_file is resolved) but are not auto-registered.
VARIANT_CFG = ROOT / "artifacts/case_study/tier2_variants"
BASE_EPJSON = BLD / "5ZoneTankFixed_final.epJSON"
BASE_YAML = CFG / "5ZoneTankFixed.yaml"

GAIN_SCHEDULES = ["BLDG_OCC_SCH", "BLDG_EQUIP_SCH", "BLDG_LIGHT_SCH"]
WEEKDAY_TAG = "Weekdays"


def _parse_until(field: str) -> int:
    m = re.match(r"Until:\s*(\d{1,2}):(\d{2})", field)
    h = int(m.group(1))
    return 24 if h == 0 and m.group(2) == "00" and field.strip().endswith("0:00") else h


def compact_weekday_to_hourly(sched: dict) -> np.ndarray:
    """Expand the 'For: Weekdays' block of a Schedule:Compact into 24 hourly values."""
    data = [d["field"] for d in sched["data"]]
    # find the Weekdays day-type block (until the next 'For:' or end)
    i = 0
    while i < len(data):
        f = data[i]
        if isinstance(f, str) and f.startswith("For:") and WEEKDAY_TAG in f:
            break
        i += 1
    i += 1
    hourly = np.zeros(24)
    prev = 0
    while i < len(data):
        f = data[i]
        if isinstance(f, str) and f.startswith("For:"):
            break
        if isinstance(f, str) and f.startswith("Until:"):
            until = _parse_until(f)
            val = float(data[i + 1])
            hourly[prev:until] = val
            prev = until
            i += 2
        else:
            i += 1
    return hourly


def hourly_to_compact_block(hourly: np.ndarray) -> list:
    """Emit 24 hourly (Until: HH:00, value) field dicts."""
    out = []
    for h in range(1, 25):
        tag = f"Until: {h:02d}:00" if h < 24 else "Until: 24:00"
        out.append({"field": tag})
        out.append({"field": float(round(hourly[h - 1], 4))})
    return out


def roll_weekday_block(sched: dict, delta: int) -> dict:
    """Return a copy of the Schedule:Compact with its Weekdays block rolled by
    +delta hours (later), other day-types untouched."""
    sched = copy.deepcopy(sched)
    hourly = compact_weekday_to_hourly(sched)
    rolled = np.roll(hourly, delta)
    data = sched["data"]
    # locate Weekdays block span [start, end)
    start = -1
    for idx, d in enumerate(data):
        f = d["field"]
        if isinstance(f, str) and f.startswith("For:") and WEEKDAY_TAG in f:
            start = idx + 1
            break
    if start < 0:
        raise ValueError("Weekdays block not found")
    end = start
    while end < len(data):
        f = data[end]["field"]
        if isinstance(f, str) and f.startswith("For:"):
            break
        end += 1
    new_block = hourly_to_compact_block(rolled)
    sched["data"] = data[:start] + new_block + data[end:]
    return sched


def make_variant(delta: int) -> tuple[Path, Path]:
    ep = json.load(open(BASE_EPJSON))
    for s in GAIN_SCHEDULES:
        ep["Schedule:Compact"][s] = roll_weekday_block(ep["Schedule:Compact"][s], delta)
    out_ep = BLD / f"5ZoneOF_shift{delta}.epJSON"
    json.dump(ep, open(out_ep, "w"))

    cfg = yaml.safe_load(open(BASE_YAML))
    cfg["id_base"] = f"5zoneOFshift{delta}"
    cfg["building_file"] = f"5ZoneOF_shift{delta}.epJSON"
    VARIANT_CFG.mkdir(parents=True, exist_ok=True)
    out_yaml = VARIANT_CFG / f"5ZoneOF_shift{delta}.yaml"
    yaml.safe_dump(cfg, open(out_yaml, "w"), sort_keys=False, default_flow_style=False)
    return out_ep, out_yaml


def verify_shift(delta: int):
    """Sanity print: weekday occupancy hourly profile before/after roll."""
    ep = json.load(open(BASE_EPJSON))
    base = compact_weekday_to_hourly(ep["Schedule:Compact"]["BLDG_OCC_SCH"])
    rolled = np.roll(base, delta)
    onpeak = slice(17, 20)
    print(
        f"  delta={delta}: OCC on-peak(17-20) mean {base[onpeak].mean():.2f} "
        f"-> {rolled[onpeak].mean():.2f}  | integral {base.sum():.2f} "
        f"-> {rolled.sum():.2f} (preserved)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    print("=== timing-shift variants (office) ===")
    for d in args.deltas:
        ep, yml = make_variant(d)
        verify_shift(d)
        print(f"    wrote {ep.name}, {yml.name}")


if __name__ == "__main__":
    main()
