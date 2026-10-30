"""Extract the numbers of paper Tables 3 (source domain) and 4 (transfer).

Cost formula (validated against direct hourly-price integration, ±0.1%):
    cost_$ = mean_reward_energy_term x n_steps x 900s / 3.6e6
Best checkpoint = argmax mean_reward over the EVAL workspace's progress.csv
rows (one row per deterministic evaluation episode); the same row supplies
that checkpoint's energy term (cost) and comfort_violation_time(%).
RBC and zero-shot runs are deterministic: any row works (last used).

Outputs figures/paper_tables_<date>/paper_tables.json.

`--dc-only` prints just the DC-fleet zero-shot block (Table 9), which reads
the exports/<case>__src<seed>/ layout and does not depend on the legacy EVAL
workspaces that the source-domain sections glob (those may be cleaned).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP_FACTOR = 900.0 / 3.6e6  # $ per (W-mean x steps) at $/kWh prices
DAYS = 122

import itertools as _it

BLDGS = ["office", "lbuilding", "warehouse", "mediumoffice"]
PAD_BLDGS = (
    BLDGS  # (a) 2026-06-27: max_zones=15, ALL 5 buildings have padding (MO included)
)
PAIRS = [f"{s}_to_{t}" for s, t in _it.permutations(BLDGS, 2)]  # 20 TLD-RL zs pairs
PADRL_PAIRS = [
    f"{s}_to_{t}" for s, t in _it.permutations(PAD_BLDGS, 2)
]  # 20 Padding padrl pairs


def rows_of(path: str):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run_cost_from_progress(rows, row_idx=-1):
    r = rows[row_idx]
    n = int(float(r["length(timesteps)"]))
    cost = float(r["mean_reward_energy_term"]) * n * STEP_FACTOR
    viol = float(r["comfort_violation_time(%)"])
    return cost, viol


def best_eval_cost(case: str, seed: int):
    """Cost/violation of the best deterministic checkpoint of one seeded run."""
    pats = sorted(
        glob.glob(
            f"{ROOT}/artifacts/case_study/workspaces/{case}/*/*_EVAL_*-res1/progress.csv"
        )
    )
    # Map run dir -> seed by content-matching the run's TRAIN progress against
    # the exported seed progress (identical file). TRAIN dir is whichever
    # *-res1 sibling is not the EVAL one (custom env_name for transfers).
    exp = Path(f"{ROOT}/artifacts/case_study/exports/{case}/seed_{seed}/progress.csv")
    exp_txt = exp.read_text()
    for ev in pats:
        run_dir = Path(ev).parent.parent
        train = [
            f
            for f in glob.glob(str(run_dir / "*-res1/progress.csv"))
            if "_EVAL_" not in f
        ]
        if train and Path(train[0]).read_text() == exp_txt:
            ev_rows = rows_of(ev)
            best = max(
                range(len(ev_rows)), key=lambda i: float(ev_rows[i]["mean_reward"])
            )
            return run_cost_from_progress(ev_rows, best)
    raise FileNotFoundError(f"no matching eval workspace for {case} seed {seed}")


def seeded_stats(case: str):
    vals = [best_eval_cost(case, s) for s in (42, 123, 777, 2024, 2025)]
    costs = [v[0] for v in vals]
    viols = [v[1] for v in vals]
    n = len(costs)
    mean = sum(costs) / n
    std = (sum((c - mean) ** 2 for c in costs) / (n - 1)) ** 0.5
    return {"cost_mean": mean, "cost_std": std, "viol_mean": sum(viols) / n}


def flat_cost(case: str):
    rows = rows_of(f"{ROOT}/artifacts/case_study/exports/{case}/progress.csv")
    return run_cost_from_progress(rows)


# ---- DC fleet (paper Table 9): zero-shot under the rbclog protocol, one
# deterministic episode per source seed, exported as <case>__src<seed>/.
# The data-center fleet extension is not part of this release.
DC_FLEET_PAIRS = []
SRC_SEEDS = (42, 123, 777, 2024, 2025)


def src_seed_zs_stats(pair_case: str, rbc_cost: float):
    """mean±sd savings over the five __src<seed> deterministic zs exports."""
    vals = []
    for s in SRC_SEEDS:
        path = f"{ROOT}/artifacts/case_study/exports/{pair_case}__src{s}/progress.csv"
        if not Path(path).exists():
            return None
        cost, _ = run_cost_from_progress(rows_of(path))
        vals.append(100 * (rbc_cost - cost) / rbc_cost)
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
    return {"savings_mean": mean, "savings_std": std, "per_seed": vals}


def dc_fleet_table() -> dict:
    """Compute and print the DC-fleet zero-shot cells (paper Table 9)."""
    result = {}
    print("=== DC fleet zs (five source seeds; Table 9) ===")
    for src, tgt in DC_FLEET_PAIRS:
        case = f"transfer_{src}_to_{tgt}_smud_sacp_zs"
        rbc_cost, _ = flat_cost(f"rbc_{tgt}_smud_tou")
        stats = src_seed_zs_stats(case, rbc_cost)
        if stats is None:
            print(f"[dc_transfer] skip {case}: __src seed exports incomplete")
            continue
        result[f"{src}_to_{tgt}"] = stats
        per_seed = " ".join(f"{v:+.2f}" for v in stats["per_seed"])
        print(
            f"{f'{src}_to_{tgt}':30s} {stats['savings_mean']:+6.2f} ± {stats['savings_std']:4.2f}   [{per_seed}]"
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dc-only",
        action="store_true",
        help="Print only the DC-fleet zero-shot block (no legacy EVAL globbing)",
    )
    if ap.parse_args().dc_only:
        dc_fleet_table()
        return

    out = {"source": {}, "transfer": {}}

    # ---- Table 3: source-domain performance ----
    for b in BLDGS:
        rbc_cost, rbc_viol = flat_cost(f"rbc_{b}_smud_tou")
        e2e = seeded_stats(f"pretrain_{b}_smud_sac_e2e")
        plus = seeded_stats(f"pretrain_{b}_smud_sac_plus")
        row = {
            "rbc": {
                "cost": rbc_cost,
                "cost_per_day": rbc_cost / DAYS,
                "viol": rbc_viol,
            },
            "e2e": {
                **e2e,
                "cost_per_day": e2e["cost_mean"] / DAYS,
                "savings_pct": 100 * (rbc_cost - e2e["cost_mean"]) / rbc_cost,
            },
            "proposed": {
                **plus,
                "cost_per_day": plus["cost_mean"] / DAYS,
                "savings_pct": 100 * (rbc_cost - plus["cost_mean"]) / rbc_cost,
            },
        }
        # Padding source-domain row (sacp_e2e = from-scratch padded RL); MO omitted in (b).
        if b in PAD_BLDGS:
            pad = seeded_stats(f"pretrain_{b}_smud_sacp_e2e")
            row["padding"] = {
                **pad,
                "cost_per_day": pad["cost_mean"] / DAYS,
                "savings_pct": 100 * (rbc_cost - pad["cost_mean"]) / rbc_cost,
            }
        out["source"][b] = row

    # ---- Table 4: transfer ----
    for p in PAIRS:
        tgt = p.split("_to_")[1]
        rbc_cost = out["source"][tgt]["rbc"]["cost"]
        e2e_sav = out["source"][tgt]["e2e"]["savings_pct"]

        zs_cost, _ = flat_cost(f"transfer_{p}_smud_sacp_zs")
        zs_sav = 100 * (rbc_cost - zs_cost) / rbc_cost

        entry = {
            "e2e_savings_pct": e2e_sav,
            "proposed_savings_pct": zs_sav,
            "proposed_delta_pp": zs_sav - e2e_sav,
        }
        if p in PADRL_PAIRS:
            pad = seeded_stats(f"transfer_{p}_smud_sacp_padrl")
            pad_sav = 100 * (rbc_cost - pad["cost_mean"]) / rbc_cost
            entry["padding_savings_pct"] = pad_sav
            entry["padding_delta_pp"] = pad_sav - e2e_sav
        out["transfer"][p] = entry

    # ---- Table 9: DC fleet zero-shot (five source seeds; zs cells only —
    # the E2E asymptote column uses the DC pretrains' own convention) ----
    out["dc_transfer"] = dc_fleet_table()

    outdir = Path(f"{ROOT}/artifacts/case_study/figures/paper_tables_2026-06-11")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "paper_tables.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- pretty print ----
    print("=== Table 3 (source) ===")
    print(
        f"{'bldg':12s} {'RBC $/d':>8s} {'E2E sav%':>9s} {'Pad sav%':>9s} {'Prop sav%':>9s} {'viol%':>6s}"
    )
    for b in BLDGS:
        s = out["source"][b]
        pad = f"{s['padding']['savings_pct']:9.2f}" if "padding" in s else f"{'--':>9s}"
        print(
            f"{b:12s} {s['rbc']['cost_per_day']:8.2f} {s['e2e']['savings_pct']:9.2f} "
            f"{pad} {s['proposed']['savings_pct']:9.2f} {s['rbc']['viol']:6.2f}"
        )
    print("\n=== Table 4 (transfer) ===")
    print(
        f"{'pair':30s} {'E2E sav%':>9s} {'Pad sav%':>9s} {'ZS sav%':>8s} {'ZSΔpp':>7s}"
    )
    for p in PAIRS:
        t = out["transfer"][p]
        pad = (
            f"{t['padding_savings_pct']:9.2f}"
            if "padding_savings_pct" in t
            else f"{'--':>9s}"
        )
        print(
            f"{p:30s} {t['e2e_savings_pct']:9.2f} {pad} "
            f"{t['proposed_savings_pct']:8.2f} {t['proposed_delta_pp']:7.2f}"
        )


if __name__ == "__main__":
    main()
