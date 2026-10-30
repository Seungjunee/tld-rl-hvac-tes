"""Aggregate seed_<N>/progress.csv files for a case into mean ± std.

Usage:
    python -m studies.case_study.cli.aggregate_seeds --case pretrain_restaurant_native_policy
    python -m studies.case_study.cli.aggregate_seeds --group all_pretrains

Writes `<export_root>/<case_id>/aggregate.csv` with columns:
    episode_num, mean_reward_mean, mean_reward_std, n_seeds
Also prints a compact summary table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from studies.case_study.core import list_case_ids, load_registry


def _seed_progress_files(case_export_dir: Path) -> List[Path]:
    if not case_export_dir.exists():
        return []
    return sorted(
        p / "progress.csv"
        for p in case_export_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed_") and (p / "progress.csv").exists()
    )


def aggregate_case(case_id: str, export_root: Path) -> Optional[pd.DataFrame]:
    case_dir = export_root / case_id
    files = _seed_progress_files(case_dir)
    if not files:
        return None
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed_id = int(f.parent.name.replace("seed_", ""))
        df = df[["episode_num", "mean_reward"]].copy()
        df["seed"] = seed_id
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    agg = (
        merged.groupby("episode_num")["mean_reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_reward_mean", "std": "mean_reward_std", "count": "n_seeds"})
    )
    out = case_dir / "aggregate.csv"
    agg.to_csv(out, index=False)
    return agg


def _print_summary(case_id: str, agg: pd.DataFrame) -> None:
    n = int(agg["n_seeds"].max())
    print(f"\n{case_id} ({n} seeds)")
    for ep in [1, 5, 10, 15, 20]:
        hit = agg[agg.episode_num == ep]
        if not hit.empty:
            m = hit.mean_reward_mean.iloc[0]
            s = hit.mean_reward_std.iloc[0]
            print(f"  Ep{ep:2d}: {m:7.2f} ± {s:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seeded progress.csv into mean ± std.")
    parser.add_argument("--case", dest="case_id", help="Single case id")
    parser.add_argument("--group", help="Aggregate all cases in a group")
    parser.add_argument("--all", action="store_true", help="Aggregate all registered cases")
    parser.add_argument("--config-root", default="studies/case_study/config")
    args = parser.parse_args()

    registry = load_registry(args.config_root)
    export_root = registry.paths["export_root"]

    case_ids: list[str] = []
    if args.all:
        case_ids = list_case_ids(registry)
    elif args.group:
        if args.group not in registry.groups:
            raise SystemExit(f"Unknown group: {args.group}")
        case_ids = list(registry.groups[args.group])
    elif args.case_id:
        case_ids = [args.case_id]
    else:
        raise SystemExit("Provide --case, --group, or --all")

    ok = 0
    for cid in case_ids:
        agg = aggregate_case(cid, export_root)
        if agg is None:
            continue
        _print_summary(cid, agg)
        ok += 1
    print(f"\nAggregated {ok}/{len(case_ids)} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
