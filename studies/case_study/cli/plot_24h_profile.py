"""24-h HVAC operating profile (power + tank temp + setpoint action).

Two stacked panels sharing the x-axis (hour 0-24):
    (a) HVAC electrical power [kW] = ChillerElectricity / 1000
    (b) TES tank temperature [°C] (state) + CWtank setpoint [°C] (action)

Background shading indicates TOU peak / mid-peak / off-peak bands.
Single curve per quantity, weekday-only average across all days of the
selected episode (default: best-seed Ep20).

Usage:
    python -m studies.case_study.cli.plot_24h_profile \
        --case pretrain_office_policy --episode 20 --seed best
    python -m studies.case_study.cli.plot_24h_profile \
        --case pretrain_office_policy --seed 42 --episode 20 \
        --out artifacts/case_study/figures/paper/foo.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from studies.case_study.core import load_registry


TOU_BANDS_DEFAULT = [
    (0, 7, "Off-peak", 0.10),
    (7, 19, "Peak", 0.25),
    (19, 24, "Mid-peak", 0.15),
]


def _select_best_seed(case_export_dir: Path, episode: int) -> Optional[int]:
    """Pick seed with highest mean_reward at the given episode."""
    best_seed: Optional[int] = None
    best_score = -float("inf")
    for sd in sorted(case_export_dir.glob("seed_*")):
        prog = sd / "progress.csv"
        if not prog.exists():
            continue
        try:
            df = pd.read_csv(prog)
            row = df[df.episode_num == episode]
            if row.empty:
                continue
            score = float(row.mean_reward.iloc[0])
            if score > best_score:
                best_score = score
                seed_id = int(sd.name.replace("seed_", ""))
                best_seed = seed_id
        except Exception:
            continue
    return best_seed


def _find_run_workspace(case_id: str, seed: int, manifest_root: Path) -> Optional[Path]:
    """Find the workspace path of the run that produced seed_<N> export."""
    case_manifests = manifest_root / case_id
    if not case_manifests.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for mf in case_manifests.glob("*.json"):
        try:
            payload = json.loads(mf.read_text())
            if int(payload.get("metadata", {}).get("seed", -1)) == seed:
                ws = Path(payload["workspace_path"])
                if ws.exists():
                    candidates.append((mf.stat().st_mtime, ws))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _episode_obs_path(workspace: Path, episode: int) -> Optional[Path]:
    """Locate episode-N/monitor/observations.csv inside the workspace."""
    direct = workspace / f"episode-{episode}" / "monitor" / "observations.csv"
    if direct.exists():
        return direct
    for eplus_dir in workspace.glob("Eplus-*"):
        obs = eplus_dir / f"episode-{episode}" / "monitor" / "observations.csv"
        if obs.exists():
            return obs
    return None


def _load_price_schedule(registry, reward_profile: str) -> Optional[list[float]]:
    rp = registry.reward_profiles.get(reward_profile, {})
    return rp.get("kwargs", {}).get("price_schedule")


def _bands_from_price(price_schedule: Optional[list[float]]) -> list[tuple[int, int, str, float]]:
    """Convert hourly price array (len 24) into contiguous TOU bands."""
    if not price_schedule or len(price_schedule) != 24:
        return TOU_BANDS_DEFAULT
    p_min, p_max = min(price_schedule), max(price_schedule)
    label_alpha = {p_min: ("Off-peak", 0.10), p_max: ("Peak", 0.25)}
    bands: list[tuple[int, int, str, float]] = []
    h0, cur = 0, price_schedule[0]
    for h in range(1, 24):
        if price_schedule[h] != cur:
            label, alpha = label_alpha.get(cur, ("Mid-peak", 0.15))
            bands.append((h0, h, label, alpha))
            h0, cur = h, price_schedule[h]
    label, alpha = label_alpha.get(cur, ("Mid-peak", 0.15))
    bands.append((h0, 24, label, alpha))
    return bands


def plot_profile(case_id: str, episode: int, seed: str | int, out_path: Path) -> None:
    registry = load_registry()
    export_root = registry.paths["export_root"]
    manifest_root = registry.paths["manifest_root"]
    case_dir = export_root / case_id

    if seed == "best":
        chosen = _select_best_seed(case_dir, episode)
        if chosen is None:
            raise SystemExit(f"No seeded exports for {case_id}; cannot pick best.")
        seed_int = chosen
    else:
        seed_int = int(seed)

    workspace = _find_run_workspace(case_id, seed_int, manifest_root)
    if workspace is None:
        raise SystemExit(f"Workspace not found for {case_id} seed={seed_int}")
    obs_path = _episode_obs_path(workspace, episode)
    if obs_path is None:
        raise SystemExit(f"episode-{episode}/monitor/observations.csv missing in {workspace}")

    df = pd.read_csv(obs_path)
    act_path = obs_path.parent / "agent_actions.csv"
    act = pd.read_csv(act_path)
    df = df.iloc[: len(act)].copy()
    setpoint_col = act.columns[0]
    df["__setpoint"] = act[setpoint_col].values

    hour = (np.arctan2(df["hour_sin"], df["hour_cos"]) / (2 * np.pi) * 24) % 24
    df["hour"] = np.round(hour).astype(int) % 24
    df["power_kW"] = df["ChillerElectricity"] / 1000.0

    df_wk = df[df.get("is_weekend", 0) == 0]
    if df_wk.empty:
        df_wk = df

    agg = df_wk.groupby("hour").agg(
        power_mean=("power_kW", "mean"),
        power_std=("power_kW", "std"),
        tank_mean=("CWtank_final", "mean"),
        tank_std=("CWtank_final", "std"),
        setp_mean=("__setpoint", "mean"),
        setp_std=("__setpoint", "std"),
    )
    agg.loc[24] = agg.loc[0]
    agg = agg.sort_index()

    case = registry.cases[case_id] if case_id in registry.cases else None
    reward_profile = None
    if case is not None:
        reward_profile = case.reward_profile or registry.buildings[case.building].reward_profile
    price_schedule = _load_price_schedule(registry, reward_profile) if reward_profile else None
    tou_bands = _bands_from_price(price_schedule)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax in axes:
        for h0, h1, _, alpha in tou_bands:
            ax.axvspan(h0, h1, color="gray", alpha=alpha, zorder=0)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(alpha=0.25, zorder=1)

    # Panel (a) power
    ax = axes[0]
    ax.fill_between(
        agg.index, agg.power_mean - agg.power_std, agg.power_mean + agg.power_std,
        color="tab:blue", alpha=0.15, zorder=2,
    )
    ax.plot(
        agg.index, agg.power_mean, color="tab:blue", linewidth=2.2,
        label=f"Proposed ({case_id}, Ep{episode}, seed={seed_int})", zorder=3,
    )
    ax.set_ylabel("HVAC electrical power (kW)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(a)", loc="left", fontsize=11, fontweight="bold")

    y_top = ax.get_ylim()[1]
    for h0, h1, label, _ in tou_bands:
        ax.text((h0 + h1) / 2, y_top * 0.96, label,
                ha="center", va="top", fontsize=9, color="dimgray", fontweight="bold")

    # Panel (b) tank state + setpoint action
    ax = axes[1]
    ax.fill_between(
        agg.index, agg.tank_mean - agg.tank_std, agg.tank_mean + agg.tank_std,
        color="tab:orange", alpha=0.15, zorder=2,
    )
    ax.plot(agg.index, agg.tank_mean, color="tab:orange", linewidth=2.2,
            label="Tank temperature (state)", zorder=3)
    ax.plot(agg.index, agg.setp_mean, color="tab:red", linewidth=1.8, linestyle="--",
            label="CWtank setpoint (action)", zorder=4)
    ax.fill_between(
        agg.index, agg.setp_mean - agg.setp_std, agg.setp_mean + agg.setp_std,
        color="tab:red", alpha=0.10, zorder=2,
    )
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Hour of day")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(b)", loc="left", fontsize=11, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  Power range: {agg.power_mean.min():.2f} - {agg.power_mean.max():.2f} kW")
    print(f"  Tank range:  {agg.tank_mean.min():.2f} - {agg.tank_mean.max():.2f} °C")
    print(f"  Setpoint:    {agg.setp_mean.min():.2f} - {agg.setp_mean.max():.2f} °C")
    print(f"  Source: {obs_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="24-h HVAC operating profile figure.")
    parser.add_argument("--case", required=True, help="Case id (e.g., pretrain_office_policy)")
    parser.add_argument("--episode", type=int, default=20, help="Episode index (default 20)")
    parser.add_argument("--seed", default="best",
                        help="Seed selector: 'best' (default), or specific int (e.g., 42)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: artifacts/case_study/figures/paper/<case>_24h_profile_ep<N>.png)")
    args = parser.parse_args()
    out = args.out or Path(
        f"artifacts/case_study/figures/paper/{args.case}_24h_profile_ep{args.episode}.png"
    )
    plot_profile(args.case, args.episode, args.seed, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
