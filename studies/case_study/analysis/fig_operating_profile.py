"""Operating-profile figure with D-1 / D-Day / D+1 context (case-study redesign).

Extends the source-domain 24-h dispatch (chiller power + TES tank temp, TLD-RL
vs RBC) by tiling the weekday-average daily profile across three consecutive days.
This shows the start-of-day state is not arbitrary but a repeating steady state
The average daily profile is a natural representative
of any adjacent weekday, so D-1 and D+1 are the same profile repeated.

The TLD-RL trajectory is produced by a deterministic rollout of the frozen office
policy on the base office building; RBC comes from the Tier-1 RBC workspace.
Optionally overlays an E2E policy line if its export exists.

Usage:
  python -m studies.case_study.analysis.fig_operating_profile \
      [--policy-case pretrain_office_smud_sac_plus] [--e2e-case ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
# artifacts live in the main checkout; worktrees only carry the repo files
_CANDS = [REPO, Path("/workspaces/sinergym")]
ROOT = next(p for p in _CANDS if (p / "artifacts/case_study/exports").exists())
MAN = REPO / "artifacts/case_study/figures/paper"
MAN.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SMUD = [0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4
C_PROP, C_E2E, C_RBC = "#1f77b4", "#d62728", "0.35"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def weekday_hourly_profile(obs_path: str):
    """Weekday-average chiller power [kW] and tank temp [C] by hour-of-day."""
    with open(obs_path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        col = {h: i for i, h in enumerate(hdr)}
        rows = list(r)
    acc = {h: [0.0, 0.0, 0] for h in range(24)}
    wk = col.get("is_weekend")
    for row in rows:
        if wk is not None and float(row[wk]) > 0.5:
            continue
        ang = math.atan2(float(row[col["hour_sin"]]), float(row[col["hour_cos"]]))
        h = int(round((ang % (2 * math.pi)) / (2 * math.pi) * 24)) % 24
        acc[h][0] += float(row[col["ChillerElectricity"]]) / 1000.0
        acc[h][1] += float(row[col["CWtank_final"]])
        acc[h][2] += 1
    kw = np.array([acc[h][0] / acc[h][2] if acc[h][2] else np.nan for h in range(24)])
    tk = np.array([acc[h][1] / acc[h][2] if acc[h][2] else np.nan for h in range(24)])
    return kw, tk


def deterministic_rollout(policy_case: str) -> str:
    """Run the frozen best-seed policy on base office; return its observations.csv.

    The rollout is deterministic, so an existing observations.csv from a
    previous run of the SAME case is reused (cache dir opprofile/<case>;
    delete it to force a fresh rollout). Pre-2026-07 flat-layout rollouts
    directly under opprofile/ are ignored: they mix the policy and E2E cases
    without a case marker in the path.
    """
    cached = sorted(
        glob.glob(
            f"{ROOT}/artifacts/case_study/workspaces/opprofile/{policy_case}/"
            "**/episode-*/monitor/observations.csv",
            recursive=True,
        )
    )
    if cached:
        print(f"(reusing cached rollout) {cached[-1]}")
        return cached[-1]
    from stable_baselines3 import SAC

    from studies.case_study.analysis.tier2_run import best_policy_path
    from studies.case_study.core.env_builder import build_environment
    from studies.case_study.core.registry import load_registry, resolve_case
    from studies.case_study.core.types import RunContext

    registry = load_registry(str(ROOT / "studies/case_study/config"))
    case = resolve_case(policy_case, registry)
    case.id = f"opprofile_{policy_case}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ws = ROOT / "artifacts/case_study/workspaces/opprofile" / policy_case
    ctx = RunContext(
        case_id=case.id,
        run_id=f"op_{ts}",
        started_at=ts,
        workspace_root=ws,
        export_root=ROOT / "artifacts/case_study/exports/opprofile",
        figure_root=ROOT / "artifacts/case_study/figures",
        manifest_root=ROOT / "artifacts/case_study/manifests",
        third_party_root=ROOT / "third_party",
        project_root=ROOT,
    )
    env = build_environment(case, ctx, mode="train")
    model = SAC.load(str(best_policy_path(policy_case)), env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _r, term, trunc, _i = env.step(a)
        done = term or trunc
    env.close()
    files = sorted(
        glob.glob(f"{ws}/**/episode-*/monitor/observations.csv", recursive=True),
        key=lambda p: int(p.split("episode-")[1].split("/")[0]),
    )
    return files[-1]


def rollout_seed(policy_case: str, seed: int) -> str:
    """Deterministic one-season rollout of one seed's final model.zip (cached)."""
    ws = ROOT / "artifacts/case_study/workspaces/opprofile" / policy_case / f"seed_{seed}"
    cached = sorted(glob.glob(f"{ws}/**/episode-*/monitor/observations.csv", recursive=True))
    if cached:
        return cached[-1]
    from stable_baselines3 import SAC

    from studies.case_study.core.env_builder import build_environment
    from studies.case_study.core.registry import load_registry, resolve_case
    from studies.case_study.core.types import RunContext

    registry = load_registry(str(ROOT / "studies/case_study/config"))
    case = resolve_case(policy_case, registry)
    case.id = f"opprofile_{policy_case}_s{seed}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ctx = RunContext(
        case_id=case.id,
        run_id=f"op_{ts}",
        started_at=ts,
        workspace_root=ws,
        export_root=ROOT / "artifacts/case_study/exports/opprofile",
        figure_root=ROOT / "artifacts/case_study/figures",
        manifest_root=ROOT / "artifacts/case_study/manifests",
        third_party_root=ROOT / "third_party",
        project_root=ROOT,
    )
    env = build_environment(case, ctx, mode="train")
    zip_ = ROOT / f"artifacts/case_study/exports/{policy_case}/seed_{seed}/model.zip"
    model = SAC.load(str(zip_), env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _r, term, trunc, _i = env.step(a)
        done = term or trunc
    env.close()
    files = sorted(
        glob.glob(f"{ws}/**/episode-*/monitor/observations.csv", recursive=True),
        key=lambda p: int(p.split("episode-")[1].split("/")[0]),
    )
    return files[-1]


def rbc_obs() -> str:
    # The canonical TMY3 export comes first. The
    # workspace glob below picked the Pittsburgh held-out-weather RBC run,
    # which shares the case id.
    exp = ROOT / "artifacts/case_study/exports/rbc_office_smud_tou/observations.csv"
    if exp.exists():
        return str(exp)
    files = sorted(
        glob.glob(
            f"{ROOT}/artifacts/case_study/workspaces/rbc_office_smud_tou/"
            "**/episode-*/monitor/observations.csv",
            recursive=True,
        ),
        key=lambda p: int(p.split("episode-")[1].split("/")[0]),
    )
    if files:
        return files[-1]
    # fallback: exported observations (same CSVLogger schema)
    exp = ROOT / "artifacts/case_study/exports/rbc_office_smud_tou/observations.csv"
    assert exp.exists(), "no RBC observations in workspaces or exports"
    return str(exp)


def shade_tou_days(ax, ndays=3):
    for day in range(ndays):
        for h in range(24):
            c = {0.3765: "#ffcccc", 0.2139: "#fff0cc"}.get(SMUD[h])
            if c:
                ax.axvspan(
                    day * 24 + h, day * 24 + h + 1, color=c, alpha=0.55, lw=0, zorder=0
                )


def tile3(arr):
    """Repeat a 24-value daily profile across D-1/D-Day/D+1 (centered at hours)."""
    x = np.concatenate([np.arange(24) + 0.5 + 24 * d for d in range(3)])
    y = np.concatenate([arr, arr, arr])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-case", default="pretrain_office_smud_sac_plus")
    ap.add_argument("--e2e-case", default="pretrain_office_smud_sac_e2e")
    ap.add_argument(
        "--window",
        choices=["3day", "28h"],
        default="28h",
        help="'3day' = D-1/D/D+1 tiling; '28h' = 20:00 of the "
        "preceding day through 24:00 of the day (compact).",
    )
    args = ap.parse_args()

    prop_obs = deterministic_rollout(args.policy_case)
    prop_kw, prop_tk = weekday_hourly_profile(prop_obs)
    rbc_kw, rbc_tk = weekday_hourly_profile(rbc_obs())
    e2e = None
    e2e_export = ROOT / f"artifacts/case_study/exports/{args.e2e_case}"
    if args.window == "3day" and (  # the 28h paper figure never plots E2E
        list(e2e_export.glob("seed_*/model.zip")) or (e2e_export / "model.zip").exists()
    ):
        try:
            e2e = weekday_hourly_profile(deterministic_rollout(args.e2e_case))
        except Exception as exc:  # noqa: BLE001
            print(f"  (E2E overlay skipped: {exc})")

    if args.window == "28h":
        # Final from-scratch E2E policies, mean of the five
        # seeds' weekday profiles (deterministic rollouts of each final model)
        e2e_runs = [
            weekday_hourly_profile(rollout_seed(args.e2e_case, s))
            for s in (42, 123, 777, 2024, 2025)
        ]
        e2e_kw = np.nanmean([r[0] for r in e2e_runs], axis=0)
        e2e_tk = np.nanmean([r[1] for r in e2e_runs], axis=0)

        def onpeak(kw):
            return 100.0 * kw[17:20].sum() / kw.sum()

        print(
            f"on-peak share of chiller power: TLD {onpeak(prop_kw):.1f} "
            f"E2E {onpeak(e2e_kw):.1f} RBC {onpeak(rbc_kw):.1f} | 17h tank "
            f"TLD {prop_tk[17]:.2f} E2E {e2e_tk[17]:.2f} RBC {rbc_tk[17]:.2f}"
        )
        # Weekday-average profile plotted from 20:00 of the preceding day to
        # 24:00 of the day: the lead-in shows both controllers reaching
        # midnight at their repeating start-of-day states (steady state),
        # without tiling identical days.
        def w28(arr):
            x = np.concatenate([np.arange(-4, 0) + 0.5, np.arange(24) + 0.5])
            y = np.concatenate([arr[20:], arr])
            return x, y

        # final printed size (\columnwidth = 3.5 in): rcParams sizes are page sizes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.11), sharex=True)
        for ax in (ax1, ax2):
            for h in range(-4, 24):
                price = SMUD[h % 24]
                color = (
                    "#fde9e9"
                    if price > 0.30
                    else "#fdf6e3" if price > 0.20 else "white"
                )
                ax.axvspan(h, h + 1, color=color, lw=0, zorder=0)
            ax.axvline(0, color="0.4", lw=0.8, ls=":")
        ax1.plot(
            *w28(prop_kw),
            color=C_PROP,
            lw=1.7,
            marker="o",
            ms=2.6,
            label="TLD-RL (Proposed)",
        )
        ax1.plot(
            *w28(e2e_kw), color=C_E2E, lw=1.3, marker="^", ms=2.4, label="E2E"
        )
        ax1.plot(
            *w28(rbc_kw), color=C_RBC, lw=1.5, ls="--", marker="s", ms=2.6, label="RBC"
        )
        # name the tariff bands in-figure once (the caption only says
        # "tariff bands are shaded"); tints are darker steps of the fills
        for xc, lab, col in (
            (6.0, "off-peak", "0.45"),
            (14.5, "mid-peak", "#8a6d2e"),
            (18.5, "on-peak", "#a14b4b"),
        ):
            ax1.text(
                xc, 19.55, lab, ha="center", va="top", fontsize=6.5, color=col, zorder=5
            )
        ax1.set_ylabel("Chiller power [kW]")
        # frame ends on ticks; the small pad below 0 keeps the chiller-off
        # markers from being clipped by the axis line
        ax1.set_ylim(-0.9, 20)
        ax1.set_yticks(range(0, 21, 5))
        ax2.plot(*w28(prop_tk), color=C_PROP, lw=1.7, marker="o", ms=2.6)
        ax2.plot(*w28(e2e_tk), color=C_E2E, lw=1.3, marker="^", ms=2.4)
        ax2.plot(*w28(rbc_tk), color=C_RBC, lw=1.5, ls="--", marker="s", ms=2.6)
        # legend in the bottom panel's free lower-right corner (the upper-left
        # of the top panel is crossed by RBC's post-midnight charging peak)
        handles, labels = ax1.get_legend_handles_labels()
        ax2.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=7,
            handlelength=1.6,
            columnspacing=1.0,
            borderaxespad=0.1,
        )
        ax2.set_ylabel("Tank temp. [°C]")
        ax2.set_ylim(4.0, 8.0)
        ax2.set_yticks(range(4, 9))
        # the 20:00 lead-in is stated in the caption; keep the axis short
        ax2.set_xlabel("Hour of day")
        ax2.set_xlim(-4, 24)
        ax2.set_xticks([-4, 0, 6, 12, 17, 20, 24])
        ax2.set_xticklabels(["20", "0", "6", "12", "17", "20", "24"])
        fig.tight_layout()
        fig.savefig(MAN / "fig4.pdf")  # paper Fig. 5
        fig.savefig(OUT / "fig_operating_profile_28h.pdf")
        fig.savefig(OUT / "fig_operating_profile_28h.png")
        print(f"written: {MAN}/fig4.pdf and {OUT}/fig_operating_profile_28h.pdf")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 3.4), sharex=True)
    for ax in (ax1, ax2):
        shade_tou_days(ax, 3)
        for d in (24, 48):
            ax.axvline(d, color="0.5", lw=0.7, ls=":")

    x, y = tile3(prop_kw)
    ax1.plot(x, y, color=C_PROP, lw=1.7, marker="o", ms=2.5, label="TLD-RL (Proposed)")
    xr, yr = tile3(rbc_kw)
    ax1.plot(xr, yr, color=C_RBC, lw=1.5, ls="--", marker="s", ms=2.5, label="RBC")
    if e2e is not None:
        xe, ye = tile3(e2e[0])
        ax1.plot(xe, ye, color=C_E2E, lw=1.4, ls="-.", marker="^", ms=2.5, label="E2E")
    ax1.set_ylabel("Chiller power [kW]")
    ax1.legend(loc="upper left", ncol=3, frameon=False, fontsize=7)

    ax2.plot(*tile3(prop_tk), color=C_PROP, lw=1.7, marker="o", ms=2.5)
    ax2.plot(*tile3(rbc_tk), color=C_RBC, lw=1.5, ls="--", marker="s", ms=2.5)
    if e2e is not None:
        ax2.plot(*tile3(e2e[1]), color=C_E2E, lw=1.4, ls="-.", marker="^", ms=2.5)
    ax2.set_ylabel("Tank temp. [°C]")
    ax2.set_xlabel("Hour of day (three consecutive weekdays)")
    ax2.set_xlim(0, 72)
    ax2.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax2.set_xticklabels(["0", "12", "0", "12", "0", "12", "0"])
    for d, lab in [(12, "Day 1"), (36, "Day 2"), (60, "Day 3")]:
        ax1.text(
            d,
            1.02,
            lab,
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUT / "fig_operating_profile_D1.pdf")
    fig.savefig(OUT / "fig_operating_profile_D1.png")
    plt.close(fig)
    print(f"written: {OUT}/fig_operating_profile_D1.pdf")


if __name__ == "__main__":
    main()
