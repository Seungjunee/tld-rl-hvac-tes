"""Tier-2 source-side causal sweep (redesigned 2026-06-23).

Headline test: transfer quality is governed by the SOURCE building's load profile
(shiftability), not the zone-count gap. Controlled ablation, only the source's
load timing changes:

  source = office timing-variant (M_src swept low->high, zones fixed at 5)
        -> train a TLD-RL policy on it
        -> transfer the frozen policy to a FIXED target (LB, 8 zones)
        -> measure transfer savings vs the LB RBC baseline.

Zone-count gap (|5-8|=3) and the target are identical for every point, so a rising
savings-vs-M_src curve isolates the source load profile as the cause.

Real OF->LB and WH->LB are anchors. The warehouse cannot be swept (envelope-locked,
see wh_derisk.py), which is why it is intrinsically a weak source.

Modes (for shell-level parallelism):
  --mode train    --delta D --seed S   train policy on office-variant D
  --mode transfer --delta D --seed S   transfer that policy to LB, write savings
  --mode plot                          aggregate + figure
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
PRED = ROOT / "artifacts/case_study/predictors"
SRC_EXPORT = ROOT / "artifacts/case_study/exports/tier2_source"
TRANSFER_OUT = OUT / "tier2_source_transfer"
DAYS = 122
LB_RBC_PER_DAY = 12.521641393337738  # from Tier-1 _final_tables (lbuilding RBC)


def _predictor_for(delta: int) -> Path:
    return (
        (PRED / "office_5zone_lstm.pt")
        if delta == 0
        else (PRED / f"office_shift{delta}_lstm.pt")
    )


def _ctx(case_id: str, sub: str):
    from studies.case_study.core.types import RunContext

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RunContext(
        case_id=case_id,
        run_id=f"{sub}_{ts}",
        started_at=ts,
        workspace_root=ROOT / f"artifacts/case_study/workspaces/{sub}",
        export_root=ROOT / f"artifacts/case_study/exports/{sub}",
        figure_root=ROOT / "artifacts/case_study/figures",
        manifest_root=ROOT / "artifacts/case_study/manifests",
        third_party_root=ROOT / "third_party",
        project_root=ROOT,
    )


def train_source(delta: int, seed: int, registry):
    """Train a TLD-RL policy on office-variant `delta` (refit predictor as source)."""
    from studies.case_study.analysis.tier2_run import (
        _override_env,
        prepare_dataset,
        refit_predictor,
    )
    from studies.case_study.core.registry import resolve_case
    from studies.case_study.core.runner import run_case

    dest = SRC_EXPORT / f"shift{delta}" / f"seed_{seed}"
    if (dest / "model.zip").exists():
        print(f"  [train] d={delta} seed={seed} cached")
        return
    # ensure the source predictor exists
    pt = _predictor_for(delta)
    if not pt.exists():
        csv_path = prepare_dataset(delta, registry)
        refit_predictor(delta, csv_path)
    case = deepcopy(resolve_case("pretrain_office_smud_sac_plus", registry))
    case.id = f"tier2_src_shift{delta}"
    _override_env(case, delta)
    case.wrappers = deepcopy(case.wrappers)
    case.wrappers["load_predictor"] = dict(case.wrappers["load_predictor"])
    case.wrappers["load_predictor"]["model_path"] = str(pt.relative_to(ROOT))
    tr = dict(case.training)
    mk = dict(tr.get("model_kwargs", {}))
    mk["seed"] = int(seed)
    tr["model_kwargs"] = mk
    # disable eval to halve wall time (progress.csv training return is unaffected)
    if isinstance(tr.get("evaluation"), dict):
        tr["evaluation"] = {**tr["evaluation"], "enabled": False}
    case.training = tr
    case.export = dict(case.export)
    case.export["destination"] = str(dest)
    print(f"  [train] d={delta} seed={seed} ...")
    run_case(case, registry)


def transfer_to_lb(delta: int, seed: int, registry) -> dict:
    """Deploy the office-variant policy frozen on LB; deterministic savings."""
    from stable_baselines3 import SAC

    from studies.case_study.core.env_builder import build_environment
    from studies.case_study.core.registry import resolve_case

    if delta == 0:  # delta-0 source IS the baseline office policy (no retrain)
        model_path = (
            ROOT / "artifacts/case_study/exports/"
            f"pretrain_office_smud_sac_plus/seed_{seed}/model.zip"
        )
    else:
        model_path = SRC_EXPORT / f"shift{delta}" / f"seed_{seed}" / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    lb_pt = PRED / "office_lbuilding_8zone_lstm.pt"
    if not lb_pt.exists():
        raise FileNotFoundError(f"LB target predictor missing: {lb_pt}")
    case = deepcopy(resolve_case("pretrain_lbuilding_smud_sac_plus", registry))
    case.id = f"tier2_xfer_shift{delta}_s{seed}"
    case.wrappers = deepcopy(case.wrappers)
    case.wrappers["load_predictor"] = dict(case.wrappers["load_predictor"])
    case.wrappers["load_predictor"]["model_path"] = str(lb_pt.relative_to(ROOT))
    ctx = _ctx(case.id, "tier2_xfer")
    env = build_environment(case, ctx, mode="train")
    model = SAC.load(str(model_path), env=env)
    obs, _ = env.reset()
    cost, viol, n, done = 0.0, 0, 0, False
    while not done:
        a, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        p, pr = info.get("total_power_demand"), info.get("current_price")
        if p is not None and pr is not None:
            cost += float(p) * 900 / 3.6e6 * float(pr)
        if float(info.get("total_temperature_violation", 0.0)) > 0:
            viol += 1
        n += 1
        done = term or trunc
    env.close()
    cpd = cost / DAYS
    sav = 100 * (LB_RBC_PER_DAY - cpd) / LB_RBC_PER_DAY
    rec = {
        "delta": delta,
        "seed": seed,
        "cost_per_day": cpd,
        "savings_on_lb": sav,
        "viol_pct": 100 * viol / max(n, 1),
    }
    TRANSFER_OUT.mkdir(parents=True, exist_ok=True)
    (TRANSFER_OUT / f"shift{delta}_seed{seed}.json").write_text(json.dumps(rec))
    print(f"  [transfer] d={delta} seed={seed}: savings_on_LB={sav:+.2f}%")
    return rec


def aggregate_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rbc = json.load(open(OUT / "tier2_rbc_metrics.json"))
    recs = [json.load(open(p)) for p in sorted(TRANSFER_OUT.glob("shift*_seed*.json"))]
    by_d = {}
    for r in recs:
        by_d.setdefault(r["delta"], []).append(r["savings_on_lb"])
    rows = []
    for d, sav in sorted(by_d.items()):
        m = rbc[str(d)]["M3_oracle_sched_pct"] if str(d) in rbc else None
        rows.append(
            {
                "delta": d,
                "M_src": m,
                "savings_mean": float(np.mean(sav)),
                "savings_std": float(np.std(sav)),
                "n": len(sav),
            }
        )
    (OUT / "tier2_source_sweep.json").write_text(json.dumps(rows, indent=2))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = [r["M_src"] for r in rows if r["M_src"] is not None]
    ys = [r["savings_mean"] for r in rows if r["M_src"] is not None]
    es = [r["savings_std"] for r in rows if r["M_src"] is not None]
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        fmt="o-",
        color="#1f77b4",
        capsize=3,
        lw=1.6,
        ms=6,
        mec="k",
        label="office source sweep (-> LB)",
    )
    # real anchors: OF->LB and WH->LB (canonical Table 4) at their M_src
    sm = json.load(open(OUT / "shiftability_metrics.json"))["per_building"]
    ft = json.load(
        open(
            ROOT / "artifacts/case_study/figures/"
            "metric_a_5seed_2026-06-14/_final_tables.json"
        )
    )["transfer"]
    for b, key, lab in [
        ("office", "office_to_lbuilding", "OF"),
        ("warehouse", "warehouse_to_lbuilding", "WH"),
    ]:
        xm = sm[b]["metric3_oracle"]["savings_schedule_pct"]
        ym = ft[key]["zs"]
        ax.scatter(xm, ym, marker="*", s=170, color="#d62728", edgecolor="k", zorder=5)
        ax.annotate(
            f"{lab}->LB (real)",
            (xm, ym),
            fontsize=7,
            xytext=(6, 0),
            textcoords="offset points",
            color="#d62728",
        )
    ax.axhline(0, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("Source shiftability $M_{src}$ (oracle potential) [%]")
    ax.set_ylabel("Transfer savings on fixed LB target [%]")
    ax.set_title(
        "Transfer rises with source shiftability across building types\n"
        "(complement; office fine-sweep noise-limited, M timing-bounded)",
        fontsize=8.5,
    )
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tier2_source_sweep.pdf")
    fig.savefig(OUT / "fig_tier2_source_sweep.png")
    plt.close(fig)
    print("=== source sweep ===")
    for r in rows:
        print(
            f"  d={r['delta']:+d} M_src={r['M_src']}: "
            f"savings_on_LB={r['savings_mean']:+.2f}±{r['savings_std']:.2f} "
            f"(n={r['n']})"
        )
    print(f"written: {OUT}/fig_tier2_source_sweep.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "transfer", "plot"], required=True)
    ap.add_argument("--delta", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.mode == "plot":
        aggregate_and_plot()
        return
    from studies.case_study.core.registry import load_registry

    registry = load_registry(str(ROOT / "studies/case_study/config"))
    if args.mode == "train":
        train_source(args.delta, args.seed, registry)
    elif args.mode == "transfer":
        transfer_to_lb(args.delta, args.seed, registry)


if __name__ == "__main__":
    main()
