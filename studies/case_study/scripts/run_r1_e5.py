"""Held-out weather re-evaluation of the source final models (method order).

For each of the four buildings, method source case and seed, one deterministic
season of the seed's FINAL model (export seed_<N>/model.zip) on its own
building under a held-out EPW: TMYx Chicago 2007-2021 (mixedx), New York JFK
(nyc), Pittsburgh (pit). No refit: wrappers, reward and predictor are those
of the source training (TLD-RL keeps <b>_lstm.pt). Same zero_shot runner and
EPW keys as the Table 5 __w_* runs; the RBC reference is
rbc_<b>_smud_tou__w_<w> (the RBC case on the same weather variant).

Exports to <source case>__w_<w>/seed_<N>/ (new directories; nothing canonical
is overwritten). The run id is that directory's case id, so workspace cleanup
never touches the training workspaces. After every run two sanity criteria
are checked against the weather-matched RBC (savings > 30 %,
|violation - RBC| > 1 pp); a crash is rerun once with identical settings, and
a repeat or an anomaly stops the driver.

Usage:
    python -m studies.case_study.scripts.run_r1_e5 --workers 1
    python -m studies.case_study.scripts.run_r1_e5 --summary
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/results"
SF = 900.0 / 3.6e6
MAX_SAVINGS, MAX_DVIOL = 30.0, 1.0
BUILDINGS = ["office", "lbuilding", "warehouse", "mediumoffice"]
METHODS = {
    "tldrl": "pretrain_{b}_smud_sac_plus",
    "e2e": "pretrain_{b}_smud_sac_e2e",
    "padding": "pretrain_{b}_smud_sacp_e2e",
}
WEATHERS = ["mixedx", "nyc", "pit"]
SEEDS = [42, 123, 777, 2024, 2025]


def _cost_viol(progress: Path) -> tuple[float, float]:
    import pandas as pd

    r = pd.read_csv(progress).iloc[-1]
    n = int(float(r["length(timesteps)"]))
    return float(r["mean_reward_energy_term"]) * n * SF, float(
        r["comfort_violation_time(%)"]
    )


def _dest(src_case: str, weather: str, seed: int) -> Path:
    return EXP / f"{src_case}__w_{weather}" / f"seed_{seed}"


def run_single(src_case: str, seed: int, weather: str) -> None:
    from studies.case_study.core.registry import load_registry, resolve_case
    from studies.case_study.core.runner import run_case

    reg = load_registry(ROOT / "studies/case_study/config")
    src = resolve_case(src_case, reg)
    b = src_case.split("_")[1]
    other = next(x for x in BUILDINGS if x != b)
    # method / training / export / cleanup of a Table 5 zero-shot case
    case = copy.deepcopy(resolve_case(f"transfer_{other}_to_{b}_smud_sacp_zs", reg))
    assert case.building.id == src.building.id
    case.id = f"{src_case}__w_{weather}"
    case.env = copy.deepcopy(src.env)
    env_id = str(case.env["id"])
    assert env_id.count("-mixed-") == 1, env_id
    case.env["id"] = env_id.replace("-mixed-", f"-{weather}-")
    case.reward = copy.deepcopy(src.reward)
    case.wrappers = copy.deepcopy(src.wrappers)
    case.inputs = {"source_case_id": src_case, "source_seed": seed}
    case.export = dict(case.export)
    case.export["destination"] = str(_dest(src_case, weather, seed))
    case.metadata = {
        **case.metadata,
        "weather_variant": weather,
        "e5_source_case": src_case,
        "e5_source_seed": seed,
    }
    run_case(case, reg)


def summary() -> dict:
    import numpy as np

    res = {}
    for b in BUILDINGS:
        for w in WEATHERS:
            rbc_cost, rbc_viol = _cost_viol(
                EXP / f"rbc_{b}_smud_tou__w_{w}" / "progress.csv"
            )
            for m, pat in METHODS.items():
                sc = pat.format(b=b)
                sav, vio = [], []
                for s in SEEDS:
                    p = _dest(sc, w, s) / "progress.csv"
                    if p.exists():
                        c, v = _cost_viol(p)
                        sav.append(100.0 * (rbc_cost - c) / rbc_cost)
                        vio.append(v)
                res.setdefault(b, {}).setdefault(w, {"rbc_viol": rbc_viol})[m] = {
                    "n": len(sav),
                    "savings_mean": float(np.mean(sav)) if sav else None,
                    "savings_sd": float(np.std(sav, ddof=1)) if len(sav) > 1 else None,
                    "savings_per_seed": sav,
                    "viol_mean": float(np.mean(vio)) if vio else None,
                }
    # in-distribution reference: final-model deterministic eval on TMY3
    for b in BUILDINGS:
        rbc_cost, _ = _cost_viol(EXP / f"rbc_{b}_smud_tou" / "progress.csv")
        ref = {}
        for m, pat in METHODS.items():
            sav = []
            for s in SEEDS:
                f = EXP / pat.format(b=b) / f"seed_{s}" / "evaluation_metrics.csv"
                if f.exists():
                    c, _ = _cost_viol(f)
                    sav.append(100.0 * (rbc_cost - c) / rbc_cost)
            ref[m] = {
                "savings_mean": float(np.mean(sav)) if sav else None,
                "savings_sd": float(np.std(sav, ddof=1)) if len(sav) > 1 else None,
            }
        res[b]["tmy3_final_eval"] = ref
    (OUT / "e5_heldout.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--single", nargs=3, metavar=("SRC_CASE", "SEED", "WEATHER"))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    if args.single:
        run_single(args.single[0], int(args.single[1]), args.single[2])
        return 0
    if args.summary:
        res = summary()
        for b, r in res.items():
            for w in WEATHERS:
                print(
                    f"{b:13s} {w:7s} "
                    + " | ".join(
                        (
                            f"{m} {r[w][m]['savings_mean']:+.2f}±{r[w][m]['savings_sd'] or 0:.2f} (n={r[w][m]['n']})"
                            if r[w][m]["savings_mean"] is not None
                            else f"{m} -"
                        )
                        for m in METHODS
                    )
                )
        return 0

    status_path = OUT / "e5_status.json"
    status = (
        json.loads(status_path.read_text()) if status_path.exists() else {"runs": {}}
    )
    log_dir = OUT / "logs" / "e5"
    log_dir.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    def job(item):
        sc, s, w = item
        key = f"{sc}__w_{w}/seed_{s}"
        dest = _dest(sc, w, s)
        if stop["flag"]:
            return
        if (dest / "progress.csv").exists():
            status["runs"].setdefault(key, {"state": "exists"})
            return
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        cmd = [sys.executable, "-m", "studies.case_study.scripts.run_r1_e5"]
        cmd += ["--single", sc, str(s), w]
        tries = 0
        while True:
            tries += 1
            with open(log_dir / f"{sc}__w_{w}__seed_{s}.log", "a") as fh:
                rc = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT
                ).returncode
            if rc == 0 and (dest / "progress.csv").exists():
                break
            if tries >= 2:
                status["runs"][key] = {"state": "crash (repeated)", "exit": rc}
                stop["flag"] = True
                save()
                print(f"[STOP] {key} crashed twice", flush=True)
                return
        b = sc.split("_")[1]
        rbc_cost, rbc_viol = _cost_viol(
            EXP / f"rbc_{b}_smud_tou__w_{w}" / "progress.csv"
        )
        cost, viol = _cost_viol(dest / "progress.csv")
        sav = 100.0 * (rbc_cost - cost) / rbc_cost
        issues = []
        if sav > MAX_SAVINGS:
            issues.append(f"savings {sav:.2f} % > {MAX_SAVINGS}")
        if abs(viol - rbc_viol) > MAX_DVIOL:
            issues.append(f"violation {viol:.2f} vs RBC {rbc_viol:.2f}")
        status["runs"][key] = {
            "state": "done",
            "retried": tries > 1,
            "savings": sav,
            "viol": viol,
            "issues": issues,
        }
        save()
        print(
            f"[{'ok' if not issues else 'ANOMALY'}] {key} savings {sav:+.2f} viol {viol:.2f} {issues}",
            flush=True,
        )
        if issues:
            stop["flag"] = True

    items = [
        (pat.format(b=b), s, w)
        for b in BUILDINGS
        for w in WEATHERS
        for s in SEEDS
        for pat in METHODS.values()
    ]
    status["last_start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, items))
    print(f"[e5 done] stop={stop['flag']}", flush=True)
    return 1 if stop["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
