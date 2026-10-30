"""Zero-shot driver: one deterministic season per (pair case, source seed).

Exports to <case>__src<seed>/ (the paper's per-source-seed layout; existing
exports are never overwritten). A pair whose source seed has no model yet is
left pending. After every run two sanity criteria are checked against the
target's RBC: savings above 30 %, or violation time more
than 1 pp away from the RBC's. A crash is rerun once with identical settings;
a repeat, or any anomaly, stops the driver (status JSON records why).

Usage:
    python -m studies.case_study.scripts.run_r1_zs --group paper_extension \
        --seeds 42 123 777 2024 2025 --workers 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
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


def _registry():
    from studies.case_study.core.registry import load_registry

    return load_registry(ROOT / "studies/case_study/config")


def _rbc_case(reg, building_id: str) -> str:
    for cid, spec in reg.cases.items():
        if (
            spec.building == building_id
            and spec.method == "rbc_tou"
            and re.fullmatch(r"rbc_[a-z0-9]+_smud_tou", cid)
        ):
            return cid
    raise KeyError(f"no RBC case for {building_id}")


def _cost_viol(progress: Path) -> tuple[float, float]:
    import pandas as pd

    r = pd.read_csv(progress).iloc[-1]
    n = int(float(r["length(timesteps)"]))
    return float(r["mean_reward_energy_term"]) * n * SF, float(
        r["comfort_violation_time(%)"]
    )


def _source_model(reg, case_id: str, seed: int) -> Path:
    from studies.case_study.core.registry import resolve_case

    case = resolve_case(case_id, reg)
    src = case.inputs["source_case_id"]
    name = resolve_case(src, reg).outputs.get("model_name", "model")
    return EXP / src / f"seed_{seed}" / f"{name}.zip"


def run_single(case_id: str, seed: int) -> None:
    from studies.case_study.core.registry import resolve_case
    from studies.case_study.core.runner import run_case

    reg = _registry()
    case = copy.deepcopy(resolve_case(case_id, reg))
    case.inputs = dict(case.inputs)
    case.inputs["source_seed"] = seed
    case.export = dict(case.export)
    case.export["destination"] = str(EXP / f"{case_id}__src{seed}")
    run_case(case, reg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--single", nargs=2, metavar=("CASE", "SEED"))
    args = ap.parse_args(argv)
    if args.single:
        run_single(args.single[0], int(args.single[1]))
        return 0

    from studies.case_study.core.registry import resolve_case

    reg = _registry()
    # zero-shot cases only, so mixed groups (datasets, RBC, fine-tuning) work too
    cases = [
        c
        for c in reg.groups[args.group]
        if resolve_case(c, reg).method.runner == "zero_shot"
    ]
    status_path = OUT / f"zs_{args.group}.json"
    status = (
        json.loads(status_path.read_text()) if status_path.exists() else {"runs": {}}
    )
    log_dir = OUT / "logs" / args.group
    log_dir.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    def check(case_id, seed):
        target = resolve_case(case_id, reg).building.id
        rbc_cost, rbc_viol = _cost_viol(EXP / _rbc_case(reg, target) / "progress.csv")
        cost, viol = _cost_viol(EXP / f"{case_id}__src{seed}" / "progress.csv")
        sav = 100.0 * (rbc_cost - cost) / rbc_cost
        issues = []
        if sav > MAX_SAVINGS:
            issues.append(f"savings {sav:.2f} % > {MAX_SAVINGS}")
        if abs(viol - rbc_viol) > MAX_DVIOL:
            issues.append(f"violation {viol:.2f} vs RBC {rbc_viol:.2f}")
        return sav, viol, issues

    def job(item):
        case_id, seed = item
        key = f"{case_id}__src{seed}"
        if stop["flag"]:
            return
        if (EXP / key / "progress.csv").exists():
            status["runs"].setdefault(key, {"state": "exists"})
            return
        if not _source_model(reg, case_id, seed).exists():
            status["runs"][key] = {"state": "pending (no source model)"}
            save()
            return
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        cmd = [
            sys.executable,
            "-m",
            "studies.case_study.scripts.run_r1_zs",
            "--single",
            case_id,
            str(seed),
        ]
        tries = 0
        while True:
            tries += 1
            with open(log_dir / f"{key}.log", "a") as fh:
                rc = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT
                ).returncode
            if rc == 0 and (EXP / key / "progress.csv").exists():
                break
            if tries >= 2:
                status["runs"][key] = {"state": "crash (repeated)", "exit": rc}
                stop["flag"] = True
                save()
                print(f"[STOP] {key} crashed twice", flush=True)
                return
        sav, viol, issues = check(case_id, seed)
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

    items = [(c, s) for c in cases for s in args.seeds]
    status["last_start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, items))
    pending = [k for k, v in status["runs"].items() if v["state"].startswith("pending")]
    print(f"[zs done] stop={stop['flag']} pending={len(pending)}", flush=True)
    return 1 if stop["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
