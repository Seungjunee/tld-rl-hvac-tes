"""Deterministic evaluation of an exported model -> seasonal cost ($/day).

cleanup deleted some seeds' eval workspaces, so best-eval is unrecoverable for
them. Instead we deterministically re-evaluate each exported (converged) model
once -- self-contained (model.zip always survives), uniform across seeds.
Validate against paper best-eval on surviving seeds before adopting.

Usage: det_eval.py <case_id> <seed>   -> prints JSON {cost_per_day, viol}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/workspaces/sinergym")

import numpy as np
from stable_baselines3 import SAC

from studies.case_study.core.env_builder import build_environment
from studies.case_study.core.registry import load_registry, resolve_case
from studies.case_study.core.types import RunContext

ROOT = Path(__file__).resolve().parents[3]
DAYS = 122


def main():
    case_id, seed = sys.argv[1], sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "model"
    registry = load_registry(str(ROOT / "studies/case_study/config"))
    case = resolve_case(case_id, registry)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ctx = RunContext(
        case_id=f"deteval_{case_id}_{seed}",
        run_id=f"deteval_{ts}",
        started_at=ts,
        workspace_root=Path("/tmp/deteval_ws"),
        export_root=Path("/tmp/deteval_ws/e"),
        figure_root=Path("/tmp/deteval_ws/f"),
        manifest_root=Path("/tmp/deteval_ws/m"),
        third_party_root=ROOT / "third_party",
        project_root=ROOT,
    )
    env = build_environment(case, ctx, mode="train")
    model_path = (
        ROOT / f"artifacts/case_study/exports/{case_id}/seed_{seed}/{model_name}.zip"
    )
    model = SAC.load(str(model_path), env=env)

    obs, _ = env.reset()
    cost = 0.0
    viol_steps = 0
    n = 0
    done = False
    while not done:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        obs, _reward, term, trunc, info = env.step(action)
        p = info.get("total_power_demand")
        pr = info.get("current_price")
        if p is not None and pr is not None:
            cost += float(p) * 900 / 3.6e6 * float(pr)
        tv = info.get("total_temperature_violation", 0.0)
        if tv and float(tv) > 0:
            viol_steps += 1
        n += 1
        done = term or trunc
    env.close()
    out = {
        "case": case_id,
        "seed": seed,
        "cost_per_day": cost / DAYS,
        "cost_total": cost,
        "viol_pct": 100 * viol_steps / max(n, 1),
        "steps": n,
    }
    outdir = Path("/tmp/deteval_out")
    outdir.mkdir(exist_ok=True)
    (outdir / f"{case_id}_{seed}.json").write_text(json.dumps(out))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
