from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from studies.case_study.core.exporter import export_run
from studies.case_study.core.types import RunContext, RunResult


def _run_context(tmp_path: Path) -> RunContext:
    for name in ["workspace_root", "export_root", "figure_root", "manifest_root", "third_party_root"]:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return RunContext(
        case_id="exp_test",
        run_id="20260329T000000Z_test",
        started_at="2026-03-29T00:00:00Z",
        workspace_root=tmp_path / "workspace_root",
        export_root=tmp_path / "export_root",
        figure_root=tmp_path / "figure_root",
        manifest_root=tmp_path / "manifest_root" / "exp_test",
        third_party_root=tmp_path / "third_party_root",
        project_root=tmp_path,
        git_sha="deadbeef",
    )


def test_export_run_flat_profile_with_preprocess(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    for name, content in {
        "model.zip": "model",
        "mean.txt": "1\n2\n",
        "var.txt": "3\n4\n",
        "count.txt": "5\n",
        "progress.csv": "episode_num,mean_reward\n1,0.1\n",
    }.items():
        (workspace / name).write_text(content, encoding="utf-8")

    monitor_dir = workspace / "episode-1" / "monitor"
    monitor_dir.mkdir(parents=True)
    pd.DataFrame({"CoolingRate": [1, 2, 3]}).to_csv(monitor_dir / "observations.csv", index=False)
    pd.DataFrame({"comfort_penalty": [10, 20, 30]}).to_csv(monitor_dir / "infos.csv", index=False)
    pd.DataFrame({"reward": [1.0, 2.0, 3.0]}).to_csv(monitor_dir / "rewards.csv", index=False)

    result = RunResult(
        case_id="exp_test",
        workspace_path=workspace,
        run_context=_run_context(tmp_path),
        model_path=workspace / "model.zip",
        metadata={"status": "success", "building_id": "office_5zone", "method_id": "scratch_baseline"},
    )

    destination = tmp_path / "exports" / "exp_test"
    report = export_run(
        result,
        {
            "profile": "flat_export",
            "destination": str(destination),
            "overwrite": True,
            "preprocess": {
                "observations": "observations.csv",
                "infos": "infos.csv",
                "out": "observations_processed.csv",
                "n_steps": 1,
            },
        },
    )

    assert (destination / "model.zip").exists()
    assert (destination / "mean.txt").exists()
    assert (destination / "var.txt").exists()
    assert (destination / "count.txt").exists()
    assert (destination / "progress.csv").exists()
    assert (destination / "observations.csv").exists()
    assert (destination / "infos.csv").exists()
    assert (destination / "observations_processed.csv").exists()
    assert report.preprocess_output == destination / "observations_processed.csv"
    assert report.resolved_episode_selectors["last"].resolved_episode_number == 1

    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == "exp_test"
    assert manifest["status"] == "success"
    assert manifest["resolved_episode_selectors"]["last"]["resolved_episode_number"] == 1
