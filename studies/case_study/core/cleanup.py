from __future__ import annotations

import json
import shutil
from pathlib import Path

from .types import CleanupPolicy, CleanupReport, RunResult


def _status_by_run_id(manifest_dir: Path) -> dict[str, str]:
    statuses = {}
    if not manifest_dir.exists():
        return statuses
    for path in manifest_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = payload.get("run_id") or path.stem
        statuses[run_id] = payload.get("status", "success")
    return statuses


def cleanup_case_root(case_id: str, case_workspace_root: Path, manifest_dir: Path, policy: CleanupPolicy) -> CleanupReport:
    report = CleanupReport(case_id=case_id)
    if not case_workspace_root.exists():
        return report
    statuses = _status_by_run_id(manifest_dir)

    run_dirs = [path for path in case_workspace_root.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    success_kept = 0
    failure_kept = 0
    for run_dir in run_dirs:
        status = statuses.get(run_dir.name, "success")
        keep = False
        if status == "success":
            if success_kept < policy.retain_success:
                keep = True
                success_kept += 1
        else:
            if failure_kept < policy.retain_failure:
                keep = True
                failure_kept += 1
        if keep:
            report.kept_paths.append(run_dir)
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        report.removed_paths.append(run_dir)
        manifest_path = manifest_dir / f"{run_dir.name}.json"
        if manifest_path.exists():
            manifest_path.unlink()
            report.removed_paths.append(manifest_path)
    return report


def cleanup_run(result: RunResult, policy: CleanupPolicy) -> CleanupReport:
    return cleanup_case_root(
        result.case_id,
        result.run_context.workspace_root.parent,
        result.run_context.manifest_root,
        policy,
    )
