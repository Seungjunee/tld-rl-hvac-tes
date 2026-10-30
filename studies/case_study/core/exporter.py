from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

from .episode_selector import resolve_episode, resolve_episode_map
from .preprocess import preprocess_and_merge
from .types import ExportReport, RunResult


CORE_FILENAMES = [
    "model.zip",
    "model_continued.zip",
    "model_duplicate_padded.zip",
    "model_baseline.zip",
    "model_e2e_finetuned.zip",
    "mean.txt",
    "var.txt",
    "count.txt",
    "progress.csv",
]


def _copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        try:
            if src.samefile(dst):
                return False
        except FileNotFoundError:
            pass
        if not overwrite:
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists():
        return False
    shutil.copytree(src, dst)
    return True


def _collect_monitor_files(episode_dir: Path | None) -> Iterable[Path]:
    if episode_dir is None:
        return []
    monitor_dir = episode_dir / "monitor"
    if not monitor_dir.exists():
        return []
    return sorted(path for path in monitor_dir.glob("*.csv") if path.is_file())


def _manifest_payload(result: RunResult, report: ExportReport) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "run_id": result.run_context.run_id,
        "building_id": result.metadata.get("building_id"),
        "method_id": result.metadata.get("method_id"),
        "source_bundle": str(result.source_bundle.model_path) if result.source_bundle else None,
        "workspace_path": str(result.workspace_path),
        "export_path": str(report.destination),
        "resolved_episode_selectors": {
            key: {
                "requested_selector": value.requested_selector,
                "resolved_episode_number": value.resolved_episode_number,
                "resolution_reason": value.resolution_reason,
                "source_path": str(value.source_path),
                "fallback_used": value.fallback_used,
            }
            for key, value in report.resolved_episode_selectors.items()
        },
        "git_sha": result.run_context.git_sha,
        "status": result.metadata.get("status", "success"),
        "metadata": result.metadata,
        "copied_files": [str(path) for path in report.copied_files],
        "preprocess_output": str(report.preprocess_output) if report.preprocess_output else None,
    }


def export_run(result: RunResult, export_profile: Dict[str, Any]) -> ExportReport:
    destination = Path(export_profile["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    profile_name = export_profile.get("profile", "flat_export")
    overwrite = bool(export_profile.get("overwrite", True))
    report = ExportReport(case_id=result.case_id, destination=destination, profile=profile_name)

    workspace = result.workspace_path
    snapshot_selectors = export_profile.get("snapshot_selectors", [])
    if profile_name == "workspace_export":
        for item in workspace.iterdir():
            dst = destination / item.name
            if item.is_file() and _copy_file(item, dst, overwrite):
                report.copied_files.append(dst)
            elif item.is_dir() and _copy_tree(item, dst, overwrite):
                report.copied_files.append(dst)
        report.resolved_episode_selectors = resolve_episode_map(destination, snapshot_selectors)
    elif profile_name == "episode_snapshot_export":
        selectors = snapshot_selectors or ["first", "penultimate", "last"]
        report.resolved_episode_selectors = resolve_episode_map(workspace, selectors)
        for name in CORE_FILENAMES:
            src = workspace / name
            dst = destination / name
            if _copy_file(src, dst, overwrite):
                report.copied_files.append(dst)
        if result.model_path is not None:
            dst = destination / result.model_path.name
            if _copy_file(result.model_path, dst, overwrite):
                report.copied_files.append(dst)
        for selector_name, episode_ref in report.resolved_episode_selectors.items():
            target_dir = destination / "episode_snapshots" / selector_name / f"episode-{episode_ref.resolved_episode_number}"
            if _copy_tree(episode_ref.source_path, target_dir, overwrite):
                report.copied_files.append(target_dir)
    else:
        episode_ref = resolve_episode(workspace, "last")
        if episode_ref is not None:
            report.resolved_episode_selectors["last"] = episode_ref
        for name in CORE_FILENAMES:
            src = workspace / name
            dst = destination / name
            if _copy_file(src, dst, overwrite):
                report.copied_files.append(dst)
        if result.model_path is not None:
            dst = destination / result.model_path.name
            if _copy_file(result.model_path, dst, overwrite):
                report.copied_files.append(dst)
        selected_episode_dir = episode_ref.source_path if episode_ref is not None else None
        for monitor_file in _collect_monitor_files(selected_episode_dir):
            dst = destination / monitor_file.name
            if _copy_file(monitor_file, dst, overwrite):
                report.copied_files.append(dst)

    # Copy the deterministic-eval learning curve. LoggerEvalCallback writes
    # evaluation_metrics.csv to the TRAIN env's evaluation/ subdir; surfacing it
    # in the (seeded) export dir lets figures plot the deterministic eval reward
    # instead of the stochastic training progress.csv.
    for eval_src in sorted(workspace.rglob("evaluation_metrics.csv")):
        if eval_src.parent.name != "evaluation":
            continue
        dst = destination / "evaluation_metrics.csv"
        if _copy_file(eval_src, dst, overwrite):
            report.copied_files.append(dst)
        break

    preprocess_cfg = export_profile.get("preprocess")
    if preprocess_cfg:
        report.preprocess_output = preprocess_and_merge(
            destination / preprocess_cfg.get("observations", "observations.csv"),
            destination / preprocess_cfg.get("infos", "infos.csv"),
            destination / preprocess_cfg.get("out", "observations_processed.csv"),
            n_steps=int(preprocess_cfg.get("n_steps", 96)),
        )

    canonical_manifest = result.run_context.manifest_root / f"{result.run_context.run_id}.json"
    export_manifest = destination / "run_manifest.json"
    manifest = _manifest_payload(result, report)
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    export_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report.manifest_path = canonical_manifest
    report.export_manifest_path = export_manifest
    return report
