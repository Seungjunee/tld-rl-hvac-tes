from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .types import RunContext, StudyRegistry


def ensure_runtime_dirs(paths: dict[str, Path]) -> None:
    for key in ["artifact_root", "workspace_root", "export_root", "figure_root", "manifest_root", "third_party_root"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def get_git_sha(project_root: Path) -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True)
        return output.strip()
    except Exception:
        return None


def prepare_runtime_environment(run_context: RunContext) -> None:
    third_party_root = run_context.third_party_root
    (third_party_root / "wandb").mkdir(parents=True, exist_ok=True)
    (third_party_root / "wandb-cache").mkdir(parents=True, exist_ok=True)
    (third_party_root / "wandb-config").mkdir(parents=True, exist_ok=True)
    (third_party_root / "mlruns").mkdir(parents=True, exist_ok=True)
    (third_party_root / "matplotlib").mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(third_party_root / "wandb")
    os.environ["WANDB_CACHE_DIR"] = str(third_party_root / "wandb-cache")
    os.environ["WANDB_CONFIG_DIR"] = str(third_party_root / "wandb-config")
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{(third_party_root / 'mlruns').resolve()}"
    os.environ["MPLCONFIGDIR"] = str(third_party_root / "matplotlib")


def build_run_context(case_id: str, registry: StudyRegistry) -> RunContext:
    ensure_runtime_dirs(registry.paths)
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{started_at}_{uuid4().hex[:8]}"
    return rebuild_run_context(case_id, run_id, registry, started_at=started_at)


def rebuild_run_context(
    case_id: str,
    run_id: str,
    registry: StudyRegistry,
    *,
    started_at: str | None = None,
    git_sha: str | None = None,
) -> RunContext:
    ensure_runtime_dirs(registry.paths)
    started_at = started_at or run_id.split("_", 1)[0]
    workspace_root = registry.paths["workspace_root"] / case_id / run_id
    workspace_root.mkdir(parents=True, exist_ok=True)
    export_root = registry.paths["export_root"] / case_id
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_root = registry.paths["manifest_root"] / case_id
    manifest_root.mkdir(parents=True, exist_ok=True)
    figure_root = registry.paths["figure_root"]
    third_party_root = registry.paths["third_party_root"] / case_id / run_id
    third_party_root.mkdir(parents=True, exist_ok=True)

    return RunContext(
        case_id=case_id,
        run_id=run_id,
        started_at=started_at,
        workspace_root=workspace_root,
        export_root=export_root,
        figure_root=figure_root,
        manifest_root=manifest_root,
        third_party_root=third_party_root,
        project_root=registry.paths["project_root"],
        git_sha=git_sha if git_sha is not None else get_git_sha(registry.paths["project_root"]),
    )
