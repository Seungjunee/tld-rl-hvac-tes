from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList

_ALGO_REGISTRY = {"ppo": PPO, "sac": SAC}

# PPO-only kwargs that must be stripped when running SAC (and vice versa).
# Method-level defaults assume PPO, so when a case overrides to SAC we filter
# out the incompatible ones before instantiation.
_PPO_ONLY_KWARGS = {
    "clip_range",
    "clip_range_vf",
    "n_steps",
    "n_epochs",
    "gae_lambda",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    "normalize_advantage",
}
_SAC_ONLY_KWARGS = {
    "buffer_size",
    "learning_starts",
    "tau",
    "train_freq",
    "gradient_steps",
    "action_noise",
    "replay_buffer_class",
    "replay_buffer_kwargs",
    "optimize_memory_usage",
    "target_update_interval",
    "target_entropy",
    "use_sde",
    "sde_sample_freq",
    "use_sde_at_warmup",
}


def _resolve_policy_kwargs(policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """YAML policy_kwargs -> SB3 kwargs (activation_fn "ReLU" -> torch.nn.ReLU)."""
    import torch

    resolved = dict(policy_kwargs)
    act = resolved.get("activation_fn")
    if isinstance(act, str):
        resolved["activation_fn"] = getattr(torch.nn, act)
    return resolved


def _resolve_algo_class(training: Dict[str, Any]):
    name = str(training.get("algorithm", "ppo")).lower()
    if name not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Supported: {list(_ALGO_REGISTRY)}"
        )
    return _ALGO_REGISTRY[name]


def _filter_kwargs_for_algo(
    algo_name: str, model_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if algo_name == "sac":
        return {k: v for k, v in model_kwargs.items() if k not in _PPO_ONLY_KWARGS}
    if algo_name == "ppo":
        return {k: v for k, v in model_kwargs.items() if k not in _SAC_ONLY_KWARGS}
    return dict(model_kwargs)


from sinergym.utils.callbacks import LoggerEvalCallback

from .cleanup import cleanup_run
from .env_builder import build_environment
from .exporter import export_run
from .method_registry import get_method_runner
from .registry import get_group_case_ids, resolve_case, resolve_case_id
from .types import (
    CleanupPolicy,
    ModelBundle,
    NormPaths,
    ResolvedCase,
    RunResult,
    StudyRegistry,
)
from .workspace import (
    build_run_context,
    prepare_runtime_environment,
    rebuild_run_context,
)


class RunnerServices:
    def __init__(
        self,
        case: ResolvedCase,
        registry: StudyRegistry,
        run_context=None,
        callbacks=None,
    ):
        self.case = case
        self.registry = registry
        self.run_context = run_context
        self.callbacks = callbacks

    def build_training_model(self, case: ResolvedCase, env):
        training = case.training
        algo_name = str(training.get("algorithm", "ppo")).lower()
        algo_cls = _resolve_algo_class(training)
        model_kwargs = dict(training.get("model_kwargs", {}))
        model_kwargs = _filter_kwargs_for_algo(algo_name, model_kwargs)
        if training.get("policy_kwargs") is not None:
            model_kwargs["policy_kwargs"] = _resolve_policy_kwargs(
                training["policy_kwargs"]
            )
        return algo_cls(
            training.get("policy", "MlpPolicy"),
            env,
            verbose=int(training.get("verbose", 1)),
            **model_kwargs,
        )

    def compute_total_timesteps(self, case: ResolvedCase, env) -> int:
        training = case.training
        if training.get("total_timesteps") is not None:
            return int(training["total_timesteps"])
        episodes = training.get("episodes")
        if episodes is None:
            raise ValueError(
                f"Case `{case.id}` needs training.total_timesteps or training.episodes."
            )
        timestep_per_episode = int(env.get_wrapper_attr("timestep_per_episode"))
        return int(episodes) * (timestep_per_episode - 1)

    def load_source_bundle(self, case: ResolvedCase) -> ModelBundle:
        candidates = []
        inputs = case.inputs
        source_case_id = inputs.get("source_case_id")
        if source_case_id:
            source_case = resolve_case(source_case_id, self.registry)
            source_export_dir = self.registry.paths["export_root"] / source_case.id
            source_model_name = source_case.outputs.get(
                "model_name", source_case.method.default_model_name
            )
            seed_selector = inputs.get("source_seed", "best")
            seeded_dir = _select_seeded_source(
                source_export_dir, source_model_name, seed_selector
            )
            if seeded_dir is not None:
                candidates.append(seeded_dir / f"{source_model_name}.zip")
            candidates.append(source_export_dir / f"{source_model_name}.zip")
        if inputs.get("previous_model_path"):
            candidates.append(Path(inputs["previous_model_path"]))
        for fallback in inputs.get("previous_model_fallbacks", []):
            candidates.append(Path(fallback))
        resolved_candidates = []
        for candidate in candidates:
            path = (
                candidate
                if candidate.is_absolute()
                else self.registry.paths["project_root"] / candidate
            )
            resolved_candidates.append(path)
        for candidate in resolved_candidates:
            if candidate.exists():
                export_path = candidate.parent
                return ModelBundle(
                    case_id=source_case_id or candidate.parent.name,
                    building_id="unknown",
                    method_id="unknown",
                    model_path=candidate,
                    export_path=export_path,
                    norm_paths=_extract_norm_paths(export_path),
                )
        raise FileNotFoundError(
            f"No source model found for `{case.id}`. Tried: {[str(p) for p in resolved_candidates]}"
        )

    def load_model(self, model_path, env=None):
        # Passing env rebinds the rollout buffer to the target's n_envs, which
        # is required when the source was trained with a VecEnv (e.g. multi-
        # source training) but the target is a single-env transfer.
        algo_cls = _resolve_algo_class(self.case.training)
        if env is not None:
            return algo_cls.load(str(model_path), env=env)
        return algo_cls.load(str(model_path))

    def build_learn_kwargs(self, case: ResolvedCase) -> Dict[str, Any]:
        learn_kwargs = dict(case.training.get("learn_kwargs", {}))
        if self.callbacks is not None:
            learn_kwargs["callback"] = self.callbacks
        if "log_interval" not in learn_kwargs:
            learn_kwargs["log_interval"] = int(case.training.get("log_interval", 100))
        if case.training.get("reset_num_timesteps") is not None:
            learn_kwargs.setdefault(
                "reset_num_timesteps", bool(case.training.get("reset_num_timesteps"))
            )
        return learn_kwargs


def _select_seeded_source(
    source_export_dir: Path,
    source_model_name: str,
    seed_selector: str | int,
) -> Optional[Path]:
    """Find a seed-tagged source export dir.

    seed_selector:
        "best" — pick seed with highest final mean_reward in progress.csv
        int    — pick specific seed (e.g. 42)
        "first" — pick first seeded dir found (sorted)
    Returns None if no seeded dirs exist. Raises FileNotFoundError when an
    explicit seed is requested but its seed dir (with model) is absent.
    """
    if not source_export_dir.exists():
        return None
    seed_dirs = sorted(
        p
        for p in source_export_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        return None
    seed_dirs = [p for p in seed_dirs if (p / f"{source_model_name}.zip").exists()]
    if not seed_dirs:
        return None
    if isinstance(seed_selector, int) or (
        isinstance(seed_selector, str) and seed_selector.isdigit()
    ):
        target = int(seed_selector)
        for p in seed_dirs:
            if p.name == f"seed_{target}":
                return p
        # A pinned seed must load exactly that seed: falling back silently
        # would export a mislabeled `__src<N>` result.
        raise FileNotFoundError(
            f"Requested source_seed {target} not found under {source_export_dir} "
            f"(available: {[p.name for p in seed_dirs]})."
        )
    if seed_selector == "first":
        return seed_dirs[0]
    # "best" — read progress.csv, pick highest final mean_reward
    import pandas as pd

    best_score = -float("inf")
    best_dir = seed_dirs[0]
    for p in seed_dirs:
        progress = p / "progress.csv"
        if not progress.exists():
            continue
        try:
            df = pd.read_csv(progress)
            score = float(df.iloc[-1]["mean_reward"])
            if score > best_score:
                best_score = score
                best_dir = p
        except Exception:
            continue
    return best_dir


def _extract_norm_paths(workspace_path: Path) -> NormPaths:
    mean = workspace_path / "mean.txt"
    var = workspace_path / "var.txt"
    count = workspace_path / "count.txt"
    return NormPaths(
        mean=mean if mean.exists() else None,
        var=var if var.exists() else None,
        count=count if count.exists() else None,
    )


def _latest_complete_episode_monitor_paths(workspace_path: Path) -> dict[str, Path]:
    from .episode_selector import resolve_episode

    episode_ref = resolve_episode(workspace_path, "last")
    if episode_ref is None:
        return {}
    monitor_dir = episode_ref.source_path / "monitor"
    if not monitor_dir.exists():
        return {}
    return {path.name: path for path in monitor_dir.glob("*.csv") if path.is_file()}


def _build_callbacks(
    case: ResolvedCase, registry: StudyRegistry, run_context, train_env
):
    callbacks = []
    eval_cfg = case.training.get("evaluation", {})
    if eval_cfg.get("enabled", False):
        eval_env = build_environment(case, run_context, mode="eval")
        callbacks.append(
            LoggerEvalCallback(
                eval_env=eval_env,
                train_env=train_env,
                n_eval_episodes=int(eval_cfg.get("n_eval_episodes", 1)),
                eval_freq_episodes=int(eval_cfg.get("eval_freq_episodes", 2)),
                deterministic=bool(eval_cfg.get("deterministic", True)),
            )
        )
        return CallbackList(callbacks), eval_env
    return None, None


def _save_model(model: PPO, workspace_path: Path, model_name: str) -> Path:
    target = workspace_path / model_name
    model.save(str(target))
    return workspace_path / f"{model_name}.zip"


def _write_failure_manifest(result: RunResult) -> None:
    payload = {
        "case_id": result.case_id,
        "run_id": result.run_context.run_id,
        "workspace_path": str(result.workspace_path),
        "status": "failed",
        "metadata": result.metadata,
        "git_sha": result.run_context.git_sha,
    }
    path = result.run_context.manifest_root / f"{result.run_context.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_dir(case_id: str, registry: StudyRegistry) -> Path:
    return registry.paths["manifest_root"] / case_id


def _sorted_manifest_paths(case_id: str, registry: StudyRegistry) -> list[Path]:
    manifest_dir = _manifest_dir(case_id, registry)
    if not manifest_dir.exists():
        return []
    return sorted(
        (path for path in manifest_dir.glob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def case_has_manifest(
    case_id: str, registry: StudyRegistry, *, status: str | None = None
) -> bool:
    return latest_manifest_path(case_id, registry, status=status) is not None


def latest_manifest_path(
    case_id: str, registry: StudyRegistry, *, status: str | None = None
) -> Optional[Path]:
    for path in _sorted_manifest_paths(case_id, registry):
        if status is None:
            return path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == status:
            return path
    return None


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_result(
    case: ResolvedCase, registry: StudyRegistry, *, status: str = "success"
) -> RunResult:
    manifest_path = latest_manifest_path(case.id, registry, status=status)
    if manifest_path is None:
        raise FileNotFoundError(f"No {status} manifest found for case `{case.id}`.")

    payload = _load_manifest(manifest_path)
    run_id = str(payload.get("run_id") or manifest_path.stem)
    workspace_path = Path(payload["workspace_path"])
    run_context = rebuild_run_context(
        case.id,
        run_id,
        registry,
        started_at=str(payload.get("started_at") or run_id.split("_", 1)[0]),
        git_sha=payload.get("git_sha"),
    )
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("status", payload.get("status", status))

    model_name = case.outputs.get("model_name", case.method.default_model_name)
    model_path = workspace_path / f"{model_name}.zip"
    if not model_path.exists():
        model_path = None

    source_bundle = None
    source_bundle_path = payload.get("source_bundle")
    if source_bundle_path:
        source_model_path = Path(source_bundle_path)
        source_bundle = ModelBundle(
            case_id="unknown",
            building_id="unknown",
            method_id="unknown",
            model_path=source_model_path,
            export_path=(
                source_model_path.parent if source_model_path.exists() else None
            ),
            metadata={"reconstructed_from_manifest": True},
        )

    result = RunResult(
        case_id=case.id,
        workspace_path=workspace_path,
        run_context=run_context,
        model_path=model_path,
        progress_path=(
            (workspace_path / "progress.csv")
            if (workspace_path / "progress.csv").exists()
            else None
        ),
        norm_paths=_extract_norm_paths(workspace_path),
        latest_episode_monitor_paths=_latest_complete_episode_monitor_paths(
            workspace_path
        ),
        metadata=metadata,
        source_bundle=source_bundle,
    )
    return result


def export_existing_case(case: ResolvedCase, registry: StudyRegistry) -> RunResult:
    result = load_existing_result(case, registry, status="success")
    result.export_report = export_run(result, case.export)
    if result.export_report is not None:
        result.metadata["export_destination"] = str(result.export_report.destination)
        result.metadata["manifest_path"] = str(result.export_report.manifest_path)
    return result


def run_case(case: ResolvedCase, registry: StudyRegistry) -> RunResult:
    run_context = build_run_context(case.id, registry)
    prepare_runtime_environment(run_context)
    train_env = build_environment(case, run_context, mode="train")
    callbacks = None
    eval_env = None
    result: Optional[RunResult] = None

    try:
        callbacks, eval_env = _build_callbacks(case, registry, run_context, train_env)
        services = RunnerServices(
            case, registry, run_context=run_context, callbacks=callbacks
        )
        runner = get_method_runner(case.method.runner)
        execution = runner.execute(case, train_env, services)
        workspace_path = Path(train_env.get_wrapper_attr("workspace_path"))
        model_name = case.outputs.get("model_name", case.method.default_model_name)
        model_path = _save_model(execution.model, workspace_path, model_name)
        progress = workspace_path / "progress.csv"
        metadata = {
            "status": "success",
            "building_id": case.building.id,
            "method_id": case.method.id,
            "stage": case.stage,
            "total_timesteps": execution.total_timesteps,
            "run_id": run_context.run_id,
        }
        metadata.update(case.metadata)
        metadata.update(execution.metadata)
        result = RunResult(
            case_id=case.id,
            workspace_path=workspace_path,
            run_context=run_context,
            model_path=model_path,
            progress_path=progress if progress.exists() else None,
            norm_paths=_extract_norm_paths(workspace_path),
            latest_episode_monitor_paths=_latest_complete_episode_monitor_paths(
                workspace_path
            ),
            metadata=metadata,
            source_bundle=execution.source_bundle,
        )
    except Exception:
        if result is None:
            result = RunResult(
                case_id=case.id,
                workspace_path=run_context.workspace_root,
                run_context=run_context,
                metadata={
                    "status": "failed",
                    "building_id": case.building.id,
                    "method_id": case.method.id,
                    "stage": case.stage,
                    "run_id": run_context.run_id,
                },
            )
        else:
            result.metadata["status"] = "failed"
        _write_failure_manifest(result)
        raise
    finally:
        if eval_env is not None:
            eval_env.close()
        train_env.close()

    result.export_report = export_run(result, case.export)
    if result.export_report is not None:
        result.metadata["export_destination"] = str(result.export_report.destination)
        result.metadata["manifest_path"] = str(result.export_report.manifest_path)
    cleanup_policy = CleanupPolicy(
        retain_success=int(case.cleanup.get("retain_success", 3)),
        retain_failure=int(case.cleanup.get("retain_failure", 3)),
    )
    result.cleanup_report = cleanup_run(result, cleanup_policy)
    return result


def run_group(
    group_id: str, registry: StudyRegistry, *, continue_on_error: bool | None = None
) -> list[RunResult]:
    results: list[RunResult] = []
    continue_on_error = (
        registry.runtime.get("continue_on_error", True)
        if continue_on_error is None
        else continue_on_error
    )
    for case_id in get_group_case_ids(group_id, registry):
        case = resolve_case(case_id, registry)
        try:
            results.append(run_case(case, registry))
        except Exception:
            if not continue_on_error:
                raise
    return results


def run_all(
    registry: StudyRegistry, *, continue_on_error: bool | None = None
) -> list[RunResult]:
    return run_group("all", registry, continue_on_error=continue_on_error)
