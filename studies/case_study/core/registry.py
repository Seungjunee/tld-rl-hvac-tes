from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from .types import (
    BuildingSpec,
    CaseSpec,
    FigureRunSpec,
    FigureSpec,
    MethodSpec,
    ResolvedCase,
    StudyRegistry,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_paths(project_root: Path, defaults: Dict[str, Any]) -> Dict[str, Path]:
    raw_paths = defaults.get("paths", {})
    resolved: Dict[str, Path] = {}
    for key, value in raw_paths.items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else project_root / p
    resolved.setdefault("project_root", project_root)
    return resolved


def _discover_project_root(config_root: Path) -> Path:
    config_root = config_root.resolve()
    for candidate in [config_root, *config_root.parents]:
        if (candidate / ".git").exists():
            return candidate
        if all(
            (candidate / name).exists() for name in ["sinergym", "examples", "studies"]
        ):
            return candidate
    parents = config_root.parents
    return parents[2] if len(parents) > 2 else config_root.parent


def load_registry(
    config_root: str | Path = "studies/case_study/config",
) -> StudyRegistry:
    config_root = Path(config_root)
    defaults = _read_yaml(config_root / "defaults.yaml")
    buildings_raw = _read_yaml(config_root / "buildings.yaml").get("buildings", {})
    methods_raw = _read_yaml(config_root / "methods.yaml").get("methods", {})
    recipes = _read_yaml(config_root / "recipes.yaml")
    cases_yaml = _read_yaml(config_root / "cases.yaml")
    figures_yaml = _read_yaml(config_root / "figures.yaml").get("figures", {})

    project_root = _discover_project_root(config_root)
    paths = _resolve_paths(project_root, defaults)

    buildings = {
        key: BuildingSpec(id=key, **value) for key, value in buildings_raw.items()
    }
    _method_known_keys = {
        "runner",
        "requires_source_model",
        "source_bundle_policy",
        "normalization_policy",
        "default_model_name",
        "training",
    }
    methods = {
        key: MethodSpec(
            id=key,
            metadata={k: v for k, v in value.items() if k not in _method_known_keys},
            **{k: v for k, v in value.items() if k in _method_known_keys},
        )
        for key, value in methods_raw.items()
    }

    case_order = []
    cases: Dict[str, CaseSpec] = {}
    for raw in cases_yaml.get("cases", []):
        case = CaseSpec(
            id=raw["id"],
            stage=raw["stage"],
            building=raw["building"],
            method=raw["method"],
            training_profile=raw.get("training_profile"),
            observation_profile=raw.get("observation_profile"),
            reward_profile=raw.get("reward_profile"),
            export_profile=raw.get("export_profile"),
            cleanup_policy=raw.get("cleanup_policy", "retain_recent_three"),
            env=raw.get("env", {}),
            training=raw.get("training", {}),
            inputs=raw.get("inputs", {}),
            outputs=raw.get("outputs", {}),
            export=raw.get("export", {}),
            wrappers=raw.get("wrappers", {}),
            metadata={
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "id",
                    "stage",
                    "building",
                    "method",
                    "training_profile",
                    "observation_profile",
                    "reward_profile",
                    "export_profile",
                    "cleanup_policy",
                    "env",
                    "training",
                    "inputs",
                    "outputs",
                    "export",
                    "wrappers",
                }
            },
        )
        case_order.append(case.id)
        cases[case.id] = case

    figures = {}
    for key, value in figures_yaml.items():
        figures[key] = FigureSpec(
            id=key,
            type=value["type"],
            style=value.get("style"),
            runs=[FigureRunSpec(**run) for run in value.get("runs", [])],
            output=value.get("output"),
            metadata={
                k: v
                for k, v in value.items()
                if k not in {"type", "style", "runs", "output"}
            },
        )

    return StudyRegistry(
        config_root=config_root,
        paths=paths,
        defaults=defaults,
        runtime=defaults.get("runtime", {}),
        buildings=buildings,
        methods=methods,
        reward_profiles=recipes.get("reward_profiles", {}),
        observation_profiles=recipes.get("observation_profiles", {}),
        training_profiles=recipes.get("training_profiles", {}),
        export_profiles=recipes.get("export_profiles", {}),
        cleanup_policies=recipes.get("cleanup_policies", {}),
        figure_styles=recipes.get("figure_styles", {}),
        cases=cases,
        case_order=case_order,
        groups=cases_yaml.get("groups", {}),
        figures=figures,
    )


def resolve_case_id(case_id: str, registry: StudyRegistry) -> str:
    if case_id not in registry.cases:
        raise KeyError(f"Unknown case id: {case_id}")
    return case_id


def list_case_ids(registry: StudyRegistry) -> list[str]:
    return list(registry.case_order)


def iter_cases(registry: StudyRegistry) -> Iterable[ResolvedCase]:
    for case_id in registry.case_order:
        yield resolve_case(case_id, registry)


def get_case_spec(case_id: str, registry: StudyRegistry) -> CaseSpec:
    resolved = resolve_case_id(case_id, registry)
    if resolved not in registry.cases:
        raise KeyError(f"Unknown case id: {case_id}")
    return registry.cases[resolved]


def get_group_case_ids(group_id: str, registry: StudyRegistry) -> list[str]:
    if group_id not in registry.groups:
        raise KeyError(f"Unknown group id: {group_id}")
    return [resolve_case_id(case_id, registry) for case_id in registry.groups[group_id]]


def get_figure(figure_id: str, registry: StudyRegistry) -> FigureSpec:
    if figure_id not in registry.figures:
        raise KeyError(f"Unknown figure id: {figure_id}")
    return registry.figures[figure_id]


def resolve_case(case_id: str, registry: StudyRegistry) -> ResolvedCase:
    spec = get_case_spec(case_id, registry)
    building = registry.buildings[spec.building]
    method = registry.methods[spec.method]

    reward_profile_name = spec.reward_profile or building.reward_profile
    reward = deepcopy(registry.reward_profiles[reward_profile_name])

    observation_profile_name = (
        spec.observation_profile
        or building.observation_profiles.get("raw_export")
        or ""
    )
    wrappers = _deep_merge(
        deepcopy(registry.defaults.get("wrappers", {})),
        deepcopy(registry.observation_profiles.get(observation_profile_name, {})),
    )
    # Case-level wrapper overrides (e.g. pinning a per-case predictor file)
    # merge last, mirroring the training/env/export override chain.
    wrappers = _deep_merge(wrappers, deepcopy(spec.wrappers))

    training = _deep_merge(
        deepcopy(registry.defaults.get("training", {})),
        deepcopy(registry.training_profiles.get(spec.training_profile or "", {})),
    )
    training = _deep_merge(training, deepcopy(method.training))
    training = _deep_merge(training, deepcopy(spec.training))

    export_profile = deepcopy(
        registry.export_profiles.get(spec.export_profile or "", {})
    )
    export_profile = _deep_merge(export_profile, deepcopy(spec.export))
    export_profile.setdefault(
        "destination", str(registry.paths["export_root"] / spec.id)
    )

    cleanup = deepcopy(registry.cleanup_policies.get(spec.cleanup_policy, {}))

    env = {
        "id": building.env_id,
        "name": building.env_id,
        "register_yaml": building.register_yaml,
        "env_name_base": building.env_name_base,
        "workspace_tag": building.workspace_tag,
        "runtime": deepcopy(building.runtime),
    }
    env = _deep_merge(env, deepcopy(spec.env))

    outputs = {"model_name": spec.outputs.get("model_name", method.default_model_name)}
    outputs = _deep_merge(outputs, deepcopy(spec.outputs))

    inputs = deepcopy(spec.inputs)
    metadata = deepcopy(spec.metadata)
    metadata.update(
        {
            "reward_profile": reward_profile_name,
            "observation_profile": observation_profile_name,
            "training_profile": spec.training_profile,
            "export_profile": spec.export_profile,
            "cleanup_policy": spec.cleanup_policy,
        }
    )

    return ResolvedCase(
        id=spec.id,
        stage=spec.stage,
        building=building,
        method=method,
        env=env,
        reward=reward,
        wrappers=wrappers,
        training=training,
        inputs=inputs,
        outputs=outputs,
        export=export_profile,
        cleanup=cleanup,
        paths=registry.paths,
        runtime=registry.runtime,
        metadata=metadata,
    )
