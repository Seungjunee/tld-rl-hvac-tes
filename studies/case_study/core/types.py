from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class NormPaths:
    mean: Optional[Path] = None
    var: Optional[Path] = None
    count: Optional[Path] = None


@dataclass
class BuildingSpec:
    id: str
    env_id: str
    env_name_base: str
    register_yaml: str
    workspace_tag: str
    reward_profile: str
    runtime: Dict[str, Any] = field(default_factory=dict)
    physical: Dict[str, Any] = field(default_factory=dict)
    observation_profiles: Dict[str, str] = field(default_factory=dict)


@dataclass
class MethodSpec:
    id: str
    runner: str
    requires_source_model: bool
    source_bundle_policy: str
    normalization_policy: str
    default_model_name: str
    training: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseSpec:
    id: str
    stage: str
    building: str
    method: str
    training_profile: Optional[str] = None
    observation_profile: Optional[str] = None
    reward_profile: Optional[str] = None
    export_profile: Optional[str] = None
    cleanup_policy: str = "retain_recent_three"
    env: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    export: Dict[str, Any] = field(default_factory=dict)
    wrappers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureRunSpec:
    label: str
    case_id: str
    selector: Optional[str] = None
    line_style: Optional[str] = None
    line_width: Optional[float] = None
    color_group: Optional[str] = None


@dataclass
class FigureSpec:
    id: str
    type: str
    style: Optional[str] = None
    runs: List[FigureRunSpec] = field(default_factory=list)
    output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyRegistry:
    config_root: Path
    paths: Dict[str, Path]
    defaults: Dict[str, Any]
    runtime: Dict[str, Any]
    buildings: Dict[str, BuildingSpec]
    methods: Dict[str, MethodSpec]
    reward_profiles: Dict[str, Dict[str, Any]]
    observation_profiles: Dict[str, Dict[str, Any]]
    training_profiles: Dict[str, Dict[str, Any]]
    export_profiles: Dict[str, Dict[str, Any]]
    cleanup_policies: Dict[str, Dict[str, Any]]
    figure_styles: Dict[str, Dict[str, Any]]
    cases: Dict[str, CaseSpec]
    case_order: List[str]
    groups: Dict[str, List[str]]
    figures: Dict[str, FigureSpec]


@dataclass
class ResolvedCase:
    id: str
    stage: str
    building: BuildingSpec
    method: MethodSpec
    env: Dict[str, Any]
    reward: Dict[str, Any]
    wrappers: Dict[str, Any]
    training: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    export: Dict[str, Any]
    cleanup: Dict[str, Any]
    paths: Dict[str, Path]
    runtime: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    case_id: str
    run_id: str
    started_at: str
    workspace_root: Path
    export_root: Path
    figure_root: Path
    manifest_root: Path
    third_party_root: Path
    project_root: Path
    git_sha: Optional[str] = None


@dataclass
class ModelBundle:
    case_id: str
    building_id: str
    method_id: str
    model_path: Path
    export_path: Optional[Path] = None
    norm_paths: NormPaths = field(default_factory=NormPaths)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSelector:
    selector_type: Literal[
        "first", "last", "last_complete", "penultimate", "index", "raw_last"
    ]
    fallback_policy: Literal["previous_complete", "skip", "raw_last"] = (
        "previous_complete"
    )
    index: Optional[int] = None


@dataclass
class EpisodeRef:
    requested_selector: str
    resolved_episode_number: int
    resolution_reason: str
    source_path: Path
    fallback_used: bool = False


@dataclass
class ExportReport:
    case_id: str
    destination: Path
    profile: str
    copied_files: List[Path] = field(default_factory=list)
    preprocess_output: Optional[Path] = None
    manifest_path: Optional[Path] = None
    export_manifest_path: Optional[Path] = None
    resolved_episode_selectors: Dict[str, EpisodeRef] = field(default_factory=dict)


@dataclass
class CleanupPolicy:
    retain_success: int = 3
    retain_failure: int = 3


@dataclass
class CleanupReport:
    case_id: str
    removed_paths: List[Path] = field(default_factory=list)
    kept_paths: List[Path] = field(default_factory=list)


@dataclass
class RunResult:
    case_id: str
    workspace_path: Path
    run_context: RunContext
    model_path: Optional[Path] = None
    progress_path: Optional[Path] = None
    norm_paths: NormPaths = field(default_factory=NormPaths)
    latest_episode_monitor_paths: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_bundle: Optional[ModelBundle] = None
    export_report: Optional[ExportReport] = None
    cleanup_report: Optional[CleanupReport] = None
