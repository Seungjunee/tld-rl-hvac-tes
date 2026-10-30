from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Optional

from stable_baselines3 import PPO

from studies.case_study.core.types import ModelBundle, ResolvedCase


@dataclass
class MethodExecution:
    model: PPO
    total_timesteps: int
    source_bundle: Optional[ModelBundle] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MethodServices(Protocol):
    def build_training_model(self, case: ResolvedCase, env) -> PPO: ...
    def compute_total_timesteps(self, case: ResolvedCase, env) -> int: ...
    def load_source_bundle(self, case: ResolvedCase) -> ModelBundle: ...
    def load_model(self, model_path): ...
    def build_learn_kwargs(self, case: ResolvedCase) -> Dict[str, Any]: ...


class MethodRunner(Protocol):
    method_id: str

    def execute(self, case: ResolvedCase, env, services: MethodServices) -> MethodExecution: ...
