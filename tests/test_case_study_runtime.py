import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from sinergym.config.modeling import ModelJSON
from studies.case_study.core import (
    export_existing_case,
    load_registry,
    resolve_case,
    run_case,
)
from studies.case_study.core import runner as runner_module

CONFIG_ROOT = "studies/case_study/config"


def test_modeljson_respects_workspace_root(tmp_path, VARIABLES_5ZONE, METERS_5ZONE):
    workspace_root = tmp_path / "workspace_root"
    model = ModelJSON(
        env_name="PYTESTCONFIG",
        json_file="5ZoneAutoDXVAV.epJSON",
        weather_files=["USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw"],
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        max_ep_store=10,
        building_config={
            "timesteps_per_hour": 2,
            "runperiod": (1, 2, 1993, 2, 3, 1993),
        },
        workspace_root=str(workspace_root),
    )

    assert Path(model.workspace_path).exists()
    assert Path(model.workspace_path).parent == workspace_root


def _retarget_registry_paths(registry, tmp_path: Path):
    paths = dict(registry.paths)
    paths.update(
        {
            "artifact_root": tmp_path / "artifacts",
            "workspace_root": tmp_path / "workspaces",
            "export_root": tmp_path / "exports",
            "figure_root": tmp_path / "figures",
            "manifest_root": tmp_path / "manifests",
            "third_party_root": tmp_path / "third_party",
        }
    )
    for path in paths.values():
        if isinstance(path, Path) and path != registry.paths["project_root"]:
            path.mkdir(parents=True, exist_ok=True)
    registry.paths = paths
    return registry


def test_export_existing_case_uses_latest_complete_episode(tmp_path):
    registry = _retarget_registry_paths(load_registry(CONFIG_ROOT), tmp_path)

    case = resolve_case("pretrain_office_smud_sac_plus", registry)
    run_id = "20260327T000000Z_abcd1234"
    workspace_path = (
        registry.paths["workspace_root"] / case.id / run_id / "TRAINING-res1"
    )
    workspace_path.mkdir(parents=True, exist_ok=True)

    for name in ["model.zip", "mean.txt", "var.txt", "count.txt", "progress.csv"]:
        (workspace_path / name).write_text("stub", encoding="utf-8")

    for episode in [1, 2, 3]:
        monitor_dir = workspace_path / f"episode-{episode}" / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        rewards = "reward\n1\n2\n" if episode < 3 else "reward\n5\n"
        (monitor_dir / "rewards.csv").write_text(rewards, encoding="utf-8")

    manifest_dir = registry.paths["manifest_root"] / case.id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "case_id": case.id,
                "run_id": run_id,
                "workspace_path": str(workspace_path),
                "status": "success",
                "metadata": {
                    "status": "success",
                    "building_id": case.building.id,
                    "method_id": case.method.id,
                },
            }
        ),
        encoding="utf-8",
    )

    result = export_existing_case(case, registry)

    assert result.export_report is not None
    assert (result.export_report.destination / "rewards.csv").exists()
    assert (
        result.export_report.resolved_episode_selectors["last"].resolved_episode_number
        == 2
    )
    exported_manifest = json.loads(
        (result.export_report.destination / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        exported_manifest["resolved_episode_selectors"]["last"][
            "resolved_episode_number"
        ]
        == 2
    )


class _DummyTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, obs_dim: int, workspace_path: Path, timestep_per_episode: int = 9
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._workspace_path = workspace_path
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        (self._workspace_path / "progress.csv").write_text(
            "episode_num,mean_reward\n1,0.0\n", encoding="utf-8"
        )
        self._steps = 0
        self._timestep_per_episode = timestep_per_episode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= 2
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = float(1.0 - np.abs(action).mean())
        return obs, reward, terminated, False, {}

    def close(self):
        return None

    def get_wrapper_attr(self, name: str):
        if name == "workspace_path":
            return str(self._workspace_path)
        if name == "timestep_per_episode":
            return self._timestep_per_episode
        raise AttributeError(name)


def _build_temp_registry(tmp_path: Path):
    """Copy the config tree and shrink the two paper cases under test to
    seconds-scale training (tiny SAC nets, 8 timesteps)."""
    config_root = tmp_path / "config"
    shutil.copytree(Path(CONFIG_ROOT), config_root)

    cases_path = config_root / "cases.yaml"
    cases_yaml = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    tuned = {
        "pretrain_office_smud_sac_plus": {
            "total_timesteps": 8,
            "model_kwargs": {
                "seed": 0,
                "buffer_size": 32,
                "learning_starts": 2,
                "batch_size": 2,
                "train_freq": 1,
                "gradient_steps": 1,
                "policy_kwargs": {"net_arch": [8]},
            },
        },
        "transfer_office_to_lbuilding_smud_sacp_zs": {"total_timesteps": 4},
    }
    for case in cases_yaml["cases"]:
        if case["id"] not in tuned:
            continue
        training = case.setdefault("training", {})
        training.update(tuned[case["id"]])
        training["verbose"] = 0
        training["evaluation"] = {"enabled": False}
    cases_path.write_text(yaml.safe_dump(cases_yaml, sort_keys=False), encoding="utf-8")

    return _retarget_registry_paths(load_registry(config_root), tmp_path)


def test_run_case_supports_paper_pretrain_and_zero_shot_transfer(tmp_path, monkeypatch):
    """The paper transfer path at unit scale: SAC source pretrain on the
    reduced state, then the frozen zero-shot deployment loading that source."""
    registry = _build_temp_registry(tmp_path)

    # Reduced state is zone-invariant: source and target share the obs dim.
    obs_dim = 12

    def fake_build_environment(case, run_context, mode="train"):
        workspace_path = run_context.workspace_root / f"{case.id}_{mode}-res1"
        return _DummyTransferEnv(obs_dim, workspace_path)

    monkeypatch.setattr(runner_module, "build_environment", fake_build_environment)

    pretrain_case = resolve_case("pretrain_office_smud_sac_plus", registry)
    pretrain = run_case(pretrain_case, registry)
    assert pretrain.model_path is not None
    assert pretrain.model_path.name == "model.zip"
    source_export = (
        registry.paths["export_root"] / "pretrain_office_smud_sac_plus" / "model.zip"
    )
    assert source_export.exists()

    zs_case = resolve_case("transfer_office_to_lbuilding_smud_sacp_zs", registry)
    assert zs_case.inputs["source_case_id"] == "pretrain_office_smud_sac_plus"
    zs = run_case(zs_case, registry)
    assert zs.model_path is not None
    assert zs.model_path.suffix == ".zip"

    zs_export = (
        registry.paths["export_root"] / "transfer_office_to_lbuilding_smud_sacp_zs"
    )
    manifest = json.loads((zs_export / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["method_id"] == zs_case.method.id
    assert manifest["metadata"]["building_id"] == "office_lbuilding_8zone"
    assert str(zs.workspace_path).startswith(str(registry.paths["workspace_root"]))
