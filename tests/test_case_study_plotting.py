"""Episode-selector mechanics of ``core/plotting.py`` on current case ids.

Only the selector semantics are asserted here: ``last`` returns the last
COMPLETE episode, ``raw_last`` the numerically last one even if partial (the
convention the export/profile tooling relies on).

The figure-spec subsystem (``figures.yaml`` + ``core/figure_bundle.py``) is
legacy: every registered spec references retired PPO-era case ids and the
bundle's active figure list is empty. The paper figures are produced by the
``studies/case_study/analysis/`` scripts instead, so no figure rendering is
asserted here.
"""

from pathlib import Path

import pandas as pd

from studies.case_study.core import load_registry
from studies.case_study.core.plotting import load_episode_rewards

CONFIG_ROOT = "studies/case_study/config"

# Any living case id works; the padding transfer runs 10 target episodes, so
# partial-final-episode exports are its realistic failure mode.
CASE_ID = "transfer_office_to_lbuilding_smud_sacp_padrl"


def _write_rewards(
    export_root: Path, case_id: str, episode: int, rewards: list[float]
) -> None:
    monitor_dir = export_root / case_id / f"episode-{episode}" / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reward": rewards}).to_csv(monitor_dir / "rewards.csv", index=False)


def _with_temp_export_root(tmp_path: Path):
    registry = load_registry(CONFIG_ROOT)
    paths = dict(registry.paths)
    export_root = tmp_path / "exports"
    figure_root = tmp_path / "figures"
    export_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths["export_root"] = export_root
    paths["figure_root"] = figure_root
    registry.paths = paths
    return registry


def test_load_episode_rewards_last_uses_complete_episode(tmp_path):
    registry = _with_temp_export_root(tmp_path)
    _write_rewards(registry.paths["export_root"], CASE_ID, 1, [1.0, 2.0])
    _write_rewards(registry.paths["export_root"], CASE_ID, 2, [3.0, 4.0])
    _write_rewards(registry.paths["export_root"], CASE_ID, 3, [5.0])  # partial

    rewards, episode_ref = load_episode_rewards(CASE_ID, registry, selector="last")

    assert episode_ref.resolved_episode_number == 2
    assert list(rewards.columns) == ["step", "reward", "cumulative_reward"]
    assert rewards["step"].tolist() == [1, 2]
    assert rewards["cumulative_reward"].iloc[-1] == 7.0


def test_load_episode_rewards_raw_last_uses_partial_final_episode(tmp_path):
    registry = _with_temp_export_root(tmp_path)
    _write_rewards(registry.paths["export_root"], CASE_ID, 1, [1.0, 2.0])
    _write_rewards(registry.paths["export_root"], CASE_ID, 2, [3.0, 4.0])
    _write_rewards(registry.paths["export_root"], CASE_ID, 3, [9.0])  # partial

    rewards, episode_ref = load_episode_rewards(CASE_ID, registry, selector="raw_last")

    assert episode_ref.resolved_episode_number == 3
    assert rewards["cumulative_reward"].iloc[-1] == 9.0
