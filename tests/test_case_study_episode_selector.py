from pathlib import Path

from studies.case_study.core.episode_selector import (
    list_complete_episode_numbers,
    list_episode_numbers,
    resolve_episode,
)


def _make_episode(workspace: Path, number: int) -> None:
    episode_dir = workspace / f"episode-{number}" / "monitor"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "rewards.csv").write_text("reward\n1\n2\n", encoding="utf-8")


def test_episode_selector_uses_last_complete(tmp_path):
    workspace = tmp_path / "workspace"
    for number in [1, 2, 3, 4]:
        _make_episode(workspace, number)

    assert list_episode_numbers(workspace) == [1, 2, 3, 4]
    assert list_complete_episode_numbers(workspace) == [1, 2, 3]

    last_ref = resolve_episode(workspace, "last")
    assert last_ref is not None
    assert last_ref.resolved_episode_number == 3
    assert last_ref.resolution_reason == "max_episode_minus_one"
    assert last_ref.fallback_used is False

    penultimate_ref = resolve_episode(workspace, "penultimate")
    assert penultimate_ref is not None
    assert penultimate_ref.resolved_episode_number == 2
    assert penultimate_ref.resolution_reason == "max_episode_minus_two"

    raw_last_ref = resolve_episode(workspace, "raw_last")
    assert raw_last_ref is not None
    assert raw_last_ref.resolved_episode_number == 4


def test_episode_selector_falls_back_to_previous_complete(tmp_path):
    workspace = tmp_path / "workspace"
    _make_episode(workspace, 1)
    _make_episode(workspace, 2)

    last_ref = resolve_episode(workspace, "last")
    assert last_ref is not None
    assert last_ref.resolved_episode_number == 1

    penultimate_ref = resolve_episode(workspace, "penultimate")
    assert penultimate_ref is not None
    assert penultimate_ref.resolved_episode_number == 1
    assert penultimate_ref.resolution_reason == "previous_complete_fallback"
    assert penultimate_ref.fallback_used is True


def test_episode_selector_single_episode_fallback(tmp_path):
    workspace = tmp_path / "workspace"
    _make_episode(workspace, 1)

    last_ref = resolve_episode(workspace, "last")
    assert last_ref is not None
    assert last_ref.resolved_episode_number == 1
    assert last_ref.resolution_reason == "single_episode_fallback"
    assert last_ref.fallback_used is True
