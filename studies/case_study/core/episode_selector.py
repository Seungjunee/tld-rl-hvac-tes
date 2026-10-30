from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from .types import EpisodeRef, EpisodeSelector


def parse_episode_selector(value: str | EpisodeSelector) -> EpisodeSelector:
    if isinstance(value, EpisodeSelector):
        return value
    raw = str(value)
    if raw.startswith("index(") and raw.endswith(")"):
        idx = int(raw[len("index(") : -1])
        return EpisodeSelector(selector_type="index", index=idx)
    if raw not in {"first", "last", "last_complete", "penultimate", "raw_last"}:
        raise ValueError(f"Unsupported episode selector: {value}")
    return EpisodeSelector(selector_type=raw)  # type: ignore[arg-type]


def list_episode_numbers(workspace_path: str | Path) -> list[int]:
    workspace = Path(workspace_path)
    numbers = []
    for path in workspace.glob("episode-*"):
        if not path.is_dir():
            continue
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(numbers)


def list_complete_episode_numbers(workspace_path: str | Path) -> list[int]:
    episodes = list_episode_numbers(workspace_path)
    if len(episodes) <= 1:
        return episodes
    return episodes[:-1]


def _fallback_episode(complete_episodes: list[int], all_episodes: list[int]) -> tuple[Optional[int], str, bool]:
    if complete_episodes:
        return complete_episodes[-1], "previous_complete_fallback", True
    if all_episodes:
        return all_episodes[-1], "single_episode_fallback", True
    return None, "missing_episode", True


def resolve_episode(workspace_path: str | Path, selector: str | EpisodeSelector) -> Optional[EpisodeRef]:
    workspace = Path(workspace_path)
    parsed = parse_episode_selector(selector)
    all_episodes = list_episode_numbers(workspace)
    complete_episodes = list_complete_episode_numbers(workspace)
    if not all_episodes:
        return None

    requested = parsed.selector_type
    resolved: Optional[int] = None
    reason = requested
    fallback_used = False

    if requested == "raw_last":
        resolved = all_episodes[-1]
        reason = "raw_last"
    elif requested == "first":
        if complete_episodes:
            resolved = complete_episodes[0]
            reason = "first_complete"
        else:
            resolved, reason, fallback_used = _fallback_episode(complete_episodes, all_episodes)
    elif requested in {"last", "last_complete"}:
        if complete_episodes:
            resolved = complete_episodes[-1]
            reason = "max_episode_minus_one" if len(all_episodes) > 1 else "single_episode_fallback"
            fallback_used = len(all_episodes) <= 1
        else:
            resolved, reason, fallback_used = _fallback_episode(complete_episodes, all_episodes)
    elif requested == "penultimate":
        if len(complete_episodes) >= 2:
            resolved = complete_episodes[-2]
            reason = "max_episode_minus_two"
        else:
            resolved, reason, fallback_used = _fallback_episode(complete_episodes, all_episodes)
    elif requested == "index":
        if parsed.index is not None and parsed.index < len(complete_episodes):
            resolved = complete_episodes[parsed.index]
            reason = f"index_{parsed.index}"
        else:
            resolved, reason, fallback_used = _fallback_episode(complete_episodes, all_episodes)
    else:
        raise ValueError(f"Unsupported selector type: {requested}")

    if resolved is None:
        return None
    return EpisodeRef(
        requested_selector=str(selector),
        resolved_episode_number=resolved,
        resolution_reason=reason,
        source_path=workspace / f"episode-{resolved}",
        fallback_used=fallback_used,
    )


def resolve_episode_map(workspace_path: str | Path, selectors: Iterable[str]) -> Dict[str, EpisodeRef]:
    resolved: Dict[str, EpisodeRef] = {}
    for selector in selectors:
        episode_ref = resolve_episode(workspace_path, selector)
        if episode_ref is not None:
            resolved[str(selector)] = episode_ref
    return resolved
