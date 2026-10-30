from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .episode_selector import resolve_episode
from .registry import get_figure, resolve_case_id
from .types import FigureSpec, StudyRegistry


def _case_export_dir(case_id: str, registry: StudyRegistry) -> Path:
    resolved = resolve_case_id(case_id, registry)
    canonical = registry.paths["export_root"] / resolved
    if canonical.exists():
        return canonical
    raise FileNotFoundError(f"No export directory found for case: {case_id}")


def _style(spec: FigureSpec, registry: StudyRegistry) -> dict:
    return dict(registry.figure_styles.get(spec.style or "", {}))


def _selector_label(base_label: str, episode_ref) -> str:
    if episode_ref is None or not episode_ref.fallback_used:
        return base_label
    return f"{base_label} (-> Episode {episode_ref.resolved_episode_number} fallback)"


def _resolve_group_color(color_group: str | None, color_map: dict[str, str], color_cycle: list[str], color_index: int) -> tuple[str | None, int]:
    if color_group is None:
        if color_index >= len(color_cycle):
            return None, color_index + 1
        return color_cycle[color_index], color_index + 1
    if color_group not in color_map:
        color_map[color_group] = color_cycle[color_index % len(color_cycle)]
        color_index += 1
    return color_map[color_group], color_index


def plot_learning_curves(figure_id: str, registry: StudyRegistry):
    spec = get_figure(figure_id, registry)
    style = _style(spec, registry)
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [10, 6])))
    plotted = 0
    for run in spec.runs:
        export_dir = _case_export_dir(run.case_id, registry)
        progress = export_dir / "progress.csv"
        if not progress.exists():
            continue
        df = pd.read_csv(progress)
        x_col = "episode_num" if "episode_num" in df.columns else ("time/episodes" if "time/episodes" in df.columns else None)
        y_col = "mean_reward" if "mean_reward" in df.columns else ("rollout/ep_rew_mean" if "rollout/ep_rew_mean" in df.columns else None)
        if x_col and y_col:
            plot_df = df
            max_episodes = spec.metadata.get("max_episodes") or style.get("max_episodes")
            if max_episodes is not None:
                plot_df = df[df[x_col] <= int(max_episodes)]
            plot_kwargs: dict = {"label": run.label}
            if run.line_style:
                plot_kwargs["linestyle"] = run.line_style
            if run.line_width is not None:
                plot_kwargs["linewidth"] = run.line_width
            ax.plot(plot_df[x_col], plot_df[y_col], **plot_kwargs)
            plotted += 1
    if plotted == 0:
        raise RuntimeError(f"No progress data available for figure {figure_id}")
    ax.set_title("Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=style.get("grid_alpha", 0.3))
    ax.legend()
    fig.tight_layout()
    return fig


def _load_progress_metric(case_id: str, registry: StudyRegistry) -> float:
    export_dir = _case_export_dir(case_id, registry)
    progress = export_dir / "progress.csv"
    if not progress.exists():
        raise FileNotFoundError(f"No progress.csv found for case {case_id}")
    df = pd.read_csv(progress)
    y_col = "mean_reward" if "mean_reward" in df.columns else ("rollout/ep_rew_mean" if "rollout/ep_rew_mean" in df.columns else None)
    if y_col is None:
        raise KeyError(f"Unsupported progress schema in {progress}")
    return float(df[y_col].dropna().iloc[-1])


def load_episode_rewards(case_id: str, registry: StudyRegistry, selector: str = "first") -> tuple[pd.DataFrame, object]:
    export_dir = _case_export_dir(case_id, registry)
    episode_ref = resolve_episode(export_dir, selector)
    if episode_ref is None:
        raise FileNotFoundError(f"No episode found for case {case_id} with selector {selector}")
    rewards_path = episode_ref.source_path / "monitor" / "rewards.csv"
    df = pd.read_csv(rewards_path)
    if "reward" not in df.columns:
        raise KeyError(f"reward column missing in {rewards_path}")
    rewards = df[["reward"]].copy()
    rewards.insert(0, "step", range(1, len(rewards) + 1))
    rewards["cumulative_reward"] = rewards["reward"].cumsum()
    return rewards, episode_ref


def plot_cumulative_reward(figure_id: str, registry: StudyRegistry):
    spec = get_figure(figure_id, registry)
    style = _style(spec, registry)
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [10, 6])))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map: dict[str, str] = {}
    color_index = 0
    for run in spec.runs:
        rewards, episode_ref = load_episode_rewards(run.case_id, registry, selector=run.selector or "first")
        color, color_index = _resolve_group_color(run.color_group, color_map, color_cycle, color_index)
        plot_kwargs = {"label": _selector_label(run.label, episode_ref)}
        if run.line_style:
            plot_kwargs["linestyle"] = run.line_style
        if color is not None:
            plot_kwargs["color"] = color
        ax.plot(rewards["step"], rewards["cumulative_reward"], **plot_kwargs)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.grid(True, alpha=style.get("grid_alpha", 0.3))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_summary_bar(figure_id: str, registry: StudyRegistry):
    spec = get_figure(figure_id, registry)
    style = _style(spec, registry)
    labels = []
    values = []
    for run in spec.runs:
        if run.selector:
            rewards, episode_ref = load_episode_rewards(run.case_id, registry, selector=run.selector)
            labels.append(_selector_label(run.label, episode_ref))
            values.append(float(rewards["cumulative_reward"].iloc[-1]))
        else:
            labels.append(run.label)
            values.append(_load_progress_metric(run.case_id, registry))
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [10, 6])))
    ax.bar(labels, values)
    ax.set_title("Summary Comparison")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=style.get("grid_alpha", 0.3))
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    return fig


def plot_comparison_table(figure_id: str, registry: StudyRegistry):
    spec = get_figure(figure_id, registry)
    rows = []
    for run in spec.runs:
        if run.selector:
            rewards, episode_ref = load_episode_rewards(run.case_id, registry, selector=run.selector)
            rows.append(
                {
                    "Label": _selector_label(run.label, episode_ref),
                    "Metric": "final_cumulative_reward",
                    "Value": float(rewards["cumulative_reward"].iloc[-1]),
                }
            )
        else:
            rows.append(
                {
                    "Label": run.label,
                    "Metric": "final_mean_reward",
                    "Value": _load_progress_metric(run.case_id, registry),
                }
            )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, len(frame) * 2), 2 + len(frame) * 0.4))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=list(frame.columns),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    fig.tight_layout()
    return fig


def plot_figure(figure_id: str, registry: StudyRegistry):
    spec = get_figure(figure_id, registry)
    if spec.type == "learning_curves":
        return plot_learning_curves(figure_id, registry)
    if spec.type == "cumulative_reward":
        return plot_cumulative_reward(figure_id, registry)
    if spec.type == "summary_bar":
        return plot_summary_bar(figure_id, registry)
    if spec.type == "comparison_table":
        return plot_comparison_table(figure_id, registry)
    raise ValueError(f"Unsupported figure type: {spec.type}")


def save_figure(figure, figure_id: str, registry: StudyRegistry, output: str | Path | None = None, fmt: str | None = None) -> Path:
    fmt = fmt or registry.defaults.get("plotting", {}).get("default_format", "png")
    target = Path(output) if output else registry.paths["figure_root"] / f"{figure_id}.{fmt}"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    return target
