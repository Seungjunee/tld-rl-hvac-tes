from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from .plotting import load_episode_rewards, plot_figure
from .registry import get_figure
from .types import StudyRegistry


TRANSFER_CUMULATIVE_COMPARISON_FIGURE_IDS: list[str] = []

TRANSFER_CUMULATIVE_FIGURE_IDS = list(TRANSFER_CUMULATIVE_COMPARISON_FIGURE_IDS)


@dataclass
class FigureBundleReport:
    bundle_root: Path
    manifest_path: Path
    summary_csv_path: Path
    summary_md_path: Path
    saved_files: list[Path]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _default_transfer_bundle_root(registry: StudyRegistry) -> Path:
    return registry.paths["figure_root"] / f"transfer_cumulative_bundle_{_utc_stamp()}"


def generate_transfer_cumulative_bundle(
    registry: StudyRegistry,
    bundle_root: str | Path | None = None,
    figure_ids: Iterable[str] | None = None,
) -> FigureBundleReport:
    root = Path(bundle_root) if bundle_root else _default_transfer_bundle_root(registry)
    root.mkdir(parents=True, exist_ok=True)
    active_ids = list(figure_ids or TRANSFER_CUMULATIVE_FIGURE_IDS)
    summary_rows: list[dict] = []
    saved_files: list[Path] = []
    manifest_rows: list[dict] = []

    for figure_id in active_ids:
        figure = plot_figure(figure_id, registry)
        png_path = root / f"{figure_id}.png"
        try:
            figure.savefig(png_path)
            saved_files.append(png_path)
            line_count = sum(len(ax.lines) for ax in figure.axes)
            spec = get_figure(figure_id, registry)
            category = "comparison"
            manifest_rows.append(
                {
                    "figure_id": figure_id,
                    "category": category,
                    "png": str(png_path),
                    "line_count": line_count,
                }
            )
            for run in spec.runs:
                rewards, episode_ref = load_episode_rewards(run.case_id, registry, selector=run.selector or "first")
                summary_rows.append(
                    {
                        "figure_id": figure_id,
                        "category": category,
                        "label": run.label,
                        "case_id": run.case_id,
                        "selector": run.selector or "first",
                        "line_style": run.line_style or "solid",
                        "color_group": run.color_group or "",
                        "metric": "final_cumulative_reward",
                        "value": float(rewards["cumulative_reward"].iloc[-1]),
                        "resolved_episode": episode_ref.resolved_episode_number,
                        "fallback_used": episode_ref.fallback_used,
                        "line_count": line_count,
                        "bundle": root.name,
                    }
                )
        finally:
            plt.close(figure)

    summary_csv_path = root / "figure_summary.csv"
    with summary_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md_path = root / "figure_summary.md"
    with summary_md_path.open("w") as handle:
        handle.write("# Transfer Cumulative Reward Summary\n\n")
        handle.write("## Comparison Figures\n\n")
        handle.write("| figure_id | label | case_id | selector | line_style | color_group | metric | value | resolved_episode | fallback_used |\n")
        handle.write("|---|---|---|---|---|---|---|---:|---:|---|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['figure_id']} | {row['label']} | {row['case_id']} | {row['selector']} | "
                f"{row['line_style']} | {row['color_group']} | {row['metric']} | {row['value']} | "
                f"{row['resolved_episode']} | {row['fallback_used']} |\n"
            )
        handle.write("\n")

    manifest_path = root / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle": root.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "comparison_figure_ids": TRANSFER_CUMULATIVE_COMPARISON_FIGURE_IDS,
                "figures": manifest_rows,
            },
            indent=2,
        )
    )

    return FigureBundleReport(
        bundle_root=root,
        manifest_path=manifest_path,
        summary_csv_path=summary_csv_path,
        summary_md_path=summary_md_path,
        saved_files=saved_files,
    )
