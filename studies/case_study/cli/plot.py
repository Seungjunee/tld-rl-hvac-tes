from __future__ import annotations

import argparse

from studies.case_study.core import get_figure, load_registry, plot_figure, save_figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper case-study figures.")
    parser.add_argument("--config-root", default="studies/case_study/config", help="Path to the case-study config directory")
    parser.add_argument("--figure", required=True, help="Figure id to render")
    parser.add_argument("--output", help="Optional output path")
    parser.add_argument("--format", dest="fmt", help="Optional figure format override")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = load_registry(args.config_root)
    get_figure(args.figure, registry)
    figure = plot_figure(args.figure, registry)
    saved = save_figure(figure, args.figure, registry, output=args.output, fmt=args.fmt)
    print(saved)
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
