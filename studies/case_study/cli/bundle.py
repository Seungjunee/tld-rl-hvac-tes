from __future__ import annotations

import argparse

from studies.case_study.core import generate_transfer_cumulative_bundle, load_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate case-study figure bundles.")
    parser.add_argument("--config-root", default="studies/case_study/config", help="Path to the case-study config directory")
    parser.add_argument(
        "--bundle",
        default="transfer_cumulative",
        choices=["transfer_cumulative"],
        help="Bundle id to render",
    )
    parser.add_argument("--output-root", help="Optional output directory override")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = load_registry(args.config_root)
    if args.bundle != "transfer_cumulative":
        raise ValueError(f"Unsupported bundle: {args.bundle}")
    report = generate_transfer_cumulative_bundle(registry, bundle_root=args.output_root)
    print(report.bundle_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
