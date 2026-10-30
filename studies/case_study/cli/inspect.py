from __future__ import annotations

import argparse
import json

from studies.case_study.core import get_case_spec, get_figure, list_case_ids, load_registry, resolve_case, resolve_case_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect case-study configuration.")
    parser.add_argument("--config-root", default="studies/case_study/config", help="Path to the case-study config directory")
    parser.add_argument("--case", dest="case_id", help="Canonical case id")
    parser.add_argument("--figure", help="Figure id")
    parser.add_argument("--list", action="store_true", help="List cases and groups")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = load_registry(args.config_root)
    if args.list:
        print(json.dumps({"cases": list_case_ids(registry), "groups": registry.groups, "figures": list(registry.figures)}, indent=2))
        return 0
    if args.case_id:
        case_id = resolve_case_id(args.case_id, registry)
        payload = {
            "spec": get_case_spec(case_id, registry).__dict__,
            "resolved": resolve_case(case_id, registry).__dict__,
        }
        print(json.dumps(payload, default=str, indent=2))
        return 0
    if args.figure:
        figure = get_figure(args.figure, registry)
        print(json.dumps(figure.__dict__, default=str, indent=2))
        return 0
    raise SystemExit("Provide --list, --case, or --figure.")


if __name__ == "__main__":
    raise SystemExit(main())
