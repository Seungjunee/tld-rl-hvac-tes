from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

from studies.case_study.core import (
    case_has_manifest,
    cleanup_case_root,
    export_existing_case,
    list_case_ids,
    load_registry,
    resolve_case,
    resolve_case_id,
    run_case,
)
from studies.case_study.core.types import CleanupPolicy


# The paper reports 5-seed means; the analysis scripts
# (make_paper_figures.py / extract_paper_tables.py) read exactly these 5.
DEFAULT_SEEDS = [42, 123, 777, 2024, 2025]


def _apply_seed(case, seed: int):
    cased = deepcopy(case)
    training = dict(cased.training)
    model_kwargs = dict(training.get("model_kwargs", {}))
    model_kwargs["seed"] = int(seed)
    training["model_kwargs"] = model_kwargs
    cased.training = training
    export = dict(cased.export)
    export["destination"] = str(Path(export["destination"]) / f"seed_{seed}")
    cased.export = export
    cased.metadata = {**cased.metadata, "seed": int(seed)}
    return cased


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper case study experiments.")
    parser.add_argument("--config-root", default="studies/case_study/config", help="Path to the case-study config directory")
    parser.add_argument("--case", dest="case_id", help="Canonical case id to run")
    parser.add_argument("--group", help="Run a predefined case group")
    parser.add_argument("--all", action="store_true", help="Run all cases in registry order")
    parser.add_argument("--list", action="store_true", help="List canonical case ids")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue batch execution after a case failure")
    parser.add_argument("--fail-fast", action="store_true", help="Stop batch execution immediately on first failure")
    parser.add_argument("--resume", action="store_true", help="Skip execution if a manifest already exists for the case")
    parser.add_argument("--export-only", action="store_true", help="Reserved for future use")
    parser.add_argument("--cleanup-only", action="store_true", help="Reserved for future use")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved case ids without executing")
    parser.add_argument(
        "--export-root",
        default=None,
        help="Override export_root (relative to project root). Use to run into an "
             "isolated era (e.g. exports_eval) without overwriting existing exports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a specific seed. Exports to <case>/seed_<N>/. Legacy behavior if omitted.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Run multiple seeds sequentially (e.g. --seeds 42 123 777). "
             f"Default if omitted: legacy single run. Standard set: {DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--default-seeds",
        action="store_true",
        help=f"Shortcut for --seeds {' '.join(str(s) for s in DEFAULT_SEEDS)}",
    )
    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _print_result(result) -> None:
    print(f"[OK] {result.case_id}")
    print(f"  workspace: {result.workspace_path}")
    if result.model_path:
        print(f"  model: {result.model_path}")
    if result.export_report is not None:
        print(f"  exported: {result.export_report.destination}")
        if result.export_report.manifest_path:
            print(f"  manifest: {result.export_report.manifest_path}")


def _print_cleanup(case_id: str, report) -> None:
    print(f"[OK] {case_id}")
    print(f"  kept: {len(report.kept_paths)}")
    print(f"  removed: {len(report.removed_paths)}")


def _existing_manifest(case_id: str, registry) -> bool:
    return case_has_manifest(case_id, registry, status="success")


def _target_case_ids(args: argparse.Namespace, registry) -> list[str]:
    if args.all:
        return list_case_ids(registry)
    if args.group:
        if args.group not in registry.groups:
            raise SystemExit(f"Unknown group id: {args.group}")
        return [resolve_case_id(case_id, registry) for case_id in registry.groups[args.group]]
    if args.case_id:
        return [resolve_case_id(args.case_id, registry)]
    raise SystemExit("Provide --case <case_id>, or use --group/--all/--list.")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    registry = load_registry(args.config_root)

    if args.export_root:
        root = Path(args.export_root)
        if not root.is_absolute():
            root = registry.paths["project_root"] / root
        registry.paths["export_root"] = root
        print(f"[export-root override] {root}")

    if args.list:
        for case_id in list_case_ids(registry):
            print(case_id)
        return 0

    if args.export_only and args.cleanup_only:
        raise SystemExit("Use only one of --export-only or --cleanup-only.")

    target_case_ids = _target_case_ids(args, registry) if (args.all or args.group or args.case_id) else []
    if args.dry_run:
        for case_id in target_case_ids:
            print(case_id)
        return 0

    continue_on_error = args.continue_on_error or not args.fail_fast
    failures = 0

    seeds = args.seeds
    if args.default_seeds:
        seeds = DEFAULT_SEEDS
    if seeds is None and args.seed is not None:
        seeds = [args.seed]

    for case_id in target_case_ids:
        if args.resume and not (args.export_only or args.cleanup_only) and _existing_manifest(case_id, registry):
            print(f"[SKIP] {case_id}: existing success manifest found")
            continue

        case = resolve_case(case_id, registry)
        run_cases = [_apply_seed(case, s) for s in seeds] if seeds else [case]
        for run_case_obj in run_cases:
            tag = f" [seed={run_case_obj.metadata.get('seed')}]" if seeds else ""
            print(f"[START] {run_case_obj.id}{tag}")
            try:
                if args.export_only:
                    result = export_existing_case(run_case_obj, registry)
                    _print_result(result)
                elif args.cleanup_only:
                    policy = CleanupPolicy(
                        retain_success=int(run_case_obj.cleanup.get("retain_success", 3)),
                        retain_failure=int(run_case_obj.cleanup.get("retain_failure", 3)),
                    )
                    report = cleanup_case_root(
                        run_case_obj.id,
                        registry.paths["workspace_root"] / run_case_obj.id,
                        registry.paths["manifest_root"] / run_case_obj.id,
                        policy,
                    )
                    _print_cleanup(run_case_obj.id, report)
                else:
                    result = run_case(run_case_obj, registry)
                    _print_result(result)
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {run_case_obj.id}{tag}: {exc}", file=sys.stderr)
                if not continue_on_error:
                    raise

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
