"""Guard: the TLD-RL reduced supervisory state must be identical across buildings.

The proposed (TLD-RL) policy is trained on the reduced "sac_plus" observation and then
reused *frozen* on other buildings (zero-shot). For that to be valid, every building's
reduced observation vector must contain the SAME variables in the SAME order. Equal
shapes are not enough: a building whose base env lists its variables in a different order
(e.g. an occupancy/zone variable at a different index) silently scrambles the policy
input, and Stable-Baselines3 will not catch it because the dimension still matches. This
already bit the medium-office and the data center.

The check builds each building's ``pretrain_<b>_smud_sac_plus`` env (construction only --
no EnergyPlus run / no ``reset()``) and asserts the reduced ``observation_variables`` are
all equal. Buildings are discovered dynamically from the registry, so a newly added
building is covered automatically once its load predictor (the ``*_lstm.pt``) exists.

The assertion is intentionally dimension-agnostic: it requires the orderings to be
*identical to each other*, not equal to any fixed length, so the reduced state can still
be re-sized without editing this test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from studies.case_study.core import build_environment, load_registry, resolve_case
from studies.case_study.core.types import RunContext

CONFIG_ROOT = "studies/case_study/config"

# The TLD-RL source policy is the reduced/proposed "sac_plus" pretrain, one per building.
_SAC_PLUS_RE = re.compile(r"^pretrain_[a-z0-9]+_smud_sac_plus$")


def _sac_plus_case_ids(registry) -> list[str]:
    return sorted(cid for cid in registry.cases if _SAC_PLUS_RE.match(cid))


def _predictor_available(case) -> bool:
    """True unless the profile pins a load-predictor .pt that is not on disk yet."""
    lp = case.wrappers.get("load_predictor")
    if not lp or not lp.get("model_path"):
        return True
    mp = Path(lp["model_path"])
    return mp.exists() or (Path.cwd() / mp).exists()


def _reduced_observation_order(case, workspace: Path) -> list[str]:
    workspace.mkdir(parents=True, exist_ok=True)
    run_context = RunContext(
        case_id=case.id,
        run_id="obsorder",
        started_at="2026-01-01",
        workspace_root=workspace,
        export_root=workspace,
        figure_root=workspace,
        manifest_root=workspace,
        third_party_root=workspace,
        project_root=Path.cwd(),
    )
    env = build_environment(case, run_context, mode="train")  # builds the stack; no reset()
    try:
        return list(env.get_wrapper_attr("observation_variables"))
    finally:
        try:
            env.close()
        except Exception:
            pass


def test_tldrl_reduced_state_identical_across_buildings(tmp_path):
    registry = load_registry(CONFIG_ROOT)
    case_ids = _sac_plus_case_ids(registry)
    assert case_ids, "no pretrain_<b>_smud_sac_plus cases found in the registry"

    orders: dict[str, list[str]] = {}
    skipped: list[str] = []
    for cid in case_ids:
        case = resolve_case(cid, registry)
        if not _predictor_available(case):
            skipped.append(cid)  # predictor not trained yet -> cannot build; skip it
            continue
        orders[cid] = _reduced_observation_order(case, tmp_path / cid)

    if len(orders) < 2:
        pytest.skip(
            "need >=2 buildable buildings to compare reduced-state ordering; "
            f"built={list(orders)} skipped(no predictor)={skipped}"
        )

    ref_id, ref = next(iter(orders.items()))
    for cid, order in orders.items():
        if order == ref:
            continue
        mismatches = [
            f"  idx {i}: {ref_id}={ref[i] if i < len(ref) else '<none>'!r}  "
            f"{cid}={order[i] if i < len(order) else '<none>'!r}"
            for i in range(max(len(ref), len(order)))
            if (ref[i] if i < len(ref) else None) != (order[i] if i < len(order) else None)
        ]
        pytest.fail(
            f"Reduced TLD-RL observation ordering differs between {ref_id} and {cid} "
            f"(len {len(ref)} vs {len(order)}). A frozen TLD-RL policy will not transfer "
            "correctly across these buildings. First mismatches:\n"
            + "\n".join(mismatches[:10])
        )
