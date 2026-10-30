"""Registry guards for the paper reproduction pipeline.

Every number reported in the paper (the four-building tables and figures,
the further-building matrix, and the PPO and comfort-weighted arms) is produced
by the case ids asserted here. These tests fail loudly if a referenced case is
deleted, renamed, or silently switched to a different protocol - the failure
mode that rotted the deprecated PPO-era groups.

Deliberately NOT pinned: the reduced-state dimension (a flexible design
quantity - only its cross-building invariance is guarded).
"""

import shutil
from pathlib import Path

import pytest

from studies.case_study.core import (
    get_group_case_ids,
    list_case_ids,
    load_registry,
    resolve_case,
    resolve_case_id,
)

CONFIG_ROOT = "studies/case_study/config"

# Paper matrix: 4 buildings (Table 1), 12 directed transfer pairs.
PAPER_BUILDINGS = {
    "office": "office_5zone",
    "lbuilding": "office_lbuilding_8zone",
    "warehouse": "warehouse_3zone",
    "mediumoffice": "mediumoffice_15zone",
}
PAPER_PAIRS = [(s, t) for s in PAPER_BUILDINGS for t in PAPER_BUILDINGS if s != t]

# Further commercial buildings: zero-shot targets of the four sources above
# (16 directed pairs; each target predictor is refit on the target's RBC log).
EXTENSION_BUILDINGS = {
    "restaurant": "restaurant_3zone",
    "stripmall": "stripmall_10zone",
    "retail": "retail_4zone",
    "fastfood": "fastfood_2zone",
}
EXTENSION_PAIRS = [(s, t) for s in PAPER_BUILDINGS for t in EXTENSION_BUILDINGS]

LEGACY_TOKENS = (
    "feature_continue",
    "duplicate_padding",
    "e2e_finetune",
    "native_policy",
    "feature_policy",
)


def paper_matrix_case_ids() -> list[str]:
    """The 44 case ids behind every reported number of the 4-building paper."""
    ids = [f"build_{b}_observation_dataset" for b in PAPER_BUILDINGS]
    ids += [f"rbc_{b}_smud_tou" for b in PAPER_BUILDINGS]
    ids += [f"pretrain_{b}_smud_sac_plus" for b in PAPER_BUILDINGS]
    ids += [f"pretrain_{b}_smud_sac_e2e" for b in PAPER_BUILDINGS]
    ids += [f"pretrain_{b}_smud_sacp_e2e" for b in PAPER_BUILDINGS]
    ids += [f"transfer_{s}_to_{t}_smud_sacp_zs" for s, t in PAPER_PAIRS]
    ids += [f"transfer_{s}_to_{t}_smud_sacp_padrl" for s, t in PAPER_PAIRS]
    return ids


def test_registry_loads_canonical_cases_only():
    registry = load_registry(CONFIG_ROOT)

    case_ids = list_case_ids(registry)
    allowed_prefixes = ("build_", "rbc_", "pretrain_", "transfer_")
    for cid in case_ids:
        assert cid.startswith(allowed_prefixes), cid
        for token in LEGACY_TOKENS:
            assert token not in cid, (cid, token)

    assert (
        resolve_case_id("pretrain_office_smud_sac_plus", registry)
        == "pretrain_office_smud_sac_plus"
    )
    with pytest.raises(KeyError):
        resolve_case_id("legacy_case_id", registry)
    with pytest.raises(KeyError):
        resolve_case_id("pretrain_restaurant_feature_policy", registry)


def test_registry_discovers_project_root_from_nested_config_copy():
    project_root = Path.cwd()
    temp_root = project_root / ".tmp_case_study_registry_test"
    config_copy = temp_root / "config"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    shutil.copytree(project_root / "studies/case_study/config", config_copy)
    try:
        registry = load_registry(config_copy)
        assert registry.paths["project_root"] == project_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_paper_matrix_is_complete_and_resolvable():
    """Existence + full config merge for all 44 paper-matrix cases."""
    registry = load_registry(CONFIG_ROOT)
    case_ids = set(list_case_ids(registry))

    expected_runner = {
        "build_": "scratch_baseline",
        "rbc_": "rbc_tou",
        "pretrain_": "scratch_baseline",
    }
    for cid in paper_matrix_case_ids():
        assert cid in case_ids, f"paper case missing from registry: {cid}"
        case = resolve_case(cid, registry)
        assert case.export["destination"].endswith(
            f"artifacts/case_study/exports/{cid}"
        ), cid

        if cid.startswith("transfer_"):
            runner = "zero_shot" if cid.endswith("_zs") else "padding_transfer"
            target_tag = cid.split("_to_")[1].split("_smud")[0]
            assert case.building.id == PAPER_BUILDINGS[target_tag], cid
        else:
            runner = next(v for k, v in expected_runner.items() if cid.startswith(k))
            building_tag = cid.split("_")[1]
            assert case.building.id == PAPER_BUILDINGS[building_tag], cid
        assert case.method.runner == runner, (cid, case.method.runner)


def test_paper_matrix_protocol_pins():
    """Pin the evaluation protocol the paper describes in Section 4.

    - TLD-RL zero-shot is deployed frozen: a single deterministic evaluation
      season, sourced from the reduced-state pretrain (`_sac_plus`).
    - Padding fine-tunes for 10 target seasons from the padded-state source
      (`_sacp_e2e`) - reverting this to fewer seasons silently breaks Fig. 6
      (adoption) and Table 4.
    - Source policies and E2E-from-scratch train for 10 seasons.
    """
    registry = load_registry(CONFIG_ROOT)

    for s, t in PAPER_PAIRS:
        zs = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_zs", registry)
        assert zs.training["algorithm"] == "sac"
        assert zs.training["episodes"] == 1
        assert zs.inputs["source_case_id"] == f"pretrain_{s}_smud_sac_plus"
        # Stage-3 target predictors are refit on the incumbent-RBC operation
        # log (2026-07-13 protocol promotion); the case-level wrapper override
        # must survive the profile merge with the horizon intact.
        zs_lp = zs.wrappers["load_predictor"]
        assert zs_lp["model_path"].endswith("_lstm_rbclog.pt"), (s, t, zs_lp)
        assert zs_lp["horizon"] == 12, (s, t, zs_lp)

        padrl = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_padrl", registry)
        assert padrl.training["algorithm"] == "sac"
        assert padrl.training["episodes"] == 10
        assert padrl.inputs["source_case_id"] == f"pretrain_{s}_smud_sacp_e2e"

    for b in PAPER_BUILDINGS:
        for kind in ("sac_plus", "sac_e2e", "sacp_e2e"):
            case = resolve_case(f"pretrain_{b}_smud_{kind}", registry)
            assert case.training["algorithm"] == "sac", (b, kind)
            assert case.training["episodes"] == 10, (b, kind)
        dataset = resolve_case(f"build_{b}_observation_dataset", registry)
        assert dataset.training["episodes"] == 1, b


def test_reduced_state_is_zone_invariant_across_buildings():
    """The TLD-RL policy input must be identical across buildings (gate G1).

    Identical names AND order: shapes can match while features are scrambled,
    which SB3 will not catch (the medium-office/datacenter obs-order bug
    class). The literal dimension is intentionally not asserted.
    """
    registry = load_registry(CONFIG_ROOT)

    keeps = {
        b: resolve_case(f"pretrain_{b}_smud_sac_plus", registry).wrappers[
            "reduce_observation"
        ]["keep"]
        for b in PAPER_BUILDINGS
    }
    reference = keeps["office"]
    for b, keep in keeps.items():
        assert keep == reference, f"reduced state differs on {b}"
    # The added buildings receive the frozen source policies, so every
    # zero-shot pair into them must feed the identical input.
    for s, t in EXTENSION_PAIRS:
        keep = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_zs", registry).wrappers[
            "reduce_observation"
        ]["keep"]
        assert keep == reference, f"reduced state differs on {s}->{t}"

    # V6' semantics: two-zone stratified tank statistics (mean/top/bottom).
    for var in ("CWtank_final", "CWtank_node1", "CWtank_use_outlet"):
        assert var in reference, var
    # Look-ahead horizon H=12 is a paper constant (Table 2).
    assert "load_forecast_12" in reference
    assert "load_forecast_13" not in reference
    # Comfort was removed from the supervisory state (2026-07-02); zone-level
    # channels must never enter the shared policy input.
    for var in reference:
        assert "comfort" not in var.lower(), var
        assert not var.startswith("Z"), f"zone-side channel leaked: {var}"


def test_paper_group_covers_matrix_and_is_sac_only():
    registry = load_registry(CONFIG_ROOT)

    group = get_group_case_ids("paper_4building", registry)
    assert set(paper_matrix_case_ids()) <= set(group)

    # 4-building matrix: 4 datasets + 4 RBC + 12 pretrains + 12 zs + 12 padrl.
    prefix_counts = {
        p: sum(c.startswith(p) for c in group)
        for p in ("build_", "rbc_", "pretrain_", "transfer_")
    }
    assert prefix_counts == {
        "build_": 4,
        "rbc_": 4,
        "pretrain_": 12,
        "transfer_": 24,
    }, prefix_counts

    for cid in group:
        case = resolve_case(cid, registry)
        algorithm = case.training.get("algorithm")
        if algorithm is not None:  # datasets and RBC carry no RL algorithm
            assert algorithm == "sac", (cid, algorithm)


def test_default_seeds_match_paper():
    """The paper reports five-seed means; the analysis scripts read exactly these."""
    from studies.case_study.cli.run import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == [42, 123, 777, 2024, 2025]


def extension_case_ids() -> list[str]:
    """The 28 case ids behind the further-building results (Table 7, Fig. 8)."""
    ids = [f"build_{b}_observation_dataset" for b in EXTENSION_BUILDINGS]
    ids += [f"build_{b}_observation_dataset_rbc" for b in EXTENSION_BUILDINGS]
    ids += [f"rbc_{b}_smud_tou" for b in EXTENSION_BUILDINGS]
    ids += [f"transfer_{s}_to_{t}_smud_sacp_zs" for s, t in EXTENSION_PAIRS]
    return ids


def test_extension_group_covers_further_buildings():
    """Zero-shot only: frozen four-building sources, target-refit predictors."""
    registry = load_registry(CONFIG_ROOT)

    group = get_group_case_ids("paper_extension", registry)
    assert sorted(group) == sorted(extension_case_ids())

    for s, t in EXTENSION_PAIRS:
        zs = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_zs", registry)
        assert zs.method.id == "tldrl_zs", (s, t)
        assert zs.building.id == EXTENSION_BUILDINGS[t], (s, t)
        assert zs.training["algorithm"] == "sac"
        assert zs.training["episodes"] == 1
        assert zs.inputs["source_case_id"] == f"pretrain_{s}_smud_sac_plus"
        lp = zs.wrappers["load_predictor"]
        assert lp["model_path"].endswith(
            f"{EXTENSION_BUILDINGS[t]}_lstm_rbclog.pt"
        ), (s, t, lp)
        assert lp["horizon"] == 12, (s, t, lp)
        # Shared action bounds: the frozen policy acts without rescaling.
        assert zs.building.physical["tes_setpoint_range"] == [5.0, 10.0], t


def test_ppo_comfort_and_frozen_feature_arms():
    """PPO replication, comfort-weighted reward and frozen-feature Padding."""
    registry = load_registry(CONFIG_ROOT)

    ppo = get_group_case_ids("r1_ppo", registry)
    assert len(ppo) == 8
    for cid in ppo:
        case = resolve_case(cid, registry)
        assert case.training["algorithm"] == "ppo", cid
        assert case.training["episodes"] == 10, cid
        # PPO acts in [-1, 1]; on the raw Box(5, 10) its actions would clip.
        assert case.wrappers["normalize_action"]["normalize_range"] == [-1.0, 1.0]

    for group, method in (
        ("r1_ppo_zs_tld", "tldrl_zs"),
        ("r1_ppo_zs_pad", "padding_zs"),
    ):
        ids = get_group_case_ids(group, registry)
        assert len(ids) == len(PAPER_PAIRS), group
        for cid in ids:
            case = resolve_case(cid, registry)
            assert case.method.id == method, cid
            assert case.training["episodes"] == 1, cid
            assert case.inputs["source_case_id"].endswith("__r1ppo"), cid

    comfort = get_group_case_ids("r1_comfort", registry)
    assert len(comfort) == 4 * len(PAPER_BUILDINGS)
    for cid in comfort:
        case = resolve_case(cid, registry)
        assert case.training["algorithm"] == "sac", cid
        assert case.reward["kwargs"]["comfort_form"] == "linear", cid
        assert case.reward["kwargs"]["lambda_temperature"] > 0, cid

    frozen = get_group_case_ids("r1_padff", registry)
    assert len(frozen) == len(PAPER_PAIRS)
    for cid in frozen:
        assert resolve_case(cid, registry).method.id == "padding_frozen", cid
