"""Unit tests for the dormant e2e-finetune weight-transfer machinery.

Current contract of ``apply_e2e_finetune_initialization`` (kept for the
hidden-core / padding baselines; no active paper case uses it):

- ``mlp_extractor.*`` tensors are copied whenever source/target shapes match,
  INCLUDING first-layer tensors (with zone padding the input dims align, which
  is what enables full-extractor transfer).
- A first-layer tensor whose shape mismatches (raw cross-building input dims)
  is left at the target's fresh initialization (``skipped_input_reset``).
- Output heads (``action_net.*``, ``value_net.*``, ``log_std``) are never
  copied (``skipped_output_reset``).
- Any other shape mismatch is recorded in ``skipped_shape_mismatch``.
"""

from pathlib import Path

import torch

from studies.case_study.core.types import ModelBundle
from studies.case_study.core.weight_transfer import apply_e2e_finetune_initialization
from studies.case_study.methods.e2e_finetune import RUNNER


def _state_dict(
    *,
    first_input_dim: int,
    hidden_dim: int = 64,
    action_dim: int = 1,
    offset: float = 0.0,
):
    fill = lambda shape, value: torch.full(shape, value + offset)
    return {
        "mlp_extractor.policy_net.0.weight": fill((hidden_dim, first_input_dim), 1.0),
        "mlp_extractor.policy_net.0.bias": fill((hidden_dim,), 2.0),
        "mlp_extractor.policy_net.2.weight": fill((hidden_dim, hidden_dim), 3.0),
        "mlp_extractor.policy_net.2.bias": fill((hidden_dim,), 4.0),
        "mlp_extractor.value_net.0.weight": fill((hidden_dim, first_input_dim), 5.0),
        "mlp_extractor.value_net.0.bias": fill((hidden_dim,), 6.0),
        "mlp_extractor.value_net.2.weight": fill((hidden_dim, hidden_dim), 7.0),
        "mlp_extractor.value_net.2.bias": fill((hidden_dim,), 8.0),
        "action_net.weight": fill((action_dim, hidden_dim), 9.0),
        "action_net.bias": fill((action_dim,), 10.0),
        "value_net.weight": fill((1, hidden_dim), 11.0),
        "value_net.bias": fill((1,), 12.0),
        "log_std": fill((action_dim,), 13.0),
    }


OUTPUT_HEAD_KEYS = {
    "action_net.weight",
    "action_net.bias",
    "value_net.weight",
    "value_net.bias",
    "log_std",
}
EXTRACTOR_KEYS = {
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight",
    "mlp_extractor.policy_net.2.bias",
    "mlp_extractor.value_net.0.weight",
    "mlp_extractor.value_net.0.bias",
    "mlp_extractor.value_net.2.weight",
    "mlp_extractor.value_net.2.bias",
}


def test_matching_shapes_copy_full_extractor_and_keep_target_heads():
    """Padding-aligned dims: the whole extractor (input layer included) transfers."""
    source = _state_dict(first_input_dim=18, offset=10.0)
    target = _state_dict(first_input_dim=18, offset=100.0)

    initialized, report = apply_e2e_finetune_initialization(source, target)

    for key in EXTRACTOR_KEYS:
        assert torch.equal(initialized[key], source[key]), key
    for key in OUTPUT_HEAD_KEYS:
        assert torch.equal(initialized[key], target[key]), key

    assert set(report.copied_keys) == EXTRACTOR_KEYS
    assert report.copied_key_count == 8
    assert report.skipped_input_reset == []
    assert set(report.skipped_output_reset) == OUTPUT_HEAD_KEYS


def test_input_dim_mismatch_resets_first_layer_weights_only():
    """Raw cross-building dims: only the mismatched first-layer weights reset.

    First-layer biases have building-independent shape, so they transfer along
    with the hidden layers.
    """
    source = _state_dict(first_input_dim=18, offset=10.0)
    target = _state_dict(first_input_dim=32, offset=100.0)

    initialized, report = apply_e2e_finetune_initialization(source, target)

    for key in (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    ):
        assert torch.equal(initialized[key], target[key]), key
    for key in (
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.value_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "mlp_extractor.value_net.2.weight",
        "mlp_extractor.value_net.2.bias",
    ):
        assert torch.equal(initialized[key], source[key]), key

    assert set(report.skipped_input_reset) == {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    assert set(report.skipped_output_reset) == OUTPUT_HEAD_KEYS
    assert report.copied_key_count == 6


def test_apply_e2e_finetune_initialization_skips_shape_mismatch():
    source = _state_dict(first_input_dim=18, hidden_dim=32, offset=10.0)
    target = _state_dict(first_input_dim=32, hidden_dim=64, offset=100.0)

    initialized, report = apply_e2e_finetune_initialization(source, target)

    assert torch.equal(
        initialized["mlp_extractor.policy_net.2.weight"],
        target["mlp_extractor.policy_net.2.weight"],
    )
    assert torch.equal(
        initialized["mlp_extractor.value_net.2.bias"],
        target["mlp_extractor.value_net.2.bias"],
    )
    assert "mlp_extractor.policy_net.2.weight" in report.skipped_shape_mismatch
    assert "mlp_extractor.value_net.2.bias" in report.skipped_shape_mismatch
    assert report.copied_key_count == 0


class _FakePolicy:
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self._state_dict = {key: value.clone() for key, value in state_dict.items()}
        self.loaded_state = None

    def state_dict(self):
        return {key: value.clone() for key, value in self._state_dict.items()}

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = {key: value.clone() for key, value in state_dict.items()}
        self._state_dict = self.loaded_state


class _FakeModel:
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self.policy = _FakePolicy(state_dict)
        self.learn_calls = []

    def learn(self, total_timesteps: int, **kwargs):
        self.learn_calls.append((total_timesteps, kwargs))
        return self


class _FakeServices:
    def __init__(self):
        self.source_bundle = ModelBundle(
            case_id="pretrain_office_smud_sacp_e2e",
            building_id="office_5zone",
            method_id="scratch_baseline",
            model_path=Path("/tmp/source_model.zip"),
        )
        self.source_model = _FakeModel(_state_dict(first_input_dim=18, offset=10.0))
        self.target_initial_state = _state_dict(first_input_dim=32, offset=100.0)
        self.target_model = _FakeModel(self.target_initial_state)

    def load_source_bundle(self, case):
        return self.source_bundle

    def load_model(self, model_path):
        assert model_path == self.source_bundle.model_path
        return self.source_model

    def build_training_model(self, case, env):
        return self.target_model

    def compute_total_timesteps(self, case, env):
        return 12

    def build_learn_kwargs(self, case):
        return {"log_interval": 7}


def test_e2e_runner_records_transfer_metadata():
    services = _FakeServices()

    execution = RUNNER.execute(case=None, env=None, services=services)

    assert execution.model is services.target_model
    assert execution.total_timesteps == 12
    assert execution.source_bundle == services.source_bundle
    assert services.target_model.learn_calls == [(12, {"log_interval": 7})]

    loaded = services.target_model.policy.loaded_state
    assert loaded is not None
    # hidden layers and shape-matched first-layer bias come from the source
    assert torch.equal(
        loaded["mlp_extractor.policy_net.2.weight"],
        services.source_model.policy.state_dict()["mlp_extractor.policy_net.2.weight"],
    )
    assert torch.equal(
        loaded["mlp_extractor.policy_net.0.bias"],
        services.source_model.policy.state_dict()["mlp_extractor.policy_net.0.bias"],
    )
    # mismatched first-layer weights and the output heads stay at target init
    assert torch.equal(
        loaded["mlp_extractor.policy_net.0.weight"],
        services.target_initial_state["mlp_extractor.policy_net.0.weight"],
    )
    assert torch.equal(
        loaded["action_net.weight"],
        services.target_initial_state["action_net.weight"],
    )
    assert execution.metadata["source_model"] == str(services.source_bundle.model_path)
    assert execution.metadata["transfer_init"]["policy"] == "hidden_core_only"
    assert execution.metadata["transfer_init"]["copied_key_count"] == 6
