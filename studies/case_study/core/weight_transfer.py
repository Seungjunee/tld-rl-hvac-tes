from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch


@dataclass
class WeightTransferReport:
    copied_keys: list[str] = field(default_factory=list)
    skipped_input_reset: list[str] = field(default_factory=list)
    skipped_output_reset: list[str] = field(default_factory=list)
    skipped_shape_mismatch: list[str] = field(default_factory=list)
    skipped_missing_target: list[str] = field(default_factory=list)

    @property
    def copied_key_count(self) -> int:
        return len(self.copied_keys)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["copied_key_count"] = self.copied_key_count
        return data


def _linear_layer_indices(prefix: str, state_dict: dict[str, torch.Tensor]) -> list[int]:
    indices: set[int] = set()
    prefix_with_dot = f"{prefix}."
    for key in state_dict:
        if not key.startswith(prefix_with_dot):
            continue
        suffix = key[len(prefix_with_dot):]
        if "." not in suffix:
            continue
        idx_token = suffix.split(".", 1)[0]
        if idx_token.isdigit():
            indices.add(int(idx_token))
    return sorted(indices)


def _is_first_linear_key(key: str, prefix: str, first_idx: int | None) -> bool:
    if first_idx is None:
        return False
    return key.startswith(f"{prefix}.{first_idx}.")


def _is_output_head_key(key: str) -> bool:
    return key.startswith("action_net.") or key.startswith("value_net.") or key == "log_std"


def _first_layer_index_map(source_state_dict: dict[str, torch.Tensor], target_state_dict: dict[str, torch.Tensor]) -> dict[str, int | None]:
    prefixes = (
        "mlp_extractor.policy_net",
        "mlp_extractor.value_net",
        "mlp_extractor.shared_net",
    )
    first_indices: dict[str, int | None] = {}
    for prefix in prefixes:
        source_indices = _linear_layer_indices(prefix, source_state_dict)
        target_indices = _linear_layer_indices(prefix, target_state_dict)
        if source_indices and target_indices:
            first_indices[prefix] = min(source_indices[0], target_indices[0])
        else:
            first_indices[prefix] = None
    return first_indices


def apply_e2e_finetune_initialization(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], WeightTransferReport]:
    new_state_dict = {key: value.clone() for key, value in target_state_dict.items()}
    report = WeightTransferReport()
    first_layer_indices = _first_layer_index_map(source_state_dict, target_state_dict)

    for key, source_tensor in source_state_dict.items():
        if key not in target_state_dict:
            report.skipped_missing_target.append(key)
            continue

        if _is_output_head_key(key):
            report.skipped_output_reset.append(key)
            continue

        if not key.startswith("mlp_extractor."):
            continue

        target_tensor = target_state_dict[key]
        if source_tensor.shape != target_tensor.shape:
            # Input layer typically has shape mismatch across buildings with
            # different observation dims. With padding (ZonePaddingWrapper),
            # shapes align and this clause is skipped, enabling full transfer.
            if any(
                _is_first_linear_key(key, prefix, first_idx)
                for prefix, first_idx in first_layer_indices.items()
            ):
                report.skipped_input_reset.append(key)
            else:
                report.skipped_shape_mismatch.append(key)
            continue

        new_state_dict[key] = source_tensor.clone()
        report.copied_keys.append(key)

    return new_state_dict, report
