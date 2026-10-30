from __future__ import annotations

from studies.case_study.methods.bc_pretrain import RUNNER as BC_PRETRAIN_RUNNER
from studies.case_study.methods.bc_then_finetune_ppo import RUNNER as BC_THEN_FINETUNE_PPO_RUNNER
from studies.case_study.methods.continue_finetune import RUNNER as CONTINUE_RUNNER
from studies.case_study.methods.e2e_finetune import RUNNER as E2E_FINETUNE_RUNNER
from studies.case_study.methods.padding_transfer import RUNNER as PADDING_TRANSFER_RUNNER
from studies.case_study.methods.rule_based import RUNNER as RULE_BASED_TOU_RUNNER
from studies.case_study.methods.rule_based import RUNNER_V2 as RULE_BASED_TOU_V2_RUNNER
from studies.case_study.methods.scratch_baseline import RUNNER as SCRATCH_BASELINE_RUNNER
from studies.case_study.methods.zero_shot import RUNNER as ZERO_SHOT_RUNNER

_METHOD_RUNNERS = {
    "scratch_baseline": SCRATCH_BASELINE_RUNNER,
    "continue_finetune": CONTINUE_RUNNER,
    "e2e_finetune": E2E_FINETUNE_RUNNER,
    "rbc_tou": RULE_BASED_TOU_RUNNER,
    "rbc_tou_v2": RULE_BASED_TOU_V2_RUNNER,
    "zero_shot": ZERO_SHOT_RUNNER,
    "bc_pretrain": BC_PRETRAIN_RUNNER,
    "bc_then_finetune_ppo": BC_THEN_FINETUNE_PPO_RUNNER,
    "padding_transfer": PADDING_TRANSFER_RUNNER,
}


def get_method_runner(name: str):
    if name not in _METHOD_RUNNERS:
        raise KeyError(f"Unknown method runner: {name}")
    return _METHOD_RUNNERS[name]


def list_method_runners() -> list[str]:
    return sorted(_METHOD_RUNNERS)
