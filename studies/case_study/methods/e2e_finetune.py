from __future__ import annotations

from studies.case_study.core.weight_transfer import apply_e2e_finetune_initialization
from studies.case_study.methods.base import MethodExecution
from studies.case_study.methods._norm_utils import seed_normalization_from_source


class E2EFinetuneRunner:
    method_id = "e2e_finetune"

    def execute(self, case, env, services):
        source_bundle = services.load_source_bundle(case)
        source_model = services.load_model(source_bundle.model_path)
        target_model = services.build_training_model(case, env)

        new_state, report = apply_e2e_finetune_initialization(
            source_model.policy.state_dict(),
            target_model.policy.state_dict(),
        )
        target_model.policy.load_state_dict(new_state, strict=False)

        # Seed normalization from source (keep automatic_update=True for adaptation)
        seed_normalization_from_source(env, source_bundle)

        total_timesteps = services.compute_total_timesteps(case, env)
        target_model.learn(total_timesteps=total_timesteps, **services.build_learn_kwargs(case))
        return MethodExecution(
            model=target_model,
            total_timesteps=total_timesteps,
            source_bundle=source_bundle,
            metadata={
                "source_model": str(source_bundle.model_path),
                "transfer_init": {
                    "policy": "hidden_core_only",
                    **report.to_dict(),
                },
            },
        )


RUNNER = E2EFinetuneRunner()
