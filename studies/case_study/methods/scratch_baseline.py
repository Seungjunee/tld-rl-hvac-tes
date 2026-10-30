from __future__ import annotations

from studies.case_study.methods.base import MethodExecution


class ScratchBaselineRunner:
    method_id = "scratch_baseline"

    def execute(self, case, env, services):
        model = services.build_training_model(case, env)
        total_timesteps = services.compute_total_timesteps(case, env)
        model.learn(total_timesteps=total_timesteps, **services.build_learn_kwargs(case))
        return MethodExecution(model=model, total_timesteps=total_timesteps)


RUNNER = ScratchBaselineRunner()
