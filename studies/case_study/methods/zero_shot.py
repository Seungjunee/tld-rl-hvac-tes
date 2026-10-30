from __future__ import annotations

import numpy as np

from studies.case_study.core.policy_selection import predict_action
from studies.case_study.methods._norm_utils import seed_normalization_from_source
from studies.case_study.methods.base import MethodExecution


class ZeroShotRunner:
    """Load source policy and evaluate on target with frozen weights AND
    configurable action selection.

    The policy is fixed (no gradient updates). By default actions use the
    argmax/mean of the policy distribution, matching historical behavior. For
    stochastic categorical policies that encode rare but important TES charging
    actions, cases may set ``training.action_selection.mode:
    discrete_threshold`` to deploy a deterministic calibrated selector.

    Normalization adapts to target distribution (automatic_update=true).
    """

    method_id = "zero_shot"

    def execute(self, case, env, services):
        source_bundle = services.load_source_bundle(case)
        # Pass env to PPO.load so rollout buffers are rebuilt for the target's
        # n_envs (source may have been a VecEnv model with several envs).
        model = services.load_model(source_bundle.model_path, env=env)

        # Loaded models are not reseeded by SB3; only matters here for
        # stochastic action_selection modes (deterministic ZS is seed-free).
        seed = case.training.get("model_kwargs", {}).get("seed")
        if seed is not None:
            model.set_random_seed(int(seed))

        # Seed normalization from source (keep automatic_update=true)
        seed_normalization_from_source(env, source_bundle)

        total_timesteps = services.compute_total_timesteps(case, env)
        eval_cfg = case.training.get("evaluation", {})
        action_selection = case.training.get("action_selection")
        deterministic = bool(eval_cfg.get("deterministic", True))
        if isinstance(action_selection, dict) and "deterministic" in action_selection:
            deterministic = bool(action_selection["deterministic"])
        selection_mode = (
            str(action_selection.get("mode", "")).lower()
            if isinstance(action_selection, dict)
            else ""
        )
        deterministic_actions = deterministic and selection_mode not in {
            "stochastic",
            "sample",
        }

        # Manual evaluation rollout with configurable actions. CSVLogger /
        # LoggerWrapper hook into env.step, so progress.csv is written
        # per episode without invoking PPO.learn.
        steps_done = 0
        obs, _info = env.reset()
        while steps_done < total_timesteps:
            action, _state = predict_action(
                model,
                np.asarray(obs),
                deterministic=deterministic,
                action_selection=action_selection,
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps_done += 1
            if terminated or truncated:
                if steps_done < total_timesteps:
                    obs, _info = env.reset()

        return MethodExecution(
            model=model,
            total_timesteps=steps_done,
            source_bundle=source_bundle,
            metadata={
                "source_model": str(source_bundle.model_path),
                "transfer_type": "zero_shot",
                "deterministic_actions": deterministic_actions,
                "action_selection": action_selection or {"mode": "deterministic"},
                "learning_rate": 0.0,
            },
        )


RUNNER = ZeroShotRunner()
