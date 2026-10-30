from __future__ import annotations

import torch

from studies.case_study.methods._norm_utils import seed_normalization_from_source
from studies.case_study.methods.base import MethodExecution


def _noop_lr_update(*_args, **_kwargs):
    """Replaces BaseAlgorithm._update_learning_rate on loaded models.

    Module-level (not a bound lambda) so SAC.save() can pickle the instance
    attribute by reference — a MethodType-bound lambda drags the whole model
    (and its env thread locks) into the pickle and crashes model export.
    """
    return None


class ContinueFinetuneRunner:
    method_id = "continue_finetune"

    def execute(self, case, env, services):
        source_bundle = services.load_source_bundle(case)
        # Pass env to PPO/SAC.load so rollout/buffers are rebuilt for the
        # target's n_envs (source may have been a VecEnv model with several envs).
        model = services.load_model(source_bundle.model_path, env=env)

        # SB3's load() does NOT apply a seed — the constructor path (scratch)
        # consumes model_kwargs["seed"], but loaded models keep whatever global
        # RNG state exists, making "multi-seed" FT runs byte-identical
        # (discovered 2026-06-10: ftA ep1 identical across 3 seeds). Reseed
        # explicitly so seeded transfer runs are genuine replicates.
        seed = case.training.get("model_kwargs", {}).get("seed")
        if seed is not None:
            model.set_random_seed(int(seed))

        # Reset optimizer momentum so transferred weights are preserved but
        # Adam moments don't carry source-env reward-scale biases. PPO has a
        # single shared policy.optimizer; SAC has actor.optimizer +
        # critic.optimizer (+ ent_coef_optimizer when ent_coef=auto).
        #
        # Supports asymmetric LR for SAC: case.training.{actor_lr, critic_lr}
        # override `fine_tune_lr` when present. Defaults preserve historical
        # behaviour (all optimizers share `fine_tune_lr`).
        fine_tune_lr = float(
            case.training.get("model_kwargs", {}).get("learning_rate", 3e-4)
        )
        actor_lr = float(case.training.get("actor_lr", fine_tune_lr))
        critic_lr = float(case.training.get("critic_lr", fine_tune_lr))
        model.learning_rate = critic_lr  # shown in SAC's progress logs

        # Optional controls (only activate when present in case.training) so that
        # an FC run can be made HP-identical to padding_transfer — leaving the
        # transferred weights (source SAC vs BC+DP-padding) as the only
        # difference for a controlled comparison. Absent keys preserve the
        # historical FC behaviour (auto entropy, source learning_starts).
        ent_coef_fixed = case.training.get("ent_coef_fixed", None)
        learning_starts_override = case.training.get("learning_starts", None)
        if learning_starts_override is not None:
            model.learning_starts = int(learning_starts_override)

        if hasattr(model, "actor") and hasattr(model.actor, "optimizer"):
            # SAC-style (actor/critic separate)
            model.actor.optimizer = torch.optim.Adam(
                model.actor.parameters(), lr=actor_lr
            )
            if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                model.critic.optimizer = torch.optim.Adam(
                    model.critic.parameters(), lr=critic_lr
                )
            if ent_coef_fixed is not None:
                # Freeze entropy to a fixed coefficient (same mechanism as
                # padding_transfer). Without this, SAC's auto-entropy re-inflates
                # log_std on the loaded policy and washes out the warm-start.
                ent_tensor = torch.tensor(float(ent_coef_fixed), device=model.device)
                model.ent_coef_tensor = ent_tensor
                if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
                    with torch.no_grad():
                        model.log_ent_coef.data.fill_(
                            torch.log(ent_tensor.clamp_min(1e-8)).item()
                        )
                model.ent_coef_optimizer = None  # freeze entropy coefficient
            elif getattr(model, "ent_coef_optimizer", None) is not None:
                model.ent_coef_optimizer = torch.optim.Adam(
                    [model.log_ent_coef], lr=critic_lr
                )
        else:
            # PPO-style (shared policy.optimizer)
            model.policy.optimizer = torch.optim.Adam(
                model.policy.parameters(), lr=fine_tune_lr
            )

        # Reset the Gaussian policy's log_std (PPO continuous only — SAC's
        # log_std is state-dependent inside the actor, not a fixed parameter).
        if hasattr(model.policy, "log_std"):
            with torch.no_grad():
                model.policy.log_std.data.fill_(0.0)

        # SB3's train() calls self._update_learning_rate(optimizers) on EVERY
        # gradient step, which overwrites every optimizer's lr from
        # self.lr_schedule — rebuilt at load() time from the SOURCE model's
        # saved learning_rate. That silently discarded the per-case
        # actor/critic LRs configured above: all pre-2026-06-10 FT runs
        # effectively trained at the source's uniform 3e-4 regardless of
        # case HP. Neutralise the schedule override so the rebuilt Adam
        # learning rates actually hold.
        model._update_learning_rate = _noop_lr_update

        seed_normalization_from_source(env, source_bundle)

        total_timesteps = services.compute_total_timesteps(case, env)
        model.learn(
            total_timesteps=total_timesteps, **services.build_learn_kwargs(case)
        )
        return MethodExecution(
            model=model,
            total_timesteps=total_timesteps,
            source_bundle=source_bundle,
            metadata={
                "source_model": str(source_bundle.model_path),
                "fine_tune_learning_rate": fine_tune_lr,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "ent_coef_fixed": (
                    float(ent_coef_fixed) if ent_coef_fixed is not None else "auto"
                ),
                "learning_starts_override": learning_starts_override,
            },
        )


RUNNER = ContinueFinetuneRunner()
