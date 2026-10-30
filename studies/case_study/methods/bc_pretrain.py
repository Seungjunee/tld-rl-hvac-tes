"""Behavioural-cloning pretrain method.

Produces an SAC bundle whose actor's mean head has been supervised-trained
on (state, action) pairs collected from RBC1 running on the same source
building env. The bundle is then consumed by `padding_transfer` (or
`tldrl_ft` if you want a BC-initialised TLD-RL variant) on a target
building.

This is the source-side step of Kadamala et al.'s DP40d baseline, adapted
to our SAC continuous setup. Critic networks are left at their default SAC
initialisation — paper convention is to RL-train the critic from fresh.
"""

from __future__ import annotations

from studies.case_study.core.bc_trainer import (
    BCConfig,
    collect_rbc_demos,
    train_bc_continuous_actor,
)
from studies.case_study.methods.base import MethodExecution


class BCPretrainRunner:
    method_id = "bc_pretrain"

    def execute(self, case, env, services):
        # Build a fresh SAC model on source env (only used to host the actor
        # being BC-trained — no RL learn() call is made by this runner).
        model = services.build_training_model(case, env)

        # Pull RBC1 schedule + tank range from case
        price_schedule = case.reward.get("kwargs", {}).get("price_schedule")
        if price_schedule is None or len(price_schedule) != 24:
            raise ValueError("bc_pretrain requires a 24-element price_schedule in reward kwargs")
        physical = case.building.physical or {}
        tes_range = physical.get("tes_setpoint_range", [5.0, 10.0])

        # Demo collection
        bc_cfg_dict = case.training.get("bc", {}) or {}
        n_episodes = int(bc_cfg_dict.get("n_episodes", 3))
        epochs = int(bc_cfg_dict.get("epochs", 10))
        batch_size = int(bc_cfg_dict.get("batch_size", 256))
        bc_lr = float(bc_cfg_dict.get("learning_rate", 3.0e-4))
        val_fraction = float(bc_cfg_dict.get("val_fraction", 0.2))

        states, actions = collect_rbc_demos(
            env,
            price_schedule=price_schedule,
            tes_range=(float(tes_range[0]), float(tes_range[1])),
            n_episodes=n_episodes,
        )

        bc_cfg = BCConfig(
            n_episodes=n_episodes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=bc_lr,
            val_fraction=val_fraction,
            device="cpu",
        )
        history = train_bc_continuous_actor(model, states, actions, bc_cfg)

        # Total timesteps fed downstream so runner.save_model logic can be
        # consistent with other pretrain methods. RBC rollout steps count.
        total_timesteps = int(states.shape[0])

        return MethodExecution(
            model=model,
            total_timesteps=total_timesteps,
            metadata={
                "method": "bc_pretrain",
                "n_demos": int(states.shape[0]),
                "n_episodes": n_episodes,
                "bc_epochs": epochs,
                "bc_batch_size": batch_size,
                "bc_learning_rate": bc_lr,
                "bc_val_fraction": val_fraction,
                "final_train_mse": history["train_mse"][-1] if history["train_mse"] else None,
                "final_val_mse": history["val_mse"][-1] if history["val_mse"] else None,
            },
        )


RUNNER = BCPretrainRunner()
