"""BC pretrain + PPO Discrete(2) fine-tune in one method.

Source-side BC (RBC1 demos → categorical CE) gets PPO actor close to RBC1's
schedule before RL takes over. Critic/value head starts from PPO defaults
and warms up during fine-tune. Same SB3 PPO model used throughout.
"""
from __future__ import annotations

from studies.case_study.core.bc_trainer import (
    BCConfig,
    collect_rbc_demos,
    train_bc_discrete_actor,
)
from studies.case_study.methods.base import MethodExecution


class BCThenFinetunePPORunner:
    method_id = "bc_then_finetune_ppo"

    def execute(self, case, env, services):
        # 1) Build PPO model (fresh weights)
        model = services.build_training_model(case, env)

        # 2) Collect RBC1 demos on this env
        price_schedule = case.reward.get("kwargs", {}).get("price_schedule")
        if price_schedule is None or len(price_schedule) != 24:
            raise ValueError("bc_then_finetune_ppo requires 24-element price_schedule")
        physical = case.building.physical or {}
        tes_range = physical.get("tes_setpoint_range", [5.0, 12.0])

        bc_cfg_dict = case.training.get("bc", {}) or {}
        n_episodes_demo = int(bc_cfg_dict.get("n_episodes", 1))
        bc_epochs = int(bc_cfg_dict.get("epochs", 20))
        bc_batch = int(bc_cfg_dict.get("batch_size", 256))
        bc_lr = float(bc_cfg_dict.get("learning_rate", 3.0e-4))
        bc_val = float(bc_cfg_dict.get("val_fraction", 0.2))
        bc_variant = str(bc_cfg_dict.get("variant", "rbc1"))

        states, actions = collect_rbc_demos(
            env,
            price_schedule=price_schedule,
            tes_range=(float(tes_range[0]), float(tes_range[1])),
            n_episodes=n_episodes_demo,
            variant=bc_variant,
        )

        # 3) BC train PPO actor on demos
        bc_history = train_bc_discrete_actor(
            model,
            states,
            actions,
            BCConfig(
                n_episodes=n_episodes_demo,
                epochs=bc_epochs,
                batch_size=bc_batch,
                learning_rate=bc_lr,
                val_fraction=bc_val,
                device="cpu",
            ),
        )

        # 4) PPO fine-tune on env (uses scratch_baseline-style learn())
        total_timesteps = services.compute_total_timesteps(case, env)
        learn_kwargs = services.build_learn_kwargs(case)
        model.learn(total_timesteps=total_timesteps, **learn_kwargs)

        return MethodExecution(
            model=model,
            total_timesteps=total_timesteps + int(states.shape[0]),
            metadata={
                "method": "bc_then_finetune_ppo",
                "bc_variant": bc_variant,
                "n_demos": int(states.shape[0]),
                "bc_demo_episodes": n_episodes_demo,
                "bc_epochs": bc_epochs,
                "bc_final_train_ce": bc_history["train_ce"][-1] if bc_history["train_ce"] else None,
                "bc_final_val_ce": bc_history["val_ce"][-1] if bc_history["val_ce"] else None,
                "bc_final_train_acc": bc_history["train_acc"][-1] if bc_history["train_acc"] else None,
                "bc_final_val_acc": bc_history["val_acc"][-1] if bc_history["val_acc"] else None,
                "rl_timesteps": int(total_timesteps),
            },
        )


RUNNER = BCThenFinetunePPORunner()
