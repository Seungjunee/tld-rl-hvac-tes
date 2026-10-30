"""Padding-transfer runner (Kadamala et al. ref [30] adapted to SAC).

Differs from continue_finetune in three ways that together prevent SAC from
washing out the BC-supervised actor before the critic catches up:

  1. learning_starts=0 — BC actor is used from step 1 (no 1000-step uniform
     warmup that would override BC).
  2. Slow actor LR (1e-5) but fast critic LR (3e-4). Mirrors Kadamala et al.
     (2025)'s schedule: actor warmup 0→3e-6, critic 3e-4 constant. Keeps
     the BC mean roughly frozen for the first few episodes while the critic
     learns Q from on-policy data, after which the actor can refine.
  3. Fixed low entropy coefficient (ent_coef=0.01) instead of auto-tuned.
     Auto-entropy quickly pushes log_std up to satisfy target entropy, which
     drowns the BC mean in noise; a fixed small value preserves
     near-deterministic BC behaviour through early episodes.

The weight transfer itself is identical to continue_finetune (SAC.load
restores all weights). In our pipeline source/target obs dims match via
ZonePaddingWrapper, so no explicit feature padding is required — the
paper's DP mechanism is structurally already in place at the wrapper level.

Frozen-feature variant (method `padding_frozen`):
`training.freeze_first_layer: true` fixes the first hidden layer (weight and
bias) of the actor and of both critic Q-networks and fine-tunes only the
remaining layers; everything else is the Padding fine-tuning above. The
target critic's first layers are set to the frozen online ones before
training, so the TD target uses the same fixed features (SB3's Polyak update
then keeps them equal up to float32 rounding). The run's metadata records a
check that the frozen layers did not move and the trainable ones did.
"""

from __future__ import annotations

import hashlib

import torch

from studies.case_study.methods._norm_utils import seed_normalization_from_source
from studies.case_study.methods.base import MethodExecution
from studies.case_study.methods.continue_finetune import _noop_lr_update


def _first_hidden_layers(model) -> dict[str, torch.nn.Linear]:
    """First hidden Linear of the actor and of each online critic Q-network."""
    layers = {"actor.latent_pi.0": model.actor.latent_pi[0]}
    for i, q_net in enumerate(model.critic.q_networks):
        layers[f"critic.qf{i}.0"] = q_net[0]
    for name, layer in layers.items():
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"freeze_first_layer: {name} is {type(layer).__name__}")
    return layers


def _state_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _freeze_first_layers(model) -> dict:
    """Freeze the first hidden layers in place; return the pre-training record."""
    layers = _first_hidden_layers(model)
    for layer in layers.values():
        for p in layer.parameters():
            p.requires_grad_(False)
    target_layers = [q_net[0] for q_net in model.critic_target.q_networks]
    online_critic = [layers[f"critic.qf{i}.0"] for i in range(len(target_layers))]
    with torch.no_grad():
        target_gap = max(
            float((t.weight - o.weight).abs().max().item())
            for t, o in zip(target_layers, online_critic)
        )
        for t, o in zip(target_layers, online_critic):
            t.weight.copy_(o.weight)
            t.bias.copy_(o.bias)
    snapshot = {
        f"{name}.{k}": v.detach().clone()
        for name, layer in layers.items()
        for k, v in layer.state_dict().items()
    }
    trainable = {
        n: p.detach().clone()
        for n, p in model.policy.named_parameters()
        if p.requires_grad and not n.startswith("critic_target.")
    }
    return {
        "frozen": snapshot,
        "trainable": trainable,
        "target_gap_before_copy": target_gap,
    }


@torch.no_grad()
def _freeze_check(model, record: dict) -> dict:
    """Frozen layers unchanged, trainable layers updated, target layers equal."""
    params = dict(model.policy.named_parameters())
    frozen_change = max(
        float((params[n] - v).abs().max().item()) for n, v in record["frozen"].items()
    )
    changes = {
        n: float((params[n] - v).abs().max().item())
        for n, v in record["trainable"].items()
    }
    target_dev = max(
        float((getattr(t[0], k) - getattr(o[0], k)).abs().max().item())
        for t, o in zip(model.critic_target.q_networks, model.critic.q_networks)
        for k in ("weight", "bias")
    )
    return {
        "frozen_max_abs_change": frozen_change,
        "trainable_tensors_changed": sum(v > 0.0 for v in changes.values()),
        "trainable_tensors_total": len(changes),
        "trainable_min_max_abs_change": min(changes.values()),
        "target_first_layer_max_abs_dev": target_dev,
    }


class PaddingTransferRunner:
    method_id = "padding_transfer"

    def execute(self, case, env, services):
        source_bundle = services.load_source_bundle(case)
        model = services.load_model(source_bundle.model_path, env=env)
        initial_state_sha256 = _state_sha256(model.policy)

        # SB3's load() does not apply a seed (only the scratch constructor
        # consumes model_kwargs["seed"]); reseed explicitly so seeded padding
        # runs are genuine replicates rather than identical reruns.
        seed = case.training.get("model_kwargs", {}).get("seed")
        if seed is not None:
            model.set_random_seed(int(seed))

        # Fail fast on state-shape mismatch (2026-06-07, design Part 2 R5):
        # SAC.load(env=...) tolerates a different env until the first step, so a
        # source trained on building-specific dims would otherwise fail deep
        # inside learn(). Padding transfer REQUIRES the building-invariant
        # padded profile on both sides (zone_slot_1..N via ZonePaddingWrapper).
        src_shape = tuple(model.observation_space.shape)
        tgt_shape = tuple(env.observation_space.shape)
        if src_shape != tgt_shape:
            raise ValueError(
                f"padding_transfer obs shape mismatch: source {src_shape} vs "
                f"target {tgt_shape}. Both cases must use *_sac_continuous_padded "
                "profiles (ZonePaddingWrapper) so dims are building-invariant."
            )

        # Force SAC to use the loaded policy from step 1 (no random warmup).
        model.learning_starts = 0

        # `actor_lr` and `critic_lr` come from the case's training spec.
        # ent_coef_fixed: a float freezes entropy at that value (Kadamala-faithful
        # "warm-start then near-frozen" regime); OMITTING it (None) leaves SAC's
        # auto-tuned entropy in place so the BC warm-start is actually RL
        # fine-tuned to convergence (paper claim: padding reaches scratch perf).
        training = case.training or {}
        actor_lr = float(training.get("actor_lr", 1e-5))
        critic_lr = float(training.get("critic_lr", 3e-4))
        ent_coef_value = training.get("ent_coef_fixed", None)
        freeze_first_layer = bool(training.get("freeze_first_layer", False))

        # Frozen-feature variant: freeze before the optimizers are rebuilt so
        # they hold only the trainable parameters (Adam settings unchanged).
        freeze_record = _freeze_first_layers(model) if freeze_first_layer else None

        def _trainable(module):
            return [p for p in module.parameters() if p.requires_grad]

        model.learning_rate = critic_lr
        if hasattr(model, "actor") and hasattr(model.actor, "optimizer"):
            model.actor.optimizer = torch.optim.Adam(
                _trainable(model.actor), lr=actor_lr
            )
            if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                model.critic.optimizer = torch.optim.Adam(
                    _trainable(model.critic), lr=critic_lr
                )

            # Only override entropy when a fixed coefficient is requested. SB3 SAC
            # stores ent_coef as a fixed tensor or as exp(log_ent_coef) under
            # 'auto'; rebind both to a constant and disable its optimizer so the
            # value stays put. When ent_coef_value is None, auto entropy is kept.
            if ent_coef_value is not None:
                ent_tensor = torch.tensor(float(ent_coef_value), device=model.device)
                model.ent_coef_tensor = ent_tensor
                if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
                    with torch.no_grad():
                        model.log_ent_coef.data.fill_(
                            torch.log(ent_tensor.clamp_min(1e-8)).item()
                        )
                model.ent_coef_optimizer = None  # freeze entropy coefficient

        # SB3's train() re-applies self.lr_schedule (source model's saved LR)
        # to every optimizer on each gradient step, discarding the asymmetric
        # actor/critic LRs above AND the critic-warmup lr=0 trick below.
        # Neutralise it so the explicitly-set Adam learning rates hold
        # (same fix as continue_finetune, 2026-06-10; module-level function
        # so model export stays picklable).
        model._update_learning_rate = _noop_lr_update

        seed_normalization_from_source(env, source_bundle)

        total_timesteps = services.compute_total_timesteps(case, env)

        # Critic warmup (2026-06-08): the BC actor is a good init but the critic
        # is fresh-random, so early actor updates follow a wrong Q and erode the
        # warm-start (the LBuilding ep2→ep10 dip). Optionally freeze the actor
        # (actor_lr=0) for `critic_warmup_steps` so the critic learns Q for the
        # BC policy first, then restore actor_lr for the rest of training. This
        # is a stabilisation of Kadamala's own RL-trained critic, not extra info.
        warmup_steps = int(training.get("critic_warmup_steps", 0))
        learn_kwargs = services.build_learn_kwargs(case)
        if warmup_steps > 0 and hasattr(model, "actor"):
            for g in model.actor.optimizer.param_groups:
                g["lr"] = 0.0
            model.learn(total_timesteps=warmup_steps, **learn_kwargs)
            for g in model.actor.optimizer.param_groups:
                g["lr"] = actor_lr
            remaining = max(total_timesteps - warmup_steps, 1)
            reset_kw = dict(learn_kwargs)
            reset_kw["reset_num_timesteps"] = False
            model.learn(total_timesteps=remaining, **reset_kw)
        else:
            model.learn(total_timesteps=total_timesteps, **learn_kwargs)

        metadata = {
            "method": case.method.id,
            "source_model": str(source_bundle.model_path),
            "initial_state_sha256": initial_state_sha256,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "ent_coef_fixed": (
                None if ent_coef_value is None else float(ent_coef_value)
            ),
            "learning_starts": 0,
            "freeze_first_layer": freeze_first_layer,
        }
        if freeze_record is not None:
            metadata["frozen_layers"] = sorted(
                {n.rsplit(".", 1)[0] for n in freeze_record["frozen"]}
            )
            metadata["n_frozen_params"] = int(
                sum(v.numel() for v in freeze_record["frozen"].values())
            )
            metadata["n_trainable_params"] = int(
                sum(v.numel() for v in freeze_record["trainable"].values())
            )
            metadata["target_gap_before_copy"] = freeze_record["target_gap_before_copy"]
            metadata["freeze_check"] = _freeze_check(model, freeze_record)

        return MethodExecution(
            model=model,
            total_timesteps=total_timesteps,
            source_bundle=source_bundle,
            metadata=metadata,
        )


RUNNER = PaddingTransferRunner()
