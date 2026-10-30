"""Behavioural Cloning (BC) utility for cross-building HVAC-TES transfer.

Used by the `bc_pretrain` method to supervise-train a SAC/PPO actor on
state-action pairs collected from an expert controller (typically RBC1)
running on a source building. The BC-pretrained actor is then transferred
to a target building via `padding_transfer`, mirroring the DP40d baseline
of Kadamala et al. (2025).

Demo collection follows the same TOU rule as RuleBasedTOURunner: charge
to tank_lo during off-peak hours, hold at tank_hi during peak hours and
mid-peak hours (RBC1 convention used elsewhere in the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class BCConfig:
    n_episodes: int = 3          # source-env rollouts for demo collection
    epochs: int = 10             # supervised training epochs
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    val_fraction: float = 0.2    # paper uses 80-20 split
    device: str = "cpu"


def _hours_by_tariff_tier(price_schedule: List[float]) -> Tuple[List[int], List[int], List[int]]:
    p_min = min(price_schedule)
    p_max = max(price_schedule)
    off, peak, mid = [], [], []
    for h, p in enumerate(price_schedule):
        if abs(p - p_min) < 1e-9:
            off.append(h)
        elif abs(p - p_max) < 1e-9:
            peak.append(h)
        else:
            mid.append(h)
    return off, peak, mid


def _decode_hour(raw_obs: np.ndarray, h_sin_idx: int, h_cos_idx: int) -> int:
    s = float(raw_obs[h_sin_idx])
    c = float(raw_obs[h_cos_idx])
    h = np.arctan2(s, c) / (2.0 * np.pi) * 24.0
    return int(round(h)) % 24


def _find_phys_norm_wrapper(env):
    from studies.case_study.core.wrappers import PhysicalNormalizeObservation
    e = env
    while hasattr(e, "env"):
        if isinstance(e, PhysicalNormalizeObservation):
            return e
        e = e.env
    return None


def collect_rbc_demos(
    env,
    price_schedule: List[float],
    tes_range: Tuple[float, float],
    n_episodes: int = 3,
    variant: str = "rbc1",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run RBC variant on env for n_episodes and return (states, actions) arrays.

    variant: "rbc1" (off charge, mid+peak hold) — default
             "rbc2" (mid charge, off+peak hold) — for BC lock-in diagnosis

    Returns:
        states: shape (N, obs_dim) — observations seen by the agent (post-wrappers,
                so already physically normalized if PhysicalNormalizeObservation
                is in the wrapper stack).
        actions: shape (N, action_dim) — actions taken by RBC1 in env-action space
                 (continuous [-1,1] for Box, integer for Discrete).
    """
    off_hours, peak_hours, _ = _hours_by_tariff_tier(price_schedule)
    tank_lo, tank_hi = tes_range

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        n_actions = int(env.action_space.n)
        grid = np.linspace(tank_lo, tank_hi, n_actions)
        env_low = env_high = 0.0
    else:
        n_actions = 0
        grid = np.array([])
        env_low = float(env.action_space.low[0])
        env_high = float(env.action_space.high[0])

    def _to_env_action(real_setpoint: float):
        """Action in env's native action space (passed to env.step)."""
        if is_discrete:
            return int(np.argmin(np.abs(grid - real_setpoint)))
        return np.array(
            [np.clip(real_setpoint, env_low, env_high)],
            dtype=np.float32,
        )

    def _to_bc_target(real_setpoint: float) -> np.ndarray:
        """Action target for BC training (SAC actor's tanh output space [-1, 1]).

        SAC stores actions in env's native range internally but the actor's
        mean head outputs tanh(mu) in [-1, 1]. The BC step trains tanh(mu)
        directly, so the target must be in [-1, 1].
        """
        if is_discrete:
            # Treat discrete action index as a real for BC; rarely hit since
            # SAC requires Box action space.
            return np.array([float(np.argmin(np.abs(grid - real_setpoint)))], dtype=np.float32)
        mid = 0.5 * (env_low + env_high)
        half = 0.5 * (env_high - env_low)
        normalized = (real_setpoint - mid) / max(half, 1e-9)
        return np.array([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)

    # Use info['hour'] (EnergyPlus DataExchange.hour() 0..23) directly — robust
    # to reduce_observation profiles that drop hour_sin/hour_cos features.

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            hour = int(info.get("hour", 0)) % 24
            if variant == "rbc1":
                if hour in off_hours:
                    setpoint = tank_lo
                elif hour in peak_hours:
                    setpoint = tank_hi
                else:
                    setpoint = tank_hi  # mid-peak: hold (no charge)
            elif variant == "rbc2":
                if hour in off_hours:
                    setpoint = tank_hi  # off: hold
                elif hour in peak_hours:
                    setpoint = tank_hi  # peak: hold
                else:
                    setpoint = tank_lo  # mid: charge
            else:
                raise ValueError(f"Unknown BC variant: {variant}")
            env_action = _to_env_action(setpoint)
            bc_target = _to_bc_target(setpoint)

            states.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(bc_target.copy())

            obs, _r, done, truncated, info = env.step(env_action)

    return np.stack(states), np.stack(actions)


def train_bc_discrete_actor(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    config: BCConfig,
) -> dict:
    """Train PPO categorical policy via supervised cross-entropy on demo actions.

    Used as the source-side step for PPO+Discrete(2) BC warmstart from RBC1.
    Compatible with SB3 PPO's ActorCriticPolicy whose discrete action head
    outputs class logits.

    Args:
        model: SB3 PPO model (modified in-place)
        states: (N, obs_dim) observations after wrappers
        actions: (N, action_dim) actions as int class indices (e.g. 0 or 1 for D(2))
        config: BCConfig
    """
    if not hasattr(model, "policy"):
        raise ValueError("train_bc_discrete_actor requires SB3 PPO model with .policy")

    device = torch.device(config.device)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    # Targets must be class indices (long); actions shape (N, 1) → squeeze
    if actions.ndim > 1 and actions.shape[-1] == 1:
        actions = actions[..., 0]
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)

    n = states_t.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * config.val_fraction)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_ds = TensorDataset(states_t[tr_idx], actions_t[tr_idx])
    val_ds = TensorDataset(states_t[val_idx], actions_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    policy = model.policy
    # Train the entire policy network — features_extractor + mlp_extractor.policy_net
    # + action_net. (Value head stays untouched for BC.)
    params_to_train = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(params_to_train, lr=config.learning_rate)
    ce_loss = nn.CrossEntropyLoss()

    history = {"train_ce": [], "val_ce": [], "train_acc": [], "val_acc": []}
    for _epoch in range(config.epochs):
        policy.train()
        tr_losses, tr_correct, tr_total = [], 0, 0
        for s, a in train_loader:
            features = policy.extract_features(s)
            if isinstance(features, tuple):  # shared net returns (pi_feat, vf_feat)
                pi_feat = features[0]
            else:
                pi_feat = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_feat)
            logits = policy.action_net(latent_pi)

            loss = ce_loss(logits, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_correct += (logits.argmax(dim=-1) == a).sum().item()
            tr_total += a.numel()

        policy.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for s, a in val_loader:
                features = policy.extract_features(s)
                pi_feat = features[0] if isinstance(features, tuple) else features
                latent_pi = policy.mlp_extractor.forward_actor(pi_feat)
                logits = policy.action_net(latent_pi)
                loss = ce_loss(logits, a)
                val_losses.append(loss.item())
                val_correct += (logits.argmax(dim=-1) == a).sum().item()
                val_total += a.numel()

        history["train_ce"].append(float(np.mean(tr_losses)) if tr_losses else 0.0)
        history["val_ce"].append(float(np.mean(val_losses)) if val_losses else 0.0)
        history["train_acc"].append(tr_correct / max(1, tr_total))
        history["val_acc"].append(val_correct / max(1, val_total))

    return history


def train_bc_continuous_actor(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    config: BCConfig,
) -> dict:
    """Train SAC actor's mean network via supervised MSE on demo actions.

    For SAC's SquashedDiagGaussianDistribution actor, BC trains the mean
    network output to match (atanh-untransformed) target actions. The
    log_std network is left at its initial values — RL finetune will
    adjust exploration from the BC mean.

    Args:
        model: SB3 SAC model (modified in-place)
        states: (N, obs_dim) normalized observations
        actions: (N, action_dim) actions in env space [-1, 1] (already tanh-squashed)
        config: BCConfig

    Returns:
        dict with training/val losses per epoch.
    """
    if not hasattr(model, "actor"):
        raise ValueError("train_bc_continuous_actor requires SAC-style model with .actor")

    device = torch.device(config.device)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)

    # Train/val split (paper: 80-20)
    n = states_t.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * config.val_fraction)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_ds = TensorDataset(states_t[tr_idx], actions_t[tr_idx])
    val_ds = TensorDataset(states_t[val_idx], actions_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Train only the actor's deterministic-action head (mu network); freeze
    # log_std for now — SAC will pick up exploration from default init.
    # SB3 SAC actor: .latent_pi (shared MLP) + .mu (mean head) + .log_std (head)
    actor = model.actor
    params_to_train = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    optimizer = torch.optim.Adam(params_to_train, lr=config.learning_rate)

    # Init log_std small so initial policy stays near BC mean (low noise).
    # SAC's auto-entropy would otherwise quickly push log_std up; we counter
    # this by setting bias to log(0.1)=-2.3 and zeroing the weight so the
    # network output is a constant ~-2.3 regardless of input.
    with torch.no_grad():
        if hasattr(actor, "log_std") and isinstance(actor.log_std, nn.Linear):
            actor.log_std.weight.data.zero_()
            actor.log_std.bias.data.fill_(-2.3)

    history = {"train_mse": [], "val_mse": []}
    for _epoch in range(config.epochs):
        actor.train()
        tr_losses = []
        for s, a in train_loader:
            # Get squashed mean action prediction (no exploration noise)
            with torch.no_grad():
                # actor.forward returns sampled actions; we want deterministic mean
                pass
            # SAC actor: get tanh(mu_net(latent_pi(s))) as predicted mean action
            features = actor.extract_features(s, actor.features_extractor)
            latent_pi = actor.latent_pi(features)
            mean_actions = actor.mu(latent_pi)
            # Apply tanh squashing to map to [-1, 1] (same as during inference)
            pred = torch.tanh(mean_actions)

            loss = nn.functional.mse_loss(pred, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        actor.eval()
        val_losses = []
        with torch.no_grad():
            for s, a in val_loader:
                features = actor.extract_features(s, actor.features_extractor)
                latent_pi = actor.latent_pi(features)
                mean_actions = actor.mu(latent_pi)
                pred = torch.tanh(mean_actions)
                loss = nn.functional.mse_loss(pred, a)
                val_losses.append(loss.item())

        tr_mse = float(np.mean(tr_losses)) if tr_losses else 0.0
        val_mse = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_mse"].append(tr_mse)
        history["val_mse"].append(val_mse)

    return history
