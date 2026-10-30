from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch


def select_discrete_threshold_actions(
    probabilities: np.ndarray,
    *,
    target_action: int,
    min_probability: float,
) -> np.ndarray | np.integer:
    """Choose a rare discrete action deterministically from policy probabilities.

    Argmax can erase useful low-probability actions learned by a stochastic
    categorical policy. This selector keeps argmax as the fallback, but forces
    ``target_action`` whenever its probability exceeds a calibrated threshold.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    squeeze = probs.ndim == 1
    if squeeze:
        probs = probs[None, :]
    if probs.ndim != 2:
        raise ValueError(f"Expected 1D or 2D probabilities, got shape {probs.shape}")
    if target_action < 0 or target_action >= probs.shape[1]:
        raise ValueError(
            f"target_action={target_action} is outside probability shape {probs.shape}"
        )
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError(f"min_probability must be in [0, 1], got {min_probability}")

    actions = np.argmax(probs, axis=1).astype(np.int64)
    actions[probs[:, target_action] >= min_probability] = int(target_action)
    return actions[0] if squeeze else actions


def _categorical_probabilities(model: Any, obs: Any) -> tuple[np.ndarray, bool]:
    obs_tensor, vectorized_env = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
    torch_distribution = getattr(distribution, "distribution", None)
    probabilities = getattr(torch_distribution, "probs", None)
    if probabilities is None:
        raise TypeError(
            "discrete_threshold action selection requires a categorical policy "
            "with distribution.probs"
        )
    return probabilities.detach().cpu().numpy(), bool(vectorized_env)


def predict_action(
    model: Any,
    obs: Any,
    *,
    state: Any = None,
    episode_start: Any = None,
    deterministic: bool = True,
    action_selection: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Predict an action with optional deterministic policy calibration.

    Supported ``action_selection`` modes:
      - ``stochastic``: call ``model.predict(..., deterministic=False)``.
      - ``deterministic`` or unset: preserve SB3 default deterministic behavior.
      - ``discrete_threshold``: for Discrete policies, choose ``action`` when
        its probability is at least ``min_probability``; otherwise use argmax.
    """
    selection = dict(action_selection or {})
    mode = str(selection.get("mode", "")).lower()

    if mode in {"stochastic", "sample"}:
        return model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=False,
        )

    if mode in {"", "deterministic", "argmax", "mean"}:
        return model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=bool(selection.get("deterministic", deterministic)),
        )

    if mode != "discrete_threshold":
        raise ValueError(f"Unknown action_selection mode: {mode}")

    if not isinstance(model.action_space, gym.spaces.Discrete) and not hasattr(
        model.action_space, "n"
    ):
        return model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )

    target_action = int(selection.get("action", selection.get("target_action", 0)))
    min_probability = float(
        selection.get("min_probability", selection.get("threshold", 0.5))
    )
    probabilities, vectorized_env = _categorical_probabilities(model, obs)
    actions = select_discrete_threshold_actions(
        probabilities,
        target_action=target_action,
        min_probability=min_probability,
    )
    if not vectorized_env:
        actions = int(np.asarray(actions).reshape(-1)[0])
    return actions, state
