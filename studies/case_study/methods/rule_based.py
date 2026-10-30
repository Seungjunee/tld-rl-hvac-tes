from __future__ import annotations

from typing import List, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from studies.case_study.methods.base import MethodExecution


def _hours_by_tariff_tier(
    price_schedule: List[float],
) -> Tuple[List[int], List[int], List[int]]:
    if not price_schedule or len(price_schedule) != 24:
        raise ValueError("rbc_tou requires a 24-element price_schedule")
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


def _make_hour_reader(obs_vars: List[str]):
    """Build a fn(raw_obs) -> hour for this env's observation layout.

    Prefers the cyclic hour_sin/hour_cos channels (datetime wrapper on);
    falls back to the raw `hour` channel for envs without the datetime
    wrapper (e.g. dataset-export profiles log the unencoded clock).
    """
    if "hour_sin" in obs_vars and "hour_cos" in obs_vars:
        h_sin_idx = obs_vars.index("hour_sin")
        h_cos_idx = obs_vars.index("hour_cos")

        def _read(raw_obs: np.ndarray) -> int:
            return _decode_hour(raw_obs, h_sin_idx, h_cos_idx)

    else:
        h_idx = obs_vars.index("hour")

        def _read(raw_obs: np.ndarray) -> int:
            return int(round(float(raw_obs[h_idx]))) % 24

    return _read


def _find_phys_norm_wrapper(env):
    from studies.case_study.core.wrappers import PhysicalNormalizeObservation

    e = env
    while hasattr(e, "env"):
        if isinstance(e, PhysicalNormalizeObservation):
            return e
        e = e.env
    return None


class RuleBasedTOURunner:
    """TOU-aware Rule-Based Control baseline.

    Off-peak hours (cheapest tariff): aggressive charge → setpoint = tank_min
    On-peak hours (most expensive): no charge → setpoint = tank_max (let warm)
    Mid-peak: hold at tank_max (no charge, conserve TES for on-peak)

    Tariff tiers are derived from price_schedule (min=off, max=peak, else=mid).
    """

    method_id = "rbc_tou"

    def execute(self, case, env, services):
        price_schedule = case.reward.get("kwargs", {}).get("price_schedule")
        off_hours, peak_hours, _ = _hours_by_tariff_tier(price_schedule)

        physical = case.building.physical or {}
        tank_lo, tank_hi = physical.get("tes_setpoint_range", [5.0, 10.0])

        if isinstance(env.action_space, gym.spaces.Discrete):
            # Discrete(n): map setpoint to nearest action index. Discretize wrapper
            # built equal-spaced grid between tank_lo and tank_hi (e.g. D(3) = [lo, mid, hi]).
            n = int(env.action_space.n)
            if n < 2:
                raise ValueError("rbc_tou requires Discrete(n>=2)")
            grid = np.linspace(tank_lo, tank_hi, n)

            def _to_env_action(real_setpoint: float) -> Union[int, np.ndarray]:
                return int(np.argmin(np.abs(grid - real_setpoint)))

        else:
            action_low = float(env.action_space.low[0])
            action_high = float(env.action_space.high[0])
            # NormalizeAction envs expose [-1, 1]; export/raw envs expose the
            # physical setpoint range. Sending a normalized value into a raw
            # space clips every command to the low bound (constant tank_lo).
            normalized_space = action_high <= 1.0 + 1e-6

            def _to_env_action(real_setpoint: float) -> np.ndarray:
                if normalized_space:
                    mid = 0.5 * (tank_lo + tank_hi)
                    half = 0.5 * (tank_hi - tank_lo)
                    value = (real_setpoint - mid) / max(half, 1e-9)
                else:
                    value = real_setpoint
                value = float(np.clip(value, action_low, action_high))
                return np.array([value], dtype=np.float32)

        obs_vars = list(env.get_wrapper_attr("observation_variables"))
        read_hour = _make_hour_reader(obs_vars)
        norm_w = _find_phys_norm_wrapper(env)

        episodes = int(case.training.get("episodes") or 0)
        if episodes <= 0:
            steps_per_ep = int(env.get_wrapper_attr("timestep_per_episode"))
            total = int(case.training.get("total_timesteps") or steps_per_ep)
            episodes = max(1, total // max(1, steps_per_ep - 1))

        total_steps = 0
        for _ in range(episodes):
            obs, _info = env.reset()
            done = truncated = False
            while not (done or truncated):
                raw = obs
                if norm_w is not None and norm_w.unwrapped_observation is not None:
                    raw = norm_w.unwrapped_observation
                hour = read_hour(np.asarray(raw))
                if hour in off_hours:
                    setpoint = tank_lo
                elif hour in peak_hours:
                    setpoint = tank_hi
                else:
                    setpoint = tank_hi
                obs, _r, done, truncated, _info = env.step(_to_env_action(setpoint))
                total_steps += 1

        # Build a placeholder PPO model so downstream save_model works
        model = PPO("MlpPolicy", env, verbose=0)

        return MethodExecution(
            model=model,
            total_timesteps=total_steps,
            metadata={
                "method": "rbc_tou",
                "off_hours": off_hours,
                "peak_hours": peak_hours,
                "tank_range": [tank_lo, tank_hi],
            },
        )


class RuleBasedTOUv2Runner:
    """RBC2: charge tank (sp=tank_lo) during ALL non-peak hours.

    On-peak hours (most expensive): sp=tank_hi (no charge, let warm)
    Off-peak + Mid-peak: sp=tank_lo (aggressive charge)

    Differs from rbc_tou which only charges during off-peak.
    """

    method_id = "rbc_tou_v2"

    def execute(self, case, env, services):
        price_schedule = case.reward.get("kwargs", {}).get("price_schedule")
        _, peak_hours, _ = _hours_by_tariff_tier(price_schedule)

        physical = case.building.physical or {}
        tank_lo, tank_hi = physical.get("tes_setpoint_range", [5.0, 10.0])

        if isinstance(env.action_space, gym.spaces.Discrete):
            n = int(env.action_space.n)
            if n < 2:
                raise ValueError("rbc_tou_v2 requires Discrete(n>=2)")
            grid = np.linspace(tank_lo, tank_hi, n)

            def _to_env_action(real_setpoint: float) -> Union[int, np.ndarray]:
                return int(np.argmin(np.abs(grid - real_setpoint)))

        else:
            action_low = float(env.action_space.low[0])
            action_high = float(env.action_space.high[0])
            # NormalizeAction envs expose [-1, 1]; export/raw envs expose the
            # physical setpoint range. Sending a normalized value into a raw
            # space clips every command to the low bound (constant tank_lo).
            normalized_space = action_high <= 1.0 + 1e-6

            def _to_env_action(real_setpoint: float) -> np.ndarray:
                if normalized_space:
                    mid = 0.5 * (tank_lo + tank_hi)
                    half = 0.5 * (tank_hi - tank_lo)
                    value = (real_setpoint - mid) / max(half, 1e-9)
                else:
                    value = real_setpoint
                value = float(np.clip(value, action_low, action_high))
                return np.array([value], dtype=np.float32)

        obs_vars = list(env.get_wrapper_attr("observation_variables"))
        read_hour = _make_hour_reader(obs_vars)
        norm_w = _find_phys_norm_wrapper(env)

        episodes = int(case.training.get("episodes") or 0)
        if episodes <= 0:
            steps_per_ep = int(env.get_wrapper_attr("timestep_per_episode"))
            total = int(case.training.get("total_timesteps") or steps_per_ep)
            episodes = max(1, total // max(1, steps_per_ep - 1))

        total_steps = 0
        for _ in range(episodes):
            obs, _info = env.reset()
            done = truncated = False
            while not (done or truncated):
                raw = obs
                if norm_w is not None and norm_w.unwrapped_observation is not None:
                    raw = norm_w.unwrapped_observation
                hour = read_hour(np.asarray(raw))
                if hour in peak_hours:
                    setpoint = tank_hi  # peak: no charge
                else:
                    setpoint = tank_lo  # off + mid: aggressive charge
                obs, _r, done, truncated, _info = env.step(_to_env_action(setpoint))
                total_steps += 1

        model = PPO("MlpPolicy", env, verbose=0)
        return MethodExecution(
            model=model,
            total_timesteps=total_steps,
            metadata={
                "method": "rbc_tou_v2",
                "peak_hours": peak_hours,
                "tank_range": [tank_lo, tank_hi],
            },
        )


RUNNER = RuleBasedTOURunner()
RUNNER_V2 = RuleBasedTOUv2Runner()
