from __future__ import annotations

import datetime
from collections import deque
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from sinergym.utils.constants import LOG_WRAPPERS_LEVEL
from sinergym.utils.logger import TerminalLogger


class SetpointForecastWrapper(gym.Wrapper):
    """Provides H-step ahead cooling setpoint forecast.

    Since setpoints follow deterministic EnergyPlus schedules, this wrapper
    repeats the current observed setpoint values for each zone as a forecast.
    For buildings with constant setpoints this is exact; for time-varying
    schedules it provides a reasonable approximation.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER SetpointForecastWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, setpoint_variables: List[str], horizon: int = 24):
        super().__init__(env)
        self.horizon = horizon

        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)

        self.setpoint_indices = []
        for var in setpoint_variables:
            if var not in obs_vars:
                raise ValueError(
                    f"SetpointForecastWrapper: '{var}' not in observation_variables"
                )
            self.setpoint_indices.append(obs_vars.index(var))

        n_setpoints = len(setpoint_variables)
        forecast_names = []
        for i in range(1, horizon + 1):
            for var in setpoint_variables:
                forecast_names.append(f"setpoint_forecast_{i}_{var}")

        self.observation_variables = list(obs_vars) + forecast_names
        old_space = env.observation_space
        n_new = n_setpoints * horizon
        new_low = np.concatenate(
            [old_space.low, np.full((n_new,), -np.inf, dtype=np.float32)]
        )
        new_high = np.concatenate(
            [old_space.high, np.full((n_new,), np.inf, dtype=np.float32)]
        )
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: {horizon}-step forecast for {setpoint_variables}"
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        current_setpoints = obs[self.setpoint_indices]
        # Repeat current setpoints for each forecast step
        forecast = np.tile(current_setpoints, self.horizon).astype(np.float32)
        return np.concatenate([obs, forecast]).astype(np.float32)


class CPULoadForecastWrapper(gym.Wrapper):
    """Provides H-step ahead CPU load forecast from a deterministic schedule CSV.

    Reads hourly CPU load data (8760 values for 1 year) and appends
    the next H hours of CPU load to the observation. Datacenter-specific.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER CPULoadForecastWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        csv_path: str,
        horizon: int = 24,
        forecast_error: dict | None = None,
    ):
        super().__init__(env)
        self.horizon = horizon

        schedule_df = pd.read_csv(csv_path)
        col = (
            "cpu_load" if "cpu_load" in schedule_df.columns else schedule_df.columns[-1]
        )
        self.schedule = schedule_df[col].values.astype(np.float32)  # 8760 hourly values

        # Optional IT-load forecast error: one
        # mean-reverting relative factor shared by the whole look-ahead
        # trajectory - the same perturbation form as the weather test.
        # Hourly AR(1) with the given time constant; the fixed seed gives one
        # realization per season, identical across cases and source seeds.
        # The building's actual IT schedule is untouched.
        cfg = forecast_error or {}
        sigma = float(cfg.get("sigma", 0.0))
        if sigma > 0.0:
            tau_h = float(cfg.get("time_constant_h", 24.0))
            phi = float(np.exp(-1.0 / tau_h))
            rng = np.random.default_rng(int(cfg.get("seed", 2026)))
            n = len(self.schedule)
            u = np.empty(n, dtype=np.float64)
            u[0] = rng.standard_normal()
            for t in range(1, n):
                u[t] = phi * u[t - 1] + np.sqrt(1.0 - phi * phi) * rng.standard_normal()
            self._err_factor = (1.0 + sigma * u).astype(np.float32)
        else:
            self._err_factor = None

        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)
        self.observation_variables = list(obs_vars) + [
            f"cpu_load_forecast_{i}" for i in range(1, horizon + 1)
        ]

        old_space = env.observation_space
        forecast_low = np.zeros((horizon,), dtype=np.float32)
        forecast_high = np.ones((horizon,), dtype=np.float32)
        new_low = np.concatenate([old_space.low, forecast_low])
        new_high = np.concatenate([old_space.high, forecast_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: {horizon}-step CPU load forecast from {csv_path}"
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray, info: dict) -> np.ndarray:
        hour_idx = self._get_hour_of_year(info)
        forecast = np.array(
            [
                self.schedule[(hour_idx + 1 + i) % len(self.schedule)]
                for i in range(self.horizon)
            ],
            dtype=np.float32,
        )
        if self._err_factor is not None:
            forecast = np.clip(forecast * self._err_factor[hour_idx], 0.0, 1.0)
        return np.concatenate([obs, forecast]).astype(np.float32)

    def _get_hour_of_year(self, info: dict) -> int:
        month = int(info.get("month", 1))
        day = int(info.get("day", 1))
        # EnergyPlus DataExchange.hour() returns 0..23. Old code subtracted 1
        # under a 1..24 assumption, causing a 1-hour lag in forecast lookup.
        # Aligned with PriceForecastWrapper / TimeVaryingCostLinearReward (2026-06-01).
        hour = int(info.get("hour", 0)) % 24
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_of_year = sum(days_in_month[:month]) + day - 1
        return (day_of_year * 24 + hour) % len(self.schedule)


class ComfortViolationWrapper(gym.Wrapper):
    """Computes a scalar comfort violation metric from zone temperatures.

    Adds 'comfort_violation' = (1/N) * Σ [max(0, T - T_max)² + max(0, T_min - T)²]
    to the observation. This is building-invariant (normalized by zone count)
    and directly corresponds to the comfort penalty term in the reward (eq.14).
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER ComfortViolationWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        temperature_variables: List[str],
        t_min: float = 23.0,
        t_max: float = 26.0,
        occupied_hours: Optional[List[int]] = None,
        exclude_days: Optional[List[int]] = None,
        simulation_year: int = 1991,
    ):
        super().__init__(env)
        self.t_min = t_min
        self.t_max = t_max
        self.occupied_hours = occupied_hours  # [start, end] e.g. [8, 18], None=24h
        self.exclude_days = set(
            exclude_days or []
        )  # Mon=0..Sun=6; e.g. [6]=skip Sunday
        self.simulation_year = int(simulation_year)

        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)

        self.temp_indices = []
        for var in temperature_variables:
            if var not in obs_vars:
                raise ValueError(
                    f"ComfortViolationWrapper: '{var}' not in observation_variables"
                )
            self.temp_indices.append(obs_vars.index(var))
        self.n_zones = len(temperature_variables)

        self.observation_variables = list(obs_vars) + ["comfort_violation"]
        old_space = env.observation_space
        new_low = np.concatenate([old_space.low, [0.0]])
        new_high = np.concatenate([old_space.high, [np.inf]])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: comfort_violation from {len(temperature_variables)} zones, "
            f"comfort band [{t_min}, {t_max}]°C"
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        # Check occupied hours
        if self.occupied_hours is not None and info is not None:
            # EnergyPlus DataExchange.hour() returns 0..23. Old code subtracted 1
            # under a 1..24 assumption, causing a 1-hour lag in occupied-hours check
            # (e.g. occupied_hours [8,18] effectively applied to 7-17h).
            # Aligned with PriceForecastWrapper / TimeVaryingCostLinearReward (2026-06-01).
            hour_idx = int(info.get("hour", 0)) % 24
            start_h, end_h = self.occupied_hours
            if not (start_h <= hour_idx < end_h):
                return np.concatenate([obs, [np.float32(0.0)]]).astype(np.float32)

        # Day-of-week exclusion (Mon=0..Sun=6)
        if self.exclude_days and info is not None:
            month = int(info.get("month", 1))
            day = int(info.get("day_of_month", info.get("day", 1)))
            try:
                dow = datetime.date(self.simulation_year, month, day).weekday()
                if dow in self.exclude_days:
                    return np.concatenate([obs, [np.float32(0.0)]]).astype(np.float32)
            except ValueError:
                pass

        violation = 0.0
        for idx in self.temp_indices:
            t = obs[idx]
            violation += max(0.0, t - self.t_max) ** 2 + max(0.0, self.t_min - t) ** 2
        violation /= self.n_zones
        return np.concatenate([obs, [np.float32(violation)]]).astype(np.float32)


class ChillerStatusWrapper(gym.Wrapper):
    """Converts ChillerElectricity (continuous W) to binary chiller_status (δ).

    Adds 'chiller_status' (1.0 if chiller on, 0.0 if off) to observations.
    The original ChillerElectricity should be removed via ReduceObservationWrapper
    so only the binary signal reaches the RL agent. ChillerElectricity remains
    available in the info dict for reward computation.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER ChillerStatusWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        source_variable: str = "ChillerElectricity",
        threshold: float = 0.0,
    ):
        super().__init__(env)
        self.threshold = threshold

        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)
        if source_variable not in obs_vars:
            raise ValueError(
                f"ChillerStatusWrapper: '{source_variable}' not in observation_variables"
            )
        self.source_idx = obs_vars.index(source_variable)

        self.observation_variables = list(obs_vars) + ["chiller_status"]
        old_space = env.observation_space
        new_low = np.concatenate([old_space.low, [0.0]])
        new_high = np.concatenate([old_space.high, [1.0]])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: binary chiller_status from {source_variable}"
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        status = np.float32(1.0 if obs[self.source_idx] > self.threshold else 0.0)
        return np.concatenate([obs, [status]]).astype(np.float32)


class PriceForecastWrapper(gym.Wrapper):
    logger = TerminalLogger().getLogger(
        name="WRAPPER PriceForecastWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, price_schedule: List[float], horizon: int = 24):
        super().__init__(env)
        if len(price_schedule) != 24:
            raise ValueError(
                f"price_schedule must have 24 hourly values, got {len(price_schedule)}"
            )
        self.price_schedule = list(price_schedule)
        self.horizon = horizon

        old_var_names = env.get_wrapper_attr("observation_variables")
        if isinstance(old_var_names, tuple):
            old_var_names = list(old_var_names)
        self.observation_variables = (
            old_var_names
            + ["current_price"]
            + [f"price_forecast_{i}" for i in range(1, horizon + 1)]
        )

        old_space = env.observation_space
        extra_low = np.zeros((horizon + 1,), dtype=np.float32)
        extra_high = np.full((horizon + 1,), np.inf, dtype=np.float32)
        new_low = np.concatenate([old_space.low, extra_low])
        new_high = np.concatenate([old_space.high, extra_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: current_price + {horizon}-step price forecast"
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray, info: dict) -> np.ndarray:
        hour_idx = self._get_hour_idx(info)
        current = np.float32(self.price_schedule[hour_idx])
        forecast = np.array(
            [self.price_schedule[(hour_idx + 1 + i) % 24] for i in range(self.horizon)],
            dtype=np.float32,
        )
        return np.concatenate([obs, [current], forecast]).astype(np.float32)

    def _get_hour_idx(self, info: dict) -> int:
        # EnergyPlus DataExchange.hour() returns 0..23 (current hour of day).
        # Previous code subtracted 1 under the assumption of 1..24 indexing,
        # which caused a 1-hour lag (e.g. step at 12:00-12:59 used price[11]).
        return int(info.get("hour", 0)) % 24


class LoadPredictorWrapper(gym.Wrapper):
    """Wrapper that runs LSTM load predictor on building-side observations.

    Replaces CSVFeatureWrapper for CoolingRate by predicting the 24-hour
    load trajectory from building-side obs instead of looking up CSV data.
    Corresponds to g_θ(o1_t) in the paper.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER LoadPredictorWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        model_path: str | Path,
        input_variables: Optional[List[str]] = None,
        seq_len: int = 24,
        horizon: int = 24,
        constant_load: Optional[float] = None,
    ):
        super().__init__(env)
        from .load_predictor import load_predictor

        self._lstm_model, self.checkpoint = load_predictor(str(model_path))
        self.seq_len = seq_len
        self.horizon = horizon
        # Knockout modes (interface ablation): replace the LSTM output with
        # a constant (float, W — e.g. the target's season-mean load) or with
        # "persistence" (current CoolingRate repeated over the horizon: level
        # stays correct, future information is removed). The LSTM is still
        # loaded so input plumbing stays identical.
        self._constant_load = constant_load
        self._persist_idx: Optional[int] = None
        if constant_load == "persistence":
            obs_vars_all = env.get_wrapper_attr("observation_variables")
            self._persist_idx = list(obs_vars_all).index("CoolingRate")

        # Use input_variables from checkpoint if not explicitly provided
        if not input_variables and "input_variables" in self.checkpoint:
            input_variables = list(self.checkpoint["input_variables"])

        if not input_variables:
            raise ValueError(
                "LoadPredictorWrapper: input_variables must be provided or saved in checkpoint"
            )

        # Resolve input variable indices in the current observation space
        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)
        self.input_indices = []
        for var in input_variables:
            if var not in obs_vars:
                raise ValueError(
                    f"LoadPredictorWrapper: input variable '{var}' not in observation_variables"
                )
            self.input_indices.append(obs_vars.index(var))
        self.input_indices = np.array(self.input_indices)

        # Normalization stats from training
        self.input_mean = torch.from_numpy(self.checkpoint["input_mean"]).float()
        self.input_std = torch.from_numpy(self.checkpoint["input_std"]).float()
        self.target_mean = float(self.checkpoint["target_mean"])
        self.target_std = float(self.checkpoint["target_std"])

        # Observation history buffer
        self._history: deque = deque(maxlen=seq_len)

        # Extend observation space
        self.observation_variables = list(obs_vars) + [
            f"load_forecast_{i}" for i in range(1, horizon + 1)
        ]
        old_space = env.observation_space
        forecast_low = np.full((horizon,), -np.inf, dtype=np.float32)
        forecast_high = np.full((horizon,), np.inf, dtype=np.float32)
        new_low = np.concatenate([old_space.low, forecast_low])
        new_high = np.concatenate([old_space.high, forecast_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: LSTM load predictor from {model_path} "
            f"({len(input_variables)} inputs, seq_len={seq_len}, horizon={horizon})"
        )
        if self._constant_load is not None:
            self.logger.info(
                f"KNOCKOUT: load forecast replaced by {self._constant_load!r} "
                f"(LSTM output ignored)"
            )

    def reset(self, seed=None, options=None):
        self._history.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_history(obs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_history(obs)
        return self._augment(obs), reward, terminated, truncated, info

    def _update_history(self, obs: np.ndarray):
        features = obs[self.input_indices].astype(np.float32)
        self._history.append(features)

    @torch.no_grad()
    def _augment(self, obs: np.ndarray) -> np.ndarray:
        if self._persist_idx is not None:
            pred = np.full(
                (self.horizon,), float(obs[self._persist_idx]), dtype=np.float32
            )
            return np.concatenate([obs, pred]).astype(np.float32)
        if self._constant_load is not None:
            pred = np.full((self.horizon,), self._constant_load, dtype=np.float32)
            return np.concatenate([obs, pred]).astype(np.float32)
        # Build input sequence with zero-padding if history < seq_len
        if len(self._history) < self.seq_len:
            pad_len = self.seq_len - len(self._history)
            seq = np.zeros((self.seq_len, len(self.input_indices)), dtype=np.float32)
            for i, h in enumerate(self._history):
                seq[pad_len + i] = h
        else:
            seq = np.array(list(self._history), dtype=np.float32)

        # Normalize input
        x = torch.from_numpy(seq).unsqueeze(0)  # (1, seq_len, input_dim)
        x = (x - self.input_mean) / self.input_std

        # LSTM inference
        pred_norm = self._lstm_model(x).squeeze(0).numpy()  # (checkpoint horizon,)
        pred = pred_norm * self.target_std + self.target_mean

        # Truncate to the wrapper's declared horizon: checkpoints trained with
        # a longer horizon (e.g. the CLI's default 24 under horizon-12
        # profiles) would otherwise append UNNAMED channels after the declared
        # load_forecast_* names, silently shifting every downstream CSV column
        # (root cause of the historical observations.csv scrambling).
        if pred.shape[0] < self.horizon:
            raise ValueError(
                f"Predictor horizon {pred.shape[0]} < wrapper horizon {self.horizon}"
            )
        pred = pred[: self.horizon]
        # Physical floor (cooling load >= 0) AFTER denormalisation — the
        # documented 2026-06-04 fix, re-instated (never clip in normalised
        # space; see lstm-clamp-bug memory).
        pred = np.clip(pred, 0.0, None)

        return np.concatenate([obs, pred]).astype(np.float32)


class DayOfWeekWrapper(gym.Wrapper):
    """Append `dow_cos` and `dow_sin` to observation based on current sim date.

    Rationale: sinergym's DatetimeWrapper emits only `is_weekend` (binary), which
    collapses Saturday/Sunday. Per-building BLDG_OCC_SCH shows 2-6x occupancy
    differences between Sat and Sun, so an LSTM relying on is_weekend alone
    cannot distinguish them. This wrapper computes the weekday (Mon=0..Sun=6)
    from (simulation_year, month, day_of_month) and emits a sin/cos cyclic
    encoding so the downstream network sees a 7-class distinction.

    NOTE: must run BEFORE DatetimeWrapper since DatetimeWrapper replaces the
    `day_of_month` raw time variable with `is_weekend`.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER DayOfWeekWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, simulation_year: int = 1991):
        super().__init__(env)
        self.simulation_year = int(simulation_year)
        obs_vars = list(env.get_wrapper_attr("observation_variables"))
        if "month" not in obs_vars or "day_of_month" not in obs_vars:
            raise ValueError(
                "DayOfWeekWrapper requires 'month' and 'day_of_month' in observation — "
                "place this wrapper BEFORE DatetimeWrapper."
            )
        self.month_idx = obs_vars.index("month")
        self.day_idx = obs_vars.index("day_of_month")
        self.observation_variables = obs_vars + ["dow_cos", "dow_sin"]
        old = env.observation_space
        new_low = np.concatenate([old.low, np.array([-1.0, -1.0], dtype=np.float32)])
        new_high = np.concatenate([old.high, np.array([1.0, 1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: dow features from year={self.simulation_year}"
        )

    def _dow_features(self, month: float, day: float) -> tuple[float, float]:
        weekday = datetime.date(
            self.simulation_year, max(1, int(month)), max(1, int(day))
        ).weekday()
        angle = 2 * np.pi * weekday / 7
        return float(np.cos(angle)), float(np.sin(angle))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        dow_cos, dow_sin = self._dow_features(obs[self.month_idx], obs[self.day_idx])
        return np.concatenate([obs, [dow_cos, dow_sin]]).astype(np.float32)


class OracleLoadForecastWrapper(gym.Wrapper):
    """Ground-truth load_forecast from dataset CSV (no LSTM).

    Diagnostic wrapper that replaces LoadPredictorWrapper with an oracle lookup
    of future CoolingRate values collected during dataset building. Used to
    measure the upper bound of Feature TL performance if LSTM prediction were
    perfect. Matches the output shape of LoadPredictorWrapper
    (load_forecast_1..horizon).

    Caveat: the CSV trajectory reflects the policy that collected the dataset,
    not the current agent's actions. So the "oracle" approximates macro load
    patterns (daily cycle + weather response) but is not a true oracle under
    different control decisions.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER OracleLoadForecastWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, csv_path: str | Path, horizon: int = 12):
        super().__init__(env)
        self.horizon = horizon

        df = pd.read_csv(csv_path)
        n_samples = horizon * 4  # 4 timesteps per hour
        cr_cols = [f"CoolingRate{i}" for i in range(1, n_samples + 1)]
        missing = [c for c in cr_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"OracleLoadForecastWrapper: dataset missing {missing[:3]}..."
            )
        cr_raw = df[cr_cols].values.astype(np.float32)
        # Hourly average: (N, horizon*4) -> (N, horizon, 4) -> (N, horizon)
        self.future_loads = cr_raw.reshape(len(cr_raw), horizon, 4).mean(axis=2)
        self._fallback = np.zeros(horizon, dtype=np.float32)
        self._step_counter = 0

        obs_vars = env.get_wrapper_attr("observation_variables")
        if isinstance(obs_vars, tuple):
            obs_vars = list(obs_vars)
        self.observation_variables = list(obs_vars) + [
            f"load_forecast_{i}" for i in range(1, horizon + 1)
        ]
        old_space = env.observation_space
        forecast_low = np.full((horizon,), -np.inf, dtype=np.float32)
        forecast_high = np.full((horizon,), np.inf, dtype=np.float32)
        new_low = np.concatenate([old_space.low, forecast_low])
        new_high = np.concatenate([old_space.high, forecast_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: oracle load forecast from {csv_path} "
            f"({len(self.future_loads)} rows, horizon={horizon})"
        )

    def reset(self, seed=None, options=None):
        self._step_counter = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_counter += 1
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        if self._step_counter < len(self.future_loads):
            future = self.future_loads[self._step_counter]
        else:
            future = self._fallback
        return np.concatenate([obs, future]).astype(np.float32)


class CSVFeatureWrapper(gym.Wrapper):
    logger = TerminalLogger().getLogger(
        name="WRAPPER CSVFeatureWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, csv_path: str, columns: List[str]):
        super().__init__(env)
        self.csv_path = csv_path
        self.columns = columns
        try:
            self.df = pd.read_csv(csv_path)
            self.features_data = self.df[columns].values.astype(np.float32)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc
        except KeyError as exc:
            raise KeyError(f"CSV file missing expected columns: {exc}") from exc

        old_var_names = env.get_wrapper_attr("observation_variables")
        if isinstance(old_var_names, tuple):
            old_var_names = list(old_var_names)
        self.observation_variables = old_var_names + columns

        old_space = env.observation_space
        feature_low = np.full((len(columns),), -np.inf, dtype=np.float32)
        feature_high = np.full((len(columns),), np.inf, dtype=np.float32)
        new_low = np.concatenate([old_space.low, feature_low])
        new_high = np.concatenate([old_space.high, feature_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.current_step_idx = 0
        self.logger.info(f"Wrapper initialized with features from: {csv_path}")

    def reset(self, seed=None, options=None):
        self.current_step_idx = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step_idx += 1
        return self._augment_observation(obs), reward, terminated, truncated, info

    def _augment_observation(self, obs):
        if len(self.features_data) == 0:
            return np.asarray(obs, dtype=np.float32)
        idx = min(self.current_step_idx, len(self.features_data) - 1)
        features = self.features_data[idx]
        return np.concatenate([obs, features]).astype(np.float32)


class PhysicalNormalizeObservation(gym.ObservationWrapper):
    """Min-max normalization using physics-based reference values.

    Instead of z-score running statistics, uses fixed (min, max) bounds
    derived from physical system parameters. This makes the normalization
    deterministic and building-invariant for shared variables, while
    building-specific references (e.g. Q_rated for load) align the scale.

    Args:
        env: The environment to wrap.
        variable_bounds: Maps variable name or prefix to (min, max).
            Prefix matching is supported: 'load_forecast' matches
            'load_forecast_1' through 'load_forecast_24'.
            Priority: exact match > longest prefix match.
        epsilon: Stability constant to avoid division by zero.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER PhysicalNormalizeObservation",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        variable_bounds: dict[str, tuple[float, float]],
        epsilon: float = 1e-8,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.unwrapped_observation: Optional[np.ndarray] = None

        obs_vars = list(env.get_wrapper_attr("observation_variables"))
        n = len(obs_vars)
        self._min = np.zeros(n, dtype=np.float32)
        self._max = np.ones(n, dtype=np.float32)

        # Sort bound keys by length (longest first) for prefix matching
        sorted_keys = sorted(variable_bounds.keys(), key=len, reverse=True)

        for i, var_name in enumerate(obs_vars):
            # Exact match first
            if var_name in variable_bounds:
                lo, hi = variable_bounds[var_name]
                self._min[i] = lo
                self._max[i] = hi
                continue
            # Longest prefix match
            matched = False
            for key in sorted_keys:
                if var_name.startswith(key):
                    lo, hi = variable_bounds[key]
                    self._min[i] = lo
                    self._max[i] = hi
                    matched = True
                    break
            if not matched:
                # Suffix matching for patterns like *_temperature
                for key in sorted_keys:
                    if key.startswith("*") and var_name.endswith(key[1:]):
                        lo, hi = variable_bounds[key]
                        self._min[i] = lo
                        self._max[i] = hi
                        matched = True
                        break
            if not matched:
                self.logger.warning(
                    f"No bounds for '{var_name}', defaulting to [0, 1] pass-through"
                )

        self._range = self._max - self._min
        self.logger.info(f"Wrapper initialized: physical min-max for {n} variables")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self.unwrapped_observation = obs.copy()
        return ((obs - self._min) / (self._range + self.epsilon)).astype(np.float32)


class ZoneAggregationWrapper(gym.ObservationWrapper):
    """Replace per-zone observations with building-invariant aggregate statistics.

    For cross-building transfer (different zone counts between source/target),
    raw per-zone features ('Z1_temperature', 'Dining_temperature', etc.) are not
    directly transferable because zone slot 1 in source has different semantics
    than in target. This wrapper aggregates zone-level observations to
    fixed-dim statistics (max, min, mean) so all buildings produce the same
    observation shape and each aggregate has consistent semantics across buildings.

    Aggregates produced (building-invariant):
        zone_temperature_max, zone_temperature_min, zone_temperature_mean
        zone_setpoint_mean  (single representative setpoint)

    The individual zone variables listed in `zone_temperature_vars` and
    `zone_setpoint_vars` are REMOVED from the observation; aggregates are
    appended at the end.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER ZoneAggregationWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        zone_temperature_vars: List[str],
        zone_setpoint_vars: List[str],
    ):
        super().__init__(env)
        obs_vars = list(env.get_wrapper_attr("observation_variables"))

        for v in zone_temperature_vars + zone_setpoint_vars:
            if v not in obs_vars:
                raise ValueError(
                    f"ZoneAggregationWrapper: '{v}' not in observation_variables"
                )

        self._temp_idx = np.array([obs_vars.index(v) for v in zone_temperature_vars])
        self._setp_idx = np.array([obs_vars.index(v) for v in zone_setpoint_vars])
        drop = set(zone_temperature_vars + zone_setpoint_vars)
        self._keep_idx = np.array([i for i, v in enumerate(obs_vars) if v not in drop])
        keep_vars = [obs_vars[i] for i in self._keep_idx]
        agg_vars = [
            "zone_temperature_max",
            "zone_temperature_min",
            "zone_temperature_mean",
            "zone_setpoint_mean",
        ]
        self.observation_variables = keep_vars + agg_vars

        old = env.observation_space
        kept_low = old.low[self._keep_idx]
        kept_high = old.high[self._keep_idx]
        agg_low = np.array([15.0, 15.0, 15.0, 20.0], dtype=np.float32)
        agg_high = np.array([40.0, 40.0, 40.0, 30.0], dtype=np.float32)
        new_low = np.concatenate([kept_low, agg_low])
        new_high = np.concatenate([kept_high, agg_high])
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        self.logger.info(
            f"Wrapper initialized: {len(zone_temperature_vars)} zone temps + "
            f"{len(zone_setpoint_vars)} setpoints aggregated to 4 fixed features"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        temps = obs[self._temp_idx]
        setps = obs[self._setp_idx]
        kept = obs[self._keep_idx]
        agg = np.array(
            [temps.max(), temps.min(), temps.mean(), setps.mean()],
            dtype=np.float32,
        )
        return np.concatenate([kept, agg]).astype(np.float32)


class ZonePaddingWrapper(gym.ObservationWrapper):
    """Pad zone-level observations to a fixed number of slots for transfer.

    Produces building-invariant observation shape by padding per-zone features
    (zone_temperature_*, zone_setpoint_*) to `max_zones` slots. Enables
    transfer of ALL network layers (input + hidden) across buildings with
    different zone counts, per Kadamala et al. 2025 (Energy & AI).

    Padding modes:
        "zero":       missing zone slots filled with 0.0 (may go out of
                      normalization range → noisy learning).
        "duplicate":  each zone value duplicated to neighbouring slots via
                      np.repeat with even distribution. Keeps values in valid
                      range, matches source distribution. Paper's technique.

    Input zone variables are REMOVED from obs; padded slots appended:
        zone_slot_1_temperature, ..., zone_slot_N_temperature
        (optional) zone_slot_1_setpoint, ..., zone_slot_N_setpoint

    Setpoint padding is optional: if `zone_setpoint_vars` is empty or None,
    no setpoint slots are added (setpoints typically follow deterministic
    schedules with few unique values and do not aid control decisions).
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER ZonePaddingWrapper",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(
        self,
        env: gym.Env,
        zone_temperature_vars: List[str],
        zone_setpoint_vars: Optional[List[str]] = None,
        max_zones: int = 10,
        padding_mode: str = "duplicate",
    ):
        super().__init__(env)
        if padding_mode not in ("zero", "duplicate"):
            raise ValueError(
                f"ZonePaddingWrapper: unknown padding_mode '{padding_mode}'"
            )
        self._max_zones = int(max_zones)
        self._padding_mode = padding_mode
        zone_setpoint_vars = list(zone_setpoint_vars) if zone_setpoint_vars else []
        self._has_setpoints = len(zone_setpoint_vars) > 0

        obs_vars = list(env.get_wrapper_attr("observation_variables"))
        for v in zone_temperature_vars + zone_setpoint_vars:
            if v not in obs_vars:
                raise ValueError(
                    f"ZonePaddingWrapper: '{v}' not in observation_variables"
                )
        if len(zone_temperature_vars) > self._max_zones:
            raise ValueError(
                f"ZonePaddingWrapper: {len(zone_temperature_vars)} temp vars "
                f"exceeds max_zones={self._max_zones}"
            )
        if len(zone_setpoint_vars) > self._max_zones:
            raise ValueError(
                f"ZonePaddingWrapper: {len(zone_setpoint_vars)} setpoint vars "
                f"exceeds max_zones={self._max_zones}"
            )

        self._temp_idx = np.array(
            [obs_vars.index(v) for v in zone_temperature_vars], dtype=np.int64
        )
        self._setp_idx = np.array(
            [obs_vars.index(v) for v in zone_setpoint_vars], dtype=np.int64
        )
        drop = set(zone_temperature_vars + zone_setpoint_vars)
        self._keep_idx = np.array(
            [i for i, v in enumerate(obs_vars) if v not in drop], dtype=np.int64
        )
        keep_vars = [obs_vars[i] for i in self._keep_idx]
        padded_temp_names = [
            f"zone_slot_{i + 1}_temperature" for i in range(self._max_zones)
        ]
        padded_setp_names = (
            [f"zone_slot_{i + 1}_setpoint" for i in range(self._max_zones)]
            if self._has_setpoints
            else []
        )
        self.observation_variables = keep_vars + padded_temp_names + padded_setp_names

        old = env.observation_space
        kept_low = old.low[self._keep_idx]
        kept_high = old.high[self._keep_idx]
        temp_low = np.full(self._max_zones, 15.0, dtype=np.float32)
        temp_high = np.full(self._max_zones, 40.0, dtype=np.float32)
        low_pieces = [kept_low, temp_low]
        high_pieces = [kept_high, temp_high]
        if self._has_setpoints:
            setp_low = np.full(self._max_zones, 20.0, dtype=np.float32)
            setp_high = np.full(self._max_zones, 30.0, dtype=np.float32)
            low_pieces.append(setp_low)
            high_pieces.append(setp_high)
        new_low = np.concatenate(low_pieces)
        new_high = np.concatenate(high_pieces)
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )
        setpoint_desc = (
            f" + {len(zone_setpoint_vars)} setpoints"
            if self._has_setpoints
            else " (setpoints excluded)"
        )
        self.logger.info(
            f"Wrapper initialized: {len(zone_temperature_vars)} temps"
            f"{setpoint_desc} padded to {self._max_zones} slots via "
            f"'{padding_mode}' padding"
        )

    def _pad(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n == self._max_zones:
            return values
        if n > self._max_zones:
            return values[: self._max_zones]
        if self._padding_mode == "zero":
            pad = np.zeros(self._max_zones - n, dtype=values.dtype)
            return np.concatenate([values, pad])
        # Duplication (neighbour) padding (Kadamala 2025, Energy & AI):
        # Each feature is duplicated to its neighbouring positions.
        # Repetitions distributed evenly so slot_i ≈ proportional zone position
        # across buildings with different zone counts. Remainder distributed to
        # the earliest features so that every feature is duplicated at least once.
        #
        #   n=2, max=10  → repeats=[5,5] → [A,A,A,A,A,B,B,B,B,B]
        #   n=5, max=10  → repeats=[2,2,2,2,2] → [A,A,B,B,C,C,D,D,E,E]
        #   n=8, max=10  → repeats=[2,2,1,1,1,1,1,1] → [A,A,B,B,C,D,E,F,G,H]
        repeats = np.full(n, self._max_zones // n, dtype=np.int64)
        remainder = self._max_zones - int(repeats.sum())
        if remainder > 0:
            repeats[:remainder] += 1
        return np.repeat(values, repeats)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        temps = obs[self._temp_idx]
        kept = obs[self._keep_idx]
        padded_temps = self._pad(temps).astype(np.float32)
        pieces = [kept, padded_temps]
        if self._has_setpoints:
            setps = obs[self._setp_idx]
            padded_setps = self._pad(setps).astype(np.float32)
            pieces.append(padded_setps)
        return np.concatenate(pieces).astype(np.float32)


class PhysicalNormalizeReward(gym.Wrapper):
    """Normalizes reward by dividing by a physics-based reference value.

    The reference should be chosen so that the maximum expected reward
    magnitude per timestep is approximately 1.0. For HVAC-TES control:
    R_ref = p_peak * Q_rated / COP_rated (max energy cost per timestep).

    Args:
        env: The environment to wrap.
        reward_ref: The reference reward magnitude for normalization.
    """

    logger = TerminalLogger().getLogger(
        name="WRAPPER PhysicalNormalizeReward",
        level=LOG_WRAPPERS_LEVEL,
    )

    def __init__(self, env: gym.Env, reward_ref: float):
        super().__init__(env)
        if reward_ref <= 0:
            raise ValueError(f"reward_ref must be positive, got {reward_ref}")
        self.reward_ref = reward_ref
        self.logger.info(f"Wrapper initialized: reward / {reward_ref:.2f}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = float(reward)
        return obs, float(reward) / self.reward_ref, terminated, truncated, info


def resolve_latest_workspace(results_root: str | Path, keyword: str) -> Optional[Path]:
    results_root = Path(results_root)

    def score(ws: Path) -> float:
        p = ws / "progress.csv"
        return p.stat().st_mtime if p.exists() else ws.stat().st_mtime

    obs_files = list(
        results_root.rglob(f"*{keyword}*/episode-*/monitor/observations.csv")
    )
    workspaces = {f.parents[2].resolve() for f in obs_files}
    if not workspaces:
        for directory in results_root.rglob(f"*{keyword}*"):
            if directory.is_dir() and (directory / "progress.csv").exists():
                workspaces.add(directory.resolve())
    if not workspaces:
        return None
    return max(workspaces, key=score)
