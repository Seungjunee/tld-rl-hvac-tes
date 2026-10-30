from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import (
    CSVLogger,
    DatetimeWrapper,
    DiscretizeEnv,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
    ReduceObservationWrapper,
    WeatherForecastingWrapper,
)

from .rewards import TimeVaryingCostLinearReward
from .types import ResolvedCase, RunContext
from .wrappers import (
    ChillerStatusWrapper,
    ComfortViolationWrapper,
    CPULoadForecastWrapper,
    CSVFeatureWrapper,
    DayOfWeekWrapper,
    LoadPredictorWrapper,
    OracleLoadForecastWrapper,
    PhysicalNormalizeObservation,
    PhysicalNormalizeReward,
    PriceForecastWrapper,
    SetpointForecastWrapper,
    ZonePaddingWrapper,
)


def _resolve_path(value: Any, project_root: Path) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)


def _reward_class(name: str):
    if name == "TimeVaryingCostLinearReward":
        return TimeVaryingCostLinearReward
    raise ValueError(f"Unsupported reward class: {name}")


def _resolve_mode_wrappers(case: ResolvedCase, mode: str) -> Dict[str, Any]:
    wrappers = dict(case.wrappers)
    mode_wrappers = wrappers.get(mode, {})
    merged = dict(wrappers)
    if isinstance(mode_wrappers, dict):
        merged.update(mode_wrappers)
    merged.pop("train", None)
    merged.pop("eval", None)
    return merged


def _resolve_feature_columns(cfg: Dict[str, Any]) -> list[str]:
    if cfg.get("columns"):
        return list(cfg["columns"])
    columns = []
    for group in cfg.get("column_groups", []):
        prefix = group["prefix"]
        start = int(group.get("start", 1))
        end = int(group["end"])
        columns.extend([f"{prefix}{i}" for i in range(start, end + 1)])
    return columns


# Universal physical bounds (building-invariant)
_UNIVERSAL_BOUNDS: Dict[str, tuple[float, float]] = {
    # TES tank temperature -- action range [7, 12] + deadband=1.0; mixing/standing loss
    # can push tank slightly outside (~4–15°C). Widen for clean normalization.
    "CWtank_final": (4.0, 15.0),
    # Stratified per-node temps (Node 1=top warm ~ Node 6=bottom cold, end-of-timestep)
    "CWtank_node": (4.0, 15.0),
    # Use side outlet (cold supply temp leaving tank to load coils)
    "CWtank_use_outlet": (4.0, 15.0),
    # Extra tank-loop temps (V4/V5t state-ablation): use-side inlet = warm coil
    # return; source-side = chiller charge loop. Warm sides widened to 20°C.
    "CWtank_use_inlet": (4.0, 20.0),
    "CWtank_source_inlet": (4.0, 15.0),
    "CWtank_source_outlet": (4.0, 20.0),
    # Binary chiller on/off status
    "chiller_status": (0.0, 1.0),
    # Zone-averaged squared comfort deviation; ref = (T_max - T_min)^2
    "comfort_violation": (0.0, 9.0),
    # TOU electricity price ($/kWh) — OEB 2x peak amplification: 0.098/0.157/0.406
    "price_forecast": (0.098, 0.406),
    "current_price": (0.098, 0.406),
    # Time encodings from DatetimeWrapper
    "month_cos": (-1.0, 1.0),
    "month_sin": (-1.0, 1.0),
    "hour_cos": (-1.0, 1.0),
    "hour_sin": (-1.0, 1.0),
    "is_weekend": (0.0, 1.0),
    "dow_cos": (-1.0, 1.0),
    "dow_sin": (-1.0, 1.0),
    # EnergyPlus minute-of-step occasionally exceeds 60 due to floor/ceil bookkeeping.
    "minutes": (0.0, 75.0),
    # Weather
    "outdoor_temperature": (-10.0, 50.0),
    "outdoor_humidity": (0.0, 100.0),
    "weather_forecast": (-10.0, 50.0),
    # CPU load (datacenter-specific)
    "cpu_loading_fraction": (0.0, 1.0),
    "cpu_load_forecast": (0.0, 1.0),
    # Building occupancy fraction from BLDG_OCC_SCH (LBuilding: People_office_Sch)
    "occupancy_fraction": (0.0, 1.0),
}

# Suffix-based bounds for zone-level variables (*_temperature, *_setpoint)
_SUFFIX_BOUNDS: Dict[str, tuple[float, float]] = {
    # Zone temp: lower 14.0 to cover winter design day dip (~14.7°C observed).
    "*_temperature": (14.0, 40.0),
    "*_zone_temperature": (14.0, 40.0),
    "*H_setpoint": (15.0, 25.0),
    "*_heating_setpoint": (15.0, 25.0),
    # Cooling SP: upper 35.0 covers weekday-only convention (weekend=33°C unoccupied).
    "*C_setpoint": (20.0, 35.0),
    "*_cooling_setpoint": (20.0, 35.0),
    "*_setpoint": (20.0, 35.0),  # matches padded zone_slot_<i>_setpoint
    # Weather forecast columns (forecast_<i>_<column> names from WeatherForecastingWrapper)
    "*_Dry Bulb Temperature": (-10.0, 50.0),
    "*_Relative Humidity": (0.0, 100.0),
}

# Peak-price reward-normalization reference ($/kWh). Historical OEB-2x value,
# retained above the active SMUD peak (0.3765) as a fixed headroom constant:
# it is uniform across buildings, so it cancels in every savings ratio, and
# changing it would invalidate all trained policies (reward rescale). The
# price OBSERVATION bounds (0.098, 0.406) share the same headroom convention.
_P_PEAK = 0.406


def _build_variable_bounds(physical: Dict[str, Any]) -> Dict[str, tuple[float, float]]:
    """Build variable_bounds dict from universal constants + building physical params."""
    q_rated = float(physical["q_rated"])
    cop = float(physical["cop_rated"])

    bounds: Dict[str, tuple[float, float]] = dict(_UNIVERSAL_BOUNDS)
    bounds.update(_SUFFIX_BOUNDS)

    # Building-specific: load normalized by Q_rated (PLR)
    bounds["load_forecast"] = (0.0, q_rated)
    # Building-specific: raw energy/cooling variables
    bounds["CoolingRate"] = (0.0, q_rated)
    bounds["ChillerElectricity"] = (0.0, q_rated / cop)

    return bounds


def _compute_reward_ref(physical: Dict[str, Any]) -> float:
    """Compute reward normalization reference: p_peak * Q_rated / COP.

    Note: /100 reward boost tested Jun 02 — Office cherry % degraded -2.35pp,
    LB -1.28pp, WH +0.78pp (mixed). Reverted to original scaling because
    ent_coef auto rebalance was not the bottleneck for non-LB buildings.
    """
    q_rated = float(physical["q_rated"])
    cop = float(physical["cop_rated"])
    return _P_PEAK * q_rated / cop


def apply_wrappers(
    env: gym.Env, wrappers_cfg: Dict[str, Any], project_root: Path
) -> gym.Env:
    order = wrappers_cfg.get(
        "order",
        [
            "day_of_week",
            "datetime",
            "chiller_status",
            "comfort_violation",
            "weather_forecast",
            "setpoint_forecast",
            "cpu_load_forecast",
            "price_forecast",
            "load_predictor",
            "oracle_load_forecast",
            "csv_features",
            "zone_padding",
            "logger",
            "csv_logger",
            "reduce_observation",
            "physical_normalize_observation",
            "normalize_observation",
            "physical_normalize_reward",
            "normalize_action",
        ],
    )
    for name in order:
        if name == "day_of_week" and wrappers_cfg.get("day_of_week"):
            cfg = wrappers_cfg["day_of_week"]
            if isinstance(cfg, bool):
                cfg = {}
            env = DayOfWeekWrapper(
                env, simulation_year=int(cfg.get("simulation_year", 1991))
            )
        elif name == "datetime" and wrappers_cfg.get("datetime", False):
            env = DatetimeWrapper(env)
        elif name == "chiller_status" and wrappers_cfg.get("chiller_status", False):
            env = ChillerStatusWrapper(env)
        elif name == "comfort_violation" and wrappers_cfg.get("comfort_violation"):
            cfg = wrappers_cfg["comfort_violation"]
            if isinstance(cfg, bool):
                cfg = {}
            env = ComfortViolationWrapper(
                env,
                temperature_variables=cfg["temperature_variables"],
                t_min=cfg.get("t_min", 23.0),
                t_max=cfg.get("t_max", 26.0),
                occupied_hours=cfg.get("occupied_hours"),
                exclude_days=cfg.get("exclude_days"),
                simulation_year=cfg.get("simulation_year", 1991),
            )
        elif name == "weather_forecast" and wrappers_cfg.get("weather_forecast"):
            cfg = wrappers_cfg["weather_forecast"]
            if isinstance(cfg, bool):
                cfg = {}
            env = WeatherForecastingWrapper(
                env,
                n=cfg.get("horizon", 24),
                delta=cfg.get("delta", 1),
                columns=cfg.get("columns", ["Dry Bulb Temperature"]),
            )
        elif name == "setpoint_forecast" and wrappers_cfg.get("setpoint_forecast"):
            cfg = wrappers_cfg["setpoint_forecast"]
            if isinstance(cfg, bool):
                cfg = {}
            env = SetpointForecastWrapper(
                env,
                setpoint_variables=cfg["setpoint_variables"],
                horizon=cfg.get("horizon", 24),
            )
        elif name == "cpu_load_forecast" and wrappers_cfg.get("cpu_load_forecast"):
            cfg = wrappers_cfg["cpu_load_forecast"]
            if isinstance(cfg, bool):
                cfg = {}
            env = CPULoadForecastWrapper(
                env,
                csv_path=_resolve_path(
                    cfg.get(
                        "csv_path",
                        "sinergym/data/buildings/GoogleClusteData_CPU_Data_Hourly_1.csv",
                    ),
                    project_root,
                ),
                horizon=cfg.get("horizon", 24),
                forecast_error=cfg.get("forecast_error"),
            )
        elif name == "price_forecast" and wrappers_cfg.get("price_forecast"):
            cfg = wrappers_cfg["price_forecast"]
            if isinstance(cfg, bool):
                cfg = {}
            env = PriceForecastWrapper(
                env,
                price_schedule=cfg["price_schedule"],
                horizon=cfg.get("horizon", 24),
            )
        elif name == "load_predictor" and wrappers_cfg.get("load_predictor"):
            cfg = wrappers_cfg["load_predictor"]
            env = LoadPredictorWrapper(
                env,
                model_path=_resolve_path(cfg["model_path"], project_root),
                input_variables=cfg.get("input_variables"),
                seq_len=cfg.get("seq_len", 24),
                horizon=cfg.get("horizon", 24),
                constant_load=cfg.get("constant_load"),
            )
        elif name == "oracle_load_forecast" and wrappers_cfg.get(
            "oracle_load_forecast"
        ):
            cfg = wrappers_cfg["oracle_load_forecast"]
            env = OracleLoadForecastWrapper(
                env,
                csv_path=_resolve_path(cfg["csv_path"], project_root),
                horizon=cfg.get("horizon", 12),
            )
        elif name == "csv_features" and wrappers_cfg.get("csv_features"):
            cfg = wrappers_cfg["csv_features"]
            env = CSVFeatureWrapper(
                env,
                csv_path=_resolve_path(cfg["csv_path"], project_root),
                columns=_resolve_feature_columns(cfg),
            )
        elif name == "zone_padding" and wrappers_cfg.get("zone_padding"):
            cfg = wrappers_cfg["zone_padding"]
            if isinstance(cfg, bool):
                cfg = {}
            env = ZonePaddingWrapper(
                env,
                zone_temperature_vars=cfg["zone_temperature_vars"],
                zone_setpoint_vars=cfg.get("zone_setpoint_vars"),
                max_zones=int(cfg.get("max_zones", 10)),
                padding_mode=cfg.get("padding_mode", "duplicate"),
            )
        elif name == "logger" and wrappers_cfg.get("logger", False):
            env = LoggerWrapper(env)
        elif name == "csv_logger" and wrappers_cfg.get("csv_logger", False):
            env = CSVLogger(env)
        elif name == "reduce_observation" and wrappers_cfg.get("reduce_observation"):
            cfg = wrappers_cfg["reduce_observation"]
            if isinstance(cfg, dict) and cfg.get("keep"):
                # keep mode: compute obs_reduction as everything NOT in keep list
                keep_vars = set(cfg["keep"])
                all_vars = list(env.get_wrapper_attr("observation_variables"))
                obs_reduction = [v for v in all_vars if v not in keep_vars]
            elif isinstance(cfg, list):
                obs_reduction = cfg
            else:
                obs_reduction = cfg.get("obs_reduction", [])
            env = ReduceObservationWrapper(env, obs_reduction=obs_reduction)
        elif name == "physical_normalize_observation" and wrappers_cfg.get(
            "physical_normalize_observation"
        ):
            physical = wrappers_cfg.get("_physical", {})
            if not physical:
                raise ValueError(
                    "physical_normalize_observation requires building physical config "
                    "(q_rated, cop_rated). Ensure the case's building has a 'physical' block."
                )
            bounds = _build_variable_bounds(physical)
            env = PhysicalNormalizeObservation(env, variable_bounds=bounds)
        elif name == "normalize_observation" and wrappers_cfg.get(
            "normalize_observation"
        ):
            cfg = wrappers_cfg["normalize_observation"]
            if cfg is True:
                cfg = {}
            mean = cfg.get("mean")
            var = cfg.get("var")
            count = cfg.get("count", 1e-4)
            if isinstance(mean, (str, Path)):
                mean = _resolve_path(mean, project_root)
            if isinstance(var, (str, Path)):
                var = _resolve_path(var, project_root)
            if isinstance(count, (str, Path)):
                count = _resolve_path(count, project_root)
            env = NormalizeObservation(
                env,
                automatic_update=cfg.get("automatic_update", True),
                mean=mean,
                var=var,
                count=count,
            )
        elif name == "physical_normalize_reward" and wrappers_cfg.get(
            "physical_normalize_reward"
        ):
            physical = wrappers_cfg.get("_physical", {})
            if not physical:
                raise ValueError(
                    "physical_normalize_reward requires building physical config."
                )
            reward_ref = _compute_reward_ref(physical)
            env = PhysicalNormalizeReward(env, reward_ref=reward_ref)
        elif name == "normalize_action" and wrappers_cfg.get("normalize_action", False):
            cfg = wrappers_cfg.get("normalize_action")
            if isinstance(cfg, dict):
                env = NormalizeAction(
                    env, normalize_range=tuple(cfg.get("normalize_range", (-1.0, 1.0)))
                )
            else:
                env = NormalizeAction(env)
        elif name == "discretize_action" and wrappers_cfg.get("discretize_action"):
            cfg = wrappers_cfg["discretize_action"]
            n = int(cfg.get("n", 2))
            values = cfg.get("values", [5.0, 10.0])
            assert len(values) == n, "values length must match n"
            mapping = {
                i: np.array([float(v)], dtype=np.float32) for i, v in enumerate(values)
            }
            env = DiscretizeEnv(
                env,
                discrete_space=gym.spaces.Discrete(n),
                action_mapping=lambda a, _m=mapping: _m[int(a)],
            )
    return env


def _env_name(case: ResolvedCase, run_context: RunContext, mode: str) -> str:
    env_cfg = case.env
    if mode == "eval" and env_cfg.get("eval_env_name"):
        return str(env_cfg["eval_env_name"])
    if mode == "train" and env_cfg.get("env_name"):
        explicit = str(env_cfg["env_name"])
        if env_cfg.get("append_timestamp", False):
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            return f"{explicit}_{ts}"
        return explicit
    base = env_cfg.get("name") or env_cfg.get("env_name_base") or case.id
    return f"{base}_{mode.upper()}_{run_context.run_id}"


def build_environment(
    case: ResolvedCase, run_context: RunContext, mode: str = "train"
) -> gym.Env:
    env_cfg = case.env
    project_root = case.paths["project_root"]
    register_yaml = env_cfg.get("register_yaml")
    if register_yaml:
        sinergym.register_envs_from_yaml(_resolve_path(register_yaml, project_root))

    make_kwargs: Dict[str, Any] = {
        "env_name": _env_name(case, run_context, mode),
        "workspace_root": str(run_context.workspace_root),
        "max_ep_store": int(
            env_cfg.get("runtime", {}).get(
                "max_episode_store", case.building.runtime.get("max_episode_store", 10)
            )
        ),
    }
    # Case-level building-model override (e.g. a 2x-plant sizing variant of the
    # same building); variables, actuators and weather stay those of the env.
    if env_cfg.get("building_file"):
        make_kwargs["building_file"] = str(env_cfg["building_file"])
    reward_cfg = case.reward
    if reward_cfg:
        reward_name = reward_cfg.get("class", "TimeVaryingCostLinearReward")
        make_kwargs["reward"] = _reward_class(reward_name)
        make_kwargs["reward_kwargs"] = reward_cfg.get("kwargs", {})

    env = gym.make(env_cfg["id"], **make_kwargs)
    wrappers_cfg = _resolve_mode_wrappers(case, mode)

    # Inject building physical config for normalization wrappers
    if case.building.physical:
        wrappers_cfg["_physical"] = dict(case.building.physical)

    if wrappers_cfg.get("price_forecast"):
        pf = wrappers_cfg["price_forecast"]
        if isinstance(pf, bool):
            pf = {}
            wrappers_cfg["price_forecast"] = pf
        if "price_schedule" not in pf and reward_cfg:
            schedule = reward_cfg.get("kwargs", {}).get("price_schedule")
            if schedule:
                pf["price_schedule"] = schedule

    return apply_wrappers(env, wrappers_cfg, project_root)
