from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

from sinergym.utils.rewards import LinearReward


class TimeVaryingCostLinearReward(LinearReward):
    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        coolingsetpoint_variables: Optional[List[str]] = None,
        range_comfort_winter: Tuple[float, float] = (20.0, 22.0),
        range_comfort_summer: Tuple[float, float] = (23.0, 26.0),
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0e-4,
        lambda_temperature: float = 1.0,
        price_schedule: Optional[List[float]] = None,
        occupied_hours: Optional[List[int]] = None,
        exclude_days: Optional[List[int]] = None,
        simulation_year: int = 1991,
        comfort_form: str = "squared",
        zone_comfort_ranges: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=energy_weight,
            lambda_energy=lambda_energy,
            lambda_temperature=lambda_temperature,
        )
        self.coolingsetpoint_names = coolingsetpoint_variables or []
        self.temp_names = temperature_variables
        self.base_lambda_energy = float(lambda_energy)
        self.energy_names = energy_variables
        self.occupied_hours = occupied_hours  # [start, end] e.g. [8, 18], None=24h
        self.exclude_days = set(
            exclude_days or []
        )  # Mon=0..Sun=6; e.g. [6]=skip Sunday
        self.simulation_year = int(simulation_year)
        # "squared": per-zone band violation squared (setpoint deviation when
        # coolingsetpoint_variables are given). "linear": per-zone band
        # violation max(T_min - T, 0, T - T_max), Sinergym's LinearReward form.
        if comfort_form not in ("squared", "linear"):
            raise ValueError(
                f"comfort_form must be squared or linear, got {comfort_form}"
            )
        self.comfort_form = comfort_form
        # Optional per-zone band {temperature variable: [lo, hi]} for buildings
        # whose zones have different occupied cooling setpoints (rule:
        # band = occupied setpoint +- 2 C). None keeps the single seasonal band.
        self.zone_comfort_ranges = (
            {k: (float(v[0]), float(v[1])) for k, v in zone_comfort_ranges.items()}
            if zone_comfort_ranges
            else None
        )
        if self.zone_comfort_ranges:
            missing = [
                v for v in temperature_variables if v not in self.zone_comfort_ranges
            ]
            if missing:
                raise ValueError(f"zone_comfort_ranges missing {missing}")
        if price_schedule is None:
            self.price_schedule = [1.0] * 24
        else:
            if len(price_schedule) != 24:
                raise ValueError(
                    "price_schedule must have length 24 (one value per hour)."
                )
            self.price_schedule = [float(x) for x in price_schedule]
        self.last_price = 1.0

    def __call__(self, obs_dict: Dict[str, Any]):
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # EnergyPlus DataExchange.hour() returns 0..23. Old code subtracted 1
        # under a 1..24 assumption, causing a 1-hour lag in price lookup.
        hour_idx = int(obs_dict.get("hour", 0)) % 24

        price = self.price_schedule[hour_idx]
        self.last_price = price
        self.lambda_energy = price

        # Check if currently in occupied hours
        is_occupied = True
        if self.occupied_hours is not None:
            start_h, end_h = self.occupied_hours
            is_occupied = start_h <= hour_idx < end_h

        # Day-of-week exclusion (Mon=0 .. Sun=6)
        if is_occupied and self.exclude_days:
            month = int(obs_dict.get("month", 1))
            day = int(obs_dict.get("day_of_month", 1))
            try:
                dow = datetime.date(self.simulation_year, month, day).weekday()
                if dow in self.exclude_days:
                    is_occupied = False
            except ValueError:
                pass

        if is_occupied and self.comfort_form == "linear":
            self.total_temp_violation = sum(self._get_temperature_violation(obs_dict))
        elif is_occupied:
            # Quadratic comfort penalty: each zone's violation is squared so the
            # implemented reward matches the squared comfort term in the reward eq.
            if self.coolingsetpoint_names and len(self.coolingsetpoint_names) == len(
                self.temp_names
            ):
                temp_values = [obs_dict[v] for v in self.temp_names]
                setpoint_values = [obs_dict[v] for v in self.coolingsetpoint_names]
                temp_violations = [
                    (t - sp) ** 2 for t, sp in zip(temp_values, setpoint_values)
                ]
            else:
                temp_violations = [
                    v**2 for v in self._get_temperature_violation(obs_dict)
                ]
            self.total_temp_violation = sum(temp_violations)
        else:
            self.total_temp_violation = 0.0

        self.comfort_penalty = -self.total_temp_violation

        reward, energy_term, comfort_term = self._get_reward()
        reward_terms = {
            "energy_term": -energy_term,
            "comfort_term": -comfort_term,
            "energy_penalty": self.energy_penalty,
            "comfort_penalty": price,
            "total_power_demand": self.total_energy,
            "total_temperature_violation": self.total_temp_violation,
            "reward_weight": 0,
            "current_price": price,
        }
        return reward, reward_terms

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        if not self.zone_comfort_ranges:
            return super()._get_temperature_violation(obs_dict)
        out = []
        for v in self.temp_names:
            lo, hi = self.zone_comfort_ranges[v]
            t = float(obs_dict[v])
            out.append(max(lo - t, 0.0, t - hi))
        return out

    def _get_reward(self):
        energy_term = self.lambda_energy * self.energy_penalty
        n_zones = max(len(self.temp_names), 1)
        comfort_term = (self.lambda_temp / n_zones) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
