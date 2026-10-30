from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from studies.case_study.core.wrappers import CSVFeatureWrapper


class DummyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_variables = ["base_a", "base_b"]
        self.workspace_path = "."

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def reset(self, *, seed=None, options=None):
        return np.array([1.0, 2.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([3.0, 4.0], dtype=np.float32), 0.0, False, False, {}


def test_csv_feature_wrapper_keeps_feature_order_and_dimension(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "CoolingRate1": [10.0, 11.0],
            "comfort_penalty1": [20.0, 21.0],
        }
    ).to_csv(csv_path, index=False)

    env = CSVFeatureWrapper(DummyEnv(), csv_path=str(csv_path), columns=["CoolingRate1", "comfort_penalty1"])

    obs, _ = env.reset()
    assert obs.shape[0] == 4
    assert np.allclose(obs, np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32))
    assert env.observation_variables[-2:] == ["CoolingRate1", "comfort_penalty1"]

    obs2, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert np.allclose(obs2, np.array([3.0, 4.0, 11.0, 21.0], dtype=np.float32))


