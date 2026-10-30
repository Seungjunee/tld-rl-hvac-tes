from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from studies.case_study.core.policy_selection import (
    predict_action,
    select_discrete_threshold_actions,
)


def test_select_discrete_threshold_actions_preserves_argmax_fallback():
    probabilities = np.array(
        [
            [0.20, 0.80],
            [0.04, 0.96],
            [0.70, 0.30],
        ],
        dtype=np.float32,
    )

    actions = select_discrete_threshold_actions(
        probabilities,
        target_action=0,
        min_probability=0.10,
    )

    assert actions.tolist() == [0, 1, 0]


class _FakePolicy:
    def __init__(self, probabilities):
        self.probabilities = torch.as_tensor(probabilities, dtype=torch.float32)

    def obs_to_tensor(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        return obs_tensor, False

    def get_distribution(self, obs_tensor):
        distribution = torch.distributions.Categorical(probs=self.probabilities)
        return SimpleNamespace(distribution=distribution)


class _FakeDiscreteModel:
    action_space = gym.spaces.Discrete(2)

    def __init__(self, probabilities):
        self.policy = _FakePolicy(probabilities)

    def predict(self, obs, **kwargs):
        return 1, None


def test_predict_action_discrete_threshold_can_choose_non_argmax_action():
    model = _FakeDiscreteModel([[0.20, 0.80]])

    action, state = predict_action(
        model,
        np.zeros(3, dtype=np.float32),
        action_selection={
            "mode": "discrete_threshold",
            "action": 0,
            "min_probability": 0.10,
        },
    )

    assert action == 0
    assert state is None


def test_predict_action_discrete_threshold_falls_back_to_argmax_below_threshold():
    model = _FakeDiscreteModel([[0.20, 0.80]])

    action, _state = predict_action(
        model,
        np.zeros(3, dtype=np.float32),
        action_selection={
            "mode": "discrete_threshold",
            "action": 0,
            "min_probability": 0.30,
        },
    )

    assert action == 1
