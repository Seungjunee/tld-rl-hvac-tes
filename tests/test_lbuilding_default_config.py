import json
from pathlib import Path

import gymnasium as gym
import yaml

import sinergym


LBUILDING_CONFIG = Path("sinergym/data/default_configuration/LBuilding.yaml")
LBUILDING_MODEL = Path("sinergym/data/buildings/LBuilding.epJSON")


def test_lbuilding_yaml_contains_expected_tank_aliases():
    config = yaml.safe_load(LBUILDING_CONFIG.read_text(encoding="utf-8"))
    variables = config["variables"]

    assert config["id_base"] == "LBuilding"
    assert config["building_file"] == "LBuilding.epJSON"

    assert variables["Chilled Water Thermal Storage Tank Temperature"]["variable_names"] == "CWtank_temperature"
    assert variables["Chilled Water Thermal Storage Final Tank Temperature"]["variable_names"] == "CWtank_final"
    assert variables["Chilled Water Thermal Storage Tank Heat Gain Rate"]["variable_names"] == "CWtank_heat_gain_rate"
    assert (
        variables["Chilled Water Thermal Storage Use Side Heat Transfer Rate"]["variable_names"]
        == "CWtank_use_side_heat_transfer_rate"
    )
    assert (
        variables["Chilled Water Thermal Storage Source Side Heat Transfer Rate"]["variable_names"]
        == "CWtank_source_side_heat_transfer_rate"
    )

    model = json.loads(LBUILDING_MODEL.read_text(encoding="utf-8"))
    output_vars = [entry["variable_name"] for entry in model["Output:Variable"].values()]
    assert "Chilled Water Thermal Storage Final Tank Temperature" in output_vars


def test_lbuilding_env_ids_are_registered():
    expected_env_ids = [
        "Eplus-LBuilding-hot-continuous-v1",
        "Eplus-LBuilding-mixed-continuous-v1",
        "Eplus-LBuilding-cool-continuous-v1",
        "Eplus-LBuilding-hot-continuous-stochastic-v1",
        "Eplus-LBuilding-mixed-continuous-stochastic-v1",
        "Eplus-LBuilding-cool-continuous-stochastic-v1",
    ]

    for env_id in expected_env_ids:
        assert gym.spec(env_id).id == env_id


def test_lbuilding_env_make_exposes_expected_observation_variables(tmp_path):
    env = gym.make(
        "Eplus-LBuilding-hot-continuous-stochastic-v1",
        env_name="PYTEST_LBUILDING",
        workspace_root=str(tmp_path / "workspace_root"),
    )
    try:
        raw_env = env.unwrapped
        assert "CWtank_temperature" in raw_env.observation_variables
        assert "CWtank_final" in raw_env.observation_variables
        assert "CWtank_heat_gain_rate" in raw_env.observation_variables
        assert "CWtank_use_side_heat_transfer_rate" in raw_env.observation_variables
        assert "CWtank_source_side_heat_transfer_rate" in raw_env.observation_variables
        assert "Z8_temperature" in raw_env.observation_variables
        assert "Z8_humidity" in raw_env.observation_variables
        assert Path(raw_env.model.workspace_path).exists()
        assert Path(raw_env.model.workspace_path).parent == tmp_path / "workspace_root"
    finally:
        env.close()
