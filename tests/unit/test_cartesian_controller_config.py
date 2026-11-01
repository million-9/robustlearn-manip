"""Regression checks for opt-in Cartesian trajectory configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ros2_ws/src/robustlearn_description/config"

DEFAULT = CONFIG / "mujoco_trajectory_controllers.yaml"
CARTESIAN = CONFIG / "mujoco_trajectory_controllers_cartesian.yaml"


def test_default_controller_configuration_is_unchanged():
    assert "interpolate_from_desired_state:" not in DEFAULT.read_text()


def test_cartesian_variant_only_changes_interpolation_source():
    original = DEFAULT.read_text()
    specialized = CARTESIAN.read_text()
    setting = "    interpolate_from_desired_state: true\n"

    assert specialized.count(setting) == 1
    assert specialized.replace(setting, "", 1) == original
