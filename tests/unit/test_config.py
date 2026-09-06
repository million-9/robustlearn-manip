"""Tests for Panda insertion task configuration."""

from typing import Any

import pytest

from robustlearn.config import (
    FloatRange,
    PandaInsertionRandomizationConfig,
    PandaInsertionTaskConfig,
)


def test_float_range_accepts_ordered_finite_bounds() -> None:
    value_range = FloatRange(-0.01, 0.02)

    assert value_range.low == -0.01
    assert value_range.high == 0.02


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (1.0, 0.0),
        (float("inf"), 1.0),
        (0.0, float("inf")),
        (float("nan"), 1.0),
        (0.0, float("nan")),
    ],
)
def test_float_range_rejects_invalid_bounds(
    low: float,
    high: float,
) -> None:
    with pytest.raises(ValueError):
        FloatRange(low, high)


def test_default_randomization_is_disabled_and_zero_width() -> None:
    config = PandaInsertionRandomizationConfig()

    assert config.enabled is False
    assert config.receptacle_x_offset_m == FloatRange(0.0, 0.0)
    assert config.receptacle_y_offset_m == FloatRange(0.0, 0.0)
    assert config.receptacle_yaw_offset_rad == FloatRange(0.0, 0.0)


def test_default_task_config_preserves_week5_values() -> None:
    config = PandaInsertionTaskConfig()

    assert config.frame_skip == 5
    assert config.max_episode_steps == 200
    assert config.success_lateral_tolerance_m == 0.001
    assert config.success_insertion_depth_m == 0.010
    assert config.failure_lateral_error_m == 0.0022
    assert config.randomization.enabled is False


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("frame_skip", 0),
        ("max_episode_steps", 0),
        ("success_lateral_tolerance_m", 0.0),
        ("success_insertion_depth_m", 0.0),
        ("failure_lateral_error_m", 0.0),
    ],
)
def test_task_config_rejects_non_positive_values(
    field_name: str,
    value: int | float,
) -> None:
    kwargs: dict[str, Any] = {field_name: value}

    with pytest.raises(ValueError):
        PandaInsertionTaskConfig(**kwargs)


def test_task_config_rejects_failure_threshold_inside_success_tolerance() -> None:
    with pytest.raises(ValueError):
        PandaInsertionTaskConfig(
            success_lateral_tolerance_m=0.002,
            failure_lateral_error_m=0.002,
        )


def test_task_config_serialization_is_stable() -> None:
    config = PandaInsertionTaskConfig()

    assert config.to_dict() == {
        "frame_skip": 5,
        "max_episode_steps": 200,
        "success_lateral_tolerance_m": 0.001,
        "success_insertion_depth_m": 0.010,
        "failure_lateral_error_m": 0.0022,
        "randomization": {
            "enabled": False,
            "receptacle_x_offset_m": {
                "low": 0.0,
                "high": 0.0,
            },
            "receptacle_y_offset_m": {
                "low": 0.0,
                "high": 0.0,
            },
            "receptacle_yaw_offset_rad": {
                "low": 0.0,
                "high": 0.0,
            },
        },
    }
