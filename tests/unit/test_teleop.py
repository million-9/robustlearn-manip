import math

import numpy as np
import pytest

from robustlearn.teleop import (
    KEYBOARD_JOINT_BINDINGS,
    PANDA_JOINT_COUNT,
    PANDA_JOINT_LOWER_RAD,
    PANDA_JOINT_UPPER_RAD,
    TELEOP_SOURCE,
    TeleopConfig,
    bounded_joint_target,
    command_due,
    deadman_active,
    keyboard_joint_delta,
)


def test_teleop_source_is_explicit() -> None:
    assert TELEOP_SOURCE == "teleop"


def test_all_seven_joints_have_positive_and_negative_keys() -> None:
    covered: dict[int, set[float]] = {}

    for joint_index, direction in (
        KEYBOARD_JOINT_BINDINGS.values()
    ):
        covered.setdefault(
            joint_index,
            set(),
        ).add(direction)

    assert set(covered) == set(
        range(PANDA_JOINT_COUNT)
    )

    for directions in covered.values():
        assert directions == {-1.0, 1.0}


@pytest.mark.parametrize(
    ("key", "joint_index", "direction"),
    [
        ("q", 0, 1.0),
        ("a", 0, -1.0),
        ("w", 1, 1.0),
        ("s", 1, -1.0),
        ("u", 6, 1.0),
        ("j", 6, -1.0),
    ],
)
def test_keyboard_joint_delta_mapping(
    key: str,
    joint_index: int,
    direction: float,
) -> None:
    config = TeleopConfig(
        joint_step_rad=0.02,
    )

    delta = keyboard_joint_delta(
        key,
        config=config,
    )

    expected = np.zeros(
        PANDA_JOINT_COUNT,
        dtype=np.float64,
    )
    expected[joint_index] = (
        direction * 0.02
    )

    np.testing.assert_allclose(
        delta,
        expected,
    )


def test_keyboard_mapping_is_case_insensitive() -> None:
    np.testing.assert_allclose(
        keyboard_joint_delta("Q"),
        keyboard_joint_delta("q"),
    )


def test_keyboard_mapping_rejects_unknown_key() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported teleoperation key",
    ):
        keyboard_joint_delta("x")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_step_rad", 0.0),
        ("joint_step_rad", math.inf),
        ("deadman_timeout_s", -1.0),
        ("deadman_timeout_s", math.nan),
        ("min_command_period_s", 0.0),
    ],
)
def test_config_rejects_invalid_values(
    field: str,
    value: float,
) -> None:
    kwargs = {
        "joint_step_rad": 0.01,
        "deadman_timeout_s": 0.25,
        "min_command_period_s": 0.05,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        TeleopConfig(**kwargs)


def test_deadman_is_inactive_without_refresh() -> None:
    assert not deadman_active(
        now_s=10.0,
        last_refresh_s=None,
    )


def test_deadman_remains_active_inside_timeout() -> None:
    config = TeleopConfig(
        deadman_timeout_s=0.25,
    )

    assert deadman_active(
        now_s=10.20,
        last_refresh_s=10.0,
        config=config,
    )


def test_deadman_expires_after_timeout() -> None:
    config = TeleopConfig(
        deadman_timeout_s=0.25,
    )

    assert not deadman_active(
        now_s=10.26,
        last_refresh_s=10.0,
        config=config,
    )


def test_command_rate_limit() -> None:
    config = TeleopConfig(
        min_command_period_s=0.05,
    )

    assert command_due(
        now_s=1.0,
        last_command_s=None,
        config=config,
    )

    assert not command_due(
        now_s=1.04,
        last_command_s=1.0,
        config=config,
    )

    assert command_due(
        now_s=1.05,
        last_command_s=1.0,
        config=config,
    )


def test_disabled_command_does_not_progress() -> None:
    current = np.zeros(
        PANDA_JOINT_COUNT,
        dtype=np.float64,
    )
    current[3] = -1.0
    current[5] = 1.0

    target = bounded_joint_target(
        current,
        np.ones(PANDA_JOINT_COUNT),
        enabled=False,
    )

    np.testing.assert_allclose(
        target,
        current,
    )


def test_requested_delta_is_bounded() -> None:
    config = TeleopConfig(
        joint_step_rad=0.01,
    )

    current = np.asarray(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    requested = np.asarray(
        [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0],
        dtype=np.float64,
    )

    target = bounded_joint_target(
        current,
        requested,
        enabled=True,
        config=config,
    )

    delta = target - current

    assert np.all(
        np.abs(delta) <= 0.01 + 1e-12
    )


def test_target_is_clamped_to_joint_limits() -> None:
    config = TeleopConfig(
        joint_step_rad=0.1,
    )

    current = np.asarray(
        PANDA_JOINT_UPPER_RAD,
        dtype=np.float64,
    )

    target = bounded_joint_target(
        current,
        np.ones(PANDA_JOINT_COUNT),
        enabled=True,
        config=config,
    )

    np.testing.assert_allclose(
        target,
        PANDA_JOINT_UPPER_RAD,
    )


@pytest.mark.parametrize(
    "bad_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_joint_delta_is_rejected(
    bad_value: float,
) -> None:
    current = np.asarray(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    requested = np.zeros(
        PANDA_JOINT_COUNT,
        dtype=np.float64,
    )
    requested[0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        bounded_joint_target(
            current,
            requested,
            enabled=True,
        )


def test_current_position_outside_limits_is_rejected() -> None:
    current = np.asarray(
        PANDA_JOINT_LOWER_RAD,
        dtype=np.float64,
    )
    current[0] -= 0.01

    with pytest.raises(
        ValueError,
        match="outside Panda joint limits",
    ):
        bounded_joint_target(
            current,
            np.zeros(PANDA_JOINT_COUNT),
            enabled=True,
        )
