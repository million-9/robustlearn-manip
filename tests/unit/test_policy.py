"""Tests for the Panda insertion policy API."""

from typing import Any

import numpy as np
import pytest

from robustlearn.policy import (
    POLICY_ACTION_HIGH,
    POLICY_ACTION_LOW,
    POLICY_ACTION_NAMES,
    POLICY_ACTION_SIZE,
    POLICY_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SLICES,
    build_policy_observation,
    policy_action,
)
from robustlearn.sim.simulation import MuJoCoSimulation


def test_policy_action_contract_is_explicit() -> None:
    assert POLICY_ACTION_SIZE == 4
    assert POLICY_ACTION_NAMES == (
        "delta_x_m",
        "delta_y_m",
        "delta_z_m",
        "delta_yaw_rad",
    )

    assert POLICY_ACTION_LOW.shape == (4,)
    assert POLICY_ACTION_HIGH.shape == (4,)
    assert POLICY_ACTION_LOW.dtype == np.float64
    assert POLICY_ACTION_HIGH.dtype == np.float64

    assert np.all(np.isfinite(POLICY_ACTION_LOW))
    assert np.all(np.isfinite(POLICY_ACTION_HIGH))
    assert np.all(POLICY_ACTION_LOW < POLICY_ACTION_HIGH)


def test_policy_action_accepts_valid_input() -> None:
    action = policy_action(
        [0.001, -0.001, 0.0005, 0.02],
    )

    np.testing.assert_array_equal(
        action,
        np.asarray(
            [0.001, -0.001, 0.0005, 0.02],
            dtype=np.float64,
        ),
    )


def test_policy_action_clips_out_of_bounds_values_by_default() -> None:
    action = policy_action(
        [0.01, -0.01, 0.01, -0.5],
    )

    np.testing.assert_array_equal(
        action,
        np.asarray(
            [
                POLICY_ACTION_HIGH[0],
                POLICY_ACTION_LOW[1],
                POLICY_ACTION_HIGH[2],
                POLICY_ACTION_LOW[3],
            ],
            dtype=np.float64,
        ),
    )


def test_policy_action_can_reject_out_of_bounds_values() -> None:
    with pytest.raises(
        ValueError,
        match="outside configured bounds",
    ):
        policy_action(
            [0.01, 0.0, 0.0, 0.0],
            clip=False,
        )


@pytest.mark.parametrize(
    "invalid_action",
    (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0, 0.0]],
    ),
)
def test_policy_action_rejects_invalid_shape(
    invalid_action: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="policy action must have shape",
    ):
        policy_action(invalid_action)


@pytest.mark.parametrize(
    "invalid_value",
    (
        np.nan,
        np.inf,
        -np.inf,
    ),
)
def test_policy_action_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        policy_action(
            [0.0, 0.0, invalid_value, 0.0],
        )


def test_policy_action_returns_independent_array() -> None:
    source = np.asarray(
        [0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    action = policy_action(
        source,
        clip=False,
    )

    source[0] = 1.0

    assert action[0] == 0.0


def test_policy_observation_contract_is_explicit() -> None:
    assert POLICY_OBSERVATION_SIZE == 38

    assert POLICY_OBSERVATION_SLICES == {
        "joint_positions": slice(0, 7),
        "joint_velocities": slice(7, 14),
        "end_effector_position": slice(14, 17),
        "end_effector_quaternion": slice(17, 21),
        "target_relative_position": slice(21, 24),
        "target_relative_quaternion": slice(24, 28),
        "wrist_force": slice(28, 31),
        "wrist_torque": slice(31, 34),
        "previous_action": slice(34, 38),
    }


def test_policy_observation_is_finite_and_has_expected_shape() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    observation = build_policy_observation(
        simulation,
        np.zeros(4, dtype=np.float64),
    )

    assert observation.shape == (38,)
    assert observation.dtype == np.float64
    assert np.all(np.isfinite(observation))


def test_policy_observation_contains_sensor_state_in_documented_order() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    sensors = simulation.sensor_snapshot()

    observation = build_policy_observation(
        simulation,
        np.zeros(4, dtype=np.float64),
    )

    np.testing.assert_array_equal(
        observation[POLICY_OBSERVATION_SLICES["joint_positions"]],
        sensors.joint_positions,
    )
    np.testing.assert_array_equal(
        observation[POLICY_OBSERVATION_SLICES["joint_velocities"]],
        sensors.joint_velocities,
    )
    np.testing.assert_array_equal(
        observation[POLICY_OBSERVATION_SLICES["wrist_force"]],
        sensors.wrist_force,
    )
    np.testing.assert_array_equal(
        observation[POLICY_OBSERVATION_SLICES["wrist_torque"]],
        sensors.wrist_torque,
    )


def test_policy_observation_contains_previous_policy_action() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    previous_action = np.asarray(
        [0.001, -0.001, 0.0005, 0.02],
        dtype=np.float64,
    )

    observation = build_policy_observation(
        simulation,
        previous_action,
    )

    np.testing.assert_array_equal(
        observation[POLICY_OBSERVATION_SLICES["previous_action"]],
        previous_action,
    )


def test_policy_observation_is_deterministic_for_identical_state() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    previous_action = np.asarray(
        [0.001, 0.0, -0.001, 0.01],
        dtype=np.float64,
    )

    first = build_policy_observation(
        simulation,
        previous_action,
    )
    second = build_policy_observation(
        simulation,
        previous_action,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_policy_observation_rejects_invalid_previous_action() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    with pytest.raises(
        ValueError,
        match="policy action must have shape",
    ):
        build_policy_observation(
            simulation,
            [0.0, 0.0, 0.0],
        )


def test_target_relative_position_is_expressed_in_end_effector_frame() -> None:
    import mujoco

    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    observation = build_policy_observation(
        simulation,
        np.zeros(4, dtype=np.float64),
    )

    model = simulation.model
    data = simulation.data

    peg_tip_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "peg_tip",
        )
    )
    target_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "receptacle_center",
        )
    )

    world_delta = np.asarray(
        data.site_xpos[target_id] - data.site_xpos[peg_tip_id],
        dtype=np.float64,
    )

    ee_quaternion = observation[
        POLICY_OBSERVATION_SLICES["end_effector_quaternion"]
    ]

    inverse_ee_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(
        inverse_ee_quaternion,
        ee_quaternion,
    )

    expected_relative_position = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(
        expected_relative_position,
        world_delta,
        inverse_ee_quaternion,
    )

    np.testing.assert_allclose(
        observation[
            POLICY_OBSERVATION_SLICES["target_relative_position"]
        ],
        expected_relative_position,
    )


def test_policy_pose_quaternions_are_unit_length() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    observation = build_policy_observation(
        simulation,
        np.zeros(4, dtype=np.float64),
    )

    ee_quaternion = observation[
        POLICY_OBSERVATION_SLICES["end_effector_quaternion"]
    ]
    relative_quaternion = observation[
        POLICY_OBSERVATION_SLICES["target_relative_quaternion"]
    ]

    assert np.linalg.norm(ee_quaternion) == pytest.approx(1.0)
    assert np.linalg.norm(relative_quaternion) == pytest.approx(1.0)
