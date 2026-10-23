"""Tests for real task-relative teleoperation observations."""

import numpy as np
import pytest

from robustlearn.sim.shadow_kinematics import (
    ShadowInsertionKinematics,
)
from robustlearn.teleop_demo_collection import (
    build_teleop_observation_frame,
)

JOINTS = (
    0.1, -0.1, 0.08, -1.7, 0.1, 1.7, -0.7
)


def make_frame(**overrides):
    values = {
        "frame_index": 0,
        "timestamp_s": 10.0,
        "joint_position": JOINTS,
        "joint_velocity": (0.01,) * 7,
        "wrist_force_sensor": (1.0, 2.0, 3.0),
        "wrist_torque_sensor": (0.1, 0.2, 0.3),
        "submitted_joint_target": JOINTS,
    }

    values.update(overrides)

    return build_teleop_observation_frame(
        kinematics=ShadowInsertionKinematics(),
        **values,
    )


def test_observation_matches_shared_geometry():
    shadow = ShadowInsertionKinematics()
    snapshot = shadow.update(np.asarray(JOINTS))

    frame = make_frame()

    expected = (
        snapshot.receptacle_center_position
        - snapshot.peg_tip_position
    )

    np.testing.assert_allclose(
        frame.task_relative_state,
        expected,
        atol=1e-10,
    )

    assert frame.joint_position == JOINTS
    assert frame.joint_velocity == (0.01,) * 7
    assert frame.expert_action == JOINTS
    assert not frame.terminated


def test_wrist_wrench_uses_world_frame():
    shadow = ShadowInsertionKinematics()
    shadow.update(np.asarray(JOINTS))

    rotation = shadow.wrist_rotation_world()

    frame = make_frame()

    expected_force = rotation @ np.array(
        (1.0, 2.0, 3.0)
    )

    expected_torque = rotation @ np.array(
        (0.1, 0.2, 0.3)
    )

    np.testing.assert_allclose(
        frame.wrist_wrench,
        np.concatenate(
            (expected_force, expected_torque)
        ),
        atol=1e-10,
    )


def test_missing_velocity_is_explicitly_unavailable():
    frame = make_frame(joint_velocity=None)

    assert frame.joint_velocity is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "joint_position",
        "wrist_force_sensor",
        "wrist_torque_sensor",
    ],
)
def test_missing_required_feedback_is_rejected(missing_field):
    with pytest.raises(ValueError):
        make_frame(**{missing_field: None})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_position", (0.0,) * 6),
        ("joint_velocity", (0.0,) * 6),
        ("wrist_force_sensor", (0.0,) * 2),
        ("wrist_torque_sensor", (np.nan, 0.0, 0.0)),
        ("submitted_joint_target", (np.inf,) * 7),
    ],
)
def test_invalid_observation_or_action_is_rejected(
    field, value
):
    with pytest.raises(ValueError):
        make_frame(**{field: value})


def test_observation_preserves_timestamp_and_frame_index():
    frame = make_frame(
        frame_index=4,
        timestamp_s=17.5,
    )

    assert frame.frame_index == 4
    assert frame.timestamp_s == 17.5
