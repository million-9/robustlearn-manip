import numpy as np
import pytest

from robustlearn.cartesian_teleop import (
    CartesianTeleopConfig,
    cartesian_joint_target,
    keyboard_cartesian_delta,
)
from robustlearn.teleop import PANDA_JOINT_UPPER_RAD


@pytest.fixture
def home_joints():
    return np.array(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    )


@pytest.fixture
def independent_jacobian():
    # Synthetic six-DOF controllable Jacobian.
    return np.eye(6, 7)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("w", [0.001, 0.0, 0.0]),
        ("s", [-0.001, 0.0, 0.0]),
        ("a", [0.0, 0.001, 0.0]),
        ("d", [0.0, -0.001, 0.0]),
        ("r", [0.0, 0.0, 0.001]),
        ("f", [0.0, 0.0, -0.001]),
    ],
)
def test_cartesian_keyboard_mapping(key, expected):
    np.testing.assert_allclose(
        keyboard_cartesian_delta(key), expected
    )


def test_keyboard_rejects_unknown_key():
    with pytest.raises(ValueError):
        keyboard_cartesian_delta("x")


def test_damped_ik_moves_requested_axis(
    home_joints, independent_jacobian
):
    target = cartesian_joint_target(
        home_joints,
        independent_jacobian,
        [0.001, 0.0, 0.0],
    )

    assert target[0] > home_joints[0]
    assert target[0] - home_joints[0] <= 0.02

    np.testing.assert_allclose(
        target[1:], home_joints[1:], atol=1e-12
    )


def test_damped_ik_preserves_orientation_to_first_order(
    home_joints, independent_jacobian
):
    target = cartesian_joint_target(
        home_joints,
        independent_jacobian,
        [0.0, 0.001, 0.0],
    )

    predicted = independent_jacobian @ (target - home_joints)

    np.testing.assert_allclose(
        predicted[3:], np.zeros(3), atol=1e-12
    )


def test_singular_zero_jacobian_rejected(home_joints):
    with pytest.raises(ValueError, match="progress"):
        cartesian_joint_target(
            home_joints,
            np.zeros((6, 7)),
            [0.001, 0.0, 0.0],
        )


def test_nonfinite_jacobian_rejected(home_joints):
    jacobian = np.eye(6, 7)
    jacobian[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        cartesian_joint_target(
            home_joints, jacobian, [0.001, 0.0, 0.0]
        )


def test_oversized_cartesian_request_rejected(
    home_joints, independent_jacobian
):
    with pytest.raises(ValueError, match="magnitude"):
        cartesian_joint_target(
            home_joints,
            independent_jacobian,
            [0.1, 0.0, 0.0],
        )


def test_joint_limit_violation_rejected(independent_jacobian):
    joints = np.array(PANDA_JOINT_UPPER_RAD)

    with pytest.raises(ValueError, match="joint limits"):
        cartesian_joint_target(
            joints,
            independent_jacobian,
            [0.001, 0.0, 0.0],
        )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        CartesianTeleopConfig(linear_step_m=0.0)
