"""Conservative Cartesian path safety regression tests."""

import math

import numpy as np
import pytest

from robustlearn.cartesian_safety import (
    CartesianPathSafetyConfig,
    reject_excessive_penetration,
    validate_free_space_cartesian_path,
)
from robustlearn.cartesian_teleop import (
    cartesian_joint_target,
    keyboard_cartesian_delta,
)
from robustlearn.sim.shadow_kinematics import (
    ShadowInsertionKinematics,
)

POSES = {
    "approach": (
        0.10, -0.10, 0.10, -1.70, 0.10, 1.70, -0.70
    ),
    "staging": (
        0.0, 0.02, 0.0, -1.50, 0.0, 1.51, -0.78
    ),
}


@pytest.fixture(scope="module")
def shadow():
    return ShadowInsertionKinematics()


@pytest.mark.parametrize("pose_name", POSES)
@pytest.mark.parametrize("key", ["w", "s", "a", "d", "r", "f"])
def test_actual_model_free_space_jog_is_accepted(
    shadow, pose_name, key
):
    joints = np.asarray(POSES[pose_name])
    initial = shadow.update(joints)

    target = cartesian_joint_target(
        joints,
        initial.peg_tip_jacobian,
        keyboard_cartesian_delta(key),
    )

    report = validate_free_space_cartesian_path(
        shadow, joints, target
    )

    assert report.samples_checked == 21
    assert 0.0001 < report.tip_displacement_m <= 0.0015
    assert report.orientation_change_rad <= 0.004

    # Validation must restore the original shadow state.
    np.testing.assert_allclose(
        shadow.update(joints).peg_tip_position,
        initial.peg_tip_position,
        atol=1e-10,
    )


def test_workspace_violation_rejected(shadow):
    joints = np.asarray(POSES["approach"])
    snapshot = shadow.update(joints)

    target = cartesian_joint_target(
        joints,
        snapshot.peg_tip_jacobian,
        [0.001, 0.0, 0.0],
    )

    narrow = CartesianPathSafetyConfig(
        min_xyz_m=(0.35, -0.05, 0.405),
        max_xyz_m=(0.75, 0.05, 0.70),
    )

    with pytest.raises(ValueError, match="workspace"):
        validate_free_space_cartesian_path(
            shadow, joints, target, config=narrow
        )


def test_oversized_joint_motion_rejected(shadow):
    joints = np.asarray(POSES["staging"])
    target = joints.copy()
    target[0] += 0.05

    with pytest.raises(ValueError, match="joint step"):
        validate_free_space_cartesian_path(
            shadow, joints, target
        )


def test_zero_motion_rejected(shadow):
    joints = np.asarray(POSES["staging"])

    with pytest.raises(ValueError, match="zero-motion"):
        validate_free_space_cartesian_path(
            shadow, joints, joints
        )


def test_excessive_penetration_rejected():
    with pytest.raises(ValueError, match="collision"):
        reject_excessive_penetration(
            [0.001, -0.0002],
            tolerance_m=0.00005,
        )


def test_near_zero_contact_tolerated():
    reject_excessive_penetration(
        [0.001, -0.00001],
        tolerance_m=0.00005,
    )


def test_nonfinite_contact_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        reject_excessive_penetration(
            [math.nan],
            tolerance_m=0.00005,
        )


def test_invalid_workspace_configuration_rejected():
    with pytest.raises(ValueError, match="workspace"):
        CartesianPathSafetyConfig(
            min_xyz_m=(0.75, -0.22, 0.405),
            max_xyz_m=(0.35, 0.22, 0.70),
        )


def test_invalid_sampling_configuration_rejected():
    with pytest.raises(ValueError, match="samples"):
        CartesianPathSafetyConfig(samples=2)
