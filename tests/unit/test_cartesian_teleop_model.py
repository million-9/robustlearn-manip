"""Validate Cartesian teleoperation against real Panda MuJoCo kinematics."""

import mujoco
import numpy as np
import pytest

from robustlearn.cartesian_teleop import (
    cartesian_joint_target,
    keyboard_cartesian_delta,
)
from robustlearn.sim.shadow_kinematics import (
    ShadowInsertionKinematics,
)
from robustlearn.teleop import (
    PANDA_JOINT_LOWER_RAD,
    PANDA_JOINT_UPPER_RAD,
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
def test_actual_peg_jacobian_matches_finite_difference(
    shadow, pose_name
):
    joints = np.asarray(POSES[pose_name], dtype=np.float64)

    initial = shadow.update(joints)
    jacobian = initial.peg_tip_jacobian.copy()

    assert jacobian.shape == (6, 7)
    assert np.all(np.isfinite(jacobian))

    epsilon = 1e-6

    for index in range(7):
        perturbed = joints.copy()
        perturbed[index] += epsilon

        updated = shadow.update(perturbed)

        measured = (
            updated.peg_tip_position
            - initial.peg_tip_position
        ) / epsilon

        np.testing.assert_allclose(
            measured,
            jacobian[:3, index],
            atol=3e-5,
            rtol=1e-4,
            err_msg=(
                f"Jacobian mismatch: {pose_name}, "
                f"joint {index + 1}"
            ),
        )


@pytest.mark.parametrize("pose_name", POSES)
@pytest.mark.parametrize("key", ["w", "s", "a", "d", "r", "f"])
def test_cartesian_jog_against_actual_forward_kinematics(
    shadow, pose_name, key
):
    joints = np.asarray(POSES[pose_name], dtype=np.float64)

    initial = shadow.update(joints)
    jacobian = initial.peg_tip_jacobian.copy()

    peg_site_id = mujoco.mj_name2id(
        shadow.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "peg_tip",
    )

    initial_rotation = (
        shadow.data.site_xmat[peg_site_id]
        .reshape(3, 3)
        .copy()
    )

    requested = keyboard_cartesian_delta(key)

    target = cartesian_joint_target(
        joints,
        jacobian,
        requested,
    )

    lower = np.asarray(PANDA_JOINT_LOWER_RAD)
    upper = np.asarray(PANDA_JOINT_UPPER_RAD)

    assert np.all(target >= lower)
    assert np.all(target <= upper)
    assert np.all(np.isfinite(target))

    joint_delta = target - joints

    assert np.max(np.abs(joint_delta)) <= 0.020000001

    predicted = jacobian @ joint_delta

    updated = shadow.update(target)

    achieved = (
        updated.peg_tip_position
        - initial.peg_tip_position
    )

    # Actual forward kinematics should agree with the
    # differential Jacobian approximation for this small jog.
    prediction_error = float(
        np.linalg.norm(achieved - predicted[:3])
    )

    assert prediction_error <= 0.0002, (
        f"{pose_name}/{key}: "
        f"FK prediction error={prediction_error:.6f} m"
    )

    progress = float(
        np.dot(achieved, requested)
        / np.dot(requested, requested)
    )

    assert progress >= 0.15, (
        f"{pose_name}/{key}: progress={progress:.3f}"
    )

    cross_axis = achieved - progress * requested

    cross_axis_error = float(np.linalg.norm(cross_axis))

    assert cross_axis_error <= 0.00055, (
        f"{pose_name}/{key}: "
        f"cross-axis error={cross_axis_error:.6f} m"
    )

    final_rotation = (
        shadow.data.site_xmat[peg_site_id]
        .reshape(3, 3)
        .copy()
    )

    relative_rotation = (
        final_rotation @ initial_rotation.T
    )

    cosine = float(
        np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = float(np.arccos(cosine))

    assert angle <= 0.004, (
        f"{pose_name}/{key}: "
        f"orientation drift={angle:.6f} rad"
    )
