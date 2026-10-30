"""Tests for shared expert/teleop task geometry."""

import mujoco
import numpy as np
import pytest

from robustlearn.sim.insertion import load_insertion_model
from robustlearn.sim.shadow_kinematics import ShadowInsertionKinematics


def initial_positions():
    return np.array(
        [0.1, -0.1, 0.08, -1.7, 0.1, 1.7, -0.7],
        dtype=np.float64,
    )


def test_shared_geometry_matches_independent_mujoco_forward_kinematics():
    positions = initial_positions()

    shadow = ShadowInsertionKinematics()
    snapshot = shadow.update(positions)

    model = load_insertion_model()
    data = mujoco.MjData(model)

    for index, position in enumerate(positions, start=1):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"joint{index}",
        )
        assert joint_id >= 0

        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address] = position

    mujoco.mj_forward(model, data)

    def site_position(name):
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        assert site_id >= 0
        return data.site_xpos[site_id].copy()

    expected_relative = (
        site_position("receptacle_center")
        - site_position("peg_tip")
    )

    measured_relative = (
        snapshot.receptacle_center_position
        - snapshot.peg_tip_position
    )

    np.testing.assert_allclose(
        measured_relative,
        expected_relative,
        atol=1e-10,
    )

    assert measured_relative.shape == (3,)
    assert np.all(np.isfinite(measured_relative))


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros(6),
        np.full(7, np.nan),
        np.full(7, np.inf),
    ],
)
def test_invalid_joint_feedback_is_rejected(positions):
    shadow = ShadowInsertionKinematics()

    with pytest.raises(ValueError):
        shadow.update(positions)


def test_wrist_rotation_is_finite():
    shadow = ShadowInsertionKinematics()
    shadow.update(initial_positions())

    rotation = shadow.wrist_rotation_world()

    assert rotation.shape == (3, 3)
    assert np.all(np.isfinite(rotation))

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-8,
    )
