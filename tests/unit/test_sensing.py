"""Tests for Panda joint-state and wrist-wrench sensor definitions."""

import mujoco
import numpy as np

from robustlearn.sim import MuJoCoSimulation, load_insertion_model
from robustlearn.sim.panda import PANDA_ARM_JOINT_NAMES
from robustlearn.sim.sensing import (
    JOINT_POSITION_SENSOR_NAMES,
    JOINT_VELOCITY_SENSOR_NAMES,
    PANDA_SENSOR_NAMES,
    WRIST_FORCE_SENSOR_NAME,
    WRIST_FT_SITE_NAME,
    WRIST_TORQUE_SENSOR_NAME,
)


def required_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """Resolve a required named MuJoCo object."""
    object_id = int(
        mujoco.mj_name2id(
            model,
            object_type,
            name,
        )
    )

    assert object_id >= 0, f"Missing MuJoCo object: {name}"

    return object_id


def object_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str:
    """Return the required name of a MuJoCo object."""
    name = mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )

    assert name is not None

    return name


def test_expected_sensor_count_and_dimensions() -> None:
    model = load_insertion_model()

    assert len(PANDA_SENSOR_NAMES) == 16
    assert model.nsensor == 16
    assert model.nsensordata == 20


def test_joint_position_sensors_target_panda_arm_joints() -> None:
    model = load_insertion_model()

    for joint_name, sensor_name in zip(
        PANDA_ARM_JOINT_NAMES,
        JOINT_POSITION_SENSOR_NAMES,
        strict=True,
    ):
        joint_id = required_object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        sensor_id = required_object_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )

        assert (
            model.sensor_type[sensor_id]
            == mujoco.mjtSensor.mjSENS_JOINTPOS
        )
        assert model.sensor_dim[sensor_id] == 1
        assert model.sensor_objid[sensor_id] == joint_id


def test_joint_velocity_sensors_target_panda_arm_joints() -> None:
    model = load_insertion_model()

    for joint_name, sensor_name in zip(
        PANDA_ARM_JOINT_NAMES,
        JOINT_VELOCITY_SENSOR_NAMES,
        strict=True,
    ):
        joint_id = required_object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        sensor_id = required_object_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )

        assert (
            model.sensor_type[sensor_id]
            == mujoco.mjtSensor.mjSENS_JOINTVEL
        )
        assert model.sensor_dim[sensor_id] == 1
        assert model.sensor_objid[sensor_id] == joint_id


def test_wrist_sensor_site_is_on_tool_child_of_hand() -> None:
    model = load_insertion_model()

    site_id = required_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        WRIST_FT_SITE_NAME,
    )

    tool_body_id = int(model.site_bodyid[site_id])
    parent_body_id = int(model.body_parentid[tool_body_id])

    assert (
        object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            tool_body_id,
        )
        == "peg_tool"
    )

    assert (
        object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_body_id,
        )
        == "hand"
    )


def test_wrist_force_and_torque_sensors_target_wrist_site() -> None:
    model = load_insertion_model()

    site_id = required_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        WRIST_FT_SITE_NAME,
    )

    expected = (
        (
            WRIST_FORCE_SENSOR_NAME,
            mujoco.mjtSensor.mjSENS_FORCE,
        ),
        (
            WRIST_TORQUE_SENSOR_NAME,
            mujoco.mjtSensor.mjSENS_TORQUE,
        ),
    )

    for sensor_name, sensor_type in expected:
        sensor_id = required_object_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )

        assert model.sensor_type[sensor_id] == sensor_type
        assert model.sensor_dim[sensor_id] == 3
        assert model.sensor_objid[sensor_id] == site_id

def test_sensor_snapshot_has_expected_shapes_and_finite_values() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=2026)

    snapshot = sim.sensor_snapshot()

    assert snapshot.joint_positions.shape == (7,)
    assert snapshot.joint_velocities.shape == (7,)
    assert snapshot.wrist_force.shape == (3,)
    assert snapshot.wrist_torque.shape == (3,)

    assert np.all(np.isfinite(snapshot.joint_positions))
    assert np.all(np.isfinite(snapshot.joint_velocities))
    assert np.all(np.isfinite(snapshot.wrist_force))
    assert np.all(np.isfinite(snapshot.wrist_torque))


def test_joint_sensor_values_match_simulator_joint_state() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=2026)

    snapshot = sim.sensor_snapshot()

    joint_qpos = np.asarray(
        sim.data.qpos[:7],
        dtype=np.float64,
    )

    joint_qvel = np.asarray(
        sim.data.qvel[:7],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        snapshot.joint_positions,
        joint_qpos,
    )

    np.testing.assert_array_equal(
        snapshot.joint_velocities,
        joint_qvel,
    )


def test_same_seed_reproduces_sensor_snapshot() -> None:
    sim_a = MuJoCoSimulation()
    sim_b = MuJoCoSimulation()

    sim_a.reset(seed=2026)
    sim_b.reset(seed=2026)

    snapshot_a = sim_a.sensor_snapshot()
    snapshot_b = sim_b.sensor_snapshot()

    np.testing.assert_array_equal(
        snapshot_a.joint_positions,
        snapshot_b.joint_positions,
    )

    np.testing.assert_array_equal(
        snapshot_a.joint_velocities,
        snapshot_b.joint_velocities,
    )

    np.testing.assert_array_equal(
        snapshot_a.wrist_force,
        snapshot_b.wrist_force,
    )

    np.testing.assert_array_equal(
        snapshot_a.wrist_torque,
        snapshot_b.wrist_torque,
    )


def test_sensor_snapshot_is_independent_of_live_sensordata() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=2026)

    snapshot = sim.sensor_snapshot()

    saved_joint_positions = snapshot.joint_positions.copy()
    saved_joint_velocities = snapshot.joint_velocities.copy()
    saved_wrist_force = snapshot.wrist_force.copy()
    saved_wrist_torque = snapshot.wrist_torque.copy()

    sim.data.sensordata[:] += 123.0

    np.testing.assert_array_equal(
        snapshot.joint_positions,
        saved_joint_positions,
    )
    np.testing.assert_array_equal(
        snapshot.joint_velocities,
        saved_joint_velocities,
    )
    np.testing.assert_array_equal(
        snapshot.wrist_force,
        saved_wrist_force,
    )
    np.testing.assert_array_equal(
        snapshot.wrist_torque,
        saved_wrist_torque,
    )
def test_controlled_tool_load_produces_measurable_wrist_wrench() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=2026)

    tool_body_id = int(
        mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "peg_tool",
        )
    )

    assert tool_body_id >= 0

    baseline = sim.sensor_snapshot()

    sim.data.xfrc_applied[tool_body_id, :3] = np.asarray(
        [12.0, -7.0, 5.0],
        dtype=np.float64,
    )
    sim.data.xfrc_applied[tool_body_id, 3:] = np.asarray(
        [1.2, -0.7, 0.4],
        dtype=np.float64,
    )

    mujoco.mj_forward(
        sim.model,
        sim.data,
    )

    loaded = sim.sensor_snapshot()

    force_delta = (
        loaded.wrist_force
        - baseline.wrist_force
    )
    torque_delta = (
        loaded.wrist_torque
        - baseline.wrist_torque
    )

    assert np.all(np.isfinite(force_delta))
    assert np.all(np.isfinite(torque_delta))

    assert np.linalg.norm(force_delta) > 0.1
    assert np.linalg.norm(torque_delta) > 0.01

    sim.data.xfrc_applied[tool_body_id, :] = 0.0

    mujoco.mj_forward(
        sim.model,
        sim.data,
    )

    restored = sim.sensor_snapshot()

    np.testing.assert_array_equal(
        restored.wrist_force,
        baseline.wrist_force,
    )
    np.testing.assert_array_equal(
        restored.wrist_torque,
        baseline.wrist_torque,
    )