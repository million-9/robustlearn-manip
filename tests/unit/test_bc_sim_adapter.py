"""Tests for the BC-to-MuJoCo compatibility adapter."""

import mujoco
import numpy as np
import pytest

from robustlearn.bc_sim_adapter import (
    bc_action_to_env,
    bc_observation_from_env,
)
from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.sim.control import (
    panda_arm_actuator_ids,
    panda_gripper_actuator_id,
)
from robustlearn.sim.sensing import WRIST_FT_SITE_NAME
from robustlearn.sim.simulation import TASK_SITE_NAMES


@pytest.fixture
def env():
    """Create and reset the real headless insertion simulator."""
    environment = PandaInsertionEnv()
    environment.reset(seed=2026)
    yield environment
    environment.close()


def test_observation_matches_live_sensor_contract(env):
    observation = bc_observation_from_env(env)
    simulation = env.simulation
    sensors = simulation.sensor_snapshot()
    snapshot = simulation.snapshot()

    sites = dict(
        zip(TASK_SITE_NAMES, snapshot.task_site_xpos, strict=True)
    )
    site_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_SITE,
        WRIST_FT_SITE_NAME,
    )
    rotation = simulation.data.site_xmat[site_id].reshape(3, 3)

    expected = np.concatenate(
        (
            sensors.joint_positions,
            sensors.joint_velocities,
            sites["receptacle_center"] - sites["peg_tip"],
            rotation @ sensors.wrist_force,
            rotation @ sensors.wrist_torque,
        )
    ).astype(np.float32)

    assert observation.shape == (23,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    np.testing.assert_array_equal(observation, expected)


def test_same_seed_produces_same_bc_observation(env):
    first = bc_observation_from_env(env)
    env.reset(seed=2026)
    second = bc_observation_from_env(env)

    np.testing.assert_array_equal(first, second)


def test_action_maps_arm_targets_and_preserves_gripper(env):
    simulation = env.simulation
    arm_ids = panda_arm_actuator_ids(simulation.model)
    gripper_id = panda_gripper_actuator_id(simulation.model)

    original_gripper = float(simulation.data.ctrl[gripper_id])
    targets = np.asarray(
        simulation.data.ctrl[list(arm_ids)],
        dtype=np.float64,
    ).copy()

    command = bc_action_to_env(env, targets)

    assert command.shape == (8,)
    assert command.dtype == np.float64
    np.testing.assert_array_equal(command[list(arm_ids)], targets)
    assert command[gripper_id] == original_gripper

    # Adapter construction must not modify live controls.
    assert simulation.data.ctrl[gripper_id] == original_gripper


def test_rejects_wrong_action_dimension(env):
    with pytest.raises(ValueError, match="seven joint targets"):
        bc_action_to_env(env, np.zeros(8))


def test_rejects_nonfinite_action(env):
    targets = np.zeros(7)
    targets[0] = np.nan

    with pytest.raises(ValueError, match="Nonfinite"):
        bc_action_to_env(env, targets)


def test_rejects_out_of_range_action(env):
    arm_ids = panda_arm_actuator_ids(env.simulation.model)
    targets = np.asarray(
        env.simulation.data.ctrl[list(arm_ids)],
        dtype=np.float64,
    ).copy()
    targets[0] = env.action_space.high[arm_ids[0]] + 1.0

    with pytest.raises(ValueError, match="control limits"):
        bc_action_to_env(env, targets)


def test_valid_bc_action_executes_real_simulation_step(env):
    simulation = env.simulation
    arm_ids = panda_arm_actuator_ids(simulation.model)

    targets = np.asarray(
        simulation.data.ctrl[list(arm_ids)],
        dtype=np.float64,
    ).copy()

    command = bc_action_to_env(env, targets)
    before_time = float(simulation.data.time)

    observation, _, terminated, truncated, info = env.step(command)

    assert observation.shape == (30,)
    assert np.all(np.isfinite(observation))
    assert info["simulation_time"] > before_time
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
