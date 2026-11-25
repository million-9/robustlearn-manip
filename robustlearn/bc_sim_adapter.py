"""Compatibility adapter between BC training data and Panda MuJoCo."""

import mujoco
import numpy as np
from numpy.typing import NDArray

from robustlearn.bc_dataset import ACTION_DIM, OBSERVATION_DIM
from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.sim.control import (
    current_control_targets,
    panda_arm_actuator_ids,
)
from robustlearn.sim.sensing import WRIST_FT_SITE_NAME
from robustlearn.sim.simulation import TASK_SITE_NAMES

FloatArray = NDArray[np.float32]
ControlArray = NDArray[np.float64]


def bc_observation_from_env(env: PandaInsertionEnv) -> FloatArray:
    """Construct the BC observation from live MuJoCo state."""
    simulation = env.simulation
    sensors = simulation.sensor_snapshot()
    snapshot = simulation.snapshot()

    sites = dict(
        zip(TASK_SITE_NAMES, snapshot.task_site_xpos, strict=True)
    )

    relative_position = (
        sites["receptacle_center"] - sites["peg_tip"]
    )

    site_id = int(
        mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_SITE,
            WRIST_FT_SITE_NAME,
        )
    )
    if site_id < 0:
        raise RuntimeError("Missing wrist force/torque site")

    rotation_world = np.asarray(
        simulation.data.site_xmat[site_id],
        dtype=np.float64,
    ).reshape(3, 3)

    force_world = rotation_world @ sensors.wrist_force
    torque_world = rotation_world @ sensors.wrist_torque

    observation = np.concatenate(
        (
            sensors.joint_positions,
            sensors.joint_velocities,
            relative_position,
            force_world,
            torque_world,
        )
    ).astype(np.float32)

    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError("BC observation dimension mismatch")

    if not np.all(np.isfinite(observation)):
        raise ValueError("Nonfinite BC observation")

    return observation


def bc_action_to_env(
    env: PandaInsertionEnv,
    joint_targets: NDArray[np.float32] | NDArray[np.float64],
) -> ControlArray:
    """Map seven BC joint targets to simulator actuator controls."""
    targets = np.asarray(joint_targets, dtype=np.float64)

    if targets.shape != (ACTION_DIM,):
        raise ValueError("BC action must contain seven joint targets")

    if not np.all(np.isfinite(targets)):
        raise ValueError("Nonfinite BC action")

    simulation = env.simulation
    actuator_ids = panda_arm_actuator_ids(simulation.model)

    if len(set(actuator_ids)) != ACTION_DIM:
        raise RuntimeError("Invalid arm actuator mapping")

    action = current_control_targets(simulation.data)

    for actuator_id, target in zip(
        actuator_ids, targets, strict=True
    ):
        action[actuator_id] = target

    if not env.action_space.contains(action):
        raise ValueError(
            "BC prediction exceeds MuJoCo actuator control limits"
        )

    return action
