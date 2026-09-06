"""State-based policy API for Panda insertion."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

POLICY_ACTION_SIZE = 4
POLICY_ACTION_NAMES: tuple[str, ...] = (
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "delta_yaw_rad",
)

# Initial conservative per-policy-step Cartesian limits.
POLICY_ACTION_LOW = np.asarray(
    [-0.002, -0.002, -0.002, -0.05],
    dtype=np.float64,
)
POLICY_ACTION_HIGH = np.asarray(
    [0.002, 0.002, 0.002, 0.05],
    dtype=np.float64,
)


def policy_action(
    action: ArrayLike,
    *,
    clip: bool = True,
) -> FloatArray:
    """Return a validated 4D Cartesian Panda insertion policy action.

    Ordering:
        [delta_x_m, delta_y_m, delta_z_m, delta_yaw_rad]

    When ``clip`` is true, finite values outside the project-owned action
    limits are clipped component-wise. When false, out-of-bounds values are
    rejected.
    """
    action_array = np.asarray(
        action,
        dtype=np.float64,
    )

    if action_array.shape != (POLICY_ACTION_SIZE,):
        raise ValueError(
            "policy action must have shape "
            f"({POLICY_ACTION_SIZE},); got {action_array.shape}"
        )

    if not np.all(np.isfinite(action_array)):
        raise ValueError("policy action must contain only finite values")

    if clip:
        return np.asarray(
            np.clip(
                action_array,
                POLICY_ACTION_LOW,
                POLICY_ACTION_HIGH,
            ),
            dtype=np.float64,
        )

    if np.any(action_array < POLICY_ACTION_LOW) or np.any(
        action_array > POLICY_ACTION_HIGH
    ):
        raise ValueError("policy action is outside configured bounds")

    return action_array.copy()


POLICY_OBSERVATION_SIZE = 38

POLICY_OBSERVATION_SLICES: dict[str, slice] = {
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


def _site_pose(
    model: object,
    data: object,
    site_name: str,
) -> tuple[FloatArray, FloatArray]:
    """Return one named MuJoCo site pose in world coordinates."""
    import mujoco

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a mujoco.MjModel")

    if not isinstance(data, mujoco.MjData):
        raise TypeError("data must be a mujoco.MjData")

    site_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
    )

    if site_id < 0:
        raise RuntimeError(
            f"MuJoCo model does not contain required site {site_name!r}"
        )

    position = np.asarray(
        data.site_xpos[site_id],
        dtype=np.float64,
    ).copy()

    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.asarray(
            data.site_xmat[site_id],
            dtype=np.float64,
        ),
    )

    return position, quaternion


def build_policy_observation(
    simulation: object,
    previous_action: ArrayLike,
) -> FloatArray:
    """Build the stable state-based Panda insertion policy observation."""
    from robustlearn.sim.simulation import MuJoCoSimulation

    if not isinstance(simulation, MuJoCoSimulation):
        raise TypeError("simulation must be a MuJoCoSimulation")

    previous = policy_action(
        previous_action,
        clip=False,
    )

    sensors = simulation.sensor_snapshot()

    ee_position, ee_quaternion = _site_pose(
        simulation.model,
        simulation.data,
        "peg_tip",
    )
    target_position, target_quaternion = _site_pose(
        simulation.model,
        simulation.data,
        "receptacle_center",
    )

    world_target_delta = target_position - ee_position

    import mujoco

    inverse_ee_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(
        inverse_ee_quaternion,
        ee_quaternion,
    )

    relative_position = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(
        relative_position,
        world_target_delta,
        inverse_ee_quaternion,
    )

    relative_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(
        relative_quaternion,
        inverse_ee_quaternion,
        target_quaternion,
    )

    observation = np.concatenate(
        (
            sensors.joint_positions,
            sensors.joint_velocities,
            ee_position,
            ee_quaternion,
            relative_position,
            relative_quaternion,
            sensors.wrist_force,
            sensors.wrist_torque,
            previous,
        ),
        dtype=np.float64,
    )

    if observation.shape != (POLICY_OBSERVATION_SIZE,):
        raise RuntimeError(
            "policy observation has unexpected shape "
            f"{observation.shape}"
        )

    if not np.all(np.isfinite(observation)):
        raise RuntimeError(
            "policy observation contains non-finite values"
        )

    return observation
