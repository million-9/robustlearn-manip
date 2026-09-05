"""Named Panda actuator and Jacobian utilities for project control code."""

import mujoco
import numpy as np
from numpy.typing import NDArray

from robustlearn.sim.panda import PANDA_ARM_JOINT_NAMES

FloatArray = NDArray[np.float64]

PANDA_ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
)

PANDA_GRIPPER_ACTUATOR_NAME = "actuator8"


def actuator_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    """Resolve one required actuator by name."""
    resolved_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
    )

    if resolved_id < 0:
        raise RuntimeError(
            f"MuJoCo model does not contain required actuator {name!r}"
        )

    return resolved_id


def panda_arm_actuator_ids(
    model: mujoco.MjModel,
) -> tuple[int, ...]:
    """Return the seven Panda arm actuator IDs in joint order."""
    return tuple(
        actuator_id(model, name)
        for name in PANDA_ARM_ACTUATOR_NAMES
    )


def panda_gripper_actuator_id(
    model: mujoco.MjModel,
) -> int:
    """Return the Panda gripper actuator ID."""
    return actuator_id(
        model,
        PANDA_GRIPPER_ACTUATOR_NAME,
    )


def current_control_targets(
    data: mujoco.MjData,
) -> FloatArray:
    """Return an independent copy of the current actuator targets."""
    return np.asarray(
        data.ctrl,
        dtype=np.float64,
    ).copy()


def _site_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    """Resolve one required MuJoCo site by name."""
    resolved_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
    )

    if resolved_id < 0:
        raise RuntimeError(
            f"MuJoCo model does not contain required site {name!r}"
        )

    return resolved_id


def _joint_dof_address(
    model: mujoco.MjModel,
    name: str,
) -> int:
    """Resolve the velocity-DOF address for one Panda arm joint."""
    joint_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
    )

    if joint_id < 0:
        raise RuntimeError(
            f"MuJoCo model does not contain required joint {name!r}"
        )

    joint_type = int(model.jnt_type[joint_id])

    if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise RuntimeError(
            f"Joint {name!r} is not a one-DOF hinge joint"
        )

    return int(model.jnt_dofadr[joint_id])


def panda_site_spatial_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> FloatArray:
    """Return a 6x7 translational-and-rotational Panda site Jacobian."""
    site_id = _site_id(
        model,
        site_name,
    )

    jacobian_position = np.zeros(
        (3, model.nv),
        dtype=np.float64,
    )
    jacobian_rotation = np.zeros(
        (3, model.nv),
        dtype=np.float64,
    )

    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )

    arm_dof_addresses = tuple(
        _joint_dof_address(
            model,
            joint_name,
        )
        for joint_name in PANDA_ARM_JOINT_NAMES
    )

    position = jacobian_position[:, arm_dof_addresses]
    rotation = jacobian_rotation[:, arm_dof_addresses]

    return np.asarray(
        np.vstack((position, rotation)),
        dtype=np.float64,
    ).copy()


def panda_site_translation_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> FloatArray:
    """Return a 3x7 translational Jacobian for a named Panda task site."""
    return panda_site_spatial_jacobian(
        model,
        data,
        site_name,
    )[:3].copy()
