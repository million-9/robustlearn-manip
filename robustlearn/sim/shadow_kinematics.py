"""Shared MuJoCo shadow kinematics for insertion observations."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from robustlearn.sim.control import panda_site_spatial_jacobian
from robustlearn.sim.insertion import load_insertion_model

FloatArray = NDArray[np.float64]

MUJOCO_PANDA_ARM_JOINT_NAMES = tuple(
    f"joint{i}" for i in range(1, 8)
)

PEG_TIP_SITE_NAME = "peg_tip"
RECEPTACLE_CENTER_SITE_NAME = "receptacle_center"
INSERTION_AXIS_SITE_NAME = "insertion_axis"
WRIST_FT_SITE_NAME = "panda_wrist_ft"


@dataclass(frozen=True)
class ShadowKinematicsSnapshot:
    """Finite kinematic state reconstructed from ROS joint feedback."""

    joint_positions: FloatArray
    peg_tip_position: FloatArray
    receptacle_center_position: FloatArray
    insertion_axis_position: FloatArray
    peg_tip_jacobian: FloatArray


class ShadowInsertionKinematics:
    """Project-owned MuJoCo model used only for FK/Jacobian reconstruction."""

    def __init__(self) -> None:
        self.model = load_insertion_model()
        self.data = mujoco.MjData(self.model)

        self._joint_qpos_addresses = tuple(
            self._joint_qpos_address(name)
            for name in MUJOCO_PANDA_ARM_JOINT_NAMES
        )

        self._peg_tip_site_id = self._site_id(
            PEG_TIP_SITE_NAME
        )

        self._receptacle_center_site_id = self._site_id(
            RECEPTACLE_CENTER_SITE_NAME
        )

        self._insertion_axis_site_id = self._site_id(
            INSERTION_AXIS_SITE_NAME
        )

        self._wrist_ft_site_id = self._site_id(
            WRIST_FT_SITE_NAME
        )

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
        )

        if joint_id < 0:
            raise RuntimeError(
                f'Shadow insertion model is missing joint {name!r}'
            )

        if int(self.model.jnt_type[joint_id]) != int(
            mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise RuntimeError(
                f'Shadow insertion joint {name!r} is not a hinge'
            )

        return int(
            self.model.jnt_qposadr[joint_id]
        )

    def _site_id(self, name: str) -> int:
        site_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
        )

        if site_id < 0:
            raise RuntimeError(
                f'Shadow insertion model is missing site {name!r}'
            )

        return site_id

    def update(
        self,
        joint_positions: FloatArray,
    ) -> ShadowKinematicsSnapshot:
        """Reconstruct task geometry from observed Panda joint positions."""
        positions = np.asarray(
            joint_positions,
            dtype=np.float64,
        )

        if positions.shape != (7,):
            raise ValueError(
                f'joint_positions must have shape (7,), got {positions.shape}'
            )

        if not np.all(np.isfinite(positions)):
            raise ValueError(
                'joint_positions must contain only finite values'
            )

        for qpos_address, position in zip(
            self._joint_qpos_addresses,
            positions,
            strict=True,
        ):
            self.data.qpos[qpos_address] = position

        self.data.qvel.fill(0.0)

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        peg_tip_position = np.asarray(
            self.data.site_xpos[self._peg_tip_site_id],
            dtype=np.float64,
        ).copy()

        receptacle_center_position = np.asarray(
            self.data.site_xpos[
                self._receptacle_center_site_id
            ],
            dtype=np.float64,
        ).copy()

        insertion_axis_position = np.asarray(
            self.data.site_xpos[
                self._insertion_axis_site_id
            ],
            dtype=np.float64,
        ).copy()

        peg_tip_jacobian = panda_site_spatial_jacobian(
            self.model,
            self.data,
            PEG_TIP_SITE_NAME,
        )

        arrays = (
            peg_tip_position,
            receptacle_center_position,
            insertion_axis_position,
            peg_tip_jacobian,
        )

        if not all(
            np.all(np.isfinite(array))
            for array in arrays
        ):
            raise RuntimeError(
                'Shadow insertion kinematics produced non-finite state'
            )

        return ShadowKinematicsSnapshot(
            joint_positions=positions.copy(),
            peg_tip_position=peg_tip_position,
            receptacle_center_position=receptacle_center_position,
            insertion_axis_position=insertion_axis_position,
            peg_tip_jacobian=peg_tip_jacobian,
        )

    def wrist_rotation_world(self) -> FloatArray:
        """Return the world rotation matrix of the wrist F/T site."""
        rotation = np.asarray(
            self.data.site_xmat[self._wrist_ft_site_id],
            dtype=np.float64,
        ).reshape(3, 3).copy()

        if not np.all(np.isfinite(rotation)):
            raise RuntimeError(
                'Shadow wrist rotation contains non-finite values'
            )

        return rotation
