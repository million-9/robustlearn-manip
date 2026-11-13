"""Conservative free-space guard for Panda Cartesian jogging.

This performs sampled kinematic collision checks, not continuous
collision detection or dynamic force-safety verification.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from robustlearn.sim.shadow_kinematics import ShadowInsertionKinematics
from robustlearn.teleop import (
    PANDA_JOINT_LOWER_RAD,
    PANDA_JOINT_UPPER_RAD,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CartesianPathSafetyConfig:
    """Provisional, conservative insertion-workspace limits."""

    min_xyz_m: tuple[float, float, float] = (
        0.35, -0.22, 0.405
    )
    max_xyz_m: tuple[float, float, float] = (
        0.75, 0.22, 0.70
    )

    max_joint_delta_rad: float = 0.02
    max_tip_displacement_m: float = 0.0015
    max_orientation_change_rad: float = 0.004
    max_penetration_m: float = 0.00005
    samples: int = 21

    def __post_init__(self) -> None:
        lower = np.asarray(self.min_xyz_m, dtype=np.float64)
        upper = np.asarray(self.max_xyz_m, dtype=np.float64)

        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("workspace bounds must be XYZ vectors")

        if not np.all(np.isfinite(lower)):
            raise ValueError("workspace minimum must be finite")

        if not np.all(np.isfinite(upper)):
            raise ValueError("workspace maximum must be finite")

        if np.any(lower >= upper):
            raise ValueError("invalid workspace bounds")

        for name in (
            "max_joint_delta_rad",
            "max_tip_displacement_m",
            "max_orientation_change_rad",
            "max_penetration_m",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        if type(self.samples) is not int or self.samples < 11:
            raise ValueError("samples must be an integer >= 11")


DEFAULT_CARTESIAN_PATH_SAFETY_CONFIG = CartesianPathSafetyConfig()


@dataclass(frozen=True)
class CartesianPathSafetyResult:
    """Diagnostics for a validated free-space joint path."""

    tip_displacement_m: float
    orientation_change_rad: float
    samples_checked: int


def reject_excessive_penetration(
    contact_distances_m: Iterable[float],
    *,
    tolerance_m: float,
) -> None:
    """Fail closed for nonfinite or excessive contact penetration."""

    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("invalid penetration tolerance")

    for distance in contact_distances_m:
        value = float(distance)

        if not math.isfinite(value):
            raise ValueError("nonfinite contact distance")

        if value < -tolerance_m:
            raise ValueError(
                f"free-space collision: penetration "
                f"{-value * 1000:.4f} mm"
            )


def _validated_joints(
    name: str,
    values: ArrayLike,
) -> FloatArray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if vector.shape != (7,):
        raise ValueError(f"{name} must have seven joints")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")

    lower = np.asarray(PANDA_JOINT_LOWER_RAD)
    upper = np.asarray(PANDA_JOINT_UPPER_RAD)

    if np.any(vector < lower) or np.any(vector > upper):
        raise ValueError(f"{name} exceeds joint limits")

    return vector


def _check_workspace(
    xyz: ArrayLike,
    config: CartesianPathSafetyConfig,
) -> None:
    position = np.asarray(xyz, dtype=np.float64)

    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("invalid peg-tip position")

    if (
        np.any(position < config.min_xyz_m)
        or np.any(position > config.max_xyz_m)
    ):
        raise ValueError("peg tip outside configured workspace")


def _check_contacts(
    shadow: ShadowInsertionKinematics,
    config: CartesianPathSafetyConfig,
) -> None:
    distances = (
        float(shadow.data.contact[i].dist)
        for i in range(shadow.data.ncon)
    )

    reject_excessive_penetration(
        distances,
        tolerance_m=config.max_penetration_m,
    )


def validate_free_space_cartesian_path(
    shadow: ShadowInsertionKinematics,
    current_joints: ArrayLike,
    target_joints: ArrayLike,
    *,
    config: CartesianPathSafetyConfig = DEFAULT_CARTESIAN_PATH_SAFETY_CONFIG,
) -> CartesianPathSafetyResult:
    """Validate a short free-space joint trajectory before submission.

    Interpolates in joint space and checks the peg workspace,
    MuJoCo contact penetration, tip displacement and orientation.

    The caller must separately verify fresh measured feedback,
    controller state, and motion-safety interlocks.
    """

    current = _validated_joints(
        "current_joints", current_joints
    )
    target = _validated_joints(
        "target_joints", target_joints
    )

    largest_delta = float(
        np.max(np.abs(target - current))
    )

    if largest_delta <= 0.0:
        raise ValueError("zero-motion Cartesian target")

    if largest_delta > config.max_joint_delta_rad + 1e-10:
        raise ValueError("Cartesian joint step exceeds safety limit")

    site_id = int(
        mujoco.mj_name2id(
            shadow.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "peg_tip",
        )
    )

    if site_id < 0:
        raise ValueError("peg-tip site missing")

    initial = shadow.update(current)

    start_position = initial.peg_tip_position.copy()
    start_rotation = (
        shadow.data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    try:
        final_position = start_position.copy()
        final_rotation = start_rotation.copy()

        for index in range(config.samples):
            alpha = index / (config.samples - 1)

            intermediate = (
                current + alpha * (target - current)
            )

            snapshot = shadow.update(intermediate)

            _check_workspace(
                snapshot.peg_tip_position,
                config,
            )

            _check_contacts(shadow, config)

            if index == config.samples - 1:
                final_position = (
                    snapshot.peg_tip_position.copy()
                )
                final_rotation = (
                    shadow.data.site_xmat[site_id]
                    .reshape(3, 3)
                    .copy()
                )

        displacement = float(
            np.linalg.norm(
                final_position - start_position
            )
        )

        if displacement > config.max_tip_displacement_m:
            raise ValueError(
                "peg-tip displacement exceeds safety limit"
            )

        relative_rotation = (
            final_rotation @ start_rotation.T
        )

        cosine = float(
            np.clip(
                (np.trace(relative_rotation) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )

        angle = float(math.acos(cosine))

        if angle > config.max_orientation_change_rad:
            raise ValueError(
                "peg orientation change exceeds safety limit"
            )

        return CartesianPathSafetyResult(
            tip_displacement_m=displacement,
            orientation_change_rad=angle,
            samples_checked=config.samples,
        )

    finally:
        # Restore the shadow model to the original measured state.
        shadow.update(current)
