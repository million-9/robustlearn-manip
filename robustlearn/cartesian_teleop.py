"""Pure bounded Cartesian jogging for the Panda peg tip."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robustlearn.teleop import (
    PANDA_JOINT_LOWER_RAD,
    PANDA_JOINT_UPPER_RAD,
)

FloatArray = NDArray[np.float64]

# World-frame Cartesian displacement: positive / negative.
CARTESIAN_KEY_BINDINGS: dict[str, tuple[int, float]] = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "a": (1, 1.0),
    "d": (1, -1.0),
    "r": (2, 1.0),
    "f": (2, -1.0),
}


@dataclass(frozen=True)
class CartesianTeleopConfig:
    """Numerical limits for one Cartesian jog."""

    linear_step_m: float = 0.001
    max_joint_step_rad: float = 0.02
    damping: float = 0.025
    min_progress_fraction: float = 0.2
    max_cross_axis_error_m: float = 0.0005
    max_angular_drift_rad: float = 0.003
    max_reference_tracking_error_rad: float = 0.01

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        if self.min_progress_fraction > 1.0:
            raise ValueError("min_progress_fraction must be <= 1")


DEFAULT_CARTESIAN_CONFIG = CartesianTeleopConfig()


def _vector(
    name: str,
    values: ArrayLike,
    shape: tuple[int, ...],
) -> FloatArray:
    """Validate numerical shape and finiteness."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")

    return array


def keyboard_cartesian_delta(
    key: str,
    *,
    config: CartesianTeleopConfig = DEFAULT_CARTESIAN_CONFIG,
) -> FloatArray:
    """Return one requested world-frame XYZ displacement."""

    if not isinstance(key, str) or len(key) != 1:
        raise ValueError("Cartesian key must be one character")

    binding = CARTESIAN_KEY_BINDINGS.get(key.lower())

    if binding is None:
        raise ValueError(f"unsupported Cartesian key: {key!r}")

    axis, direction = binding

    delta = np.zeros(3, dtype=np.float64)
    delta[axis] = direction * config.linear_step_m

    return delta


def cartesian_joint_target(
    current_joints: ArrayLike,
    spatial_jacobian: ArrayLike,
    requested_xyz: ArrayLike,
    *,
    config: CartesianTeleopConfig = DEFAULT_CARTESIAN_CONFIG,
) -> FloatArray:
    """Compute a bounded differential-IK Panda joint target.

    The Jacobian must be 6x7: world-frame linear rows followed
    by angular rows. Angular velocity is requested to be zero.

    This is a local numerical target generator, not a complete
    collision checker, controller, or execution safety layer.
    """

    joints = _vector(
        "current_joints", current_joints, (7,)
    )
    jacobian = _vector(
        "spatial_jacobian", spatial_jacobian, (6, 7)
    )
    displacement = _vector(
        "requested_xyz", requested_xyz, (3,)
    )

    lower = np.asarray(PANDA_JOINT_LOWER_RAD)
    upper = np.asarray(PANDA_JOINT_UPPER_RAD)

    if np.any(joints < lower) or np.any(joints > upper):
        raise ValueError("current joints outside Panda limits")

    magnitude = float(np.linalg.norm(displacement))

    if magnitude <= 0.0 or magnitude > config.linear_step_m + 1e-12:
        raise ValueError("invalid Cartesian displacement magnitude")

    twist = np.zeros(6, dtype=np.float64)
    twist[:3] = displacement

    # Damped least-squares differential inverse kinematics.
    matrix = (
        jacobian @ jacobian.T
        + config.damping**2 * np.eye(6)
    )

    try:
        joint_delta = jacobian.T @ np.linalg.solve(
            matrix, twist
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("Cartesian IK solve failed") from exc

    if not np.all(np.isfinite(joint_delta)):
        raise ValueError("Cartesian IK produced non-finite motion")

    largest_step = float(np.max(np.abs(joint_delta)))

    if largest_step > config.max_joint_step_rad:
        joint_delta *= config.max_joint_step_rad / largest_step

    predicted_twist = jacobian @ joint_delta
    predicted_xyz = predicted_twist[:3]

    progress = float(
        np.dot(predicted_xyz, displacement)
        / np.dot(displacement, displacement)
    )

    if progress < config.min_progress_fraction:
        raise ValueError("Cartesian IK cannot achieve sufficient progress")

    cross_axis = predicted_xyz - progress * displacement

    if float(np.linalg.norm(cross_axis)) > config.max_cross_axis_error_m:
        raise ValueError("Cartesian IK has excessive cross-axis motion")

    if (
        float(np.linalg.norm(predicted_twist[3:]))
        > config.max_angular_drift_rad
    ):
        raise ValueError("Cartesian IK has excessive angular drift")

    target = joints + joint_delta

    # Reject rather than silently clamping a distorted IK target.
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError("Cartesian target exceeds Panda joint limits")

    if not np.all(np.isfinite(target)):
        raise ValueError("Cartesian target is non-finite")

    return np.asarray(target, dtype=np.float64)



def reference_relative_cartesian_target(
    measured_joints: ArrayLike,
    controller_reference: ArrayLike,
    measured_ik_target: ArrayLike,
    *,
    config: CartesianTeleopConfig = DEFAULT_CARTESIAN_CONFIG,
) -> FloatArray:
    """Apply a measured-state IK increment to the controller reference.

    The controller's steady holding reference may differ from the
    measured gravity-loaded Panda equilibrium. Preserve the IK
    increment while retaining that reference offset.

    The caller must validate both measured and reference joint paths
    and verify freshness of the controller-reference feedback.
    """
    measured = _vector(
        "measured_joints", measured_joints, (7,)
    )
    reference = _vector(
        "controller_reference", controller_reference, (7,)
    )
    ik_target = _vector(
        "measured_ik_target", measured_ik_target, (7,)
    )

    lower = np.asarray(PANDA_JOINT_LOWER_RAD)
    upper = np.asarray(PANDA_JOINT_UPPER_RAD)

    for name, vector in (
        ("measured joints", measured),
        ("controller reference", reference),
        ("measured IK target", ik_target),
    ):
        if np.any(vector < lower) or np.any(vector > upper):
            raise ValueError(f"{name} outside Panda joint limits")

    tracking_error = float(
        np.max(np.abs(reference - measured))
    )

    if tracking_error > config.max_reference_tracking_error_rad:
        raise ValueError(
            "controller reference differs excessively from measured joints"
        )

    delta = ik_target - measured
    largest_step = float(np.max(np.abs(delta)))

    if (
        largest_step <= 0.0
        or largest_step > config.max_joint_step_rad + 1e-10
    ):
        raise ValueError("invalid Cartesian IK joint increment")

    commanded = reference + delta

    if not np.all(np.isfinite(commanded)):
        raise ValueError("nonfinite reference-relative target")

    if np.any(commanded < lower) or np.any(commanded > upper):
        raise ValueError(
            "reference-relative target exceeds Panda joint limits"
        )

    return np.asarray(commanded, dtype=np.float64)
