"""Pure safety and input-mapping helpers for Panda teleoperation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

TELEOP_SOURCE = "teleop"
PANDA_JOINT_COUNT = 7

# Panda limits mirrored from robustlearn_description/urdf/panda.urdf.xacro.
PANDA_JOINT_LOWER_RAD: tuple[float, ...] = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
)

PANDA_JOINT_UPPER_RAD: tuple[float, ...] = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
)

# Keyboard pairs: positive direction / negative direction.
KEYBOARD_JOINT_BINDINGS: dict[str, tuple[int, float]] = {
    "q": (0, 1.0),
    "a": (0, -1.0),
    "w": (1, 1.0),
    "s": (1, -1.0),
    "e": (2, 1.0),
    "d": (2, -1.0),
    "r": (3, 1.0),
    "f": (3, -1.0),
    "t": (4, 1.0),
    "g": (4, -1.0),
    "y": (5, 1.0),
    "h": (5, -1.0),
    "u": (6, 1.0),
    "j": (6, -1.0),
}


@dataclass(frozen=True)
class TeleopConfig:
    """Project-owned safety bounds for joint-space teleoperation."""

    joint_step_rad: float = 0.01
    deadman_timeout_s: float = 0.25
    min_command_period_s: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("joint_step_rad", self.joint_step_rad),
            ("deadman_timeout_s", self.deadman_timeout_s),
            ("min_command_period_s", self.min_command_period_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )


DEFAULT_TELEOP_CONFIG = TeleopConfig()


def _joint_vector(
    name: str,
    values: ArrayLike,
) -> FloatArray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != (PANDA_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape ({PANDA_JOINT_COUNT},); "
            f"got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return array


def keyboard_joint_delta(
    key: str,
    *,
    config: TeleopConfig = DEFAULT_TELEOP_CONFIG,
) -> FloatArray:
    """Map one keyboard jog key to a bounded seven-joint delta."""
    normalized = key.lower()

    if normalized not in KEYBOARD_JOINT_BINDINGS:
        raise ValueError(
            f"unsupported teleoperation key: {key!r}"
        )

    joint_index, direction = KEYBOARD_JOINT_BINDINGS[
        normalized
    ]

    delta = np.zeros(
        PANDA_JOINT_COUNT,
        dtype=np.float64,
    )
    delta[joint_index] = (
        direction * config.joint_step_rad
    )

    return delta


def deadman_active(
    *,
    now_s: float,
    last_refresh_s: float | None,
    config: TeleopConfig = DEFAULT_TELEOP_CONFIG,
) -> bool:
    """Return whether the deadman refresh is still valid."""
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")

    if last_refresh_s is None:
        return False

    if not math.isfinite(last_refresh_s):
        raise ValueError(
            "last_refresh_s must be finite when supplied"
        )

    age_s = now_s - last_refresh_s

    if age_s < 0.0:
        raise ValueError(
            "last_refresh_s cannot be later than now_s"
        )

    return age_s <= config.deadman_timeout_s


def command_due(
    *,
    now_s: float,
    last_command_s: float | None,
    config: TeleopConfig = DEFAULT_TELEOP_CONFIG,
) -> bool:
    """Return whether rate limiting permits another command."""
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")

    if last_command_s is None:
        return True

    if not math.isfinite(last_command_s):
        raise ValueError(
            "last_command_s must be finite when supplied"
        )

    elapsed_s = now_s - last_command_s

    if elapsed_s < 0.0:
        raise ValueError(
            "last_command_s cannot be later than now_s"
        )

    return elapsed_s >= config.min_command_period_s


def bounded_joint_target(
    current_positions: ArrayLike,
    requested_delta: ArrayLike,
    *,
    enabled: bool,
    config: TeleopConfig = DEFAULT_TELEOP_CONFIG,
) -> FloatArray:
    """Return a safe Panda target for one teleoperation command.

    When ``enabled`` is false, the current joint positions are returned
    unchanged. This is the core deadman stop behavior.

    When enabled, every requested joint delta is component-wise bounded by
    ``joint_step_rad`` and the resulting target is clamped to the Panda URDF
    joint limits.
    """
    current = _joint_vector(
        "current_positions",
        current_positions,
    )
    requested = _joint_vector(
        "requested_delta",
        requested_delta,
    )

    lower = np.asarray(
        PANDA_JOINT_LOWER_RAD,
        dtype=np.float64,
    )
    upper = np.asarray(
        PANDA_JOINT_UPPER_RAD,
        dtype=np.float64,
    )

    if np.any(current < lower) or np.any(current > upper):
        raise ValueError(
            "current_positions are outside Panda joint limits"
        )

    if not enabled:
        return current.copy()

    bounded_delta = np.clip(
        requested,
        -config.joint_step_rad,
        config.joint_step_rad,
    )

    target = np.clip(
        current + bounded_delta,
        lower,
        upper,
    )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "teleoperation target is non-finite"
        )

    return np.asarray(
        target,
        dtype=np.float64,
    )
