"""Fail-closed wrist-wrench validation for Cartesian teleoperation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CartesianWrenchSafetyConfig:
    """Provisional simulation-only free-space wrench limits."""

    feedback_timeout_s: float = 0.25
    max_force_norm_n: float = 5.0
    max_torque_norm_nm: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("feedback_timeout_s", self.feedback_timeout_s),
            ("max_force_norm_n", self.max_force_norm_n),
            ("max_torque_norm_nm", self.max_torque_norm_nm),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )


DEFAULT_CARTESIAN_WRENCH_SAFETY_CONFIG = (
    CartesianWrenchSafetyConfig()
)


def _validated_norm(
    name: str,
    vector: tuple[float, float, float] | None,
) -> float:
    if vector is None:
        raise ValueError(f"{name} feedback unavailable")

    if len(vector) != 3:
        raise ValueError(f"{name} must have three components")

    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} feedback is non-finite")

    norm = math.hypot(*vector)

    if not math.isfinite(norm):
        raise ValueError(f"{name} norm is non-finite")

    return norm


def validate_cartesian_wrench(
    *,
    force_xyz: tuple[float, float, float] | None,
    torque_xyz: tuple[float, float, float] | None,
    received_at_s: float | None,
    now_s: float,
    config: CartesianWrenchSafetyConfig = (
        DEFAULT_CARTESIAN_WRENCH_SAFETY_CONFIG
    ),
) -> None:
    """Reject unsafe or stale measured wrist feedback.

    Timestamps use the local monotonic clock at receipt.
    No ROS message-header clock synchronization is assumed.
    """
    if not math.isfinite(now_s):
        raise ValueError("current time is non-finite")

    if received_at_s is None:
        raise ValueError("wrist wrench feedback unavailable")

    if not math.isfinite(received_at_s):
        raise ValueError("wrist wrench timestamp is non-finite")

    age_s = now_s - received_at_s

    if age_s < 0.0 or age_s > config.feedback_timeout_s:
        raise ValueError("wrist wrench feedback stale")

    force_norm = _validated_norm("force", force_xyz)
    torque_norm = _validated_norm("torque", torque_xyz)

    if force_norm >= config.max_force_norm_n:
        raise ValueError(
            f"wrist force limit reached: {force_norm:.3f} N"
        )

    if torque_norm >= config.max_torque_norm_nm:
        raise ValueError(
            f"wrist torque limit reached: {torque_norm:.3f} Nm"
        )
