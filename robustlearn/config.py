"""Validated configuration for the Panda insertion task."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class FloatRange:
    """Inclusive finite range used by randomization configuration."""

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate finite, ordered range bounds."""
        if not isfinite(self.low) or not isfinite(self.high):
            raise ValueError("range bounds must be finite")

        if self.low > self.high:
            raise ValueError("range low bound must not exceed high bound")

    def to_dict(self) -> dict[str, float]:
        """Return a stable serializable representation."""
        return {
            "low": self.low,
            "high": self.high,
        }


@dataclass(frozen=True)
class PandaInsertionRandomizationConfig:
    """Episode-level randomization configuration."""

    enabled: bool = False
    receptacle_x_offset_m: FloatRange = field(
        default_factory=lambda: FloatRange(0.0, 0.0)
    )
    receptacle_y_offset_m: FloatRange = field(
        default_factory=lambda: FloatRange(0.0, 0.0)
    )
    receptacle_yaw_offset_rad: FloatRange = field(
        default_factory=lambda: FloatRange(0.0, 0.0)
    )

    def to_dict(self) -> dict[str, object]:
        """Return a stable serializable representation."""
        return {
            "enabled": self.enabled,
            "receptacle_x_offset_m": self.receptacle_x_offset_m.to_dict(),
            "receptacle_y_offset_m": self.receptacle_y_offset_m.to_dict(),
            "receptacle_yaw_offset_rad": self.receptacle_yaw_offset_rad.to_dict(),
        }


@dataclass(frozen=True)
class PandaInsertionTaskConfig:
    """Validated runtime configuration for the Panda insertion task."""

    frame_skip: int = 5
    max_episode_steps: int = 200

    success_lateral_tolerance_m: float = 0.001
    success_insertion_depth_m: float = 0.010
    failure_lateral_error_m: float = 0.0022

    randomization: PandaInsertionRandomizationConfig = field(
        default_factory=PandaInsertionRandomizationConfig
    )

    def __post_init__(self) -> None:
        """Validate task runtime parameters."""
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        thresholds = {
            "success_lateral_tolerance_m": self.success_lateral_tolerance_m,
            "success_insertion_depth_m": self.success_insertion_depth_m,
            "failure_lateral_error_m": self.failure_lateral_error_m,
        }

        for name, value in thresholds.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        if self.failure_lateral_error_m <= self.success_lateral_tolerance_m:
            raise ValueError(
                "failure_lateral_error_m must exceed "
                "success_lateral_tolerance_m"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable serializable representation."""
        return {
            "frame_skip": self.frame_skip,
            "max_episode_steps": self.max_episode_steps,
            "success_lateral_tolerance_m": self.success_lateral_tolerance_m,
            "success_insertion_depth_m": self.success_insertion_depth_m,
            "failure_lateral_error_m": self.failure_lateral_error_m,
            "randomization": self.randomization.to_dict(),
        }
