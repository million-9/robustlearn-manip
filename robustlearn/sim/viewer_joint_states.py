"""Validate and map Panda ROS joint-state observations."""

from collections.abc import Sequence

import numpy as np


def extract_panda_joint_positions(
    names: Sequence[str],
    positions: Sequence[float],
) -> np.ndarray | None:
    """Return seven ordered finite Panda joints, or None."""
    if len(names) != len(positions):
        return None

    mapping = dict(zip(names, positions, strict=True))

    for prefix in ("panda_joint", "joint"):
        joint_names = [
            f"{prefix}{index}" for index in range(1, 8)
        ]

        if not all(name in mapping for name in joint_names):
            continue

        try:
            result = np.asarray(
                [mapping[name] for name in joint_names],
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError):
            return None

        if np.all(np.isfinite(result)):
            return result

        return None

    return None
