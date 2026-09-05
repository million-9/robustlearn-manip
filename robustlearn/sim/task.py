"""Geometric task evaluation for Panda insertion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SUCCESS_LATERAL_TOLERANCE = 0.001
SUCCESS_INSERTION_DEPTH = 0.010
FAILURE_LATERAL_ERROR = 0.0022


@dataclass(frozen=True)
class InsertionTaskStatus:
    """Geometric status of the current insertion state."""

    lateral_error: float
    axial_offset: float
    insertion_depth: float
    success: bool
    failure: bool

    @property
    def terminated(self) -> bool:
        """Return whether task semantics terminate the episode."""
        return self.success or self.failure


def evaluate_insertion_task(
    peg_tip_position: FloatArray,
    receptacle_center_position: FloatArray,
    insertion_axis_position: FloatArray,
) -> InsertionTaskStatus:
    """Evaluate insertion progress relative to the configured insertion axis."""
    peg_tip = np.asarray(
        peg_tip_position,
        dtype=np.float64,
    )
    receptacle_center = np.asarray(
        receptacle_center_position,
        dtype=np.float64,
    )
    insertion_axis = np.asarray(
        insertion_axis_position,
        dtype=np.float64,
    )

    positions = {
        "peg_tip_position": peg_tip,
        "receptacle_center_position": receptacle_center,
        "insertion_axis_position": insertion_axis,
    }

    for name, position in positions.items():
        if position.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,), got {position.shape}"
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                "task positions must contain only finite values"
            )

    axis_vector = insertion_axis - receptacle_center
    axis_norm = float(np.linalg.norm(axis_vector))

    if axis_norm <= np.finfo(np.float64).eps:
        raise ValueError(
            "insertion axis must differ from receptacle center"
        )

    axis_direction = axis_vector / axis_norm

    delta = peg_tip - receptacle_center

    axial_offset = float(
        np.dot(
            delta,
            axis_direction,
        )
    )

    lateral_vector = (
        delta
        - axial_offset * axis_direction
    )

    lateral_error = float(
        np.linalg.norm(lateral_vector)
    )

    insertion_depth = -axial_offset

    success = (
        lateral_error <= SUCCESS_LATERAL_TOLERANCE
        and insertion_depth >= SUCCESS_INSERTION_DEPTH
    )

    failure = (
        insertion_depth > 0.0
        and lateral_error >= FAILURE_LATERAL_ERROR
    )

    return InsertionTaskStatus(
        lateral_error=lateral_error,
        axial_offset=axial_offset,
        insertion_depth=insertion_depth,
        success=success,
        failure=failure,
    )
