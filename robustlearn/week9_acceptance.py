"""Week 9 clean hybrid-acceptance metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robustlearn.config import PandaInsertionTaskConfig
from robustlearn.sim.task import evaluate_insertion_task

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Week9InsertionMetrics:
    """Finite task and contact metrics for one Week 9 observation."""

    lateral_error_m: float
    axial_offset_m: float
    insertion_depth_m: float
    axial_force_n: float
    lateral_force_n: float
    force_norm_n: float
    task_success: bool
    task_failure: bool


def evaluate_week9_insertion_metrics(
    peg_tip_position: FloatArray,
    receptacle_center_position: FloatArray,
    insertion_axis_position: FloatArray,
    wrist_force_world_n: FloatArray,
    *,
    task_config: PandaInsertionTaskConfig | None = None,
) -> Week9InsertionMetrics:
    """Evaluate finite Week 9 insertion geometry and contact-force metrics."""

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

    wrist_force = np.asarray(
        wrist_force_world_n,
        dtype=np.float64,
    )

    vectors = {
        'peg_tip_position': peg_tip,
        'receptacle_center_position': receptacle_center,
        'insertion_axis_position': insertion_axis,
        'wrist_force_world_n': wrist_force,
    }

    for name, vector in vectors.items():
        if vector.shape != (3,):
            raise ValueError(
                f'{name} must have shape (3,), got {vector.shape}'
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f'{name} must contain only finite values'
            )

    axis_vector = (
        insertion_axis
        - receptacle_center
    )

    axis_norm = float(
        np.linalg.norm(
            axis_vector
        )
    )

    if (
        not np.isfinite(axis_norm)
        or axis_norm <= np.finfo(np.float64).eps
    ):
        raise ValueError(
            'insertion axis must differ from receptacle center'
        )

    axis_direction = (
        axis_vector
        / axis_norm
    )

    task_status = evaluate_insertion_task(
        peg_tip,
        receptacle_center,
        insertion_axis,
        config=task_config,
    )

    signed_axial_force = float(
        np.dot(
            wrist_force,
            axis_direction,
        )
    )

    axial_force = abs(
        signed_axial_force
    )

    lateral_force_vector = (
        wrist_force
        - signed_axial_force * axis_direction
    )

    lateral_force = float(
        np.linalg.norm(
            lateral_force_vector
        )
    )

    force_norm = float(
        np.linalg.norm(
            wrist_force
        )
    )

    metrics = np.asarray(
        [
            task_status.lateral_error,
            task_status.axial_offset,
            task_status.insertion_depth,
            axial_force,
            lateral_force,
            force_norm,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(metrics)
    ):
        raise RuntimeError(
            'Week 9 acceptance metrics became non-finite'
        )

    return Week9InsertionMetrics(
        lateral_error_m=float(
            task_status.lateral_error
        ),
        axial_offset_m=float(
            task_status.axial_offset
        ),
        insertion_depth_m=float(
            task_status.insertion_depth
        ),
        axial_force_n=axial_force,
        lateral_force_n=lateral_force,
        force_norm_n=force_norm,
        task_success=bool(
            task_status.success
        ),
        task_failure=bool(
            task_status.failure
        ),
    )
