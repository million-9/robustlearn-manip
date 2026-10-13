"""Tests for Week 9 hybrid-acceptance metric helpers."""

import numpy as np
import pytest

from robustlearn.config import PandaInsertionTaskConfig
from robustlearn.week9_acceptance import evaluate_week9_insertion_metrics


def test_week9_metrics_decompose_axial_and_lateral_force() -> None:
    metrics = evaluate_week9_insertion_metrics(
        peg_tip_position=np.asarray(
            [0.001, 0.0, -0.010],
            dtype=np.float64,
        ),
        receptacle_center_position=np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        insertion_axis_position=np.asarray(
            [0.0, 0.0, 0.050],
            dtype=np.float64,
        ),
        wrist_force_world_n=np.asarray(
            [3.0, 4.0, -12.0],
            dtype=np.float64,
        ),
    )

    assert metrics.lateral_error_m == pytest.approx(
        0.001
    )

    assert metrics.axial_offset_m == pytest.approx(
        -0.010
    )

    assert metrics.insertion_depth_m == pytest.approx(
        0.010
    )

    assert metrics.axial_force_n == pytest.approx(
        12.0
    )

    assert metrics.lateral_force_n == pytest.approx(
        5.0
    )

    assert metrics.force_norm_n == pytest.approx(
        13.0
    )


def test_week9_metrics_follow_rotated_insertion_axis() -> None:
    metrics = evaluate_week9_insertion_metrics(
        peg_tip_position=np.asarray(
            [-0.020, 0.003, 0.0],
            dtype=np.float64,
        ),
        receptacle_center_position=np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        insertion_axis_position=np.asarray(
            [0.050, 0.0, 0.0],
            dtype=np.float64,
        ),
        wrist_force_world_n=np.asarray(
            [-7.0, 2.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert metrics.axial_offset_m == pytest.approx(
        -0.020
    )

    assert metrics.insertion_depth_m == pytest.approx(
        0.020
    )

    assert metrics.lateral_error_m == pytest.approx(
        0.003
    )

    assert metrics.axial_force_n == pytest.approx(
        7.0
    )

    assert metrics.lateral_force_n == pytest.approx(
        2.0
    )


def test_week9_metrics_preserve_task_success_semantics() -> None:
    config = PandaInsertionTaskConfig()

    metrics = evaluate_week9_insertion_metrics(
        peg_tip_position=np.asarray(
            [
                0.0,
                0.0,
                -config.success_insertion_depth_m,
            ],
            dtype=np.float64,
        ),
        receptacle_center_position=np.zeros(
            3,
            dtype=np.float64,
        ),
        insertion_axis_position=np.asarray(
            [0.0, 0.0, 0.050],
            dtype=np.float64,
        ),
        wrist_force_world_n=np.zeros(
            3,
            dtype=np.float64,
        ),
        task_config=config,
    )

    assert metrics.task_success
    assert not metrics.task_failure


@pytest.mark.parametrize(
    'bad_force',
    [
        np.asarray(
            [np.nan, 0.0, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            [np.inf, 0.0, 0.0],
            dtype=np.float64,
        ),
    ],
)
def test_week9_metrics_reject_nonfinite_force(
    bad_force: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match='wrist_force_world_n must contain only finite values',
    ):
        evaluate_week9_insertion_metrics(
            peg_tip_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            receptacle_center_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            insertion_axis_position=np.asarray(
                [0.0, 0.0, 0.050],
                dtype=np.float64,
            ),
            wrist_force_world_n=bad_force,
        )


def test_week9_metrics_reject_zero_length_axis() -> None:
    with pytest.raises(
        ValueError,
        match='insertion axis must differ from receptacle center',
    ):
        evaluate_week9_insertion_metrics(
            peg_tip_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            receptacle_center_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            insertion_axis_position=np.zeros(
                3,
                dtype=np.float64,
            ),
            wrist_force_world_n=np.zeros(
                3,
                dtype=np.float64,
            ),
        )
