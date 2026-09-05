"""Tests for geometric Panda insertion task evaluation."""

import numpy as np
import pytest

from robustlearn.sim.task import (
    FAILURE_LATERAL_ERROR,
    SUCCESS_INSERTION_DEPTH,
    SUCCESS_LATERAL_TOLERANCE,
    evaluate_insertion_task,
)


def z_axis() -> np.ndarray:
    """Return a unit insertion-axis marker above the origin."""
    return np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )


def test_canonical_pre_insertion_state_is_in_progress() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [0.554499478, 0.0, 0.454502429],
            dtype=np.float64,
        ),
        np.asarray(
            [0.5545, 0.0, 0.435],
            dtype=np.float64,
        ),
        np.asarray(
            [0.5545, 0.0, 0.485],
            dtype=np.float64,
        ),
    )

    assert status.success is False
    assert status.failure is False
    assert status.terminated is False

    assert status.lateral_error == pytest.approx(
        0.000000522,
        abs=1.0e-9,
    )
    assert status.axial_offset == pytest.approx(
        0.019502429,
    )
    assert status.insertion_depth == pytest.approx(
        -0.019502429,
    )


def test_success_at_exact_geometric_boundary() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [
                SUCCESS_LATERAL_TOLERANCE,
                0.0,
                -SUCCESS_INSERTION_DEPTH,
            ],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        z_axis(),
    )

    assert status.success is True
    assert status.failure is False
    assert status.terminated is True


def test_state_just_outside_success_lateral_tolerance_is_not_success() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [
                SUCCESS_LATERAL_TOLERANCE + 1.0e-6,
                0.0,
                -SUCCESS_INSERTION_DEPTH,
            ],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        z_axis(),
    )

    assert status.success is False
    assert status.failure is False
    assert status.terminated is False


def test_state_just_short_of_success_depth_is_not_success() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [
                0.0,
                0.0,
                -(SUCCESS_INSERTION_DEPTH - 1.0e-6),
            ],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        z_axis(),
    )

    assert status.success is False
    assert status.failure is False
    assert status.terminated is False


def test_misaligned_inserted_state_is_failure() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [
                FAILURE_LATERAL_ERROR,
                0.0,
                -0.001,
            ],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        z_axis(),
    )

    assert status.success is False
    assert status.failure is True
    assert status.terminated is True


def test_misaligned_state_above_receptacle_is_not_failure() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [
                FAILURE_LATERAL_ERROR,
                0.0,
                0.001,
            ],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        z_axis(),
    )

    assert status.success is False
    assert status.failure is False
    assert status.terminated is False


def test_metrics_follow_rotated_insertion_axis() -> None:
    status = evaluate_insertion_task(
        np.asarray(
            [-0.010, 0.002, 0.0],
            dtype=np.float64,
        ),
        np.zeros(3, dtype=np.float64),
        np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert status.axial_offset == pytest.approx(-0.010)
    assert status.insertion_depth == pytest.approx(0.010)
    assert status.lateral_error == pytest.approx(0.002)


@pytest.mark.parametrize(
    "peg_tip,receptacle,axis",
    [
        (
            np.zeros(2, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            z_axis(),
        ),
        (
            np.zeros(3, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            z_axis(),
        ),
        (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
    ],
)
def test_invalid_position_shape_is_rejected(
    peg_tip: np.ndarray,
    receptacle: np.ndarray,
    axis: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        evaluate_insertion_task(
            peg_tip,
            receptacle,
            axis,
        )


def test_nonfinite_position_is_rejected() -> None:
    peg_tip = np.asarray(
        [0.0, np.nan, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        evaluate_insertion_task(
            peg_tip,
            np.zeros(3, dtype=np.float64),
            z_axis(),
        )


def test_zero_length_insertion_axis_is_rejected() -> None:
    receptacle = np.asarray(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="insertion axis must differ",
    ):
        evaluate_insertion_task(
            np.zeros(3, dtype=np.float64),
            receptacle,
            receptacle.copy(),
        )
