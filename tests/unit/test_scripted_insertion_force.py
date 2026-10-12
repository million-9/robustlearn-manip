"""Focused force-safety tests for the classical insertion expert."""

import numpy as np
import pytest

from robustlearn.scripted_insertion import (
    ClassicalInsertionExpert,
    ClassicalInsertionExpertConfig,
    InsertionPhase,
)


def _insertion_state() -> dict[str, np.ndarray | float]:
    """Return a simple finite state that places the expert in INSERT."""
    jacobian = np.zeros((6, 7), dtype=np.float64)

    # Give the controller direct translational authority through the first
    # three joints. This keeps the tests independent of MuJoCo while still
    # exercising the real damped least-squares expert calculation.
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[2, 2] = 1.0

    return {
        "current_joint_targets": np.zeros(7, dtype=np.float64),
        "peg_tip_position": np.asarray(
            [0.0, 0.0, 0.005],
            dtype=np.float64,
        ),
        "receptacle_center_position": np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "insertion_axis_position": np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        "peg_tip_jacobian": jacobian,
        "success_insertion_depth_m": 0.010,
    }


def test_force_limits_are_positive_by_default() -> None:
    """Default contact thresholds must be finite positive safety limits."""
    config = ClassicalInsertionExpertConfig()

    assert np.isfinite(config.max_axial_force_n)
    assert np.isfinite(config.max_lateral_force_n)

    assert config.max_axial_force_n > 0.0
    assert config.max_lateral_force_n > 0.0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_axial_force_n", 0.0),
        ("max_lateral_force_n", 0.0),
        ("max_axial_force_n", -1.0),
        ("max_lateral_force_n", -1.0),
        ("max_axial_force_n", np.nan),
        ("max_lateral_force_n", np.inf),
    ],
)
def test_force_limits_reject_invalid_values(
    field_name: str,
    value: float,
) -> None:
    """Contact limits must remain finite and strictly positive."""
    with pytest.raises(ValueError):
        ClassicalInsertionExpertConfig(
            **{field_name: value}
        )


def test_subthreshold_force_allows_insertion_motion() -> None:
    """Nominal contact must not suppress the insertion command."""
    expert = ClassicalInsertionExpert()
    state = _insertion_state()

    current = np.asarray(
        state["current_joint_targets"],
        dtype=np.float64,
    )

    targets = expert.joint_targets(
        **state,
        wrist_force_world_n=np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
    )

    assert expert.phase is InsertionPhase.INSERT

    assert targets.shape == (7,)
    assert np.all(np.isfinite(targets))

    assert not np.array_equal(
        targets,
        current,
    )


def test_axial_force_limit_holds_current_joint_targets() -> None:
    """Excessive axial contact must stop further insertion progression."""
    config = ClassicalInsertionExpertConfig(
        max_axial_force_n=4.0,
    )

    expert = ClassicalInsertionExpert(
        config=config,
    )

    state = _insertion_state()

    current = np.asarray(
        state["current_joint_targets"],
        dtype=np.float64,
    )

    targets = expert.joint_targets(
        **state,
        wrist_force_world_n=np.asarray(
            [0.0, 0.0, 4.0],
            dtype=np.float64,
        ),
    )

    assert expert.phase is InsertionPhase.INSERT

    np.testing.assert_array_equal(
        targets,
        current,
    )


def test_axial_force_limit_uses_force_magnitude() -> None:
    """Axial protection must work for either force direction."""
    config = ClassicalInsertionExpertConfig(
        max_axial_force_n=4.0,
    )

    expert = ClassicalInsertionExpert(
        config=config,
    )

    state = _insertion_state()

    current = np.asarray(
        state["current_joint_targets"],
        dtype=np.float64,
    )

    targets = expert.joint_targets(
        **state,
        wrist_force_world_n=np.asarray(
            [0.0, 0.0, -5.0],
            dtype=np.float64,
        ),
    )

    assert expert.phase is InsertionPhase.INSERT

    np.testing.assert_array_equal(
        targets,
        current,
    )


def test_lateral_force_limit_aborts_expert() -> None:
    """Excessive lateral contact must enter the ABORT phase."""
    config = ClassicalInsertionExpertConfig(
        max_lateral_force_n=3.0,
    )

    expert = ClassicalInsertionExpert(
        config=config,
    )

    state = _insertion_state()

    with pytest.raises(
        RuntimeError,
        match="lateral contact force",
    ):
        expert.joint_targets(
            **state,
            wrist_force_world_n=np.asarray(
                [3.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        )

    assert expert.phase is InsertionPhase.ABORT


def test_non_finite_force_aborts_expert() -> None:
    """The expert must never process non-finite contact feedback."""
    expert = ClassicalInsertionExpert()
    state = _insertion_state()

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        expert.joint_targets(
            **state,
            wrist_force_world_n=np.asarray(
                [np.nan, 0.0, 0.0],
                dtype=np.float64,
            ),
        )

    assert expert.phase is InsertionPhase.ABORT


def test_force_with_invalid_shape_aborts_expert() -> None:
    """Malformed wrench-derived force input must fail safely."""
    expert = ClassicalInsertionExpert()
    state = _insertion_state()

    with pytest.raises(
        ValueError,
        match="wrist_force_world_n",
    ):
        expert.joint_targets(
            **state,
            wrist_force_world_n=np.zeros(
                6,
                dtype=np.float64,
            ),
        )

    assert expert.phase is InsertionPhase.ABORT
