"""Reference-relative Cartesian controller tests."""

import numpy as np
import pytest

from robustlearn.cartesian_teleop import (
    reference_relative_cartesian_target,
)

MEASURED = np.array([
    0.10, -0.0881082, 0.1002129, -1.7129679,
    0.0992131, 1.6975185, -0.6999972,
])

REFERENCE = np.array([
    0.10, -0.0940809, 0.1001099, -1.7064780,
    0.0996066, 1.6987537, -0.6999986,
])

DELTA = np.array([
    -0.00010823, 0.00308753, -0.00007505,
    0.00347922, -0.00030953, -0.00041295,
    -0.00009844,
])


def test_reference_relative_command_preserves_ik_increment():
    """A gravity-offset holding reference must not shift the intended jog."""
    result = reference_relative_cartesian_target(
        MEASURED, REFERENCE, MEASURED + DELTA
    )

    np.testing.assert_allclose(
        result,
        REFERENCE + DELTA,
        rtol=0,
        atol=1e-12,
    )

    assert np.max(np.abs(result - MEASURED)) > np.max(
        np.abs(DELTA)
    )


def test_zero_reference_offset_matches_original_ik():
    result = reference_relative_cartesian_target(
        MEASURED, MEASURED, MEASURED + DELTA
    )

    np.testing.assert_allclose(
        result, MEASURED + DELTA, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "invalid_reference",
    [
        np.full(7, np.nan),
        np.full(7, np.inf),
        np.zeros(6),
        MEASURED + np.array(
            [0, 0.011, 0, 0, 0, 0, 0]
        ),
        MEASURED + np.array(
            [0, 0, 0, 5, 0, 0, 0]
        ),
    ],
)
def test_invalid_reference_fails_closed(invalid_reference):
    with pytest.raises(ValueError):
        reference_relative_cartesian_target(
            MEASURED,
            invalid_reference,
            MEASURED + DELTA,
        )


def test_excessive_joint_increment_rejected():
    excessive = DELTA.copy()
    excessive[1] = 0.025

    with pytest.raises(ValueError):
        reference_relative_cartesian_target(
            MEASURED,
            REFERENCE,
            MEASURED + excessive,
        )


def test_zero_increment_rejected():
    with pytest.raises(ValueError):
        reference_relative_cartesian_target(
            MEASURED, REFERENCE, MEASURED
        )
