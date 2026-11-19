"""Tests for the ROS-synchronized viewer joint mapping."""

import numpy as np
import pytest

from robustlearn.sim.viewer_joint_states import (
    extract_panda_joint_positions,
)


def test_panda_joint_names():
    names = [f"panda_joint{i}" for i in range(1, 8)]
    positions = [float(i) for i in range(1, 8)]

    result = extract_panda_joint_positions(names, positions)

    np.testing.assert_array_equal(result, positions)


def test_generic_joint_names():
    names = [f"joint{i}" for i in range(1, 8)]
    positions = [float(i) for i in range(1, 8)]

    result = extract_panda_joint_positions(names, positions)

    np.testing.assert_array_equal(result, positions)


def test_unordered_joint_names():
    names = [f"joint{i}" for i in range(7, 0, -1)]
    positions = [float(i) for i in range(7, 0, -1)]

    result = extract_panda_joint_positions(names, positions)

    np.testing.assert_array_equal(
        result, [1, 2, 3, 4, 5, 6, 7]
    )


def test_extra_joint_is_allowed():
    names = [f"joint{i}" for i in range(1, 8)]
    names.append("finger_joint")

    positions = [float(i) for i in range(1, 8)]
    positions.append(0.02)

    result = extract_panda_joint_positions(names, positions)

    np.testing.assert_array_equal(
        result, [1, 2, 3, 4, 5, 6, 7]
    )


def test_mismatched_lengths_rejected():
    names = [f"joint{i}" for i in range(1, 8)]

    result = extract_panda_joint_positions(
        names, [0.0] * 6
    )

    assert result is None


def test_missing_joint_rejected():
    names = [f"joint{i}" for i in range(1, 7)]

    result = extract_panda_joint_positions(
        names, [0.0] * 6
    )

    assert result is None


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nonfinite_joint_rejected(invalid_value):
    names = [f"joint{i}" for i in range(1, 8)]
    positions = [0.0] * 7
    positions[3] = invalid_value

    result = extract_panda_joint_positions(
        names, positions
    )

    assert result is None


def test_invalid_joint_value_rejected():
    names = [f"joint{i}" for i in range(1, 8)]
    positions = [0.0] * 7
    positions[2] = "invalid"

    result = extract_panda_joint_positions(
        names, positions
    )

    assert result is None
