"""Regression tests for the deterministic scripted Panda insertion sequence."""

import numpy as np

from robustlearn.scripted_insertion import run_scripted_insertion


def _scalar_trace(
    diagnostics: list[dict[str, object]],
) -> np.ndarray:
    """Return deterministic scalar diagnostics for trace comparison."""
    return np.asarray(
        [
            [
                record["step"],
                record["simulation_time"],
                record["lateral_error"],
                record["axial_offset"],
                record["insertion_depth"],
                record["success"],
                record["failure"],
                record["truncated"],
            ]
            for record in diagnostics
        ],
        dtype=np.float64,
    )


def test_scripted_insertion_reaches_success_deterministically() -> None:
    first = run_scripted_insertion(seed=2026)
    second = run_scripted_insertion(seed=2026)

    assert first
    assert second

    final = first[-1]

    assert final["success"] is True
    assert final["failure"] is False
    assert final["truncated"] is False

    first_trace = _scalar_trace(first)
    second_trace = _scalar_trace(second)

    assert np.all(np.isfinite(first_trace))
    assert np.all(np.isfinite(second_trace))

    np.testing.assert_array_equal(
        first_trace,
        second_trace,
    )

    for record in first:
        wrist_force = np.asarray(
            record["wrist_force"],
            dtype=np.float64,
        )
        wrist_torque = np.asarray(
            record["wrist_torque"],
            dtype=np.float64,
        )

        assert wrist_force.shape == (3,)
        assert wrist_torque.shape == (3,)
        assert np.all(np.isfinite(wrist_force))
        assert np.all(np.isfinite(wrist_torque))
