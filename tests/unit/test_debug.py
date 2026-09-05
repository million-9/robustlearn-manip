"""Tests for Panda insertion debug-data generation."""

import numpy as np

from robustlearn.sim.debug import insertion_debug_snapshot
from robustlearn.sim.simulation import MuJoCoSimulation


def test_insertion_debug_snapshot_is_headless_and_finite() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    debug = insertion_debug_snapshot(simulation)

    assert np.isfinite(debug.simulation_time)
    assert np.isfinite(debug.lateral_error)
    assert np.isfinite(debug.axial_offset)
    assert np.isfinite(debug.insertion_depth)

    assert debug.wrist_force.shape == (3,)
    assert debug.wrist_torque.shape == (3,)
    assert np.all(np.isfinite(debug.wrist_force))
    assert np.all(np.isfinite(debug.wrist_torque))

    assert debug.success is False
    assert debug.failure is False
