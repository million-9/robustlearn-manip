"""Headless diagnostic data for Panda insertion debugging."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robustlearn.sim.simulation import MuJoCoSimulation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InsertionDebugSnapshot:
    """Task and wrist-wrench diagnostics for developer tooling."""

    simulation_time: float
    lateral_error: float
    axial_offset: float
    insertion_depth: float
    success: bool
    failure: bool
    wrist_force: FloatArray
    wrist_torque: FloatArray


def insertion_debug_snapshot(
    simulation: MuJoCoSimulation,
) -> InsertionDebugSnapshot:
    """Collect task and sensing diagnostics without requiring a GUI."""
    task = simulation.task_status()
    sensors = simulation.sensor_snapshot()

    return InsertionDebugSnapshot(
        simulation_time=float(simulation.data.time),
        lateral_error=task.lateral_error,
        axial_offset=task.axial_offset,
        insertion_depth=task.insertion_depth,
        success=task.success,
        failure=task.failure,
        wrist_force=sensors.wrist_force.copy(),
        wrist_torque=sensors.wrist_torque.copy(),
    )
