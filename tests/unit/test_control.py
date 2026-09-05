"""Tests for named Panda control utilities."""

import numpy as np

from robustlearn.sim.control import (
    panda_arm_actuator_ids,
    panda_gripper_actuator_id,
    panda_site_translation_jacobian,
)
from robustlearn.sim.simulation import MuJoCoSimulation


def test_named_panda_actuators_resolve() -> None:
    simulation = MuJoCoSimulation()

    arm_ids = panda_arm_actuator_ids(simulation.model)
    gripper_id = panda_gripper_actuator_id(simulation.model)

    assert len(arm_ids) == 7
    assert len(set(arm_ids)) == 7
    assert gripper_id not in arm_ids


def test_peg_tip_translation_jacobian_is_finite() -> None:
    simulation = MuJoCoSimulation()
    simulation.reset(seed=2026)

    jacobian = panda_site_translation_jacobian(
        simulation.model,
        simulation.data,
        "peg_tip",
    )

    assert jacobian.shape == (3, 7)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.norm(jacobian) > 0.0
