"""Regression tests for the deterministic scripted Panda insertion sequence."""

import gymnasium as gym
import numpy as np
import pytest

from robustlearn.config import (
    FloatRange,
    PandaInsertionRandomizationConfig,
    PandaInsertionTaskConfig,
)
from robustlearn.envs.panda_insertion import (
    PandaInsertionEnv,
)
from robustlearn.scripted_insertion import (
    ClassicalInsertionExpert,
    ClassicalInsertionExpertConfig,
    InsertionPhase,
    run_scripted_insertion,
)
from robustlearn.sim.control import (
    current_control_targets,
    panda_arm_actuator_ids,
    panda_site_spatial_jacobian,
)


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


def test_classical_expert_preserves_non_arm_actuator_targets() -> None:
    env = PandaInsertionEnv()
    env.reset(seed=2026)

    baseline = current_control_targets(env.simulation.data)

    arm_ids = set(panda_arm_actuator_ids(env.simulation.model))

    expert = ClassicalInsertionExpert()
    action = expert.action(env)

    for actuator_index, (
        before,
        after,
    ) in enumerate(
        zip(
            baseline,
            action,
            strict=True,
        )
    ):
        if actuator_index not in arm_ids:
            assert after == before


def test_classical_expert_aligns_before_inserting_from_fixed_offset() -> None:
    config = PandaInsertionTaskConfig(
        randomization=(
            PandaInsertionRandomizationConfig(
                enabled=True,
                receptacle_x_offset_m=FloatRange(
                    0.001,
                    0.001,
                ),
            )
        )
    )

    env = PandaInsertionEnv(config=config)
    env.reset(seed=2026)

    expert = ClassicalInsertionExpert()

    phases: list[InsertionPhase] = []

    final_info: dict[
        str,
        object,
    ] = {}

    for _ in range(config.max_episode_steps):
        action = expert.action(env)

        phases.append(expert.phase)

        (
            _,
            _,
            terminated,
            truncated,
            final_info,
        ) = env.step(action)

        if terminated or truncated:
            break

    assert phases
    assert phases[0] is InsertionPhase.ALIGN

    assert InsertionPhase.INSERT in phases

    first_insert_index = phases.index(InsertionPhase.INSERT)

    assert all(phase is InsertionPhase.ALIGN for phase in phases[:first_insert_index])

    assert final_info["task_success"] is True

    assert final_info["task_failure"] is False


def test_scripted_insertion_records_expert_phases() -> None:
    diagnostics = run_scripted_insertion(seed=2026)

    phases = [record["phase"] for record in diagnostics]

    assert phases

    assert InsertionPhase.INSERT.value in phases

    assert InsertionPhase.ABORT.value not in phases

    assert set(phases).issubset(
        {
            InsertionPhase.ALIGN.value,
            InsertionPhase.INSERT.value,
            InsertionPhase.HOLD.value,
        }
    )


def test_classical_expert_action_is_finite_and_bounded() -> None:
    env = PandaInsertionEnv()
    env.reset(seed=2026)

    expert = ClassicalInsertionExpert()
    action = expert.action(env)

    assert np.all(np.isfinite(action))

    assert isinstance(
        env.action_space,
        gym.spaces.Box,
    )

    np.testing.assert_array_less(
        action,
        np.asarray(
            env.action_space.high,
            dtype=np.float64,
        )
        + 1.0e-12,
    )

    np.testing.assert_array_less(
        np.asarray(
            env.action_space.low,
            dtype=np.float64,
        )
        - 1.0e-12,
        action,
    )


def test_classical_expert_action_is_deterministic_for_identical_state() -> None:
    first_env = PandaInsertionEnv()
    second_env = PandaInsertionEnv()

    first_env.reset(seed=2026)
    second_env.reset(seed=2026)

    first_expert = ClassicalInsertionExpert()

    second_expert = ClassicalInsertionExpert()

    first_action = first_expert.action(first_env)

    second_action = second_expert.action(second_env)

    np.testing.assert_array_equal(
        first_action,
        second_action,
    )

    assert first_expert.phase is second_expert.phase


def test_classical_expert_phase_progression() -> None:
    expert = ClassicalInsertionExpert()

    phase = expert.update_phase(
        lateral_error_m=0.001,
        insertion_depth_m=0.0,
        success_insertion_depth_m=0.010,
    )

    assert phase is InsertionPhase.ALIGN

    phase = expert.update_phase(
        lateral_error_m=0.0001,
        insertion_depth_m=0.0,
        success_insertion_depth_m=0.010,
    )

    assert phase is InsertionPhase.INSERT

    phase = expert.update_phase(
        lateral_error_m=0.0001,
        insertion_depth_m=0.010,
        success_insertion_depth_m=0.010,
    )

    assert phase is InsertionPhase.HOLD


def test_classical_expert_enters_abort_on_failure() -> None:
    expert = ClassicalInsertionExpert()

    phase = expert.update_phase(
        lateral_error_m=0.0,
        insertion_depth_m=0.0,
        success_insertion_depth_m=0.010,
        failed=True,
    )

    assert phase is InsertionPhase.ABORT


def test_classical_expert_enters_abort_on_non_finite_state() -> None:
    expert = ClassicalInsertionExpert()

    phase = expert.update_phase(
        lateral_error_m=np.nan,
        insertion_depth_m=0.0,
        success_insertion_depth_m=0.010,
    )

    assert phase is InsertionPhase.ABORT


def test_classical_expert_starts_in_align_phase() -> None:
    expert = ClassicalInsertionExpert()

    assert expert.phase is InsertionPhase.ALIGN


def test_classical_expert_reset_restores_align_phase() -> None:
    expert = ClassicalInsertionExpert()

    expert.phase = InsertionPhase.INSERT

    expert.reset()

    assert expert.phase is InsertionPhase.ALIGN


def test_classical_expert_phases_are_stable_string_values() -> None:
    assert InsertionPhase.ALIGN.value == "align"

    assert InsertionPhase.INSERT.value == "insert"

    assert InsertionPhase.HOLD.value == "hold"

    assert InsertionPhase.ABORT.value == "abort"


def test_classical_expert_default_config_is_valid() -> None:
    config = ClassicalInsertionExpertConfig()

    assert config.cartesian_step_m > 0.0

    assert config.damping > 0.0

    assert config.max_joint_step_rad > 0.0

    assert config.alignment_tolerance_m > 0.0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        (
            "cartesian_step_m",
            0.0,
        ),
        (
            "damping",
            0.0,
        ),
        (
            "max_joint_step_rad",
            0.0,
        ),
        (
            "alignment_tolerance_m",
            0.0,
        ),
        (
            "cartesian_step_m",
            np.nan,
        ),
        (
            "damping",
            np.inf,
        ),
        (
            "max_joint_step_rad",
            -np.inf,
        ),
    ],
)
def test_classical_expert_config_rejects_non_positive_values(
    field_name: str,
    value: float,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValueError):
        ClassicalInsertionExpertConfig(**kwargs)


def test_scripted_insertion_reaches_success_deterministically() -> None:
    first = run_scripted_insertion(seed=2026)

    second = run_scripted_insertion(seed=2026)

    assert first
    assert second

    final = first[-1]

    assert final["success"] is True
    assert final["failure"] is False
    assert final["truncated"] is False

    assert final["phase"] in {
        InsertionPhase.INSERT.value,
        InsertionPhase.HOLD.value,
    }

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


def test_classical_expert_joint_targets_returns_seven_finite_targets() -> None:
    env = PandaInsertionEnv()
    env.reset(seed=2026)

    expert = ClassicalInsertionExpert()

    simulation = env.simulation
    snapshot = simulation.snapshot()

    site_positions = {
        name: position
        for name, position in zip(
            ("peg_tip", "receptacle_center", "insertion_axis", "pre_insertion"),
            snapshot.task_site_xpos,
            strict=True,
        )
    }

    arm_ids = panda_arm_actuator_ids(simulation.model)

    current = np.asarray(
        [
            simulation.data.ctrl[actuator_index]
            for actuator_index in arm_ids
        ],
        dtype=np.float64,
    )

    targets = expert.joint_targets(
        current_joint_targets=current,
        peg_tip_position=site_positions["peg_tip"],
        receptacle_center_position=site_positions["receptacle_center"],
        insertion_axis_position=site_positions["insertion_axis"],
        peg_tip_jacobian=panda_site_spatial_jacobian(
            simulation.model,
            simulation.data,
            "peg_tip",
        ),
        success_insertion_depth_m=env.config.success_insertion_depth_m,
    )

    assert targets.shape == (7,)
    assert np.all(np.isfinite(targets))

    np.testing.assert_array_less(
        np.abs(targets - current),
        np.full(7, expert.config.max_joint_step_rad + 1.0e-12),
    )


def test_classical_expert_joint_targets_is_deterministic() -> None:
    env = PandaInsertionEnv()
    env.reset(seed=2026)

    simulation = env.simulation
    snapshot = simulation.snapshot()

    site_positions = {
        name: position
        for name, position in zip(
            ("peg_tip", "receptacle_center", "insertion_axis", "pre_insertion"),
            snapshot.task_site_xpos,
            strict=True,
        )
    }

    arm_ids = panda_arm_actuator_ids(simulation.model)

    current = np.asarray(
        [
            simulation.data.ctrl[actuator_index]
            for actuator_index in arm_ids
        ],
        dtype=np.float64,
    )

    jacobian = panda_site_spatial_jacobian(
        simulation.model,
        simulation.data,
        "peg_tip",
    )

    first = ClassicalInsertionExpert().joint_targets(
        current_joint_targets=current,
        peg_tip_position=site_positions["peg_tip"],
        receptacle_center_position=site_positions["receptacle_center"],
        insertion_axis_position=site_positions["insertion_axis"],
        peg_tip_jacobian=jacobian,
        success_insertion_depth_m=env.config.success_insertion_depth_m,
    )

    second = ClassicalInsertionExpert().joint_targets(
        current_joint_targets=current,
        peg_tip_position=site_positions["peg_tip"],
        receptacle_center_position=site_positions["receptacle_center"],
        insertion_axis_position=site_positions["insertion_axis"],
        peg_tip_jacobian=jacobian,
        success_insertion_depth_m=env.config.success_insertion_depth_m,
    )

    np.testing.assert_array_equal(first, second)


def test_classical_expert_joint_targets_rejects_non_finite_input() -> None:
    expert = ClassicalInsertionExpert()

    current = np.zeros(7, dtype=np.float64)

    peg_tip = np.zeros(3, dtype=np.float64)
    peg_tip[0] = np.nan

    with pytest.raises(ValueError):
        expert.joint_targets(
            current_joint_targets=current,
            peg_tip_position=peg_tip,
            receptacle_center_position=np.zeros(3, dtype=np.float64),
            insertion_axis_position=np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            ),
            peg_tip_jacobian=np.zeros((6, 7), dtype=np.float64),
            success_insertion_depth_m=0.01,
        )

    assert expert.phase is InsertionPhase.ABORT


def test_classical_expert_joint_targets_rejects_invalid_shapes() -> None:
    expert = ClassicalInsertionExpert()

    with pytest.raises(ValueError):
        expert.joint_targets(
            current_joint_targets=np.zeros(6, dtype=np.float64),
            peg_tip_position=np.zeros(3, dtype=np.float64),
            receptacle_center_position=np.zeros(3, dtype=np.float64),
            insertion_axis_position=np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            ),
            peg_tip_jacobian=np.zeros((6, 7), dtype=np.float64),
            success_insertion_depth_m=0.01,
        )

    assert expert.phase is InsertionPhase.ABORT
