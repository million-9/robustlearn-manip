"""Tests for deterministic MuJoCo simulation resets."""

import numpy as np
import pytest

from robustlearn.sim.simulation import MuJoCoSimulation, SimulationSnapshot


def assert_snapshots_equal(
    first: SimulationSnapshot,
    second: SimulationSnapshot,
) -> None:
    """Assert that all determinism-controlled snapshot fields match exactly."""
    assert first.time == second.time

    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.qvel, second.qvel)
    np.testing.assert_array_equal(first.act, second.act)
    np.testing.assert_array_equal(first.ctrl, second.ctrl)
    np.testing.assert_array_equal(first.mocap_pos, second.mocap_pos)
    np.testing.assert_array_equal(first.mocap_quat, second.mocap_quat)
    np.testing.assert_array_equal(
        first.qfrc_applied,
        second.qfrc_applied,
    )
    np.testing.assert_array_equal(
        first.xfrc_applied,
        second.xfrc_applied,
    )
    np.testing.assert_array_equal(
        first.task_site_xpos,
        second.task_site_xpos,
    )


def test_same_seed_matches_across_independent_simulators() -> None:
    sim_a = MuJoCoSimulation()
    sim_b = MuJoCoSimulation()

    snapshot_a = sim_a.reset(seed=42)
    snapshot_b = sim_b.reset(seed=42)

    assert_snapshots_equal(snapshot_a, snapshot_b)


def test_repeated_same_seed_reset_reproduces_state() -> None:
    sim = MuJoCoSimulation()

    first = sim.reset(seed=123)
    second = sim.reset(seed=123)
    third = sim.reset(seed=123)

    assert_snapshots_equal(first, second)
    assert_snapshots_equal(second, third)


def test_reset_does_not_depend_on_previous_episode_state() -> None:
    sim = MuJoCoSimulation()

    baseline = sim.reset(seed=42)

    sim.data.time = 123.0
    sim.data.qpos[:] += 0.123
    sim.data.qvel[:] = 5.0
    sim.data.ctrl[:] = -3.0
    sim.data.qfrc_applied[:] = 17.0
    sim.data.xfrc_applied[:] = 23.0
    sim.data.qacc_warmstart[:] = 31.0

    if sim.data.act.size:
        sim.data.act[:] = 7.0

    if sim.data.mocap_pos.size:
        sim.data.mocap_pos[:] = 4.0

    if sim.data.mocap_quat.size:
        sim.data.mocap_quat[:] = 2.0

    restored = sim.reset(seed=42)

    assert_snapshots_equal(baseline, restored)
    np.testing.assert_array_equal(
        sim.data.qacc_warmstart,
        np.zeros_like(sim.data.qacc_warmstart),
    )


def test_reset_after_rollout_reproduces_baseline() -> None:
    sim = MuJoCoSimulation()

    baseline = sim.reset(seed=2026)

    sim.step(500)

    assert sim.data.time > 0.0

    restored = sim.reset(seed=2026)

    assert_snapshots_equal(baseline, restored)


def test_same_seed_reconstructs_rng_stream() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=2026)
    first_sequence = sim.rng.random(10)

    sim.rng.random(100)

    sim.reset(seed=2026)
    second_sequence = sim.rng.random(10)

    np.testing.assert_array_equal(
        first_sequence,
        second_sequence,
    )


def test_same_seed_rng_matches_across_independent_simulators() -> None:
    sim_a = MuJoCoSimulation()
    sim_b = MuJoCoSimulation()

    sim_a.reset(seed=777)
    sim_b.reset(seed=777)

    draws_a = sim_a.rng.random(10)
    draws_b = sim_b.rng.random(10)

    np.testing.assert_array_equal(
        draws_a,
        draws_b,
    )


def test_different_seeds_produce_different_rng_streams() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=1)
    first_sequence = sim.rng.random(10)

    sim.reset(seed=2)
    second_sequence = sim.rng.random(10)

    assert not np.array_equal(
        first_sequence,
        second_sequence,
    )


def test_step_advances_simulation_after_reset() -> None:
    sim = MuJoCoSimulation()

    sim.reset(seed=42)

    initial_time = sim.data.time
    steps = 10

    sim.step(steps)

    assert sim.data.time == pytest.approx(
        initial_time + steps * sim.timestep
    )


def test_step_rejects_non_positive_step_count() -> None:
    sim = MuJoCoSimulation()

    with pytest.raises(
        ValueError,
        match="steps must be at least 1",
    ):
        sim.step(0)


def test_snapshot_is_independent_of_live_simulation_state() -> None:
    sim = MuJoCoSimulation()

    snapshot = sim.reset(seed=42)
    saved_qpos = snapshot.qpos.copy()

    sim.data.qpos[:] += 0.5

    np.testing.assert_array_equal(
        snapshot.qpos,
        saved_qpos,
    )


def test_task_status_uses_live_task_site_geometry() -> None:
    sim = MuJoCoSimulation()
    sim.reset(seed=2026)

    status = sim.task_status()

    assert status.success is False
    assert status.failure is False
    assert status.terminated is False

    assert status.lateral_error == pytest.approx(
        0.000000522,
        abs=1.0e-9,
    )
    assert status.insertion_depth == pytest.approx(
        -0.019502429,
        abs=1.0e-9,
    )
