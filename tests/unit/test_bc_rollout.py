"""Integration and safety tests for the BC closed-loop runner."""

from pathlib import Path

import numpy as np
import pytest

import robustlearn.bc_rollout as rollout
from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.sim.control import panda_arm_actuator_ids


@pytest.fixture
def checkpoint_stub(monkeypatch):
    """Replace checkpoint inference with deterministic test actions."""
    env = PandaInsertionEnv()
    try:
        env.reset(seed=2026)
        ids = panda_arm_actuator_ids(env.simulation.model)
        targets = env.simulation.data.ctrl[list(ids)].copy()
    finally:
        env.close()

    class TestPolicy:
        def __init__(self, checkpoint, *, expected_dataset_sha256):
            assert checkpoint == Path("test-checkpoint.pt")
            assert expected_dataset_sha256 == "test-fingerprint"

        def predict(self, observation):
            assert observation.shape == (23,)
            return targets.astype(np.float32)

    monkeypatch.setattr(rollout, "BCInference", TestPolicy)
    monkeypatch.setattr(
        rollout.hashlib,
        "sha256",
        lambda data: type("Hash", (), {"hexdigest": lambda self: "test"})(),
    )
    monkeypatch.setattr(
        rollout.Path,
        "read_bytes",
        lambda self: b"test checkpoint",
    )


def test_rollout_executes_policy_controlled_steps(checkpoint_stub):
    result = rollout.run_bc_rollout(
        "test-checkpoint.pt",
        "test-fingerprint",
        seed=2026,
        max_steps=2,
        max_force_n=1e9,
    )

    assert result["steps_executed"] >= 1
    assert result["steps_executed"] <= 2
    assert result["elapsed_simulation_time_s"] > 0
    assert len(result["trajectory"]) == result["steps_executed"]

    first = result["trajectory"][0]
    assert len(first["observation"]) == 23
    assert len(first["policy_targets_rad"]) == 7
    assert len(first["applied_actuator_controls"]) == 8
    assert first["time_s"] > 0


def test_same_seed_reproduces_trajectory(checkpoint_stub):
    kwargs = {
        "seed": 2026,
        "max_steps": 2,
        "max_force_n": 1e9,
    }

    first = rollout.run_bc_rollout(
        "test-checkpoint.pt", "test-fingerprint", **kwargs
    )
    second = rollout.run_bc_rollout(
        "test-checkpoint.pt", "test-fingerprint", **kwargs
    )

    assert first["outcome"] == second["outcome"]
    assert first["trajectory"] == second["trajectory"]


def test_out_of_range_policy_action_stops_safely(
    checkpoint_stub, monkeypatch
):
    class UnsafePolicy:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, observation):
            return np.full(7, 1000.0, dtype=np.float32)

    monkeypatch.setattr(rollout, "BCInference", UnsafePolicy)

    result = rollout.run_bc_rollout(
        "test-checkpoint.pt",
        "test-fingerprint",
        max_steps=2,
    )

    assert result["outcome"] == "unsafe_action"
    assert result["success"] is False
    assert result["steps_executed"] == 0


def test_rejects_invalid_safety_configuration(checkpoint_stub):
    with pytest.raises(ValueError, match="max_steps"):
        rollout.run_bc_rollout(
            "test-checkpoint.pt",
            "test-fingerprint",
            max_steps=0,
        )

    with pytest.raises(ValueError, match="max_force_n"):
        rollout.run_bc_rollout(
            "test-checkpoint.pt",
            "test-fingerprint",
            max_force_n=float("nan"),
        )


def test_invalid_live_observation_stops_safely(
    checkpoint_stub, monkeypatch
):
    """Invalid sensor data must not trigger a policy action."""

    def invalid_observation(env):
        raise ValueError("Nonfinite BC observation")

    monkeypatch.setattr(
        rollout,
        "bc_observation_from_env",
        invalid_observation,
    )

    result = rollout.run_bc_rollout(
        "test-checkpoint.pt",
        "test-fingerprint",
        max_steps=2,
    )

    assert result["outcome"] == "sensor_failure"
    assert result["success"] is False
    assert result["steps_executed"] == 0
    assert "Nonfinite" in result["failure_reason"]


def test_inference_failure_stops_safely(
    checkpoint_stub, monkeypatch
):
    """Failed inference must not submit an actuator command."""

    class FailingPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, observation):
            raise RuntimeError("Inference failure")

    monkeypatch.setattr(rollout, "BCInference", FailingPolicy)

    result = rollout.run_bc_rollout(
        "test-checkpoint.pt",
        "test-fingerprint",
        max_steps=2,
    )

    assert result["outcome"] == "unsafe_action"
    assert result["success"] is False
    assert result["steps_executed"] == 0
    assert "Inference failure" in result["failure_reason"]


def test_rejected_action_records_targets_and_limits(
    checkpoint_stub, monkeypatch
):
    """Preserve the rejected prediction for failure analysis."""

    class UnsafePolicy:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, observation):
            return np.full(7, 1000.0, dtype=np.float32)

    monkeypatch.setattr(rollout, "BCInference", UnsafePolicy)

    result = rollout.run_bc_rollout(
        "test-checkpoint.pt",
        "test-fingerprint",
        max_steps=2,
    )

    assert result["outcome"] == "unsafe_action"
    assert result["steps_executed"] == 0

    rejected = result["rejected_action"]

    assert rejected is not None
    assert rejected["step"] == 1
    assert len(rejected["observation"]) == 23
    assert len(rejected["policy_targets_rad"]) == 7
    assert len(rejected["joint_lower_limits_rad"]) == 7
    assert len(rejected["joint_upper_limits_rad"]) == 7

    np.testing.assert_array_equal(
        rejected["policy_targets_rad"],
        np.full(7, 1000.0),
    )

    assert all(
        target > upper
        for target, upper in zip(
            rejected["policy_targets_rad"],
            rejected["joint_upper_limits_rad"],
            strict=True,
        )
    )
