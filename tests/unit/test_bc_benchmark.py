"""Regression tests for multi-seed BC evaluation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import robustlearn.bc_benchmark as benchmark


def make_episode(
    seed: int,
    *,
    outcome: str = "success",
    steps: int = 2,
    force: float = 3.0,
    observation: float = 0.0,
) -> dict[str, Any]:
    """Construct a deterministic rollout result."""
    success = outcome == "success"

    return {
        "seed": seed,
        "outcome": outcome,
        "success": success,
        "failure_reason": None if success else "test_failure",
        "steps_executed": steps,
        "completion_time_s": 0.02 if success else None,
        "max_observed_force_n": force,
        "mean_action_change_rad": 0.1 if steps > 1 else None,
        "rejected_action": None,
        "trajectory": [
            {"observation": [observation] * 23}
            for _ in range(steps)
        ],
    }


@pytest.mark.parametrize(
    "seeds",
    [[], [1, 1], [-1], [True], [1.5]],
)
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        benchmark.validate_seeds(seeds)


def test_seed_order_preserved():
    assert benchmark.validate_seeds([9, 2, 5]) == (9, 2, 5)


def test_empty_summary_rejected():
    with pytest.raises(ValueError):
        benchmark.summarize_episodes([])


def test_aggregate_mixed_outcomes():
    episodes = [
        make_episode(1, force=3.0),
        make_episode(2, outcome="unsafe_action", force=7.0),
        make_episode(3, outcome="timeout", force=5.0),
        make_episode(4, observation=1.0),
    ]

    summary = benchmark.summarize_episodes(episodes)

    assert summary["episode_count"] == 4
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["outcome_counts"] == {
        "success": 2,
        "timeout": 1,
        "unsafe_action": 1,
    }
    assert summary["failure_reason_counts"] == {"test_failure": 2}
    assert summary["total_steps_executed"] == 8
    assert summary["mean_recorded_peak_force_n"] == 4.5
    assert summary["max_recorded_peak_force_n"] == 7.0
    assert summary["completion_time_sample_count"] == 2
    assert summary["mean_completion_time_s"] == 0.02
    assert summary["unique_initial_observation_count"] == 2


def test_all_failures_remain_in_denominator():
    episodes = [
        make_episode(1, outcome="unsafe_action", steps=0),
        make_episode(2, outcome="sensor_failure", steps=0),
    ]

    summary = benchmark.summarize_episodes(episodes)

    assert summary["success_rate"] == 0.0
    assert summary["failure_count"] == 2
    assert summary["zero_step_episode_count"] == 2
    assert summary["mean_completion_time_s"] is None
    assert summary["mean_action_change_rad"] is None
    assert summary["mean_recorded_peak_force_n"] is None
    assert summary["peak_force_sample_count"] == 0


def test_identical_initial_observations_do_not_imply_independence():
    episodes = [
        make_episode(seed)
        for seed in range(10)
    ]

    summary = benchmark.summarize_episodes(episodes)

    assert summary["initial_observation_sample_count"] == 10
    assert summary["unique_initial_observation_count"] == 1
    assert summary["success_rate_ci95"] is None


def test_rejected_first_action_preserves_initial_observation():
    episode = make_episode(
        1,
        outcome="unsafe_action",
        steps=0,
    )
    episode["rejected_action"] = {
        "step": 1,
        "observation": [0.0] * 23,
    }

    summary = benchmark.summarize_episodes([episode])

    assert summary["rejected_action_count"] == 1
    assert summary["initial_observation_sample_count"] == 1
    assert summary["unique_initial_observation_count"] == 1


def test_inconsistent_outcome_rejected():
    episode = make_episode(1)
    episode["success"] = False

    with pytest.raises(ValueError, match="Inconsistent"):
        benchmark.summarize_episodes([episode])


def test_duplicate_episode_seeds_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.summarize_episodes([
            make_episode(1),
            make_episode(1),
        ])


def test_benchmark_forwards_configuration(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    calls = []

    def fake_rollout(
        checkpoint_path,
        dataset_sha256,
        *,
        seed,
        max_steps,
        max_force_n,
    ):
        calls.append(seed)

        assert checkpoint_path == checkpoint
        assert dataset_sha256 == "test-dataset"
        assert max_steps == 100
        assert max_force_n == 50.0

        result = make_episode(seed)
        result["checkpoint_sha256"] = digest
        result["dataset_sha256"] = dataset_sha256
        return result

    monkeypatch.setattr(benchmark, "run_bc_rollout", fake_rollout)

    kwargs = {
        "seeds": [2026, 2027, 2028],
        "max_steps": 100,
        "max_force_n": 50.0,
    }

    result = benchmark.run_bc_benchmark(
        checkpoint,
        "test-dataset",
        **kwargs,
    )

    assert calls == [2026, 2027, 2028]
    assert result["configuration"]["checkpoint_sha256"] == digest
    assert result["summary"]["episode_count"] == 3
    assert len(result["episodes"]) == 3

    repeated = benchmark.run_bc_benchmark(
        checkpoint,
        "test-dataset",
        **kwargs,
    )

    assert result == repeated


def test_checkpoint_mismatch_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_rollout(*args, **kwargs):
        result = make_episode(kwargs["seed"])
        result["checkpoint_sha256"] = "wrong-checkpoint"
        result["dataset_sha256"] = "test-dataset"
        return result

    monkeypatch.setattr(benchmark, "run_bc_rollout", fake_rollout)

    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        benchmark.run_bc_benchmark(
            checkpoint,
            "test-dataset",
            seeds=[2026],
        )


def test_strict_json_serialization(tmp_path):
    output = tmp_path / "results" / "benchmark.json"

    result = {
        "summary": {"success_rate": 0.0},
        "episodes": [
            {
                "outcome": "sensor_failure",
                "diagnostic": float("nan"),
                "trajectory": [
                    {"force": float("inf")},
                ],
            }
        ],
    }

    benchmark.write_benchmark(result, output)

    restored = json.loads(output.read_text(encoding="utf-8"))

    assert restored["summary"]["success_rate"] == 0.0
    assert restored["episodes"][0]["diagnostic"] is None
    assert restored["episodes"][0]["trajectory"][0]["force"] is None
    assert "NaN" not in output.read_text(encoding="utf-8")
    assert "Infinity" not in output.read_text(encoding="utf-8")


def test_missing_checkpoint_rejected(tmp_path):
    checkpoint = Path(tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError):
        benchmark.run_bc_benchmark(
            checkpoint,
            "test-dataset",
            seeds=[2026],
        )
