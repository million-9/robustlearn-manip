"""Reproducible multi-seed closed-loop BC evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import mujoco
import numpy as np
import torch

from robustlearn.bc_rollout import run_bc_rollout

SCHEMA_VERSION = 1
DEFAULT_SEEDS = tuple(range(2026, 2046))


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Require a nonempty collection of unique nonnegative seeds."""
    values = tuple(seeds)

    if not values:
        raise ValueError("At least one evaluation seed is required")

    if any(type(seed) is not int or seed < 0 for seed in values):
        raise ValueError("Evaluation seeds must be nonnegative integers")

    if len(set(values)) != len(values):
        raise ValueError("Duplicate evaluation seeds are not permitted")

    return values


def optional_mean(values: Sequence[float]) -> float | None:
    """Return no estimate when no measurements are available."""
    return float(mean(values)) if values else None


def initial_observation(
    episode: dict[str, Any],
) -> list[float] | None:
    """Recover the initial observation when recorded by the rollout."""
    trajectory = episode["trajectory"]

    if trajectory:
        return [float(value) for value in trajectory[0]["observation"]]

    rejected = episode.get("rejected_action")

    if rejected is not None and rejected["step"] == 1:
        return [float(value) for value in rejected["observation"]]

    return None


def summarize_episodes(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate outcomes without excluding failed episodes."""
    if not episodes:
        raise ValueError("Cannot summarize an empty benchmark")

    seeds = validate_seeds([episode["seed"] for episode in episodes])

    outcomes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    forces: list[float] = []
    completion_times: list[float] = []
    smoothness: list[float] = []
    initial_observations: set[tuple[float, ...]] = set()
    observed_initial_count = 0
    unsafe_rejections = 0
    success_count = 0
    executed_steps: list[int] = []

    for episode in episodes:
        outcome = str(episode["outcome"])
        success = bool(episode["success"])

        if success != (outcome == "success"):
            raise ValueError("Inconsistent success and outcome fields")

        outcomes[outcome] += 1
        executed_steps.append(int(episode["steps_executed"]))

        if success:
            success_count += 1
            completion_time = episode["completion_time_s"]

            if completion_time is not None:
                completion_times.append(float(completion_time))
        else:
            reason = episode.get("failure_reason") or "unspecified"
            failures[str(reason)] += 1

        force = float(episode["max_observed_force_n"])

        if episode["steps_executed"] > 0 and math.isfinite(force):
            forces.append(force)

        action_change = episode["mean_action_change_rad"]

        if action_change is not None:
            value = float(action_change)

            if math.isfinite(value):
                smoothness.append(value)

        if episode.get("rejected_action") is not None:
            unsafe_rejections += 1

        observation = initial_observation(episode)

        if observation is not None:
            if all(math.isfinite(value) for value in observation):
                observed_initial_count += 1
                initial_observations.add(tuple(observation))

    count = len(episodes)

    return {
        "episode_count": count,
        "seeds": list(seeds),
        "success_count": success_count,
        "failure_count": count - success_count,
        "success_rate": success_count / count,
        "outcome_counts": dict(sorted(outcomes.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "rejected_action_count": unsafe_rejections,
        "total_steps_executed": sum(executed_steps),
        "mean_steps_executed": float(mean(executed_steps)),
        "zero_step_episode_count": executed_steps.count(0),
        "mean_completion_time_s": optional_mean(completion_times),
        "median_completion_time_s": (
            float(median(completion_times))
            if completion_times
            else None
        ),
        "completion_time_sample_count": len(completion_times),
        "mean_recorded_peak_force_n": optional_mean(forces),
        "max_recorded_peak_force_n": max(forces) if forces else None,
        "peak_force_sample_count": len(forces),
        "mean_action_change_rad": optional_mean(smoothness),
        "action_change_sample_count": len(smoothness),
        "initial_observation_sample_count": observed_initial_count,
        "unique_initial_observation_count": len(initial_observations),
        "success_rate_ci95": None,
        "confidence_interval_status": (
            "not_reported_independent_scenarios_not_established"
        ),
    }


def run_bc_benchmark(
    checkpoint: str | Path,
    dataset_sha256: str,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_steps: int = 100,
    max_force_n: float = 50.0,
) -> dict[str, Any]:
    """Evaluate the same checkpoint across explicit evaluation seeds."""
    evaluation_seeds = validate_seeds(seeds)

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    if not math.isfinite(max_force_n) or max_force_n <= 0:
        raise ValueError("max_force_n must be finite and positive")

    if not dataset_sha256:
        raise ValueError("Dataset fingerprint is required")

    checkpoint_path = Path(checkpoint).resolve()

    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    episodes: list[dict[str, Any]] = []

    for seed in evaluation_seeds:
        episode = run_bc_rollout(
            checkpoint_path,
            dataset_sha256,
            seed=seed,
            max_steps=max_steps,
            max_force_n=max_force_n,
        )

        if episode["seed"] != seed:
            raise RuntimeError("Rollout returned an unexpected seed")

        if episode["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed during benchmark")

        if episode["dataset_sha256"] != dataset_sha256:
            raise RuntimeError("Dataset fingerprint mismatch")

        episodes.append(episode)

    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_sha256": dataset_sha256,
            "evaluation_seeds": list(evaluation_seeds),
            "max_steps": max_steps,
            "max_force_n": max_force_n,
            "domain_randomization": False,
            "inference_device": "cpu",
            "policy": "behavior_cloning",
            "action_representation": "absolute_joint_position_rad",
        },
        "software": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "summary": summarize_episodes(episodes),
        "episodes": episodes,
        "limitations": [
            "Different seeds do not establish independent scenarios.",
            "Initial observation equality is checked exactly.",
            "Recorded peak force excludes pre-action-only measurements.",
            "Force is sampled at environment-step boundaries.",
            "Completion times include successful episodes only.",
            "Action smoothness excludes episodes with insufficient actions.",
            "No independent-scenario confidence interval is claimed.",
        ],
    }


def json_safe(value: Any) -> Any:
    """Represent nonfinite diagnostic values as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def write_benchmark(
    result: dict[str, Any],
    output: str | Path,
) -> None:
    """Write strict JSON with complete per-episode diagnostics."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json_safe(result)
    payload["serialization"] = {
        "nonfinite_diagnostic_values": "represented_as_null",
    }

    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-force-n", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        args.dataset_manifest.read_text(encoding="utf-8")
    )

    result = run_bc_benchmark(
        args.checkpoint,
        manifest["dataset_sha256"],
        seeds=args.seeds,
        max_steps=args.max_steps,
        max_force_n=args.max_force_n,
    )

    write_benchmark(result, args.output)

    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"Benchmark saved to: {args.output}")


if __name__ == "__main__":
    main()
