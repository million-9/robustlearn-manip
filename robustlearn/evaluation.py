"""Structured evaluation results for Panda insertion."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from robustlearn.config import PandaInsertionTaskConfig
from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.scripted_insertion import scripted_insertion_action


@dataclass(frozen=True)
class EpisodeEvaluationResult:
    """Structured result from one seeded Panda insertion episode."""

    seed: int
    success: bool
    failure: bool
    truncated: bool
    step_count: int
    simulation_duration_s: float
    final_lateral_error_m: float
    final_insertion_depth_m: float
    task_config: dict[str, object]
    randomization_sample: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a stable serializable episode record."""
        return {
            "seed": self.seed,
            "success": self.success,
            "failure": self.failure,
            "truncated": self.truncated,
            "step_count": self.step_count,
            "simulation_duration_s": self.simulation_duration_s,
            "final_lateral_error_m": self.final_lateral_error_m,
            "final_insertion_depth_m": self.final_insertion_depth_m,
            "task_config": self.task_config,
            "randomization_sample": self.randomization_sample,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate statistics for a seeded evaluation run."""

    episode_count: int
    success_count: int
    failure_count: int
    truncated_count: int
    success_rate: float
    failure_categories: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a stable serializable aggregate summary."""
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "truncated_count": self.truncated_count,
            "success_rate": self.success_rate,
            "failure_categories": self.failure_categories,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Complete reusable result of a multi-episode evaluation."""

    episodes: tuple[EpisodeEvaluationResult, ...]
    summary: EvaluationSummary

    def to_dict(self) -> dict[str, object]:
        """Return a stable serializable evaluation report."""
        return {
            "episodes": [
                episode.to_dict()
                for episode in self.episodes
            ],
            "summary": self.summary.to_dict(),
        }



def evaluate_scripted_episode(
    *,
    seed: int,
    config: PandaInsertionTaskConfig | None = None,
) -> EpisodeEvaluationResult:
    """Run one seeded scripted Panda insertion episode."""
    task_config = (
        PandaInsertionTaskConfig()
        if config is None
        else config
    )

    env = PandaInsertionEnv(config=task_config)

    try:
        observation, info = env.reset(seed=seed)

        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "evaluation reset produced a non-finite observation"
            )

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = scripted_insertion_action(env)

            if not np.all(np.isfinite(action)):
                raise RuntimeError(
                    "evaluation controller produced a non-finite action"
                )

            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if not np.all(np.isfinite(observation)):
                raise RuntimeError(
                    "evaluation episode produced a non-finite observation"
                )

        simulation_duration_s = float(info["simulation_time"])
        final_lateral_error_m = float(info["task_lateral_error"])
        final_insertion_depth_m = float(info["task_insertion_depth"])

        finite_scalars = {
            "simulation_duration_s": simulation_duration_s,
            "final_lateral_error_m": final_lateral_error_m,
            "final_insertion_depth_m": final_insertion_depth_m,
        }

        for name, value in finite_scalars.items():
            if not isfinite(value):
                raise RuntimeError(
                    f"evaluation result {name} is non-finite"
                )

        randomization = info["randomization_sample"]

        if not isinstance(randomization, dict):
            raise RuntimeError(
                "evaluation result is missing randomization metadata"
            )

        randomization_sample = {
            str(name): float(value)
            for name, value in randomization.items()
        }

        if not all(
            isfinite(value)
            for value in randomization_sample.values()
        ):
            raise RuntimeError(
                "evaluation randomization metadata contains non-finite values"
            )

        return EpisodeEvaluationResult(
            seed=seed,
            success=bool(info["task_success"]),
            failure=bool(info["task_failure"]),
            truncated=bool(truncated),
            step_count=int(info["elapsed_steps"]),
            simulation_duration_s=simulation_duration_s,
            final_lateral_error_m=final_lateral_error_m,
            final_insertion_depth_m=final_insertion_depth_m,
            task_config=task_config.to_dict(),
            randomization_sample=randomization_sample,
        )
    finally:
        env.close()



def resolve_episode_seeds(
    *,
    base_seed: int | None = None,
    episode_count: int | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the explicit deterministic seed sequence for an evaluation."""
    if seeds is not None:
        if base_seed is not None or episode_count is not None:
            raise ValueError(
                "explicit seeds cannot be combined with base_seed or episode_count"
            )

        resolved = tuple(seeds)

        if not resolved:
            raise ValueError("seeds must contain at least one seed")
    else:
        if base_seed is None or episode_count is None:
            raise ValueError(
                "base_seed and episode_count are required when seeds are omitted"
            )

        if episode_count <= 0:
            raise ValueError("episode_count must be positive")

        resolved = tuple(
            base_seed + offset
            for offset in range(episode_count)
        )

    for seed in resolved:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("episode seeds must be integers")

        if seed < 0:
            raise ValueError("episode seeds must be non-negative")

    return resolved


def summarize_evaluation_results(
    episodes: Sequence[EpisodeEvaluationResult],
) -> EvaluationSummary:
    """Build aggregate statistics from completed episode results."""
    episode_records = tuple(episodes)

    if not episode_records:
        raise ValueError("at least one episode result is required")

    success_count = sum(
        int(episode.success)
        for episode in episode_records
    )
    failure_count = sum(
        int(episode.failure)
        for episode in episode_records
    )
    truncated_count = sum(
        int(episode.truncated)
        for episode in episode_records
    )

    episode_count = len(episode_records)
    success_rate = success_count / episode_count

    if not isfinite(success_rate):
        raise RuntimeError("evaluation success rate is non-finite")

    return EvaluationSummary(
        episode_count=episode_count,
        success_count=success_count,
        failure_count=failure_count,
        truncated_count=truncated_count,
        success_rate=success_rate,
        failure_categories={
            "task_failure": failure_count,
            "truncated": truncated_count,
        },
    )


def evaluate_scripted_episodes(
    *,
    base_seed: int | None = None,
    episode_count: int | None = None,
    seeds: Sequence[int] | None = None,
    config: PandaInsertionTaskConfig | None = None,
) -> EvaluationReport:
    """Run multiple deterministic scripted Panda insertion episodes."""
    resolved_seeds = resolve_episode_seeds(
        base_seed=base_seed,
        episode_count=episode_count,
        seeds=seeds,
    )

    episodes = tuple(
        evaluate_scripted_episode(
            seed=seed,
            config=config,
        )
        for seed in resolved_seeds
    )

    return EvaluationReport(
        episodes=episodes,
        summary=summarize_evaluation_results(episodes),
    )
