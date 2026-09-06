"""Tests for Panda insertion evaluation result structures."""

import pytest

from robustlearn.config import (
    PandaInsertionTaskConfig,
)
from robustlearn.evaluation import (
    EpisodeEvaluationResult,
    EvaluationReport,
    EvaluationSummary,
    evaluate_scripted_episode,
    evaluate_scripted_episodes,
    resolve_episode_seeds,
    summarize_evaluation_results,
)


def example_episode() -> EpisodeEvaluationResult:
    """Return one deterministic example episode result."""
    return EpisodeEvaluationResult(
        seed=2026,
        success=True,
        failure=False,
        truncated=False,
        step_count=34,
        simulation_duration_s=0.68,
        final_lateral_error_m=0.000057,
        final_insertion_depth_m=0.010737,
        task_config={
            "frame_skip": 5,
            "max_episode_steps": 100,
        },
        randomization_sample={
            "receptacle_x_offset_m": 0.0,
            "receptacle_y_offset_m": 0.0,
            "receptacle_yaw_offset_rad": 0.0,
        },
    )


def test_episode_result_serialization_is_stable() -> None:
    result = example_episode()

    assert result.to_dict() == {
        "seed": 2026,
        "success": True,
        "failure": False,
        "truncated": False,
        "step_count": 34,
        "simulation_duration_s": 0.68,
        "final_lateral_error_m": 0.000057,
        "final_insertion_depth_m": 0.010737,
        "task_config": {
            "frame_skip": 5,
            "max_episode_steps": 100,
        },
        "randomization_sample": {
            "receptacle_x_offset_m": 0.0,
            "receptacle_y_offset_m": 0.0,
            "receptacle_yaw_offset_rad": 0.0,
        },
    }


def test_summary_serialization_is_stable() -> None:
    summary = EvaluationSummary(
        episode_count=3,
        success_count=2,
        failure_count=1,
        truncated_count=0,
        success_rate=2.0 / 3.0,
        failure_categories={
            "task_failure": 1,
            "truncated": 0,
        },
    )

    assert summary.to_dict() == {
        "episode_count": 3,
        "success_count": 2,
        "failure_count": 1,
        "truncated_count": 0,
        "success_rate": 2.0 / 3.0,
        "failure_categories": {
            "task_failure": 1,
            "truncated": 0,
        },
    }


def test_report_serialization_contains_episode_records_and_summary() -> None:
    episode = example_episode()
    summary = EvaluationSummary(
        episode_count=1,
        success_count=1,
        failure_count=0,
        truncated_count=0,
        success_rate=1.0,
        failure_categories={
            "task_failure": 0,
            "truncated": 0,
        },
    )

    report = EvaluationReport(
        episodes=(episode,),
        summary=summary,
    )

    assert report.to_dict() == {
        "episodes": [
            episode.to_dict(),
        ],
        "summary": summary.to_dict(),
    }


def test_evaluate_scripted_episode_completes_clean_task() -> None:
    result = evaluate_scripted_episode(seed=2026)

    assert result.seed == 2026
    assert result.success is True
    assert result.failure is False
    assert result.truncated is False
    assert result.step_count > 0
    assert result.simulation_duration_s > 0.0
    assert result.final_lateral_error_m >= 0.0
    assert result.final_insertion_depth_m > 0.0

    assert result.randomization_sample == {
        "receptacle_x_offset_m": 0.0,
        "receptacle_y_offset_m": 0.0,
        "receptacle_yaw_offset_rad": 0.0,
    }


def test_evaluate_scripted_episode_is_reproducible_for_same_seed() -> None:
    first = evaluate_scripted_episode(seed=2026)
    second = evaluate_scripted_episode(seed=2026)

    assert first.to_dict() == second.to_dict()



def test_resolve_episode_seeds_from_base_seed() -> None:
    assert resolve_episode_seeds(
        base_seed=2026,
        episode_count=4,
    ) == (
        2026,
        2027,
        2028,
        2029,
    )


def test_resolve_episode_seeds_preserves_explicit_seed_order() -> None:
    assert resolve_episode_seeds(
        seeds=[17, 3, 99],
    ) == (
        17,
        3,
        99,
    )


def test_resolve_episode_seeds_rejects_mixed_seed_modes() -> None:
    with pytest.raises(ValueError):
        resolve_episode_seeds(
            base_seed=2026,
            episode_count=2,
            seeds=[1, 2],
        )


def test_resolve_episode_seeds_rejects_non_positive_episode_count() -> None:
    with pytest.raises(ValueError):
        resolve_episode_seeds(
            base_seed=2026,
            episode_count=0,
        )


def test_resolve_episode_seeds_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        resolve_episode_seeds(
            seeds=[2026, -1],
        )



def test_summarize_evaluation_results_counts_outcomes() -> None:
    successful = example_episode()

    failed = EpisodeEvaluationResult(
        seed=2027,
        success=False,
        failure=True,
        truncated=False,
        step_count=12,
        simulation_duration_s=0.24,
        final_lateral_error_m=0.003,
        final_insertion_depth_m=0.0,
        task_config={},
        randomization_sample={},
    )

    truncated = EpisodeEvaluationResult(
        seed=2028,
        success=False,
        failure=False,
        truncated=True,
        step_count=100,
        simulation_duration_s=2.0,
        final_lateral_error_m=0.0015,
        final_insertion_depth_m=0.005,
        task_config={},
        randomization_sample={},
    )

    summary = summarize_evaluation_results(
        [
            successful,
            failed,
            truncated,
        ]
    )

    assert summary.episode_count == 3
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.truncated_count == 1
    assert summary.success_rate == pytest.approx(1.0 / 3.0)
    assert summary.failure_categories == {
        "task_failure": 1,
        "truncated": 1,
    }


def test_summarize_evaluation_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_evaluation_results([])



def test_evaluate_scripted_episodes_runs_multiple_seeded_episodes() -> None:
    report = evaluate_scripted_episodes(
        base_seed=2026,
        episode_count=3,
    )

    assert tuple(
        episode.seed
        for episode in report.episodes
    ) == (
        2026,
        2027,
        2028,
    )

    assert report.summary.episode_count == 3


def test_evaluate_scripted_episodes_is_reproducible() -> None:
    first = evaluate_scripted_episodes(
        seeds=[2026, 2027],
    )
    second = evaluate_scripted_episodes(
        seeds=[2026, 2027],
    )

    assert first.to_dict() == second.to_dict()


def test_task_failure_does_not_stop_later_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate_scripted_episode(
        *,
        seed: int,
        config: PandaInsertionTaskConfig | None = None,
    ) -> EpisodeEvaluationResult:
        _ = config

        failed = seed == 2026

        return EpisodeEvaluationResult(
            seed=seed,
            success=not failed,
            failure=failed,
            truncated=False,
            step_count=1,
            simulation_duration_s=0.01,
            final_lateral_error_m=0.003 if failed else 0.0,
            final_insertion_depth_m=0.001 if failed else 0.011,
            task_config={},
            randomization_sample={},
        )

    monkeypatch.setattr(
        "robustlearn.evaluation.evaluate_scripted_episode",
        fake_evaluate_scripted_episode,
    )

    report = evaluate_scripted_episodes(
        seeds=[2026, 2027],
    )

    assert tuple(
        episode.seed
        for episode in report.episodes
    ) == (
        2026,
        2027,
    )

    assert report.episodes[0].failure is True
    assert report.episodes[1].success is True
    assert report.summary.episode_count == 2
    assert report.summary.failure_count == 1
    assert report.summary.success_count == 1

