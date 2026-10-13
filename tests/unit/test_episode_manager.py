"""Unit tests for the Week 10 episode orchestration contract."""

import math

import pytest

from robustlearn.episode_manager import (
    EpisodeFailureCategory,
    EpisodePhase,
    EpisodeResult,
    EpisodeSpec,
)


def test_episode_phases_are_explicit() -> None:
    assert tuple(EpisodePhase) == (
        EpisodePhase.PENDING,
        EpisodePhase.RESETTING,
        EpisodePhase.RUNNING,
        EpisodePhase.TERMINAL,
    )


def test_episode_spec_accepts_valid_input() -> None:
    spec = EpisodeSpec(
        episode_index=7,
        seed=1234,
        scenario="week9_approach",
        timeout_s=60.0,
    )

    assert spec.episode_index == 7
    assert spec.seed == 1234
    assert spec.scenario == "week9_approach"
    assert spec.timeout_s == 60.0


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, math.inf, -math.inf, math.nan],
)
def test_episode_spec_rejects_invalid_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError):
        EpisodeSpec(
            episode_index=0,
            seed=1,
            scenario="week9_approach",
            timeout_s=timeout_s,
        )


def test_successful_result_is_finite_and_consistent() -> None:
    result = EpisodeResult(
        episode_index=0,
        seed=10,
        scenario="week9_approach",
        start_time_s=100.0,
        end_time_s=103.5,
        success=True,
        failure_category=None,
        failure_reason=None,
    )

    assert result.duration_s == pytest.approx(3.5)


def test_failed_result_requires_failure_details() -> None:
    with pytest.raises(ValueError):
        EpisodeResult(
            episode_index=0,
            seed=10,
            scenario="week9_approach",
            start_time_s=100.0,
            end_time_s=101.0,
            success=False,
            failure_category=None,
            failure_reason=None,
        )


@pytest.mark.parametrize(
    ("start_time_s", "end_time_s"),
    [
        (math.nan, 1.0),
        (0.0, math.inf),
        (2.0, 1.0),
    ],
)
def test_result_rejects_invalid_timestamps(
    start_time_s: float,
    end_time_s: float,
) -> None:
    with pytest.raises(ValueError):
        EpisodeResult(
            episode_index=0,
            seed=10,
            scenario="week9_approach",
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            success=False,
            failure_category=EpisodeFailureCategory.UNKNOWN,
            failure_reason="test failure",
        )


def test_episode_lifecycle_nominal_transitions() -> None:
    from robustlearn.episode_manager import EpisodeLifecycle

    lifecycle = EpisodeLifecycle()

    assert lifecycle.phase is EpisodePhase.PENDING
    assert not lifecycle.terminal

    lifecycle.transition_to(EpisodePhase.RESETTING)
    assert lifecycle.phase is EpisodePhase.RESETTING

    lifecycle.transition_to(EpisodePhase.RUNNING)
    assert lifecycle.phase is EpisodePhase.RUNNING

    lifecycle.transition_to(EpisodePhase.TERMINAL)
    assert lifecycle.phase is EpisodePhase.TERMINAL
    assert lifecycle.terminal


def test_episode_lifecycle_can_fail_during_reset() -> None:
    from robustlearn.episode_manager import EpisodeLifecycle

    lifecycle = EpisodeLifecycle()

    lifecycle.transition_to(EpisodePhase.RESETTING)
    lifecycle.transition_to(EpisodePhase.TERMINAL)

    assert lifecycle.terminal


def test_episode_lifecycle_rejects_skipping_reset() -> None:
    from robustlearn.episode_manager import EpisodeLifecycle

    lifecycle = EpisodeLifecycle()

    with pytest.raises(
        ValueError,
        match="pending -> running",
    ):
        lifecycle.transition_to(EpisodePhase.RUNNING)


def test_episode_lifecycle_rejects_transition_after_terminal() -> None:
    from robustlearn.episode_manager import EpisodeLifecycle

    lifecycle = EpisodeLifecycle()
    lifecycle.transition_to(EpisodePhase.RESETTING)
    lifecycle.transition_to(EpisodePhase.TERMINAL)

    with pytest.raises(
        ValueError,
        match="terminal -> resetting",
    ):
        lifecycle.transition_to(EpisodePhase.RESETTING)


def test_scenario_for_seed_is_deterministic() -> None:
    from robustlearn.episode_manager import scenario_for_seed

    assert scenario_for_seed(0) == "week9_case_01"
    assert scenario_for_seed(0) == scenario_for_seed(0)


def test_scenario_for_seed_cycles_over_week9_scenarios() -> None:
    from robustlearn.episode_manager import scenario_for_seed

    assert scenario_for_seed(19) == "week9_case_20"
    assert scenario_for_seed(20) == "week9_case_01"
    assert scenario_for_seed(21) == "week9_case_02"


def test_scenario_for_seed_handles_negative_seed_deterministically() -> None:
    from robustlearn.episode_manager import scenario_for_seed

    assert scenario_for_seed(-1) == "week9_case_20"
    assert scenario_for_seed(-21) == "week9_case_20"


def test_scenario_for_seed_rejects_empty_scenario_set() -> None:
    from robustlearn.episode_manager import scenario_for_seed

    with pytest.raises(
        ValueError,
        match="scenarios must not be empty",
    ):
        scenario_for_seed(0, ())


def test_scenario_for_seed_rejects_blank_scenario_name() -> None:
    from robustlearn.episode_manager import scenario_for_seed

    with pytest.raises(
        ValueError,
        match="non-empty names",
    ):
        scenario_for_seed(
            0,
            ("week9_case_01", ""),
        )


def test_build_episode_specs_preserves_seed_order() -> None:
    from robustlearn.episode_manager import build_episode_specs

    specs = build_episode_specs(
        seeds=(10, 20, 30),
        timeout_s=60.0,
    )

    assert tuple(spec.episode_index for spec in specs) == (0, 1, 2)
    assert tuple(spec.seed for spec in specs) == (10, 20, 30)
    assert tuple(spec.scenario for spec in specs) == (
        "week9_case_11",
        "week9_case_01",
        "week9_case_11",
    )
    assert all(spec.timeout_s == 60.0 for spec in specs)


def test_build_episode_specs_is_reproducible() -> None:
    from robustlearn.episode_manager import build_episode_specs

    seeds = (1234, 55, -7, 1234)

    first = build_episode_specs(
        seeds=seeds,
        timeout_s=45.0,
    )
    second = build_episode_specs(
        seeds=seeds,
        timeout_s=45.0,
    )

    assert first == second


def test_build_episode_specs_allows_empty_plan() -> None:
    from robustlearn.episode_manager import build_episode_specs

    assert build_episode_specs(
        seeds=(),
        timeout_s=60.0,
    ) == ()


def test_build_episode_specs_rejects_invalid_timeout() -> None:
    from robustlearn.episode_manager import build_episode_specs

    with pytest.raises(
        ValueError,
        match="timeout_s must be finite and positive",
    ):
        build_episode_specs(
            seeds=(1,),
            timeout_s=0.0,
        )


def test_build_episode_specs_supports_custom_scenarios() -> None:
    from robustlearn.episode_manager import build_episode_specs

    specs = build_episode_specs(
        seeds=(0, 1, 2, 3),
        timeout_s=5.0,
        scenarios=("alpha", "beta"),
    )

    assert tuple(spec.scenario for spec in specs) == (
        "alpha",
        "beta",
        "alpha",
        "beta",
    )


def test_execution_outcome_accepts_success() -> None:
    from robustlearn.episode_manager import EpisodeExecutionOutcome

    outcome = EpisodeExecutionOutcome(
        success=True,
        failure_category=None,
        failure_reason=None,
    )

    assert outcome.success


def test_execution_outcome_accepts_failure() -> None:
    from robustlearn.episode_manager import EpisodeExecutionOutcome

    outcome = EpisodeExecutionOutcome(
        success=False,
        failure_category=EpisodeFailureCategory.PLANNING,
        failure_reason="MoveIt planning failed",
    )

    assert not outcome.success
    assert outcome.failure_category is EpisodeFailureCategory.PLANNING


def test_execution_outcome_rejects_success_with_failure_details() -> None:
    from robustlearn.episode_manager import EpisodeExecutionOutcome

    with pytest.raises(
        ValueError,
        match="successful execution outcome",
    ):
        EpisodeExecutionOutcome(
            success=True,
            failure_category=EpisodeFailureCategory.UNKNOWN,
            failure_reason="inconsistent",
        )


def test_execution_outcome_rejects_failure_without_category() -> None:
    from robustlearn.episode_manager import EpisodeExecutionOutcome

    with pytest.raises(
        ValueError,
        match="failed execution outcome must have a failure category",
    ):
        EpisodeExecutionOutcome(
            success=False,
            failure_category=None,
            failure_reason="failed",
        )


def test_execution_outcome_rejects_blank_failure_reason() -> None:
    from robustlearn.episode_manager import EpisodeExecutionOutcome

    with pytest.raises(
        ValueError,
        match="non-empty failure reason",
    ):
        EpisodeExecutionOutcome(
            success=False,
            failure_category=EpisodeFailureCategory.UNKNOWN,
            failure_reason=" ",
        )


def test_episode_manager_runs_nominal_lifecycle() -> None:
    from robustlearn.episode_manager import (
        EpisodeExecutionOutcome,
        EpisodeManager,
    )

    events = []

    class Backend:
        def reset(self, spec):
            events.append(("reset", spec.scenario))

        def execute(self, spec):
            events.append(("execute", spec.scenario))
            return EpisodeExecutionOutcome(
                success=True,
                failure_category=None,
                failure_reason=None,
            )

    times = iter((10.0, 12.5))
    manager = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    )

    result = manager.run_one(
        EpisodeSpec(
            episode_index=3,
            seed=23,
            scenario="week9_case_04",
            timeout_s=60.0,
        )
    )

    assert events == [
        ("reset", "week9_case_04"),
        ("execute", "week9_case_04"),
    ]
    assert result.success
    assert result.episode_index == 3
    assert result.seed == 23
    assert result.scenario == "week9_case_04"
    assert result.duration_s == pytest.approx(2.5)


def test_episode_manager_preserves_backend_failure() -> None:
    from robustlearn.episode_manager import (
        EpisodeExecutionOutcome,
        EpisodeManager,
    )

    class Backend:
        def reset(self, spec):
            pass

        def execute(self, spec):
            return EpisodeExecutionOutcome(
                success=False,
                failure_category=EpisodeFailureCategory.PLANNING,
                failure_reason="MoveIt planning failed",
            )

    times = iter((20.0, 21.0))
    result = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    ).run_one(
        EpisodeSpec(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            timeout_s=60.0,
        )
    )

    assert not result.success
    assert result.failure_category is EpisodeFailureCategory.PLANNING
    assert result.failure_reason == "MoveIt planning failed"


def test_episode_manager_converts_timeout_to_terminal_result() -> None:
    from robustlearn.episode_manager import EpisodeManager

    class Backend:
        def reset(self, spec):
            pass

        def execute(self, spec):
            raise TimeoutError("60 second episode timeout")

    times = iter((30.0, 90.0))
    result = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    ).run_one(
        EpisodeSpec(
            episode_index=1,
            seed=1,
            scenario="week9_case_02",
            timeout_s=60.0,
        )
    )

    assert not result.success
    assert result.failure_category is EpisodeFailureCategory.TIMEOUT
    assert result.failure_reason == "60 second episode timeout"


def test_episode_manager_converts_reset_exception_to_terminal_result() -> None:
    from robustlearn.episode_manager import EpisodeManager

    class Backend:
        def reset(self, spec):
            raise RuntimeError("reset failed")

        def execute(self, spec):
            raise AssertionError("execute must not run")

    times = iter((40.0, 40.5))
    result = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    ).run_one(
        EpisodeSpec(
            episode_index=2,
            seed=2,
            scenario="week9_case_03",
            timeout_s=60.0,
        )
    )

    assert not result.success
    assert result.failure_category is EpisodeFailureCategory.UNKNOWN
    assert result.failure_reason == "RuntimeError: reset failed"


def test_episode_manager_runs_multiple_episodes_in_order() -> None:
    from robustlearn.episode_manager import (
        EpisodeExecutionOutcome,
        EpisodeManager,
        build_episode_specs,
    )

    executed = []

    class Backend:
        def reset(self, spec):
            executed.append(("reset", spec.episode_index))

        def execute(self, spec):
            executed.append(("execute", spec.episode_index))
            return EpisodeExecutionOutcome(
                success=True,
                failure_category=None,
                failure_reason=None,
            )

    times = iter(
        (
            0.0, 1.0,
            2.0, 3.0,
            4.0, 5.0,
        )
    )

    manager = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    )

    specs = build_episode_specs(
        seeds=(10, 11, 12),
        timeout_s=60.0,
    )

    results = manager.run_many(specs)

    assert tuple(result.episode_index for result in results) == (0, 1, 2)
    assert all(result.success for result in results)
    assert executed == [
        ("reset", 0),
        ("execute", 0),
        ("reset", 1),
        ("execute", 1),
        ("reset", 2),
        ("execute", 2),
    ]


def test_failed_episode_does_not_block_later_episode() -> None:
    from robustlearn.episode_manager import (
        EpisodeExecutionOutcome,
        EpisodeManager,
        build_episode_specs,
    )

    executed = []

    class Backend:
        def reset(self, spec):
            executed.append(("reset", spec.episode_index))

        def execute(self, spec):
            executed.append(("execute", spec.episode_index))

            if spec.episode_index == 1:
                raise RuntimeError("deliberate episode failure")

            return EpisodeExecutionOutcome(
                success=True,
                failure_category=None,
                failure_reason=None,
            )

    times = iter(
        (
            0.0, 1.0,
            2.0, 3.0,
            4.0, 5.0,
        )
    )

    manager = EpisodeManager(
        backend=Backend(),
        clock=lambda: next(times),
    )

    specs = build_episode_specs(
        seeds=(20, 21, 22),
        timeout_s=60.0,
    )

    results = manager.run_many(specs)

    assert len(results) == 3

    assert results[0].success

    assert not results[1].success
    assert (
        results[1].failure_category
        is EpisodeFailureCategory.UNKNOWN
    )
    assert (
        results[1].failure_reason
        == "RuntimeError: deliberate episode failure"
    )

    assert results[2].success

    assert executed[-2:] == [
        ("reset", 2),
        ("execute", 2),
    ]


def test_episode_manager_accepts_empty_episode_plan() -> None:
    from robustlearn.episode_manager import EpisodeManager

    class Backend:
        def reset(self, spec):
            raise AssertionError("reset must not run")

        def execute(self, spec):
            raise AssertionError("execute must not run")

    manager = EpisodeManager(
        backend=Backend(),
    )

    assert manager.run_many(()) == ()


def test_week9_backend_domain_assignment_is_deterministic() -> None:
    from robustlearn.episode_manager import Week9HybridBackend

    backend = Week9HybridBackend(
        base_domain_id=190,
        domain_pool_size=40,
    )

    first = EpisodeSpec(
        episode_index=0,
        seed=0,
        scenario="week9_case_01",
        timeout_s=60.0,
    )
    wrapped = EpisodeSpec(
        episode_index=40,
        seed=40,
        scenario="week9_case_01",
        timeout_s=60.0,
    )

    assert backend.domain_id(first) == 190
    assert backend.domain_id(wrapped) == 190


def test_week9_backend_rejects_invalid_domain_pool() -> None:
    from robustlearn.episode_manager import Week9HybridBackend

    with pytest.raises(
        ValueError,
        match="domain_pool_size must be positive",
    ):
        Week9HybridBackend(
            domain_pool_size=0,
        )


@pytest.mark.parametrize(
    ("raw_category", "expected"),
    [
        ("scenario_timeout", EpisodeFailureCategory.TIMEOUT),
        ("moveit_planning_failure", EpisodeFailureCategory.PLANNING),
        ("moveit_execution_failure", EpisodeFailureCategory.EXECUTION),
        ("handoff_failure", EpisodeFailureCategory.HANDOFF),
        ("local_expert_failure", EpisodeFailureCategory.EXPERT),
        ("acceptance_probe_failure", EpisodeFailureCategory.TASK),
        ("launch_failure", EpisodeFailureCategory.LAUNCH),
        ("hybrid_coordinator_failure", EpisodeFailureCategory.UNKNOWN),
    ],
)
def test_week9_failure_mapping(
    raw_category: str,
    expected: EpisodeFailureCategory,
) -> None:
    from robustlearn.episode_manager import WEEK9_FAILURE_CATEGORY_MAP

    assert WEEK9_FAILURE_CATEGORY_MAP[raw_category] is expected
