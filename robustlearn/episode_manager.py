"""Reusable episode orchestration contracts for hybrid manipulation rollouts.

This module deliberately contains no ROS or controller implementation.
It defines the stable episode-level data contract used by later execution,
logging, benchmarking, and dataset-writing infrastructure.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Protocol


class EpisodePhase(StrEnum):
    """High-level lifecycle of one hybrid manipulation episode."""

    PENDING = "pending"
    RESETTING = "resetting"
    RUNNING = "running"
    TERMINAL = "terminal"


class EpisodeFailureCategory(StrEnum):
    """Explicit terminal failure categories for hybrid episodes."""

    PLANNING = "planning_failure"
    EXECUTION = "execution_failure"
    HANDOFF = "handoff_failure"
    EXPERT = "local_expert_failure"
    TIMEOUT = "episode_timeout"
    TASK = "task_failure"
    LAUNCH = "launch_failure"
    UNKNOWN = "unknown_failure"


@dataclass(frozen=True)
class EpisodeSpec:
    """Deterministic input specification for one episode."""

    episode_index: int
    seed: int
    scenario: str
    timeout_s: float

    def __post_init__(self) -> None:
        """Reject invalid episode inputs before execution begins."""
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        if not self.scenario.strip():
            raise ValueError("scenario must be non-empty")

        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")


@dataclass(frozen=True)
class EpisodeResult:
    """Structured terminal result produced for every attempted episode."""

    episode_index: int
    seed: int
    scenario: str
    start_time_s: float
    end_time_s: float
    success: bool
    failure_category: EpisodeFailureCategory | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        """Validate the terminal episode-result contract."""
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        if not self.scenario.strip():
            raise ValueError("scenario must be non-empty")

        if not math.isfinite(self.start_time_s):
            raise ValueError("start_time_s must be finite")

        if not math.isfinite(self.end_time_s):
            raise ValueError("end_time_s must be finite")

        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must not precede start_time_s")

        if self.success:
            if self.failure_category is not None:
                raise ValueError(
                    "successful result must not have a failure category"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "successful result must not have a failure reason"
                )
        else:
            if self.failure_category is None:
                raise ValueError(
                    "failed result must have a failure category"
                )
            if self.failure_reason is None or not self.failure_reason.strip():
                raise ValueError(
                    "failed result must have a non-empty failure reason"
                )

    @property
    def duration_s(self) -> float:
        """Return elapsed wall-clock duration for the episode."""
        return self.end_time_s - self.start_time_s


class EpisodeLifecycle:
    """Validate high-level state transitions for one episode."""

    _ALLOWED_TRANSITIONS: dict[EpisodePhase, frozenset[EpisodePhase]] = {
        EpisodePhase.PENDING: frozenset(
            {
                EpisodePhase.RESETTING,
            }
        ),
        EpisodePhase.RESETTING: frozenset(
            {
                EpisodePhase.RUNNING,
                EpisodePhase.TERMINAL,
            }
        ),
        EpisodePhase.RUNNING: frozenset(
            {
                EpisodePhase.TERMINAL,
            }
        ),
        EpisodePhase.TERMINAL: frozenset(),
    }

    def __init__(self) -> None:
        self._phase = EpisodePhase.PENDING

    @property
    def phase(self) -> EpisodePhase:
        """Return the current episode phase."""
        return self._phase

    @property
    def terminal(self) -> bool:
        """Return whether the episode has reached a terminal state."""
        return self._phase is EpisodePhase.TERMINAL

    def transition_to(self, next_phase: EpisodePhase) -> None:
        """Advance to a valid next phase.

        Raises:
            ValueError: If the requested lifecycle transition is invalid.
        """
        allowed = self._ALLOWED_TRANSITIONS[self._phase]

        if next_phase not in allowed:
            raise ValueError(
                "invalid episode lifecycle transition: "
                f"{self._phase.value} -> {next_phase.value}"
            )

        self._phase = next_phase


WEEK9_SCENARIOS: tuple[str, ...] = tuple(
    f"week9_case_{index:02d}"
    for index in range(1, 21)
)


def scenario_for_seed(
    seed: int,
    scenarios: tuple[str, ...] = WEEK9_SCENARIOS,
) -> str:
    """Select a deterministic start scenario from an integer seed."""
    if not scenarios:
        raise ValueError("scenarios must not be empty")

    if any(not scenario.strip() for scenario in scenarios):
        raise ValueError("scenarios must contain only non-empty names")

    return scenarios[seed % len(scenarios)]


def build_episode_specs(
    seeds: tuple[int, ...],
    timeout_s: float,
    scenarios: tuple[str, ...] = WEEK9_SCENARIOS,
) -> tuple[EpisodeSpec, ...]:
    """Build a deterministic ordered episode plan from supplied seeds."""
    return tuple(
        EpisodeSpec(
            episode_index=index,
            seed=seed,
            scenario=scenario_for_seed(
                seed,
                scenarios,
            ),
            timeout_s=timeout_s,
        )
        for index, seed in enumerate(seeds)
    )


@dataclass(frozen=True)
class EpisodeExecutionOutcome:
    """Terminal outcome returned by an episode execution backend."""

    success: bool
    failure_category: EpisodeFailureCategory | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        """Validate backend terminal outcome consistency."""
        if self.success:
            if self.failure_category is not None:
                raise ValueError(
                    "successful execution outcome must not have "
                    "a failure category"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "successful execution outcome must not have "
                    "a failure reason"
                )
        else:
            if self.failure_category is None:
                raise ValueError(
                    "failed execution outcome must have "
                    "a failure category"
                )
            if self.failure_reason is None or not self.failure_reason.strip():
                raise ValueError(
                    "failed execution outcome must have "
                    "a non-empty failure reason"
                )


class EpisodeBackend(Protocol):
    """Execution boundary used by the episode manager."""

    def reset(self, spec: EpisodeSpec) -> None:
        """Reset the system to the deterministic episode start state."""
        ...

    def execute(self, spec: EpisodeSpec) -> EpisodeExecutionOutcome:
        """Execute one hybrid episode and return its terminal outcome."""
        ...


class EpisodeManager:
    """Orchestrate lifecycle and terminal-result creation for episodes."""

    def __init__(
        self,
        backend: EpisodeBackend,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._backend = backend
        self._clock = clock

    def run_one(self, spec: EpisodeSpec) -> EpisodeResult:
        """Run one episode and always produce a structured terminal result."""
        lifecycle = EpisodeLifecycle()
        start_time_s = self._clock()

        try:
            lifecycle.transition_to(EpisodePhase.RESETTING)
            self._backend.reset(spec)

            lifecycle.transition_to(EpisodePhase.RUNNING)
            outcome = self._backend.execute(spec)

        except TimeoutError as exc:
            outcome = EpisodeExecutionOutcome(
                success=False,
                failure_category=EpisodeFailureCategory.TIMEOUT,
                failure_reason=str(exc) or "episode execution timed out",
            )

        except Exception as exc:
            outcome = EpisodeExecutionOutcome(
                success=False,
                failure_category=EpisodeFailureCategory.UNKNOWN,
                failure_reason=(
                    f"{type(exc).__name__}: {exc}"
                    if str(exc)
                    else type(exc).__name__
                ),
            )

        if not lifecycle.terminal:
            lifecycle.transition_to(EpisodePhase.TERMINAL)

        end_time_s = self._clock()

        return EpisodeResult(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            success=outcome.success,
            failure_category=outcome.failure_category,
            failure_reason=outcome.failure_reason,
        )

    def run_many(
        self,
        specs: tuple[EpisodeSpec, ...],
    ) -> tuple[EpisodeResult, ...]:
        """Run episodes sequentially without one failure blocking the rest."""
        return tuple(
            self.run_one(spec)
            for spec in specs
        )


WEEK9_FAILURE_CATEGORY_MAP: dict[str, EpisodeFailureCategory] = {
    "scenario_timeout": EpisodeFailureCategory.TIMEOUT,
    "moveit_planning_failure": EpisodeFailureCategory.PLANNING,
    "moveit_execution_failure": EpisodeFailureCategory.EXECUTION,
    "handoff_failure": EpisodeFailureCategory.HANDOFF,
    "local_expert_failure": EpisodeFailureCategory.EXPERT,
    "acceptance_probe_failure": EpisodeFailureCategory.TASK,
    "launch_failure": EpisodeFailureCategory.LAUNCH,
    "hybrid_coordinator_failure": EpisodeFailureCategory.UNKNOWN,
}


class Week9HybridBackend:
    """Adapter around the validated Week 9 isolated hybrid runner."""

    def __init__(
        self,
        *,
        base_domain_id: int = 190,
        domain_pool_size: int = 40,
    ) -> None:
        if base_domain_id < 0:
            raise ValueError("base_domain_id must be non-negative")

        if domain_pool_size <= 0:
            raise ValueError("domain_pool_size must be positive")

        self._base_domain_id = base_domain_id
        self._domain_pool_size = domain_pool_size

    def reset(self, spec: EpisodeSpec) -> None:
        """Defer reset to the fresh Week 9 launch-test process."""
        return None

    def domain_id(self, spec: EpisodeSpec) -> int:
        """Return the reusable ROS domain assigned to this sequential episode."""
        return (
            self._base_domain_id
            + (spec.episode_index % self._domain_pool_size)
        )

    def execute(self, spec: EpisodeSpec) -> EpisodeExecutionOutcome:
        """Run one real Week 9 hybrid scenario through the existing runner."""
        from tools.run_week9_acceptance import run_scenario

        result = run_scenario(
            spec.scenario,
            domain_id=self.domain_id(spec),
            timeout_s=spec.timeout_s,
        )

        if result.get("success") is True:
            return EpisodeExecutionOutcome(
                success=True,
                failure_category=None,
                failure_reason=None,
            )

        raw_category = result.get("failure_category")

        if not isinstance(raw_category, str) or not raw_category:
            return EpisodeExecutionOutcome(
                success=False,
                failure_category=EpisodeFailureCategory.UNKNOWN,
                failure_reason="Week 9 runner returned no failure category",
            )

        category = WEEK9_FAILURE_CATEGORY_MAP.get(
            raw_category,
            EpisodeFailureCategory.UNKNOWN,
        )

        return EpisodeExecutionOutcome(
            success=False,
            failure_category=category,
            failure_reason=f"Week 9 failure: {raw_category}",
        )
