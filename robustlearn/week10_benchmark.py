"""Structured Week 10 classical-expert benchmark records and statistics."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from robustlearn.episode_manager import EpisodeFailureCategory

BENCHMARK_EPISODE_COUNT = 200


def benchmark_scenario_for_seed(
    seed: int,
) -> str:
    """Return the deterministic Week 9 scenario for one benchmark seed."""
    from robustlearn.episode_manager import WEEK9_SCENARIOS

    return WEEK9_SCENARIOS[
        seed % len(WEEK9_SCENARIOS)
    ]


def benchmark_domain_for_episode(
    *,
    episode_index: int,
    base_domain_id: int,
    domain_pool_size: int,
) -> int:
    """Return deterministic reusable ROS domain assignment."""
    if episode_index < 0:
        raise ValueError(
            "episode_index must be non-negative"
        )

    if base_domain_id < 0:
        raise ValueError(
            "base_domain_id must be non-negative"
        )

    if domain_pool_size <= 0:
        raise ValueError(
            "domain_pool_size must be positive"
        )

    return (
        base_domain_id
        + (episode_index % domain_pool_size)
    )


@dataclass(frozen=True)
class BenchmarkEpisodeRecord:
    """Structured terminal record for one Week 10 benchmark episode."""

    episode_index: int
    seed: int
    scenario: str
    moveit_planning_success: bool
    moveit_execution_success: bool
    handoff_error_rad: float | None
    local_expert_success: bool
    final_task_success: bool
    completion_time_s: float
    final_lateral_error_m: float | None
    final_insertion_depth_m: float | None
    peak_axial_force_n: float | None
    peak_lateral_force_n: float | None
    failure_category: EpisodeFailureCategory | None
    failure_reason: str | None
    start_base_joint_positions_rad: tuple[float, ...] | None = None
    start_perturbation_rad: tuple[float, ...] | None = None
    start_joint_positions_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent or non-finite benchmark records."""
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        if not self.scenario.strip():
            raise ValueError("scenario must be non-empty")

        if (
            not math.isfinite(self.completion_time_s)
            or self.completion_time_s < 0.0
        ):
            raise ValueError(
                "completion_time_s must be finite and non-negative"
            )

        optional_metrics = {
            "handoff_error_rad": self.handoff_error_rad,
            "final_lateral_error_m": self.final_lateral_error_m,
            "final_insertion_depth_m": self.final_insertion_depth_m,
            "peak_axial_force_n": self.peak_axial_force_n,
            "peak_lateral_force_n": self.peak_lateral_force_n,
        }

        for name, value in optional_metrics.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")

        start_vectors = (
            self.start_base_joint_positions_rad,
            self.start_perturbation_rad,
            self.start_joint_positions_rad,
        )

        present_start_vectors = tuple(
            value is not None
            for value in start_vectors
        )

        if any(present_start_vectors) and not all(present_start_vectors):
            raise ValueError(
                "benchmark start-state vectors must be all present or all absent"
            )

        if all(present_start_vectors):
            assert self.start_base_joint_positions_rad is not None
            assert self.start_perturbation_rad is not None
            assert self.start_joint_positions_rad is not None

            vectors = {
                "start_base_joint_positions_rad": (
                    self.start_base_joint_positions_rad
                ),
                "start_perturbation_rad": (
                    self.start_perturbation_rad
                ),
                "start_joint_positions_rad": (
                    self.start_joint_positions_rad
                ),
            }

            for name, values in vectors.items():
                if len(values) != PANDA_ARM_JOINT_COUNT:
                    raise ValueError(
                        f"{name} must contain exactly "
                        f"{PANDA_ARM_JOINT_COUNT} values"
                    )

                if not all(
                    math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        f"{name} must contain only finite values"
                    )

            reconstructed = tuple(
                base + delta
                for base, delta in zip(
                    self.start_base_joint_positions_rad,
                    self.start_perturbation_rad,
                    strict=True,
                )
            )

            for actual, expected in zip(
                self.start_joint_positions_rad,
                reconstructed,
                strict=True,
            ):
                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "start_joint_positions_rad must equal "
                        "base positions plus perturbation"
                    )

        if self.final_task_success:
            if not self.moveit_planning_success:
                raise ValueError(
                    "successful task requires MoveIt planning success"
                )
            if not self.moveit_execution_success:
                raise ValueError(
                    "successful task requires MoveIt execution success"
                )
            if not self.local_expert_success:
                raise ValueError(
                    "successful task requires local expert success"
                )
            if self.failure_category is not None:
                raise ValueError(
                    "successful task must not have a failure category"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "successful task must not have a failure reason"
                )

        else:
            if self.failure_category is None:
                raise ValueError(
                    "failed task must have a failure category"
                )
            if (
                self.failure_reason is None
                or not self.failure_reason.strip()
            ):
                raise ValueError(
                    "failed task must have a non-empty failure reason"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable episode record."""
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "scenario": self.scenario,
            "moveit_planning_success": self.moveit_planning_success,
            "moveit_execution_success": self.moveit_execution_success,
            "handoff_error_rad": self.handoff_error_rad,
            "local_expert_success": self.local_expert_success,
            "final_task_success": self.final_task_success,
            "completion_time_s": self.completion_time_s,
            "final_lateral_error_m": self.final_lateral_error_m,
            "final_insertion_depth_m": self.final_insertion_depth_m,
            "peak_axial_force_n": self.peak_axial_force_n,
            "peak_lateral_force_n": self.peak_lateral_force_n,
            "failure_category": (
                None
                if self.failure_category is None
                else self.failure_category.value
            ),
            "failure_reason": self.failure_reason,
            "start_base_joint_positions_rad": (
                None
                if self.start_base_joint_positions_rad is None
                else list(self.start_base_joint_positions_rad)
            ),
            "start_perturbation_rad": (
                None
                if self.start_perturbation_rad is None
                else list(self.start_perturbation_rad)
            ),
            "start_joint_positions_rad": (
                None
                if self.start_joint_positions_rad is None
                else list(self.start_joint_positions_rad)
            ),
        }


@dataclass(frozen=True)
class MetricStatistics:
    """Aggregate statistics for one finite benchmark metric."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    p95: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable metric summary."""
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate result for a completed benchmark."""

    episode_count: int
    success_count: int
    failure_count: int
    success_rate: float
    failure_counts: dict[str, int]
    completion_time_s: MetricStatistics
    handoff_error_rad: MetricStatistics | None
    final_lateral_error_m: MetricStatistics | None
    final_insertion_depth_m: MetricStatistics | None
    peak_axial_force_n: MetricStatistics | None
    peak_lateral_force_n: MetricStatistics | None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable benchmark summary."""
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "failure_counts": dict(self.failure_counts),
            "completion_time_s": self.completion_time_s.to_dict(),
            "handoff_error_rad": _stats_to_dict(
                self.handoff_error_rad
            ),
            "final_lateral_error_m": _stats_to_dict(
                self.final_lateral_error_m
            ),
            "final_insertion_depth_m": _stats_to_dict(
                self.final_insertion_depth_m
            ),
            "peak_axial_force_n": _stats_to_dict(
                self.peak_axial_force_n
            ),
            "peak_lateral_force_n": _stats_to_dict(
                self.peak_lateral_force_n
            ),
        }


def _stats_to_dict(
    stats: MetricStatistics | None,
) -> dict[str, object] | None:
    if stats is None:
        return None

    return stats.to_dict()


def metric_statistics(
    values: Sequence[float],
) -> MetricStatistics | None:
    """Summarize finite metric values with deterministic statistics."""
    resolved = tuple(float(value) for value in values)

    if not resolved:
        return None

    if not all(math.isfinite(value) for value in resolved):
        raise ValueError("metric values must all be finite")

    ordered = tuple(sorted(resolved))
    count = len(ordered)

    p95_index = max(
        0,
        math.ceil(0.95 * count) - 1,
    )

    return MetricStatistics(
        count=count,
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p95=ordered[p95_index],
    )


def summarize_benchmark(
    records: Sequence[BenchmarkEpisodeRecord],
    *,
    require_exact_count: bool = True,
) -> BenchmarkSummary:
    """Aggregate terminal episode records into the Week 10 benchmark summary."""
    episodes = tuple(records)

    if not episodes:
        raise ValueError(
            "benchmark requires at least one episode record"
        )

    if (
        require_exact_count
        and len(episodes) != BENCHMARK_EPISODE_COUNT
    ):
        raise ValueError(
            "Week 10 benchmark requires exactly "
            f"{BENCHMARK_EPISODE_COUNT} episodes"
        )

    expected_indices = tuple(range(len(episodes)))
    actual_indices = tuple(
        record.episode_index
        for record in episodes
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "benchmark episode indices must be contiguous and ordered"
        )

    seeds = tuple(record.seed for record in episodes)

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "benchmark seeds must be unique"
        )

    success_count = sum(
        int(record.final_task_success)
        for record in episodes
    )
    episode_count = len(episodes)
    failure_count = episode_count - success_count
    success_rate = success_count / episode_count

    if not math.isfinite(success_rate):
        raise RuntimeError(
            "benchmark success rate is non-finite"
        )

    failure_counts = Counter(
        record.failure_category.value
        for record in episodes
        if record.failure_category is not None
    )

    return BenchmarkSummary(
        episode_count=episode_count,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_rate,
        failure_counts=dict(
            sorted(failure_counts.items())
        ),
        completion_time_s=_required_stats(
            tuple(
                record.completion_time_s
                for record in episodes
            )
        ),
        handoff_error_rad=metric_statistics(
            tuple(
                record.handoff_error_rad
                for record in episodes
                if record.handoff_error_rad is not None
            )
        ),
        final_lateral_error_m=metric_statistics(
            tuple(
                record.final_lateral_error_m
                for record in episodes
                if record.final_lateral_error_m is not None
            )
        ),
        final_insertion_depth_m=metric_statistics(
            tuple(
                record.final_insertion_depth_m
                for record in episodes
                if record.final_insertion_depth_m is not None
            )
        ),
        peak_axial_force_n=metric_statistics(
            tuple(
                record.peak_axial_force_n
                for record in episodes
                if record.peak_axial_force_n is not None
            )
        ),
        peak_lateral_force_n=metric_statistics(
            tuple(
                record.peak_lateral_force_n
                for record in episodes
                if record.peak_lateral_force_n is not None
            )
        ),
    )


def _required_stats(
    values: Sequence[float],
) -> MetricStatistics:
    stats = metric_statistics(values)

    if stats is None:
        raise RuntimeError(
            "required benchmark metric has no values"
        )

    return stats


def benchmark_record_from_week9_result(
    *,
    episode_index: int,
    seed: int,
    completion_time_s: float,
    result: dict[str, object],
    start_state: BenchmarkStartState | None = None,
) -> BenchmarkEpisodeRecord:
    """Convert one validated Week 9 runner result into a benchmark record."""
    scenario_value = result.get("scenario")

    if not isinstance(scenario_value, str) or not scenario_value.strip():
        raise ValueError(
            "Week 9 result must contain a non-empty scenario"
        )

    success_value = result.get("success")

    if not isinstance(success_value, bool):
        raise ValueError(
            "Week 9 result success must be boolean"
        )

    planning_success = _required_bool(
        result,
        "moveit_planning_success",
    )
    execution_success = _required_bool(
        result,
        "moveit_execution_success",
    )
    expert_success = _required_bool(
        result,
        "local_expert_success",
    )

    raw_category = result.get("failure_category")

    if success_value:
        failure_category = None
        failure_reason = None
    else:
        if not isinstance(raw_category, str) or not raw_category:
            failure_category = EpisodeFailureCategory.UNKNOWN
            failure_reason = (
                "Week 9 runner returned no failure category"
            )
        else:
            from robustlearn.episode_manager import (
                WEEK9_FAILURE_CATEGORY_MAP,
            )

            failure_category = WEEK9_FAILURE_CATEGORY_MAP.get(
                raw_category,
                EpisodeFailureCategory.UNKNOWN,
            )
            failure_reason = f"Week 9 failure: {raw_category}"

    return BenchmarkEpisodeRecord(
        episode_index=episode_index,
        seed=seed,
        scenario=scenario_value,
        moveit_planning_success=planning_success,
        moveit_execution_success=execution_success,
        handoff_error_rad=_optional_finite_float(
            result,
            "handoff_error_rad",
        ),
        local_expert_success=expert_success,
        final_task_success=success_value,
        completion_time_s=float(completion_time_s),
        final_lateral_error_m=_optional_finite_float(
            result,
            "final_lateral_error_m",
        ),
        final_insertion_depth_m=_optional_finite_float(
            result,
            "final_insertion_depth_m",
        ),
        peak_axial_force_n=_optional_finite_float(
            result,
            "peak_axial_force_n",
        ),
        peak_lateral_force_n=_optional_finite_float(
            result,
            "peak_lateral_force_n",
        ),
        failure_category=failure_category,
        failure_reason=failure_reason,
        start_base_joint_positions_rad=(
            None
            if start_state is None
            else start_state.base_joint_positions_rad
        ),
        start_perturbation_rad=(
            None
            if start_state is None
            else start_state.perturbation_rad
        ),
        start_joint_positions_rad=(
            None
            if start_state is None
            else start_state.joint_positions_rad
        ),
    )


def _required_bool(
    result: dict[str, object],
    field: str,
) -> bool:
    value = result.get(field)

    if not isinstance(value, bool):
        raise ValueError(
            f"Week 9 result {field} must be boolean"
        )

    return value


def _optional_finite_float(
    result: dict[str, object],
    field: str,
) -> float | None:
    value = result.get(field)

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"Week 9 result {field} must be numeric when present"
        )

    resolved = float(value)

    if not math.isfinite(resolved):
        raise ValueError(
            f"Week 9 result {field} must be finite when present"
        )

    return resolved


@dataclass(frozen=True)
class BenchmarkConfig:
    """Reproducible configuration for the Week 10 expert benchmark."""

    seeds: tuple[int, ...]
    timeout_s: float
    base_domain_id: int
    domain_pool_size: int

    def __post_init__(self) -> None:
        """Validate benchmark configuration before execution."""
        if len(self.seeds) != BENCHMARK_EPISODE_COUNT:
            raise ValueError(
                "Week 10 benchmark configuration requires exactly "
                f"{BENCHMARK_EPISODE_COUNT} seeds"
            )

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "benchmark configuration seeds must be unique"
            )

        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(
                    "benchmark configuration seeds must be integers"
                )

            if seed < 0:
                raise ValueError(
                    "benchmark configuration seeds must be non-negative"
                )

        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError(
                "timeout_s must be finite and positive"
            )

        if self.base_domain_id < 0:
            raise ValueError(
                "base_domain_id must be non-negative"
            )

        if self.domain_pool_size <= 0:
            raise ValueError(
                "domain_pool_size must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-serializable benchmark configuration."""
        return {
            "episode_count": BENCHMARK_EPISODE_COUNT,
            "seeds": list(self.seeds),
            "timeout_s": self.timeout_s,
            "base_domain_id": self.base_domain_id,
            "domain_pool_size": self.domain_pool_size,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete reproducible Week 10 benchmark artifact."""

    config: BenchmarkConfig
    episodes: tuple[BenchmarkEpisodeRecord, ...]
    summary: BenchmarkSummary

    def __post_init__(self) -> None:
        """Reject reports inconsistent with their configuration."""
        if len(self.episodes) != BENCHMARK_EPISODE_COUNT:
            raise ValueError(
                "benchmark report must contain exactly "
                f"{BENCHMARK_EPISODE_COUNT} episodes"
            )

        episode_seeds = tuple(
            record.seed
            for record in self.episodes
        )

        if episode_seeds != self.config.seeds:
            raise ValueError(
                "benchmark episode seeds must match configuration order"
            )

        if self.summary.episode_count != len(self.episodes):
            raise ValueError(
                "benchmark summary episode count does not match records"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete stable JSON benchmark artifact."""
        return {
            "config": self.config.to_dict(),
            "episodes": [
                record.to_dict()
                for record in self.episodes
            ],
            "summary": self.summary.to_dict(),
        }


def default_benchmark_seeds() -> tuple[int, ...]:
    """Return the canonical deterministic Week 10 seed sequence."""
    return tuple(
        range(BENCHMARK_EPISODE_COUNT)
    )


def build_benchmark_config(
    *,
    seeds: tuple[int, ...] | None = None,
    timeout_s: float = 90.0,
    base_domain_id: int = 190,
    domain_pool_size: int = 40,
) -> BenchmarkConfig:
    """Build the canonical or explicitly seeded benchmark configuration."""
    resolved_seeds = (
        default_benchmark_seeds()
        if seeds is None
        else seeds
    )

    return BenchmarkConfig(
        seeds=resolved_seeds,
        timeout_s=timeout_s,
        base_domain_id=base_domain_id,
        domain_pool_size=domain_pool_size,
    )


def build_benchmark_report(
    *,
    config: BenchmarkConfig,
    episodes: Sequence[BenchmarkEpisodeRecord],
) -> BenchmarkReport:
    """Build and validate the complete Week 10 benchmark report."""
    resolved_episodes = tuple(episodes)
    summary = summarize_benchmark(
        resolved_episodes,
        require_exact_count=True,
    )

    return BenchmarkReport(
        config=config,
        episodes=resolved_episodes,
        summary=summary,
    )


def write_benchmark_report(
    path: str | Path,
    report: BenchmarkReport,
) -> Path:
    """Write the complete benchmark report using deterministic JSON."""
    import json

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return output_path


BENCHMARK_START_PERTURBATION_RAD = 0.01
PANDA_ARM_JOINT_COUNT = 7


@dataclass(frozen=True)
class BenchmarkStartState:
    """Deterministic seed-derived Panda arm start state."""

    seed: int
    scenario: str
    base_joint_positions_rad: tuple[float, ...]
    perturbation_rad: tuple[float, ...]
    joint_positions_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject malformed or non-finite benchmark start states."""
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("start-state seed must be an integer")

        if self.seed < 0:
            raise ValueError("start-state seed must be non-negative")

        if not self.scenario.strip():
            raise ValueError("start-state scenario must be non-empty")

        vectors = {
            "base_joint_positions_rad": self.base_joint_positions_rad,
            "perturbation_rad": self.perturbation_rad,
            "joint_positions_rad": self.joint_positions_rad,
        }

        for name, values in vectors.items():
            if len(values) != PANDA_ARM_JOINT_COUNT:
                raise ValueError(
                    f"{name} must contain exactly "
                    f"{PANDA_ARM_JOINT_COUNT} values"
                )

            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{name} must contain only finite values"
                )

        reconstructed = tuple(
            base + delta
            for base, delta in zip(
                self.base_joint_positions_rad,
                self.perturbation_rad,
                strict=True,
            )
        )

        for actual, expected in zip(
            self.joint_positions_rad,
            reconstructed,
            strict=True,
        ):
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "joint_positions_rad must equal "
                    "base positions plus perturbation"
                )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-serializable start-state metadata."""
        return {
            "seed": self.seed,
            "scenario": self.scenario,
            "base_joint_positions_rad": list(
                self.base_joint_positions_rad
            ),
            "perturbation_rad": list(
                self.perturbation_rad
            ),
            "joint_positions_rad": list(
                self.joint_positions_rad
            ),
        }


def build_benchmark_start_state(
    *,
    seed: int,
    scenario: str,
    base_joint_positions_rad: tuple[float, ...],
    max_perturbation_rad: float = BENCHMARK_START_PERTURBATION_RAD,
) -> BenchmarkStartState:
    """Create a deterministic bounded seven-joint benchmark start state."""
    import numpy as np

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    if not scenario.strip():
        raise ValueError("scenario must be non-empty")

    if len(base_joint_positions_rad) != PANDA_ARM_JOINT_COUNT:
        raise ValueError(
            "base_joint_positions_rad must contain exactly "
            f"{PANDA_ARM_JOINT_COUNT} values"
        )

    if not all(
        math.isfinite(value)
        for value in base_joint_positions_rad
    ):
        raise ValueError(
            "base_joint_positions_rad must contain only finite values"
        )

    if (
        not math.isfinite(max_perturbation_rad)
        or max_perturbation_rad <= 0.0
    ):
        raise ValueError(
            "max_perturbation_rad must be finite and positive"
        )

    rng = np.random.default_rng(seed)

    perturbation = tuple(
        float(value)
        for value in rng.uniform(
            -max_perturbation_rad,
            max_perturbation_rad,
            size=PANDA_ARM_JOINT_COUNT,
        )
    )

    joint_positions = tuple(
        base + delta
        for base, delta in zip(
            base_joint_positions_rad,
            perturbation,
            strict=True,
        )
    )

    return BenchmarkStartState(
        seed=seed,
        scenario=scenario,
        base_joint_positions_rad=base_joint_positions_rad,
        perturbation_rad=perturbation,
        joint_positions_rad=joint_positions,
    )


BENCHMARK_KEYFRAME_PREFIX = "week10_seed"


def benchmark_keyframe_name(seed: int) -> str:
    """Return a deterministic MuJoCo keyframe name for one benchmark seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    return f"{BENCHMARK_KEYFRAME_PREFIX}_{seed:06d}"


def _parse_float_vector(
    text: str,
    *,
    field: str,
) -> tuple[float, ...]:
    values = tuple(
        float(token)
        for token in text.split()
    )

    if not values:
        raise ValueError(
            f"{field} must contain at least one numeric value"
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"{field} must contain only finite values"
        )

    return values


def build_seeded_benchmark_mjcf(
    *,
    source_path: str | Path,
    output_path: str | Path,
    start_state: BenchmarkStartState,
) -> Path:
    """Write a deterministic MuJoCo model with one seed-specific keyframe."""
    import xml.etree.ElementTree as ET

    source = Path(source_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"benchmark source MJCF not found: {source}"
        )

    tree = ET.parse(source)
    root = tree.getroot()

    for element in root.iter():
        file_path = element.get("file")

        if file_path is None or not file_path.strip():
            continue

        resolved_file = Path(file_path)

        if resolved_file.is_absolute():
            continue

        resolved_file = (
            source.parent
            / resolved_file
        ).resolve()

        element.set(
            "file",
            str(resolved_file),
        )

    keyframe_parent = root.find("keyframe")

    if keyframe_parent is None:
        raise ValueError(
            "benchmark source MJCF does not contain <keyframe>"
        )

    base_key = None

    for key in keyframe_parent.findall("key"):
        if key.get("name") == start_state.scenario:
            base_key = key
            break

    if base_key is None:
        raise ValueError(
            "benchmark source MJCF does not contain scenario "
            f"{start_state.scenario!r}"
        )

    qpos_text = base_key.get("qpos")

    if qpos_text is None:
        raise ValueError(
            f"{start_state.scenario} keyframe has no qpos"
        )

    qpos = _parse_float_vector(
        qpos_text,
        field=f"{start_state.scenario} qpos",
    )

    if len(qpos) < PANDA_ARM_JOINT_COUNT:
        raise ValueError(
            f"{start_state.scenario} qpos has fewer than "
            f"{PANDA_ARM_JOINT_COUNT} Panda arm values"
        )

    ctrl_text = base_key.get("ctrl")
    ctrl = (
        None
        if ctrl_text is None
        else _parse_float_vector(
            ctrl_text,
            field=f"{start_state.scenario} ctrl",
        )
    )

    if (
        ctrl is not None
        and len(ctrl) < PANDA_ARM_JOINT_COUNT
    ):
        raise ValueError(
            f"{start_state.scenario} ctrl has fewer than "
            f"{PANDA_ARM_JOINT_COUNT} Panda arm values"
        )

    seeded_qpos = (
        start_state.joint_positions_rad
        + qpos[PANDA_ARM_JOINT_COUNT:]
    )

    seeded_ctrl = (
        None
        if ctrl is None
        else (
            start_state.joint_positions_rad
            + ctrl[PANDA_ARM_JOINT_COUNT:]
        )
    )

    key_name = benchmark_keyframe_name(
        start_state.seed
    )

    existing = keyframe_parent.find(
        f"./key[@name='{key_name}']"
    )

    if existing is not None:
        keyframe_parent.remove(existing)

    seeded_key = ET.Element("key")
    seeded_key.set("name", key_name)
    seeded_key.set(
        "qpos",
        " ".join(
            f"{value:.17g}"
            for value in seeded_qpos
        ),
    )

    if seeded_ctrl is not None:
        seeded_key.set(
            "ctrl",
            " ".join(
                f"{value:.17g}"
                for value in seeded_ctrl
            ),
        )

    keyframe_parent.append(seeded_key)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )

    return destination


def load_benchmark_base_joint_positions(
    *,
    source_path: str | Path,
    scenario: str,
) -> tuple[float, ...]:
    """Load the seven Panda arm positions for a named benchmark scenario."""
    import xml.etree.ElementTree as ET

    source = Path(source_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"benchmark source MJCF not found: {source}"
        )

    if not scenario.strip():
        raise ValueError("scenario must be non-empty")

    root = ET.parse(source).getroot()
    keyframe_parent = root.find("keyframe")

    if keyframe_parent is None:
        raise ValueError(
            "benchmark source MJCF does not contain <keyframe>"
        )

    key = keyframe_parent.find(
        f"./key[@name='{scenario}']"
    )

    if key is None:
        raise ValueError(
            f"benchmark source MJCF does not contain scenario {scenario!r}"
        )

    qpos_text = key.get("qpos")

    if qpos_text is None:
        raise ValueError(
            f"{scenario} keyframe has no qpos"
        )

    qpos = _parse_float_vector(
        qpos_text,
        field=f"{scenario} qpos",
    )

    if len(qpos) < PANDA_ARM_JOINT_COUNT:
        raise ValueError(
            f"{scenario} qpos has fewer than "
            f"{PANDA_ARM_JOINT_COUNT} Panda arm values"
        )

    return tuple(
        qpos[:PANDA_ARM_JOINT_COUNT]
    )


def write_benchmark_checkpoint(
    path: str | Path,
    *,
    config: BenchmarkConfig,
    episodes: Sequence[BenchmarkEpisodeRecord],
    requested_episode_count: int,
) -> Path:
    """Atomically persist partial benchmark progress."""
    import json
    import os

    if (
        requested_episode_count <= 0
        or requested_episode_count > BENCHMARK_EPISODE_COUNT
    ):
        raise ValueError(
            "requested_episode_count must be between 1 and "
            f"{BENCHMARK_EPISODE_COUNT}"
        )

    resolved_episodes = tuple(episodes)

    if len(resolved_episodes) > requested_episode_count:
        raise ValueError(
            "checkpoint cannot contain more episodes than requested"
        )

    expected_indices = tuple(
        range(len(resolved_episodes))
    )
    actual_indices = tuple(
        record.episode_index
        for record in resolved_episodes
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "checkpoint episode indices must be contiguous from zero"
        )

    expected_seeds = config.seeds[
        : len(resolved_episodes)
    ]
    actual_seeds = tuple(
        record.seed
        for record in resolved_episodes
    )

    if actual_seeds != expected_seeds:
        raise ValueError(
            "checkpoint episode seeds must match configuration order"
        )

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "mode": "checkpoint",
        "requested_episode_count": requested_episode_count,
        "completed_episode_count": len(resolved_episodes),
        "config": config.to_dict(),
        "episodes": [
            record.to_dict()
            for record in resolved_episodes
        ],
    }

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    os.replace(
        temporary_path,
        output_path,
    )

    return output_path
