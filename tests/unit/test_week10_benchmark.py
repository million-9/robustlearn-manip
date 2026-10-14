"""Tests for Week 10 classical-expert benchmark records and statistics."""

import math
from pathlib import Path

import pytest

from robustlearn.episode_manager import EpisodeFailureCategory
from robustlearn.week10_benchmark import (
    BENCHMARK_EPISODE_COUNT,
    BenchmarkEpisodeRecord,
    metric_statistics,
    summarize_benchmark,
)


def make_success(
    index: int = 0,
    *,
    seed: int | None = None,
) -> BenchmarkEpisodeRecord:
    """Return a valid successful benchmark record."""
    resolved_seed = index if seed is None else seed

    return BenchmarkEpisodeRecord(
        episode_index=index,
        seed=resolved_seed,
        scenario=f"week9_case_{(index % 20) + 1:02d}",
        moveit_planning_success=True,
        moveit_execution_success=True,
        handoff_error_rad=0.01,
        local_expert_success=True,
        final_task_success=True,
        completion_time_s=2.0 + index * 0.01,
        final_lateral_error_m=0.0001,
        final_insertion_depth_m=0.010,
        peak_axial_force_n=4.0,
        peak_lateral_force_n=1.0,
        failure_category=None,
        failure_reason=None,
    )


def make_failure(
    index: int,
) -> BenchmarkEpisodeRecord:
    """Return a valid planning-failure benchmark record."""
    return BenchmarkEpisodeRecord(
        episode_index=index,
        seed=index,
        scenario=f"week9_case_{(index % 20) + 1:02d}",
        moveit_planning_success=False,
        moveit_execution_success=False,
        handoff_error_rad=None,
        local_expert_success=False,
        final_task_success=False,
        completion_time_s=1.5,
        final_lateral_error_m=None,
        final_insertion_depth_m=None,
        peak_axial_force_n=None,
        peak_lateral_force_n=None,
        failure_category=EpisodeFailureCategory.PLANNING,
        failure_reason="MoveIt planning failed",
    )


def test_success_record_serializes_failure_category_as_none() -> None:
    record = make_success()

    assert record.to_dict()["failure_category"] is None


def test_failure_record_requires_failure_category() -> None:
    with pytest.raises(
        ValueError,
        match="failed task must have a failure category",
    ):
        BenchmarkEpisodeRecord(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            moveit_planning_success=False,
            moveit_execution_success=False,
            handoff_error_rad=None,
            local_expert_success=False,
            final_task_success=False,
            completion_time_s=1.0,
            final_lateral_error_m=None,
            final_insertion_depth_m=None,
            peak_axial_force_n=None,
            peak_lateral_force_n=None,
            failure_category=None,
            failure_reason="failed",
        )


@pytest.mark.parametrize(
    "field",
    [
        "handoff_error_rad",
        "final_lateral_error_m",
        "final_insertion_depth_m",
        "peak_axial_force_n",
        "peak_lateral_force_n",
    ],
)
def test_record_rejects_nonfinite_optional_metric(
    field: str,
) -> None:
    values = make_success().__dict__ | {field: math.nan}

    with pytest.raises(
        ValueError,
        match=f"{field} must be finite",
    ):
        BenchmarkEpisodeRecord(**values)


def test_record_rejects_nonfinite_completion_time() -> None:
    values = make_success().__dict__ | {
        "completion_time_s": math.inf,
    }

    with pytest.raises(
        ValueError,
        match="completion_time_s must be finite",
    ):
        BenchmarkEpisodeRecord(**values)


def test_success_requires_complete_success_chain() -> None:
    values = make_success().__dict__ | {
        "moveit_execution_success": False,
    }

    with pytest.raises(
        ValueError,
        match="requires MoveIt execution success",
    ):
        BenchmarkEpisodeRecord(**values)


def test_metric_statistics_are_deterministic() -> None:
    stats = metric_statistics(
        (5.0, 1.0, 3.0, 2.0, 4.0)
    )

    assert stats is not None
    assert stats.count == 5
    assert stats.minimum == 1.0
    assert stats.maximum == 5.0
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.p95 == 5.0


def test_metric_statistics_reject_nonfinite_value() -> None:
    with pytest.raises(
        ValueError,
        match="must all be finite",
    ):
        metric_statistics((1.0, math.nan))


def test_summary_requires_exactly_200_by_default() -> None:
    with pytest.raises(
        ValueError,
        match="requires exactly 200 episodes",
    ):
        summarize_benchmark(
            (make_success(),)
        )


def test_summary_rejects_noncontiguous_episode_indices() -> None:
    records = (
        make_success(0),
        make_success(2),
    )

    with pytest.raises(
        ValueError,
        match="contiguous and ordered",
    ):
        summarize_benchmark(
            records,
            require_exact_count=False,
        )


def test_summary_rejects_duplicate_seeds() -> None:
    records = (
        make_success(0, seed=7),
        make_success(1, seed=7),
    )

    with pytest.raises(
        ValueError,
        match="seeds must be unique",
    ):
        summarize_benchmark(
            records,
            require_exact_count=False,
        )


def test_summary_aggregates_success_failure_and_metrics() -> None:
    records = (
        make_success(0),
        make_failure(1),
        make_success(2),
    )

    summary = summarize_benchmark(
        records,
        require_exact_count=False,
    )

    assert summary.episode_count == 3
    assert summary.success_count == 2
    assert summary.failure_count == 1
    assert summary.success_rate == pytest.approx(2.0 / 3.0)
    assert summary.failure_counts == {
        "planning_failure": 1,
    }

    assert summary.completion_time_s.count == 3

    assert summary.handoff_error_rad is not None
    assert summary.handoff_error_rad.count == 2

    assert summary.peak_axial_force_n is not None
    assert summary.peak_axial_force_n.count == 2


def test_exact_200_episode_summary_passes() -> None:
    records = tuple(
        make_success(index)
        for index in range(
            BENCHMARK_EPISODE_COUNT
        )
    )

    summary = summarize_benchmark(records)

    assert summary.episode_count == 200
    assert summary.success_count == 200
    assert summary.failure_count == 0
    assert summary.success_rate == 1.0


def test_week9_result_conversion_preserves_success_metrics() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_record_from_week9_result,
    )

    record = benchmark_record_from_week9_result(
        episode_index=7,
        seed=107,
        completion_time_s=3.25,
        result={
            "scenario": "week9_case_08",
            "success": True,
            "moveit_planning_success": True,
            "moveit_execution_success": True,
            "handoff_error_rad": 0.012,
            "local_expert_success": True,
            "final_lateral_error_m": 0.0002,
            "final_insertion_depth_m": 0.0105,
            "peak_axial_force_n": 6.5,
            "peak_lateral_force_n": 1.8,
            "failure_category": None,
        },
    )

    assert record.episode_index == 7
    assert record.seed == 107
    assert record.scenario == "week9_case_08"
    assert record.final_task_success
    assert record.moveit_planning_success
    assert record.moveit_execution_success
    assert record.local_expert_success
    assert record.completion_time_s == pytest.approx(3.25)
    assert record.handoff_error_rad == pytest.approx(0.012)
    assert record.final_lateral_error_m == pytest.approx(0.0002)
    assert record.final_insertion_depth_m == pytest.approx(0.0105)
    assert record.peak_axial_force_n == pytest.approx(6.5)
    assert record.peak_lateral_force_n == pytest.approx(1.8)
    assert record.failure_category is None
    assert record.failure_reason is None


def test_week9_result_conversion_maps_failure_category() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_record_from_week9_result,
    )

    record = benchmark_record_from_week9_result(
        episode_index=2,
        seed=22,
        completion_time_s=1.1,
        result={
            "scenario": "week9_case_03",
            "success": False,
            "moveit_planning_success": False,
            "moveit_execution_success": False,
            "handoff_error_rad": None,
            "local_expert_success": False,
            "final_lateral_error_m": None,
            "final_insertion_depth_m": None,
            "peak_axial_force_n": None,
            "peak_lateral_force_n": None,
            "failure_category": "moveit_planning_failure",
        },
    )

    assert not record.final_task_success
    assert (
        record.failure_category
        is EpisodeFailureCategory.PLANNING
    )
    assert (
        record.failure_reason
        == "Week 9 failure: moveit_planning_failure"
    )


def test_week9_result_conversion_rejects_nonfinite_metric() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_record_from_week9_result,
    )

    with pytest.raises(
        ValueError,
        match="peak_axial_force_n must be finite",
    ):
        benchmark_record_from_week9_result(
            episode_index=0,
            seed=0,
            completion_time_s=1.0,
            result={
                "scenario": "week9_case_01",
                "success": True,
                "moveit_planning_success": True,
                "moveit_execution_success": True,
                "handoff_error_rad": 0.01,
                "local_expert_success": True,
                "final_lateral_error_m": 0.0001,
                "final_insertion_depth_m": 0.010,
                "peak_axial_force_n": math.nan,
                "peak_lateral_force_n": 1.0,
                "failure_category": None,
            },
        )


def test_week9_result_conversion_requires_boolean_stage_fields() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_record_from_week9_result,
    )

    with pytest.raises(
        ValueError,
        match="moveit_planning_success must be boolean",
    ):
        benchmark_record_from_week9_result(
            episode_index=0,
            seed=0,
            completion_time_s=1.0,
            result={
                "scenario": "week9_case_01",
                "success": False,
                "moveit_planning_success": "yes",
                "moveit_execution_success": False,
                "handoff_error_rad": None,
                "local_expert_success": False,
                "final_lateral_error_m": None,
                "final_insertion_depth_m": None,
                "peak_axial_force_n": None,
                "peak_lateral_force_n": None,
                "failure_category": "moveit_planning_failure",
            },
        )


def test_default_benchmark_seeds_are_exactly_200_unique_values() -> None:
    from robustlearn.week10_benchmark import (
        default_benchmark_seeds,
    )

    seeds = default_benchmark_seeds()

    assert len(seeds) == 200
    assert seeds == tuple(range(200))
    assert len(set(seeds)) == 200


def test_benchmark_config_requires_exactly_200_seeds() -> None:
    from robustlearn.week10_benchmark import BenchmarkConfig

    with pytest.raises(
        ValueError,
        match="requires exactly 200 seeds",
    ):
        BenchmarkConfig(
            seeds=(0, 1),
            timeout_s=90.0,
            base_domain_id=190,
            domain_pool_size=40,
        )


def test_benchmark_config_rejects_duplicate_seeds() -> None:
    from robustlearn.week10_benchmark import BenchmarkConfig

    seeds = tuple(range(199)) + (0,)

    with pytest.raises(
        ValueError,
        match="seeds must be unique",
    ):
        BenchmarkConfig(
            seeds=seeds,
            timeout_s=90.0,
            base_domain_id=190,
            domain_pool_size=40,
        )


def test_build_benchmark_config_uses_canonical_defaults() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_config,
    )

    config = build_benchmark_config()

    assert config.seeds == tuple(range(200))
    assert config.timeout_s == pytest.approx(90.0)
    assert config.base_domain_id == 190
    assert config.domain_pool_size == 40
    assert config.to_dict()["episode_count"] == 200


def test_benchmark_report_rejects_seed_order_mismatch() -> None:
    from robustlearn.week10_benchmark import (
        BenchmarkReport,
        build_benchmark_config,
        summarize_benchmark,
    )

    episodes = tuple(
        make_success(index)
        for index in range(200)
    )

    summary = summarize_benchmark(episodes)
    config = build_benchmark_config(
        seeds=tuple(reversed(range(200)))
    )

    with pytest.raises(
        ValueError,
        match="seeds must match configuration order",
    ):
        BenchmarkReport(
            config=config,
            episodes=episodes,
            summary=summary,
        )


def test_write_benchmark_report_is_deterministic(
    tmp_path,
) -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_config,
        build_benchmark_report,
        write_benchmark_report,
    )

    episodes = tuple(
        make_success(index)
        for index in range(200)
    )

    report = build_benchmark_report(
        config=build_benchmark_config(),
        episodes=episodes,
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_benchmark_report(first, report)
    write_benchmark_report(second, report)

    assert first.read_bytes() == second.read_bytes()


def test_week10_scenario_assignment_matches_seed_cycle() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_scenario_for_seed,
    )

    assert benchmark_scenario_for_seed(0) == "week9_case_01"
    assert benchmark_scenario_for_seed(19) == "week9_case_20"
    assert benchmark_scenario_for_seed(20) == "week9_case_01"
    assert benchmark_scenario_for_seed(21) == "week9_case_02"


def test_week10_domain_assignment_wraps_pool() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_domain_for_episode,
    )

    assert benchmark_domain_for_episode(
        episode_index=0,
        base_domain_id=190,
        domain_pool_size=40,
    ) == 190

    assert benchmark_domain_for_episode(
        episode_index=39,
        base_domain_id=190,
        domain_pool_size=40,
    ) == 229

    assert benchmark_domain_for_episode(
        episode_index=40,
        base_domain_id=190,
        domain_pool_size=40,
    ) == 190


def test_benchmark_start_state_is_deterministic_for_same_seed() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
    )

    base = (
        0.10,
        -0.10,
        0.10,
        -1.70,
        0.10,
        1.70,
        -0.70,
    )

    first = build_benchmark_start_state(
        seed=42,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    second = build_benchmark_start_state(
        seed=42,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    assert first == second


def test_benchmark_start_state_differs_for_different_seeds() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
    )

    base = (
        0.10,
        -0.10,
        0.10,
        -1.70,
        0.10,
        1.70,
        -0.70,
    )

    first = build_benchmark_start_state(
        seed=1,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    second = build_benchmark_start_state(
        seed=2,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    assert first.joint_positions_rad != second.joint_positions_rad


def test_benchmark_start_state_respects_perturbation_bound() -> None:
    from robustlearn.week10_benchmark import (
        BENCHMARK_START_PERTURBATION_RAD,
        build_benchmark_start_state,
    )

    base = (
        0.10,
        -0.10,
        0.10,
        -1.70,
        0.10,
        1.70,
        -0.70,
    )

    state = build_benchmark_start_state(
        seed=123,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    assert all(
        abs(delta) <= BENCHMARK_START_PERTURBATION_RAD
        for delta in state.perturbation_rad
    )


def test_benchmark_start_state_reconstructs_joint_positions() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
    )

    base = (
        0.10,
        -0.10,
        0.10,
        -1.70,
        0.10,
        1.70,
        -0.70,
    )

    state = build_benchmark_start_state(
        seed=7,
        scenario="week9_case_01",
        base_joint_positions_rad=base,
    )

    expected = tuple(
        base_value + delta
        for base_value, delta in zip(
            base,
            state.perturbation_rad,
            strict=True,
        )
    )

    assert state.joint_positions_rad == pytest.approx(expected)


def test_benchmark_start_state_rejects_wrong_joint_count() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
    )

    with pytest.raises(
        ValueError,
        match="exactly 7 values",
    ):
        build_benchmark_start_state(
            seed=0,
            scenario="week9_case_01",
            base_joint_positions_rad=(0.0,) * 6,
        )


def test_benchmark_start_state_rejects_nonfinite_base_state() -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
    )

    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        build_benchmark_start_state(
            seed=0,
            scenario="week9_case_01",
            base_joint_positions_rad=(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                math.nan,
            ),
        )


def _canonical_week9_case_01() -> tuple[float, ...]:
    return (
        0.10,
        -0.10,
        0.10,
        -1.70,
        0.10,
        1.70,
        -0.70,
    )


def test_benchmark_keyframe_name_is_deterministic() -> None:
    from robustlearn.week10_benchmark import benchmark_keyframe_name

    assert benchmark_keyframe_name(0) == "week10_seed_000000"
    assert benchmark_keyframe_name(123) == "week10_seed_000123"


def test_seeded_benchmark_mjcf_preserves_source_file(
    tmp_path,
) -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
        build_seeded_benchmark_mjcf,
    )

    source = Path(
        "robot_description/mjcf/insertion/panda_insertion.xml"
    )
    before = source.read_bytes()

    start_state = build_benchmark_start_state(
        seed=42,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    output = tmp_path / "seeded.xml"

    build_seeded_benchmark_mjcf(
        source_path=source,
        output_path=output,
        start_state=start_state,
    )

    assert source.read_bytes() == before
    assert output.is_file()


def test_seeded_benchmark_mjcf_contains_seed_specific_keyframe(
    tmp_path,
) -> None:
    import xml.etree.ElementTree as ET

    from robustlearn.week10_benchmark import (
        benchmark_keyframe_name,
        build_benchmark_start_state,
        build_seeded_benchmark_mjcf,
    )

    start_state = build_benchmark_start_state(
        seed=7,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    output = build_seeded_benchmark_mjcf(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        output_path=tmp_path / "seeded.xml",
        start_state=start_state,
    )

    root = ET.parse(output).getroot()

    key = root.find(
        f"./keyframe/key[@name='{benchmark_keyframe_name(7)}']"
    )

    assert key is not None

    qpos = tuple(
        float(value)
        for value in key.attrib["qpos"].split()
    )

    assert qpos[:7] == pytest.approx(
        start_state.joint_positions_rad
    )

    # Preserve the two gripper qpos values from the source keyframe.
    assert qpos[7:] == pytest.approx(
        (0.04, 0.04)
    )


def test_seeded_benchmark_mjcf_updates_arm_ctrl_and_preserves_tail(
    tmp_path,
) -> None:
    import xml.etree.ElementTree as ET

    from robustlearn.week10_benchmark import (
        benchmark_keyframe_name,
        build_benchmark_start_state,
        build_seeded_benchmark_mjcf,
    )

    start_state = build_benchmark_start_state(
        seed=8,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    output = build_seeded_benchmark_mjcf(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        output_path=tmp_path / "seeded.xml",
        start_state=start_state,
    )

    root = ET.parse(output).getroot()

    key = root.find(
        f"./keyframe/key[@name='{benchmark_keyframe_name(8)}']"
    )

    assert key is not None

    ctrl = tuple(
        float(value)
        for value in key.attrib["ctrl"].split()
    )

    assert ctrl[:7] == pytest.approx(
        start_state.joint_positions_rad
    )

    # Preserve the canonical gripper actuator command.
    assert ctrl[7:] == pytest.approx((255.0,))


def test_seeded_benchmark_mjcf_is_deterministic(
    tmp_path,
) -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_start_state,
        build_seeded_benchmark_mjcf,
    )

    start_state = build_benchmark_start_state(
        seed=99,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"

    build_seeded_benchmark_mjcf(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        output_path=first,
        start_state=start_state,
    )

    build_seeded_benchmark_mjcf(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        output_path=second,
        start_state=start_state,
    )

    assert first.read_bytes() == second.read_bytes()


def test_seeded_benchmark_mjcf_loads_in_mujoco(
    tmp_path,
) -> None:
    import mujoco

    from robustlearn.week10_benchmark import (
        benchmark_keyframe_name,
        build_benchmark_start_state,
        build_seeded_benchmark_mjcf,
    )

    start_state = build_benchmark_start_state(
        seed=123,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    output = build_seeded_benchmark_mjcf(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        output_path=tmp_path / "seeded.xml",
        start_state=start_state,
    )

    model = mujoco.MjModel.from_xml_path(
        str(output)
    )

    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        benchmark_keyframe_name(123),
    )

    assert key_id >= 0


def test_load_benchmark_base_joint_positions_reads_week9_case() -> None:
    from robustlearn.week10_benchmark import (
        load_benchmark_base_joint_positions,
    )

    positions = load_benchmark_base_joint_positions(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        scenario="week9_case_01",
    )

    assert positions == pytest.approx(
        _canonical_week9_case_01()
    )


def test_load_benchmark_base_joint_positions_reads_distinct_case() -> None:
    from robustlearn.week10_benchmark import (
        load_benchmark_base_joint_positions,
    )

    positions = load_benchmark_base_joint_positions(
        source_path=(
            "robot_description/mjcf/insertion/panda_insertion.xml"
        ),
        scenario="week9_case_02",
    )

    assert positions == pytest.approx(
        (
            0.08,
            -0.10,
            0.10,
            -1.70,
            0.10,
            1.70,
            -0.70,
        )
    )


def test_load_benchmark_base_joint_positions_rejects_unknown_case() -> None:
    from robustlearn.week10_benchmark import (
        load_benchmark_base_joint_positions,
    )

    with pytest.raises(
        ValueError,
        match="does not contain scenario",
    ):
        load_benchmark_base_joint_positions(
            source_path=(
                "robot_description/mjcf/insertion/panda_insertion.xml"
            ),
            scenario="week9_case_99",
        )


def test_benchmark_record_serializes_start_state_metadata() -> None:
    from robustlearn.week10_benchmark import (
        benchmark_record_from_week9_result,
        build_benchmark_start_state,
    )

    start_state = build_benchmark_start_state(
        seed=42,
        scenario="week9_case_01",
        base_joint_positions_rad=_canonical_week9_case_01(),
    )

    result = {
        "scenario": "week9_case_01",
        "success": True,
        "moveit_planning_success": True,
        "moveit_execution_success": True,
        "local_expert_success": True,
        "handoff_error_rad": 0.001,
        "final_lateral_error_m": 0.0001,
        "final_insertion_depth_m": 0.010,
        "peak_axial_force_n": 0.63,
        "peak_lateral_force_n": 0.35,
        "failure_category": None,
    }

    record = benchmark_record_from_week9_result(
        episode_index=0,
        seed=42,
        completion_time_s=15.0,
        result=result,
        start_state=start_state,
    )

    serialized = record.to_dict()

    assert serialized["start_base_joint_positions_rad"] == list(
        start_state.base_joint_positions_rad
    )
    assert serialized["start_perturbation_rad"] == list(
        start_state.perturbation_rad
    )
    assert serialized["start_joint_positions_rad"] == list(
        start_state.joint_positions_rad
    )


def test_record_rejects_partial_start_state_metadata() -> None:
    values = make_success().__dict__ | {
        "start_base_joint_positions_rad": (0.0,) * 7,
    }

    with pytest.raises(
        ValueError,
        match="all present or all absent",
    ):
        BenchmarkEpisodeRecord(**values)


def test_record_rejects_wrong_start_state_joint_count() -> None:
    values = make_success().__dict__ | {
        "start_base_joint_positions_rad": (0.0,) * 6,
        "start_perturbation_rad": (0.0,) * 6,
        "start_joint_positions_rad": (0.0,) * 6,
    }

    with pytest.raises(
        ValueError,
        match="exactly 7 values",
    ):
        BenchmarkEpisodeRecord(**values)


def test_record_rejects_inconsistent_start_state_reconstruction() -> None:
    values = make_success().__dict__ | {
        "start_base_joint_positions_rad": (0.0,) * 7,
        "start_perturbation_rad": (0.01,) * 7,
        "start_joint_positions_rad": (0.0,) * 7,
    }

    with pytest.raises(
        ValueError,
        match="base positions plus perturbation",
    ):
        BenchmarkEpisodeRecord(**values)


def test_write_benchmark_checkpoint_persists_partial_progress(
    tmp_path,
) -> None:
    import json

    from robustlearn.week10_benchmark import (
        build_benchmark_config,
        write_benchmark_checkpoint,
    )

    config = build_benchmark_config()
    records = (
        make_success(0),
        make_success(1),
    )

    output = tmp_path / "checkpoint.json"

    result = write_benchmark_checkpoint(
        output,
        config=config,
        episodes=records,
        requested_episode_count=200,
    )

    assert result == output

    payload = json.loads(
        output.read_text()
    )

    assert payload["mode"] == "checkpoint"
    assert payload["requested_episode_count"] == 200
    assert payload["completed_episode_count"] == 2
    assert len(payload["episodes"]) == 2
    assert payload["episodes"][0]["seed"] == 0
    assert payload["episodes"][1]["seed"] == 1


def test_write_benchmark_checkpoint_replaces_previous_progress(
    tmp_path,
) -> None:
    import json

    from robustlearn.week10_benchmark import (
        build_benchmark_config,
        write_benchmark_checkpoint,
    )

    config = build_benchmark_config()
    output = tmp_path / "checkpoint.json"

    write_benchmark_checkpoint(
        output,
        config=config,
        episodes=(make_success(0),),
        requested_episode_count=200,
    )

    write_benchmark_checkpoint(
        output,
        config=config,
        episodes=(
            make_success(0),
            make_success(1),
        ),
        requested_episode_count=200,
    )

    payload = json.loads(
        output.read_text()
    )

    assert payload["completed_episode_count"] == 2
    assert len(payload["episodes"]) == 2

    temporary_path = output.with_name(
        f".{output.name}.tmp"
    )

    assert not temporary_path.exists()


def test_write_benchmark_checkpoint_rejects_seed_order_mismatch(
    tmp_path,
) -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_config,
        write_benchmark_checkpoint,
    )

    config = build_benchmark_config()

    with pytest.raises(
        ValueError,
        match="seeds must match configuration order",
    ):
        write_benchmark_checkpoint(
            tmp_path / "checkpoint.json",
            config=config,
            episodes=(
                make_success(
                    0,
                    seed=99,
                ),
            ),
            requested_episode_count=200,
        )


def test_write_benchmark_checkpoint_rejects_noncontiguous_indices(
    tmp_path,
) -> None:
    from robustlearn.week10_benchmark import (
        build_benchmark_config,
        write_benchmark_checkpoint,
    )

    config = build_benchmark_config()

    with pytest.raises(
        ValueError,
        match="indices must be contiguous from zero",
    ):
        write_benchmark_checkpoint(
            tmp_path / "checkpoint.json",
            config=config,
            episodes=(make_success(1),),
            requested_episode_count=200,
        )
