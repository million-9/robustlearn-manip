"""Tests for seeded scripted expert collection orchestration."""

import pytest

from robustlearn.episode_manager import (
    EpisodeFailureCategory,
    EpisodeSpec,
)
from robustlearn.rollout_dataset import (
    RolloutFrame,
    episode_path,
    read_episode,
)
from robustlearn.scripted_demo_collection import (
    write_expert_frame_capture,
)
from robustlearn.week10_benchmark import BenchmarkEpisodeRecord


@pytest.mark.parametrize("success", [True, False])
def test_collect_one_seeded_expert_episode(
    tmp_path, monkeypatch, success
):
    from tools import collect_scripted_demos as collector

    spec = EpisodeSpec(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        timeout_s=90.0,
    )

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"

    calls = {}

    def fake_run_episode(
        received_spec,
        *,
        base_domain_id,
        domain_pool_size,
        expert_capture_path,
    ):
        calls["spec"] = received_spec
        calls["capture_path"] = expert_capture_path

        if success:
            frame = RolloutFrame(
                frame_index=0,
                timestamp_s=101.5,
                joint_position=(0.0,) * 7,
                joint_velocity=None,
                task_relative_state=(0.01, 0.0, 0.02),
                wrist_wrench=(0.0,) * 6,
                expert_action=(0.1,) * 7,
                task_success=True,
                task_failure=False,
                terminated=True,
            )

            write_expert_frame_capture(
                expert_capture_path,
                (frame,),
            )

        return BenchmarkEpisodeRecord(
            episode_index=received_spec.episode_index,
            seed=received_spec.seed,
            scenario=received_spec.scenario,
            moveit_planning_success=True,
            moveit_execution_success=True,
            handoff_error_rad=0.001,
            local_expert_success=success,
            final_task_success=success,
            completion_time_s=3.0,
            final_lateral_error_m=0.001,
            final_insertion_depth_m=0.02,
            peak_axial_force_n=4.0,
            peak_lateral_force_n=2.0,
            failure_category=(
                None
                if success
                else EpisodeFailureCategory.EXPERT
            ),
            failure_reason=(
                None
                if success
                else "local expert failed"
            ),
        )

    monkeypatch.setattr(
        collector,
        "run_episode",
        fake_run_episode,
    )

    clock_values = iter((100.0, 103.0))

    episode = collector.collect_one_episode(
        spec,
        dataset_root=dataset_root,
        capture_root=capture_root,
        git_commit="test-commit",
        config_id="test-config",
        task="peg_insertion",
        base_domain_id=190,
        domain_pool_size=40,
        clock=lambda: next(clock_values),
    )

    assert calls["spec"] == spec
    assert calls["capture_path"].parent == capture_root

    assert episode.source == "expert"
    assert episode.seed == 42
    assert episode.scenario == "week9_case_01"
    assert episode.success is success

    path = episode_path(dataset_root, episode)

    assert path.is_file()
    assert read_episode(path) == episode

    if success:
        assert len(episode.frames) == 1
        assert episode.failure_reason is None
    else:
        assert episode.frames == ()
        assert episode.failure_reason == "local expert failed"


def test_collector_rejects_existing_dataset_before_execution(
    tmp_path, monkeypatch
):
    """Never launch an episode when its dataset record exists."""
    from robustlearn.rollout_dataset import (
        RolloutEpisode,
        write_episode,
    )
    from tools import collect_scripted_demos as collector

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"

    spec = EpisodeSpec(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        timeout_s=90.0,
    )

    existing_episode = RolloutEpisode(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        source="expert",
        task="peg_insertion",
        success=False,
        failure_reason="previous attempt failed",
        start_timestamp_s=100.0,
        end_timestamp_s=103.0,
        git_commit="test-commit",
        config_id="test-config",
        frames=(),
    )

    existing_path = write_episode(
        dataset_root,
        existing_episode,
    )

    original_content = existing_path.read_bytes()

    def unexpected_execution(*args, **kwargs):
        raise AssertionError(
            "ROS2 execution must not start for an existing episode"
        )

    monkeypatch.setattr(
        collector,
        "run_episode",
        unexpected_execution,
    )

    with pytest.raises(FileExistsError, match="dataset"):
        collector.collect_one_episode(
            spec,
            dataset_root=dataset_root,
            capture_root=capture_root,
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
            base_domain_id=190,
            domain_pool_size=40,
        )

    assert existing_path.read_bytes() == original_content


def test_multi_episode_collection_resumes_and_summarizes(
    tmp_path, monkeypatch
):
    """Skip existing episodes, collect new ones, and report outcomes."""
    import json

    from robustlearn.rollout_dataset import (
        RolloutEpisode,
        write_episode,
    )
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )
    from tools import collect_scripted_demos as collector

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"
    summary_path = tmp_path / "summary.json"

    specs = [
        EpisodeSpec(
            episode_index=index,
            seed=42 + index,
            scenario=f"week9_case_{index + 1:02d}",
            timeout_s=90.0,
        )
        for index in range(3)
    ]

    def make_episode(spec, success):
        return RolloutEpisode(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            source="expert",
            task="peg_insertion",
            success=success,
            failure_reason=(
                None if success else "local expert failed"
            ),
            start_timestamp_s=100.0,
            end_timestamp_s=103.0,
            git_commit="test-commit",
            config_id="test-config",
            frames=(
                (
                    RolloutFrame(
                        frame_index=0,
                        timestamp_s=101.5,
                        joint_position=(0.0,) * 7,
                        joint_velocity=None,
                        task_relative_state=(0.01, 0.0, 0.02),
                        wrist_wrench=(0.0,) * 6,
                        expert_action=(0.1,) * 7,
                        task_success=True,
                        task_failure=False,
                        terminated=True,
                    ),
                )
                if success
                else ()
            ),
        )

    existing = make_episode(specs[0], success=False)
    existing_path = write_episode(dataset_root, existing)
    original_bytes = existing_path.read_bytes()

    executed = []

    def fake_collect_one_episode(spec, **kwargs):
        executed.append(spec.episode_index)

        if spec.episode_index == 0:
            raise AssertionError(
                "Existing episode must not be executed again"
            )

        episode = make_episode(
            spec,
            success=(spec.episode_index == 1),
        )

        persist_scripted_expert_episode(
            episode,
            output_root=dataset_root,
        )

        return episode

    monkeypatch.setattr(
        collector,
        "collect_one_episode",
        fake_collect_one_episode,
    )

    arguments = dict(
        dataset_root=dataset_root,
        capture_root=capture_root,
        summary_path=summary_path,
        git_commit="test-commit",
        config_id="test-config",
        task="peg_insertion",
        base_domain_id=190,
        domain_pool_size=40,
    )

    summary = collector.collect_episodes(specs, **arguments)

    assert executed == [1, 2]
    assert existing_path.read_bytes() == original_bytes

    assert summary["requested_count"] == 3
    assert summary["skipped_count"] == 1
    assert summary["collected_success_count"] == 1
    assert summary["collected_failure_count"] == 1

    assert [
        (item["episode_index"], item["status"])
        for item in summary["episodes"]
    ] == [
        (0, "skipped"),
        (1, "success"),
        (2, "failure"),
    ]

    assert summary["episodes"][2]["failure_reason"] == (
        "local expert failed"
    )

    assert json.loads(
        summary_path.read_text(encoding="utf-8")
    ) == summary

    # Rerunning the same collection must execute nothing.
    executed.clear()

    resumed = collector.collect_episodes(specs, **arguments)

    assert executed == []
    assert resumed["requested_count"] == 3
    assert resumed["skipped_count"] == 3
    assert resumed["collected_success_count"] == 0
    assert resumed["collected_failure_count"] == 0
    assert existing_path.read_bytes() == original_bytes


def test_batch_interruption_writes_partial_summary(
    tmp_path, monkeypatch
):
    """Preserve completed work and report unexpected interruption."""
    import json

    from robustlearn.rollout_dataset import (
        RolloutEpisode,
    )
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )
    from tools import collect_scripted_demos as collector

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"
    summary_path = tmp_path / "summary.json"

    specs = [
        EpisodeSpec(
            episode_index=index,
            seed=100 + index,
            scenario=f"week9_case_{index + 1:02d}",
            timeout_s=90.0,
        )
        for index in range(2)
    ]

    executed = []

    def fake_collect_one_episode(spec, **kwargs):
        executed.append(spec.episode_index)

        if spec.episode_index == 1:
            raise RuntimeError("simulated ROS2 launch crash")

        episode = RolloutEpisode(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            source="expert",
            task="peg_insertion",
            success=False,
            failure_reason="local expert failed",
            start_timestamp_s=100.0,
            end_timestamp_s=103.0,
            git_commit="test-commit",
            config_id="test-config",
            frames=(),
        )

        persist_scripted_expert_episode(
            episode,
            output_root=dataset_root,
        )

        return episode

    monkeypatch.setattr(
        collector,
        "collect_one_episode",
        fake_collect_one_episode,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated ROS2 launch crash",
    ):
        collector.collect_episodes(
            specs,
            dataset_root=dataset_root,
            capture_root=capture_root,
            summary_path=summary_path,
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
            base_domain_id=190,
            domain_pool_size=40,
        )

    assert executed == [0, 1]

    # Episode 0 must remain safely persisted.
    assert episode_path(dataset_root, specs[0]).is_file()

    # The interrupted attempt must be explicitly reported.
    assert summary_path.is_file()

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["requested_count"] == 2
    assert summary["collected_failure_count"] == 1
    assert summary["collection_error_count"] == 1

    assert [
        item["status"]
        for item in summary["episodes"]
    ] == ["failure", "error"]

    assert summary["episodes"][1]["episode_index"] == 1
    assert "simulated ROS2 launch crash" in (
        summary["episodes"][1]["failure_reason"]
    )


def test_batch_checkpoint_survives_abrupt_interruption(
    tmp_path, monkeypatch
):
    """Completed episodes must be checkpointed before the next run."""

    import json

    from robustlearn.rollout_dataset import RolloutEpisode
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )
    from tools import collect_scripted_demos as collector

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"
    summary_path = tmp_path / "summary.json"

    specs = [
        EpisodeSpec(
            episode_index=index,
            seed=200 + index,
            scenario=f"week9_case_{index + 1:02d}",
            timeout_s=90.0,
        )
        for index in range(2)
    ]

    def fake_collect_one_episode(spec, **kwargs):
        if spec.episode_index == 1:
            # Bypasses the collector's ordinary Exception handler.
            raise KeyboardInterrupt("simulated abrupt interruption")

        episode = RolloutEpisode(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            source="expert",
            task="peg_insertion",
            success=False,
            failure_reason="local expert failed",
            start_timestamp_s=100.0,
            end_timestamp_s=103.0,
            git_commit="test-commit",
            config_id="test-config",
            frames=(),
        )

        persist_scripted_expert_episode(
            episode,
            output_root=dataset_root,
        )

        return episode

    monkeypatch.setattr(
        collector,
        "collect_one_episode",
        fake_collect_one_episode,
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="simulated abrupt interruption",
    ):
        collector.collect_episodes(
            specs,
            dataset_root=dataset_root,
            capture_root=capture_root,
            summary_path=summary_path,
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
            base_domain_id=190,
            domain_pool_size=40,
        )

    # The completed episode must survive the interruption.
    assert episode_path(dataset_root, specs[0]).is_file()

    # Its progress summary must already be on disk.
    assert summary_path.is_file()

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["requested_count"] == 2
    assert summary["collected_failure_count"] == 1
    assert summary["episodes"][0]["episode_index"] == 0
    assert summary["episodes"][0]["status"] == "failure"
    assert len(summary["episodes"]) == 1


def test_orphan_capture_is_preserved_and_reported(
    tmp_path, monkeypatch
):
    """Never overwrite a capture left by an interrupted episode."""
    import json

    from tools import collect_scripted_demos as collector

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "captures"
    summary_path = tmp_path / "summary.json"

    spec = EpisodeSpec(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        timeout_s=90.0,
    )

    capture_root.mkdir(parents=True)

    capture_path = capture_root / (
        "episode_000000__seed_42__week9_case_01.json"
    )

    original_bytes = b"incomplete expert capture"
    capture_path.write_bytes(original_bytes)

    def unexpected_execution(*args, **kwargs):
        raise AssertionError(
            "Existing capture must prevent another ROS2 execution"
        )

    monkeypatch.setattr(
        collector,
        "run_episode",
        unexpected_execution,
    )

    with pytest.raises(FileExistsError, match="capture"):
        collector.collect_episodes(
            [spec],
            dataset_root=dataset_root,
            capture_root=capture_root,
            summary_path=summary_path,
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
            base_domain_id=190,
            domain_pool_size=40,
        )

    assert capture_path.read_bytes() == original_bytes
    assert not episode_path(dataset_root, spec).exists()

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["requested_count"] == 1
    assert summary["collection_error_count"] == 1
    assert summary["collected_success_count"] == 0

    assert len(summary["episodes"]) == 1
    assert summary["episodes"][0]["status"] == "error"
    assert summary["episodes"][0]["episode_index"] == 0
    assert "capture" in summary["episodes"][0]["failure_reason"]


def test_collector_preserves_seeded_start_vectors(
    tmp_path, monkeypatch
):
    """Preserve actual benchmark randomization in the saved episode."""
    from robustlearn.rollout_dataset import read_episode
    from tools import collect_scripted_demos as collector

    base = (0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.7)
    perturbation = (
        0.01, 0.02, 0.0, -0.01, 0.0, 0.03, -0.02,
    )
    start = tuple(
        a + b
        for a, b in zip(base, perturbation, strict=True)
    )

    spec = EpisodeSpec(
        episode_index=5,
        seed=5,
        scenario="week9_case_06",
        timeout_s=90.0,
    )

    dataset_root = tmp_path / "dataset"

    def fake_run_episode(
        received_spec,
        *,
        base_domain_id,
        domain_pool_size,
        expert_capture_path,
    ):
        assert received_spec == spec

        frame = RolloutFrame(
            frame_index=0,
            timestamp_s=101.0,
            joint_position=start,
            joint_velocity=(0.0,) * 7,
            task_relative_state=(0.01, 0.0, 0.02),
            wrist_wrench=(0.0,) * 6,
            expert_action=start,
            task_success=True,
            task_failure=False,
            terminated=True,
        )

        write_expert_frame_capture(
            expert_capture_path,
            (frame,),
        )

        return BenchmarkEpisodeRecord(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            moveit_planning_success=True,
            moveit_execution_success=True,
            handoff_error_rad=0.001,
            local_expert_success=True,
            final_task_success=True,
            completion_time_s=3.0,
            final_lateral_error_m=0.001,
            final_insertion_depth_m=0.02,
            peak_axial_force_n=4.0,
            peak_lateral_force_n=2.0,
            failure_category=None,
            failure_reason=None,
            start_base_joint_positions_rad=base,
            start_perturbation_rad=perturbation,
            start_joint_positions_rad=start,
        )

    monkeypatch.setattr(
        collector,
        "run_episode",
        fake_run_episode,
    )

    clock_values = iter((100.0, 103.0))

    episode = collector.collect_one_episode(
        spec,
        dataset_root=dataset_root,
        capture_root=tmp_path / "captures",
        git_commit="test-commit",
        config_id="test-config",
        task="peg_insertion",
        base_domain_id=190,
        domain_pool_size=40,
        clock=lambda: next(clock_values),
    )

    assert episode.start_base_joint_positions_rad == base
    assert episode.start_perturbation_rad == perturbation
    assert episode.start_joint_positions_rad == start

    persisted = read_episode(
        episode_path(dataset_root, episode)
    )

    assert persisted == episode
    assert persisted.start_base_joint_positions_rad == base
    assert persisted.start_perturbation_rad == perturbation
    assert persisted.start_joint_positions_rad == start
