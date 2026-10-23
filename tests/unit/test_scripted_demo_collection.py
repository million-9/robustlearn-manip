"""Contracts for reproducible scripted-expert demonstration storage."""

from dataclasses import replace

import pytest

from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    read_episode,
)


def make_expert_episode() -> RolloutEpisode:
    """Build a minimal valid synthetic episode for storage tests."""
    frame = RolloutFrame(
        frame_index=0,
        timestamp_s=1.5,
        joint_position=(0.0,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.0,) * 3,
        wrist_wrench=(0.0,) * 6,
        expert_action=(0.0,) * 7,
        task_success=True,
        task_failure=False,
        terminated=True,
    )

    return RolloutEpisode(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        source="expert",
        task="peg_insertion",
        success=True,
        failure_reason=None,
        start_timestamp_s=1.0,
        end_timestamp_s=2.0,
        git_commit="test-commit",
        config_id="test-config",
        frames=(frame,),
    )


def test_persist_valid_expert_episode(tmp_path):
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )

    episode = make_expert_episode()
    path = persist_scripted_expert_episode(episode, tmp_path)

    assert path.is_file()
    assert read_episode(path) == episode


def test_identical_rerun_is_idempotent(tmp_path):
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )

    episode = make_expert_episode()

    first_path = persist_scripted_expert_episode(episode, tmp_path)
    original_bytes = first_path.read_bytes()

    second_path = persist_scripted_expert_episode(episode, tmp_path)

    assert first_path == second_path
    assert second_path.read_bytes() == original_bytes


def test_conflicting_episode_cannot_overwrite(tmp_path):
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )

    episode = make_expert_episode()
    path = persist_scripted_expert_episode(episode, tmp_path)
    original_bytes = path.read_bytes()

    conflicting = replace(episode, git_commit="different-commit")

    with pytest.raises(FileExistsError):
        persist_scripted_expert_episode(conflicting, tmp_path)

    assert path.read_bytes() == original_bytes


def test_successful_episode_requires_frames(tmp_path):
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )

    episode = replace(make_expert_episode(), frames=())

    with pytest.raises(ValueError, match="frames"):
        persist_scripted_expert_episode(episode, tmp_path)


def test_scripted_collection_rejects_teleop_source(tmp_path):
    from robustlearn.scripted_demo_collection import (
        persist_scripted_expert_episode,
    )

    episode = replace(make_expert_episode(), source="teleop")

    with pytest.raises(ValueError, match="source"):
        persist_scripted_expert_episode(episode, tmp_path)


def test_expert_frame_buffer_records_submitted_command():
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    buffer.record_submitted_command(
        timestamp_s=1.5,
        joint_position=(0.0,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.02, 0.03),
        wrist_wrench=(0.0,) * 6,
        submitted_joint_target=(0.1,) * 7,
    )

    assert len(buffer.frames) == 1

    frame = buffer.frames[0]

    assert frame.frame_index == 0
    assert frame.expert_action == (0.1,) * 7
    assert frame.task_relative_state == (0.01, 0.02, 0.03)
    assert frame.terminated is False


def test_expert_frame_buffer_assigns_contiguous_indices():
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    for timestamp in (1.0, 1.1, 1.2):
        buffer.record_submitted_command(
            timestamp_s=timestamp,
            joint_position=(0.0,) * 7,
            joint_velocity=None,
            task_relative_state=(0.0,) * 3,
            wrist_wrench=(0.0,) * 6,
            submitted_joint_target=(0.1,) * 7,
        )

    assert tuple(
        frame.frame_index for frame in buffer.frames
    ) == (0, 1, 2)


def test_expert_frame_buffer_rejects_nonfinite_action():
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    with pytest.raises(ValueError, match="finite"):
        buffer.record_submitted_command(
            timestamp_s=1.0,
            joint_position=(0.0,) * 7,
            joint_velocity=None,
            task_relative_state=(0.0,) * 3,
            wrist_wrench=(0.0,) * 6,
            submitted_joint_target=(float("nan"),) + (0.0,) * 6,
        )

    assert buffer.frames == ()


def test_expert_frame_buffer_rejects_reversed_timestamps():
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    for timestamp in (2.0, 1.0):
        if timestamp == 1.0:
            with pytest.raises(ValueError, match="timestamp"):
                buffer.record_submitted_command(
                    timestamp_s=timestamp,
                    joint_position=(0.0,) * 7,
                    joint_velocity=None,
                    task_relative_state=(0.0,) * 3,
                    wrist_wrench=(0.0,) * 6,
                    submitted_joint_target=(0.1,) * 7,
                )
        else:
            buffer.record_submitted_command(
                timestamp_s=timestamp,
                joint_position=(0.0,) * 7,
                joint_velocity=None,
                task_relative_state=(0.0,) * 3,
                wrist_wrench=(0.0,) * 6,
                submitted_joint_target=(0.1,) * 7,
            )

    assert len(buffer.frames) == 1


def test_expert_frame_capture_round_trip(tmp_path):
    from robustlearn.scripted_demo_collection import (
        read_expert_frame_capture,
        write_expert_frame_capture,
    )

    frames = make_expert_episode().frames
    path = tmp_path / "expert_frames.json"

    write_expert_frame_capture(path, frames)

    assert path.is_file()
    assert read_expert_frame_capture(path) == frames


def test_expert_frame_capture_rejects_conflicting_file(tmp_path):
    from robustlearn.scripted_demo_collection import (
        write_expert_frame_capture,
    )

    frames = make_expert_episode().frames
    path = tmp_path / "expert_frames.json"

    write_expert_frame_capture(path, frames)
    original_bytes = path.read_bytes()

    conflicting_frame = replace(
        frames[0],
        expert_action=(0.2,) * 7,
    )

    with pytest.raises(FileExistsError):
        write_expert_frame_capture(
            path,
            (conflicting_frame,),
        )

    assert path.read_bytes() == original_bytes


def make_execution_result(success: bool):
    from robustlearn.episode_manager import (
        EpisodeFailureCategory,
        EpisodeResult,
    )

    return EpisodeResult(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
        start_time_s=1.0,
        end_time_s=2.0,
        success=success,
        failure_category=(
            None if success else EpisodeFailureCategory.EXPERT
        ),
        failure_reason=(
            None if success else "local expert failed"
        ),
    )


def test_assemble_successful_expert_demonstration(tmp_path):
    from robustlearn.scripted_demo_collection import (
        assemble_scripted_expert_episode,
        write_expert_frame_capture,
    )

    capture_path = tmp_path / "capture.json"
    frames = make_expert_episode().frames

    write_expert_frame_capture(capture_path, frames)

    episode = assemble_scripted_expert_episode(
        make_execution_result(success=True),
        capture_path=capture_path,
        git_commit="test-commit",
        config_id="test-config",
        task="peg_insertion",
    )

    assert episode.success is True
    assert episode.source == "expert"
    assert episode.seed == 42
    assert episode.scenario == "week9_case_01"
    assert episode.frames == frames


def test_successful_demonstration_rejects_missing_capture(tmp_path):
    from robustlearn.scripted_demo_collection import (
        assemble_scripted_expert_episode,
    )

    with pytest.raises(ValueError, match="capture"):
        assemble_scripted_expert_episode(
            make_execution_result(success=True),
            capture_path=tmp_path / "missing.json",
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
        )


def test_successful_demonstration_rejects_empty_frames(tmp_path):
    from robustlearn.scripted_demo_collection import (
        assemble_scripted_expert_episode,
        write_expert_frame_capture,
    )

    capture_path = tmp_path / "empty.json"
    write_expert_frame_capture(capture_path, ())

    with pytest.raises(ValueError, match="frames"):
        assemble_scripted_expert_episode(
            make_execution_result(success=True),
            capture_path=capture_path,
            git_commit="test-commit",
            config_id="test-config",
            task="peg_insertion",
        )


def test_failed_attempt_preserves_failure_without_capture(tmp_path):
    from robustlearn.scripted_demo_collection import (
        assemble_scripted_expert_episode,
    )

    episode = assemble_scripted_expert_episode(
        make_execution_result(success=False),
        capture_path=tmp_path / "missing.json",
        git_commit="test-commit",
        config_id="test-config",
        task="peg_insertion",
    )

    assert episode.success is False
    assert episode.failure_reason == "local expert failed"
    assert episode.source == "expert"
    assert episode.frames == ()


def make_benchmark_record(success: bool):
    """Construct a representative Week 10 benchmark record."""
    from robustlearn.episode_manager import EpisodeFailureCategory
    from robustlearn.week10_benchmark import BenchmarkEpisodeRecord

    return BenchmarkEpisodeRecord(
        episode_index=0,
        seed=42,
        scenario="week9_case_01",
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
            None if success else EpisodeFailureCategory.EXPERT
        ),
        failure_reason=(
            None if success else "local expert failed"
        ),
    )


def test_convert_successful_benchmark_to_episode_result():
    from robustlearn.scripted_demo_collection import (
        benchmark_record_to_episode_result,
    )

    record = make_benchmark_record(success=True)

    result = benchmark_record_to_episode_result(
        record,
        start_time_s=100.0,
        end_time_s=103.0,
    )

    assert result.episode_index == 0
    assert result.seed == 42
    assert result.scenario == "week9_case_01"
    assert result.start_time_s == 100.0
    assert result.end_time_s == 103.0
    assert result.success is True
    assert result.failure_category is None
    assert result.failure_reason is None


def test_convert_failed_benchmark_preserves_failure():
    from robustlearn.episode_manager import EpisodeFailureCategory
    from robustlearn.scripted_demo_collection import (
        benchmark_record_to_episode_result,
    )

    record = make_benchmark_record(success=False)

    result = benchmark_record_to_episode_result(
        record,
        start_time_s=200.0,
        end_time_s=203.0,
    )

    assert result.success is False
    assert result.failure_category == EpisodeFailureCategory.EXPERT
    assert result.failure_reason == "local expert failed"
    assert result.seed == 42
    assert result.scenario == "week9_case_01"
    assert result.duration_s == 3.0


@pytest.mark.parametrize("success", [True, False])
def test_expert_frame_buffer_marks_terminal_outcome(success):
    """Only the final expert frame receives the terminal outcome."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    for index in range(2):
        buffer.record_submitted_command(
            timestamp_s=100.0 + index,
            joint_position=(0.0,) * 7,
            joint_velocity=None,
            task_relative_state=(0.01, 0.0, 0.02),
            wrist_wrench=(0.0,) * 6,
            submitted_joint_target=(0.1,) * 7,
        )

    buffer.mark_terminal(success=success)

    first, last = buffer.frames

    assert first.terminated is False
    assert first.task_success is False
    assert first.task_failure is False

    assert last.terminated is True
    assert last.task_success is success
    assert last.task_failure is not success

    assert last.frame_index == 1
    assert last.expert_action == (0.1,) * 7


def test_expert_frame_buffer_allows_empty_failed_attempt():
    """An early failure can have no submitted expert commands."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    buffer.mark_terminal(success=False)

    assert buffer.frames == ()


def test_expert_frame_buffer_rejects_empty_success():
    """A successful expert demonstration requires recorded actions."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    with pytest.raises(ValueError, match="frames"):
        buffer.mark_terminal(success=True)


def test_pending_expert_command_requires_acceptance():
    """A submitted command is not yet an accepted demonstration."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    buffer.stage_submitted_command(
        timestamp_s=100.0,
        joint_position=(0.0,) * 7,
        joint_velocity=None,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        submitted_joint_target=(0.1,) * 7,
    )

    # Submission alone must not create a dataset frame.
    assert buffer.frames == ()

    buffer.accept_pending_command()

    assert len(buffer.frames) == 1
    assert buffer.frames[0].timestamp_s == 100.0
    assert buffer.frames[0].expert_action == (0.1,) * 7
    assert buffer.frames[0].terminated is False


def test_rejected_expert_command_is_discarded():
    """Controller-rejected commands must not become accepted frames."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    buffer.stage_submitted_command(
        timestamp_s=100.0,
        joint_position=(0.0,) * 7,
        joint_velocity=None,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        submitted_joint_target=(0.1,) * 7,
    )

    buffer.reject_pending_command()

    assert buffer.frames == ()

    # A failed attempt without accepted commands is valid.
    buffer.mark_terminal(success=False)
    assert buffer.frames == ()


def test_expert_buffer_rejects_overlapping_pending_commands():
    """One outstanding trajectory goal permits only one pending action."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    kwargs = dict(
        timestamp_s=100.0,
        joint_position=(0.0,) * 7,
        joint_velocity=None,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        submitted_joint_target=(0.1,) * 7,
    )

    buffer.stage_submitted_command(**kwargs)

    with pytest.raises(ValueError, match="pending"):
        buffer.stage_submitted_command(**kwargs)


def test_expert_buffer_rejects_acceptance_without_submission():
    """A controller acceptance requires a pending expert command."""
    from robustlearn.scripted_demo_collection import ExpertFrameBuffer

    buffer = ExpertFrameBuffer()

    with pytest.raises(ValueError, match="pending"):
        buffer.accept_pending_command()
