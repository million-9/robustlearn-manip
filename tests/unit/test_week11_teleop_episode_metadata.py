"""Validate explicit metadata for recorded teleoperation episodes."""

import pytest

from robustlearn.episode_manager import EpisodeFailureCategory
from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    build_teleop_episode_result,
)


def frame(index, timestamp):
    return RolloutFrame(
        frame_index=index,
        timestamp_s=timestamp,
        joint_position=(0.1,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        expert_action=(0.11,) * 7,
        task_success=False,
        task_failure=False,
        terminated=False,
    )


def recording():
    session = TeleopRecordingSession()
    session.record_frame(frame(0, 10.0))
    session.record_frame(frame(1, 10.25))
    return session


def build(session, **overrides):
    arguments = {
        "episode_index": 7,
        "seed": 42,
        "scenario": "nominal",
        "success": True,
    }
    arguments.update(overrides)

    return build_teleop_episode_result(
        session,
        **arguments,
    )


def test_success_metadata_uses_actual_frame_timestamps():
    session = recording()

    result = build(session)

    assert result.episode_index == 7
    assert result.seed == 42
    assert result.scenario == "nominal"
    assert result.start_time_s == 10.0
    assert result.end_time_s == 10.25
    assert result.duration_s == pytest.approx(0.25)
    assert result.success is True
    assert result.failure_category is None
    assert result.failure_reason is None

    # Metadata construction must not finalize the recording.
    assert session.status == "recording"


def test_failure_metadata_retains_operator_reason():
    result = build(
        recording(),
        success=False,
        failure_reason="Insertion depth not achieved",
    )

    assert result.success is False
    assert result.failure_category == EpisodeFailureCategory.TASK
    assert result.failure_reason == (
        "Insertion depth not achieved"
    )


@pytest.mark.parametrize(
    "reason",
    [None, "", "   "],
)
def test_failed_episode_requires_reason(reason):
    session = recording()

    with pytest.raises(ValueError):
        build(
            session,
            success=False,
            failure_reason=reason,
        )

    assert session.status == "recording"


def test_success_cannot_have_failure_reason():
    with pytest.raises(ValueError):
        build(
            recording(),
            failure_reason="unexpected reason",
        )


def test_empty_recording_cannot_generate_metadata():
    with pytest.raises(ValueError):
        build(TeleopRecordingSession())


def test_pending_controller_goal_blocks_metadata():
    session = recording()
    session.stage_frame(frame(2, 10.50))

    with pytest.raises(ValueError):
        build(session)

    assert session.has_pending_goal


@pytest.mark.parametrize(
    "finalize",
    [
        lambda session: session.reject(),
        lambda session: session.abort(),
        lambda session: session.accept(success=True),
    ],
)
def test_finalized_recording_cannot_prepare_new_metadata(finalize):
    session = recording()
    finalize(session)

    with pytest.raises(ValueError):
        build(session)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("episode_index", -1),
        ("episode_index", True),
        ("episode_index", "7"),
        ("seed", -1),
        ("seed", True),
        ("seed", "42"),
        ("scenario", ""),
        ("scenario", "   "),
        ("scenario", None),
    ],
)
def test_invalid_identity_metadata_is_rejected(field, invalid):
    with pytest.raises(ValueError):
        build(recording(), **{field: invalid})


def test_success_must_be_boolean():
    with pytest.raises(TypeError):
        build(recording(), success=1)


def test_failure_does_not_change_recording_status():
    session = recording()

    result = build(
        session,
        success=False,
        failure_reason="Task incomplete",
    )

    assert result.failure_reason == "Task incomplete"
    assert session.status == "recording"
    assert len(session.frames) == 2
