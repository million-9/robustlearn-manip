"""Tests for controller-accepted teleoperation frame recording."""

import pytest

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import TeleopRecordingSession


def frame(index=0):
    return RolloutFrame(
        frame_index=index,
        timestamp_s=10.0 + index,
        joint_position=(0.1,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        expert_action=(0.11,) * 7,
        task_success=False,
        task_failure=False,
        terminated=False,
    )


def test_frame_is_not_recorded_before_controller_acceptance():
    session = TeleopRecordingSession()

    session.stage_frame(frame())

    assert session.status == "recording"
    assert session.frames == ()

    session.accept_pending_frame()

    assert session.frames == (frame(),)


def test_rejected_controller_goal_does_not_add_frame():
    session = TeleopRecordingSession()

    session.stage_frame(frame())
    session.reject_pending_frame()

    assert session.frames == ()

    session.stage_frame(frame())
    session.accept_pending_frame()

    assert len(session.frames) == 1


def test_duplicate_pending_goals_are_rejected():
    session = TeleopRecordingSession()

    session.stage_frame(frame())

    with pytest.raises(ValueError):
        session.stage_frame(frame())

    with pytest.raises(ValueError):
        session.record_frame(frame())

    assert session.frames == ()


def test_operator_cannot_approve_pending_goal():
    session = TeleopRecordingSession()

    session.stage_frame(frame())

    with pytest.raises(ValueError):
        session.accept(success=True)

    assert session.status == "recording"
    assert session.frames == ()


def test_abort_discards_pending_controller_goal():
    session = TeleopRecordingSession()

    session.stage_frame(frame())
    session.abort()

    assert session.status == "aborted"
    assert session.frames == ()

    with pytest.raises(ValueError):
        session.accept_pending_frame()


def test_rejected_recording_cannot_commit_pending_goal():
    session = TeleopRecordingSession()

    session.stage_frame(frame())
    session.reject()

    assert session.status == "rejected"

    with pytest.raises(ValueError):
        session.accept_pending_frame()
