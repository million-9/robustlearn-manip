"""Tests for operator-controlled teleoperation recording."""

import pytest

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import TeleopRecordingSession


def make_frame(index=0):
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


def test_accepted_successful_recording():
    session = TeleopRecordingSession()

    session.record_frame(make_frame())
    session.accept(success=True)

    assert session.status == "accepted"

    frames = session.export_accepted_frames()

    assert len(frames) == 1
    assert frames[-1].terminated
    assert frames[-1].task_success
    assert not frames[-1].task_failure


def test_accepted_failed_task_preserves_failure_label():
    session = TeleopRecordingSession()

    session.record_frame(make_frame())
    session.accept(success=False)

    frames = session.export_accepted_frames()

    assert session.status == "accepted"
    assert frames[-1].terminated
    assert frames[-1].task_failure
    assert not frames[-1].task_success


def test_rejected_demonstration_cannot_be_exported():
    session = TeleopRecordingSession()

    session.record_frame(make_frame())
    session.reject()

    assert session.status == "rejected"
    assert len(session.frames) == 1

    with pytest.raises(ValueError):
        session.export_accepted_frames()


def test_aborted_demonstration_cannot_be_exported():
    session = TeleopRecordingSession()

    session.record_frame(make_frame())
    session.abort()

    assert session.status == "aborted"
    assert len(session.frames) == 1

    with pytest.raises(ValueError):
        session.export_accepted_frames()

    with pytest.raises(ValueError):
        session.accept(success=True)


def test_empty_recording_cannot_be_accepted():
    session = TeleopRecordingSession()

    with pytest.raises(ValueError):
        session.accept(success=True)

    assert session.status == "recording"


def test_finalized_recording_rejects_new_frames():
    session = TeleopRecordingSession()

    session.record_frame(make_frame())
    session.accept(success=True)

    with pytest.raises(ValueError):
        session.record_frame(make_frame(1))

    with pytest.raises(ValueError):
        session.reject()
