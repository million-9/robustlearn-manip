"""Tests for safe persistence of approved teleop demonstrations."""

import pytest

from robustlearn.episode_manager import (
    EpisodeFailureCategory,
    EpisodeResult,
)
from robustlearn.rollout_dataset import RolloutFrame, read_episode
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    persist_accepted_teleop_episode,
)


def make_frame(*, target=0.11):
    return RolloutFrame(
        frame_index=0,
        timestamp_s=11.0,
        joint_position=(0.1,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        expert_action=(target,) * 7,
        task_success=False,
        task_failure=False,
        terminated=False,
    )


def make_result(*, success=True):
    return EpisodeResult(
        episode_index=7,
        seed=42,
        scenario="week9_case_08",
        start_time_s=10.0,
        end_time_s=12.0,
        success=success,
        failure_category=(
            None if success else EpisodeFailureCategory.TASK
        ),
        failure_reason=(
            None if success else "operator marked task unsuccessful"
        ),
    )


def make_session(*, success=True, target=0.11):
    session = TeleopRecordingSession()
    session.record_frame(make_frame(target=target))
    session.accept(success=success)
    return session


def persist(session, tmp_path, *, success=True):
    return persist_accepted_teleop_episode(
        session,
        make_result(success=success),
        output_root=tmp_path,
        git_commit="test-commit",
        config_id="test-teleop-config",
        task="peg_insertion",
    )


def test_accepted_success_uses_canonical_dataset_schema(tmp_path):
    session = make_session()

    path = persist(session, tmp_path)
    episode = read_episode(path)

    assert episode.source == "teleop"
    assert episode.task == "peg_insertion"
    assert episode.episode_index == 7
    assert episode.seed == 42
    assert episode.scenario == "week9_case_08"
    assert episode.success
    assert episode.failure_reason is None
    assert episode.frames == session.export_accepted_frames()
    assert episode.frames[-1].task_success
    assert episode.frames[-1].terminated


def test_accepted_failed_task_preserves_failure_metadata(tmp_path):
    session = make_session(success=False)

    path = persist(session, tmp_path, success=False)
    episode = read_episode(path)

    assert episode.source == "teleop"
    assert not episode.success
    assert episode.failure_reason == (
        "operator marked task unsuccessful"
    )
    assert episode.frames[-1].task_failure
    assert episode.frames[-1].terminated


@pytest.mark.parametrize("decision", ("reject", "abort"))
def test_unaccepted_recordings_never_enter_dataset(
    tmp_path, decision
):
    session = TeleopRecordingSession()
    session.record_frame(make_frame())

    getattr(session, decision)()

    with pytest.raises(ValueError):
        persist(session, tmp_path)

    assert not list(tmp_path.rglob("*.json"))


def test_operator_and_episode_outcomes_must_agree(tmp_path):
    session = make_session(success=True)

    with pytest.raises(ValueError):
        persist(session, tmp_path, success=False)

    assert not list(tmp_path.rglob("*.json"))


def test_conflicting_dataset_is_never_overwritten(tmp_path):
    first = make_session(target=0.11)
    path = persist(first, tmp_path)

    original_bytes = path.read_bytes()

    second = make_session(target=0.12)

    with pytest.raises(FileExistsError):
        persist(second, tmp_path)

    assert path.read_bytes() == original_bytes
