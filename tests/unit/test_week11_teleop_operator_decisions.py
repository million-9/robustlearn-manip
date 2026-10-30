"""Tests for explicit manual demonstration decisions."""

import pytest

from robustlearn.episode_manager import (
    EpisodeFailureCategory,
    EpisodeResult,
)
from robustlearn.rollout_dataset import RolloutFrame, read_episode
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    apply_teleop_operator_decision,
)


def make_session():
    session = TeleopRecordingSession()

    session.record_frame(
        RolloutFrame(
            frame_index=0,
            timestamp_s=2.0,
            joint_position=(0.1,) * 7,
            joint_velocity=(0.0,) * 7,
            task_relative_state=(0.01, 0.0, 0.02),
            wrist_wrench=(0.0,) * 6,
            expert_action=(0.11,) * 7,
            task_success=False,
            task_failure=False,
            terminated=False,
        )
    )

    return session


def make_result(success):
    return EpisodeResult(
        episode_index=3,
        seed=17,
        scenario="nominal",
        start_time_s=1.0,
        end_time_s=3.0,
        success=success,
        failure_category=(
            None if success else EpisodeFailureCategory.TASK
        ),
        failure_reason=(
            None if success else "insertion not completed"
        ),
    )


def decide(session, decision, tmp_path, result=None):
    return apply_teleop_operator_decision(
        session,
        decision,
        result=result,
        output_root=tmp_path,
        git_commit="test-commit",
        config_id="teleop-test-config",
        task="panda_insertion",
    )


def test_accepted_success_is_persisted(tmp_path):
    session = make_session()

    path = decide(
        session,
        "accept_success",
        tmp_path,
        make_result(True),
    )

    episode = read_episode(path)

    assert session.status == "accepted"
    assert episode.source == "teleop"
    assert episode.success is True
    assert episode.episode_index == 3
    assert episode.seed == 17
    assert episode.scenario == "nominal"
    assert len(episode.frames) == 1
    assert episode.frames[-1].task_success
    assert episode.frames[-1].terminated


def test_accepted_failure_preserves_failure_reason(tmp_path):
    session = make_session()

    path = decide(
        session,
        "accept_failure",
        tmp_path,
        make_result(False),
    )

    episode = read_episode(path)

    assert episode.source == "teleop"
    assert episode.success is False
    assert episode.failure_reason == "insertion not completed"
    assert episode.frames[-1].task_failure
    assert episode.frames[-1].terminated


@pytest.mark.parametrize("decision", ["reject", "abort"])
def test_discard_decisions_never_write_dataset(
    tmp_path, decision
):
    session = make_session()

    path = decide(session, decision, tmp_path)

    assert path is None
    assert session.status == (
        "rejected" if decision == "reject" else "aborted"
    )
    assert list(tmp_path.rglob("*.json")) == []


def test_unknown_decision_does_not_change_session(tmp_path):
    session = make_session()

    with pytest.raises(ValueError):
        decide(session, "approve_automatically", tmp_path)

    assert session.status == "recording"
    assert list(tmp_path.rglob("*.json")) == []


def test_acceptance_requires_episode_metadata(tmp_path):
    session = make_session()

    with pytest.raises(ValueError):
        decide(session, "accept_success", tmp_path)

    assert session.status == "recording"
    assert list(tmp_path.rglob("*.json")) == []


def test_outcome_mismatch_is_rejected_before_finalizing(tmp_path):
    session = make_session()

    with pytest.raises(ValueError):
        decide(
            session,
            "accept_success",
            tmp_path,
            make_result(False),
        )

    assert session.status == "recording"
    assert list(tmp_path.rglob("*.json")) == []


def test_pending_controller_goal_blocks_acceptance(tmp_path):
    session = make_session()

    session.stage_frame(
        RolloutFrame(
            frame_index=1,
            timestamp_s=2.5,
            joint_position=(0.1,) * 7,
            joint_velocity=None,
            task_relative_state=(0.01, 0.0, 0.02),
            wrist_wrench=(0.0,) * 6,
            expert_action=(0.11,) * 7,
            task_success=False,
            task_failure=False,
            terminated=False,
        )
    )

    with pytest.raises(ValueError):
        decide(
            session,
            "accept_success",
            tmp_path,
            make_result(True),
        )

    assert session.status == "recording"
    assert session.has_pending_goal
    assert list(tmp_path.rglob("*.json")) == []


def test_accepted_episode_is_never_overwritten(tmp_path):
    session = make_session()
    result = make_result(True)

    path = decide(
        session, "accept_success", tmp_path, result
    )
    original_content = path.read_bytes()

    with pytest.raises(FileExistsError):
        decide(
            session, "accept_success", tmp_path, result
        )

    assert path.read_bytes() == original_content
    assert session.status == "accepted"



def test_existing_path_preflight_preserves_recording(tmp_path):
    session = make_session()
    result = make_result(True)

    path = (
        tmp_path
        / "episode_000003__seed_17__nominal.json"
    )
    path.write_text("existing episode", encoding="utf-8")

    with pytest.raises(FileExistsError):
        decide(
            session,
            "accept_success",
            tmp_path,
            result,
        )

    assert session.status == "recording"
    assert path.read_text(encoding="utf-8") == (
        "existing episode"
    )


def test_invalid_episode_time_does_not_finalize_recording(tmp_path):
    session = make_session()

    result = EpisodeResult(
        episode_index=3,
        seed=17,
        scenario="nominal",
        start_time_s=2.5,
        end_time_s=3.0,
        success=True,
        failure_category=None,
        failure_reason=None,
    )

    # The recorded frame occurs at timestamp 2.0.
    with pytest.raises(ValueError):
        decide(
            session,
            "accept_success",
            tmp_path,
            result,
        )

    assert session.status == "recording"
    assert list(tmp_path.glob("*.json")) == []


def test_transient_write_error_can_retry_without_overwrite(
    tmp_path,
    monkeypatch,
):
    import errno
    import os

    import robustlearn.teleop_demo_collection as collection

    session = make_session()
    result = make_result(True)

    real_link = os.link
    attempts = []

    def fail_once(source, destination):
        attempts.append((source, destination))

        if len(attempts) == 1:
            raise OSError(
                errno.EACCES,
                "simulated transient write failure",
            )

        return real_link(source, destination)

    monkeypatch.setattr(collection.os, "link", fail_once)

    with pytest.raises(OSError):
        decide(
            session,
            "accept_success",
            tmp_path,
            result,
        )

    # Persistence failed after approval. Preserve the accepted
    # recording for a controlled retry with the same metadata.
    assert session.status == "accepted"
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []

    path = decide(
        session,
        "accept_success",
        tmp_path,
        result,
    )

    assert path.exists()
    assert path.read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("*.json"))) == 1

    # A subsequent attempt must never overwrite the episode.
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        decide(
            session,
            "accept_success",
            tmp_path,
            result,
        )

    assert path.read_bytes() == original
