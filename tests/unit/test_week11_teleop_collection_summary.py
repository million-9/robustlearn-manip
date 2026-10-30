"""Verify summary accounting and dataset non-contamination."""

import json

import pytest

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_collection_summary import (
    build_teleop_collection_summary,
    summarize_teleop_attempt,
)
from robustlearn.teleop_demo_collection import (
    TeleopCollectionConfig,
    TeleopRecordingSession,
    apply_teleop_operator_decision,
    build_teleop_episode_result,
)


def configuration(tmp_path, *, failure_reason=None):
    return TeleopCollectionConfig(
        output_root=tmp_path / "episodes",
        episode_index=12,
        seed=42,
        scenario="nominal",
        git_commit="test-commit",
        config_id="keyboard-teleop-test",
        task="panda_insertion",
        failure_reason=failure_reason,
    )


def recording():
    session = TeleopRecordingSession()
    session.record_frame(
        RolloutFrame(
            frame_index=0,
            timestamp_s=10.0,
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


def saved_attempt(tmp_path, *, success):
    reason = None if success else "Insertion incomplete"
    config = configuration(tmp_path, failure_reason=reason)
    session = recording()

    result = build_teleop_episode_result(
        session,
        episode_index=config.episode_index,
        seed=config.seed,
        scenario=config.scenario,
        success=success,
        failure_reason=reason,
    )

    path = apply_teleop_operator_decision(
        session,
        "accept_success" if success else "accept_failure",
        result=result,
        output_root=config.output_root,
        git_commit=config.git_commit,
        config_id=config.config_id,
        task=config.task,
    )

    return config, session, path


def test_accepted_success_is_counted_only_after_save(tmp_path):
    config, session, path = saved_attempt(
        tmp_path, success=True
    )

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="accepted_success",
        episode_path=path,
    )

    assert attempt.episode_file == path.name
    assert attempt.frame_count == 1
    assert attempt.start_timestamp_s == 10.0
    assert attempt.outcome == "accepted_success"


def test_accepted_failure_retains_reason(tmp_path):
    config, session, path = saved_attempt(
        tmp_path, success=False
    )

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="accepted_failure",
        episode_path=path,
    )

    assert attempt.failure_reason == "Insertion incomplete"


def test_rejection_does_not_create_training_episode(tmp_path):
    config = configuration(tmp_path)
    session = recording()
    session.reject()

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="rejected",
    )

    assert attempt.episode_file is None
    assert not config.output_root.exists()


def test_abort_can_have_zero_frames(tmp_path):
    config = configuration(tmp_path)
    session = TeleopRecordingSession()
    session.abort()

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="aborted",
    )

    assert attempt.frame_count == 0
    assert attempt.start_timestamp_s is None
    assert attempt.end_timestamp_s is None


def test_failed_write_is_not_counted_as_saved(tmp_path):
    config = configuration(tmp_path)
    session = recording()
    session.accept(success=True)

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="persistence_failed",
        error="simulated disk write failure",
    )

    report = build_teleop_collection_summary([attempt])

    assert report["saved_episodes"] == 0
    assert report["outcome_counts"]["persistence_failed"] == 1


def test_unwritten_acceptance_is_not_saved(tmp_path):
    config = configuration(tmp_path)
    session = recording()
    session.accept(success=True)

    with pytest.raises(ValueError):
        summarize_teleop_attempt(
            config=config,
            session=session,
            outcome="accepted_success",
        )


def test_missing_saved_file_is_rejected(tmp_path):
    config = configuration(tmp_path)
    session = recording()
    session.accept(success=True)

    with pytest.raises(FileNotFoundError):
        summarize_teleop_attempt(
            config=config,
            session=session,
            outcome="accepted_success",
            episode_path=(
                config.output_root / "missing.json"
            ),
        )


def test_rejected_attempt_cannot_claim_dataset_file(tmp_path):
    config = configuration(tmp_path)
    session = recording()
    session.reject()

    with pytest.raises(ValueError):
        summarize_teleop_attempt(
            config=config,
            session=session,
            outcome="rejected",
            episode_path=config.output_root / "episode.json",
        )


def test_persistence_failure_requires_error(tmp_path):
    config = configuration(tmp_path)
    session = recording()

    with pytest.raises(ValueError):
        summarize_teleop_attempt(
            config=config,
            session=session,
            outcome="persistence_failed",
        )


def test_summary_counts_and_json_serialization(tmp_path):
    config1, session1, path1 = saved_attempt(
        tmp_path, success=True
    )

    accepted = summarize_teleop_attempt(
        config=config1,
        session=session1,
        outcome="accepted_success",
        episode_path=path1,
    )

    config2 = configuration(tmp_path)
    rejected_session = recording()
    rejected_session.reject()

    rejected = summarize_teleop_attempt(
        config=config2,
        session=rejected_session,
        outcome="rejected",
    )

    report = build_teleop_collection_summary(
        [accepted, rejected]
    )

    serialized = json.dumps(report, allow_nan=False)
    restored = json.loads(serialized)

    assert restored["schema_version"] == 1
    assert restored["source"] == "teleop"
    assert restored["total_attempts"] == 2
    assert restored["saved_episodes"] == 1
    assert restored["outcome_counts"]["accepted_success"] == 1
    assert restored["outcome_counts"]["rejected"] == 1
    assert len(restored["attempts"]) == 2


def test_empty_collection_has_zero_counts():
    report = build_teleop_collection_summary([])

    assert report["total_attempts"] == 0
    assert report["saved_episodes"] == 0
    assert all(
        count == 0
        for count in report["outcome_counts"].values()
    )


def test_invalid_aggregate_input_is_rejected():
    with pytest.raises(TypeError):
        build_teleop_collection_summary(
            [{"outcome": "accepted_success"}]
        )
