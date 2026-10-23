"""Week 11: scripted and teleop episodes share one dataset schema."""

import json
from dataclasses import replace

from robustlearn.episode_manager import EpisodeResult
from robustlearn.rollout_dataset import (
    RolloutFrame,
    episode_to_dict,
    read_episode,
    rollout_episode_from_result,
    write_episode,
)
from robustlearn.teleop_collection_summary import (
    summarize_teleop_attempt,
    teleop_collection_summary_path,
)
from robustlearn.teleop_demo_collection import (
    TeleopCollectionConfig,
    TeleopRecordingSession,
    apply_teleop_operator_decision,
    build_teleop_episode_result,
)
from robustlearn.teleop_summary_history import (
    update_teleop_summary_history,
)


def frame(timestamp, *, success=False):
    return RolloutFrame(
        frame_index=0,
        timestamp_s=timestamp,
        joint_position=(0.1,) * 7,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        expert_action=(0.11,) * 7,
        task_success=success,
        task_failure=False,
        terminated=success,
    )


def configuration(root, index, seed, reason=None):
    return TeleopCollectionConfig(
        output_root=root,
        episode_index=index,
        seed=seed,
        scenario="nominal",
        git_commit="acceptance-test",
        config_id="teleop-test",
        task="panda_insertion",
        failure_reason=reason,
    )


def test_scripted_and_teleop_share_canonical_dataset(tmp_path):
    root = tmp_path / "episodes"

    scripted_result = EpisodeResult(
        episode_index=20,
        seed=7,
        scenario="nominal",
        start_time_s=30.0,
        end_time_s=30.0,
        success=True,
        failure_category=None,
        failure_reason=None,
    )

    scripted = rollout_episode_from_result(
        scripted_result,
        source="scripted",
        task="panda_insertion",
        git_commit="acceptance-test",
        config_id="scripted-test",
        frames=(frame(30.0, success=True),),
    )

    scripted_path = write_episode(root, scripted)

    session = TeleopRecordingSession()
    session.record_frame(frame(40.0))

    teleop_result = build_teleop_episode_result(
        session,
        episode_index=21,
        seed=8,
        scenario="nominal",
        success=False,
        failure_reason="Insertion incomplete",
    )

    teleop_path = apply_teleop_operator_decision(
        session,
        "accept_failure",
        result=teleop_result,
        output_root=root,
        git_commit="acceptance-test",
        config_id="teleop-test",
        task="panda_insertion",
    )

    scripted_loaded = read_episode(scripted_path)
    teleop_loaded = read_episode(teleop_path)

    assert scripted_loaded.source == "scripted"
    assert teleop_loaded.source == "teleop"
    assert scripted_loaded.success
    assert not teleop_loaded.success
    assert teleop_loaded.failure_reason == "Insertion incomplete"
    assert teleop_loaded.frames[-1].task_failure
    assert teleop_loaded.frames[-1].terminated

    scripted_json = episode_to_dict(scripted_loaded)
    teleop_json = episode_to_dict(teleop_loaded)

    assert set(scripted_json) == set(teleop_json)

    assert set(scripted_json["frames"][0]) == set(
        teleop_json["frames"][0]
    )

    assert (
        scripted_loaded.schema_version
        == teleop_loaded.schema_version
    )

    assert len(list(root.glob("*.json"))) == 2

    config = configuration(
        root,
        21,
        8,
        reason="Insertion incomplete",
    )

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="accepted_failure",
        episode_path=teleop_path,
    )

    update_teleop_summary_history(
        replace(attempt, attempt_id="coexistence-001"),
        output_root=root,
    )

    summary_path = teleop_collection_summary_path(root)
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["saved_episodes"] == 1
    assert summary["total_attempts"] == 1
    assert summary_path.parent == root.parent
    assert summary_path not in list(root.glob("*.json"))
    assert len(list(root.glob("*.json"))) == 2


def test_aborted_recording_does_not_contaminate_dataset(tmp_path):
    root = tmp_path / "episodes"

    session = TeleopRecordingSession()
    session.record_frame(frame(10.0))
    session.abort()

    config = configuration(root, 22, 9)

    attempt = summarize_teleop_attempt(
        config=config,
        session=session,
        outcome="aborted",
    )

    update_teleop_summary_history(
        replace(attempt, attempt_id="aborted-001"),
        output_root=root,
    )

    assert not list(root.glob("*.json"))

    summary_path = teleop_collection_summary_path(root)
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["saved_episodes"] == 0
    assert summary["outcome_counts"]["aborted"] == 1
