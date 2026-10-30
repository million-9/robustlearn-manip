"""Exercise actual ROS2 approval and collection-summary methods."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from robustlearn.rollout_dataset import RolloutFrame, read_episode
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

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)


def load_methods():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    cls = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "KeyboardTeleop"
    )
    names = {
        "_approve_reviewed_recording",
        "_record_collection_outcome",
        "_retry_pending_summary",
        "_abort",
    }
    selected = [
        item for item in cls.body
        if isinstance(item, ast.FunctionDef)
        and item.name in names
    ]
    assert len(selected) == 4

    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *selected,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "Path": Path,
        "rclpy": SimpleNamespace(shutdown=lambda: None),
        "replace": replace,
        "summarize_teleop_attempt": summarize_teleop_attempt,
        "update_teleop_summary_history": (
            update_teleop_summary_history
        ),
        "build_teleop_episode_result": (
            build_teleop_episode_result
        ),
        "apply_teleop_operator_decision": (
            apply_teleop_operator_decision
        ),
    }
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def make_node(tmp_path, *, reason=None):
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
    logs = []

    node = SimpleNamespace(
        _teleop_session=session,
        _teleop_collection_config=TeleopCollectionConfig(
            output_root=tmp_path / "episodes",
            episode_index=12,
            seed=42,
            scenario="nominal",
            git_commit="test-commit",
            config_id="keyboard-teleop-test",
            task="panda_insertion",
            failure_reason=reason,
        ),
        _teleop_attempt_id="test-attempt-001",
        _teleop_pending_approval=None,
        _review_mode=True,
        _aborted=False,
        _stop=lambda: None,
        _review_ready_for_decision=lambda **kwargs: True,
        get_logger=lambda: SimpleNamespace(
            info=lambda msg: logs.append(("info", msg)),
            error=lambda msg: logs.append(("error", msg)),
            warning=lambda msg: logs.append(("warning", msg)),
        ),
    )

    namespace = load_methods()
    for name in (
        "_approve_reviewed_recording",
        "_record_collection_outcome",
        "_retry_pending_summary",
        "_abort",
    ):
        setattr(node, name, MethodType(namespace[name], node))

    return node, logs


def summary_data(tmp_path):
    path = teleop_collection_summary_path(
        tmp_path / "episodes"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_updates_dataset_and_summary(tmp_path):
    node, logs = make_node(tmp_path)

    node._approve_reviewed_recording("accept_success")

    files = list((tmp_path / "episodes").glob("*.json"))
    assert len(files) == 1
    assert read_episode(files[0]).success

    report = summary_data(tmp_path)
    assert report["total_attempts"] == 1
    assert report["saved_episodes"] == 1
    assert report["attempts"][0]["attempt_id"] == (
        "test-attempt-001"
    )
    assert report["outcome_counts"]["accepted_success"] == 1
    assert any(level == "info" for level, _ in logs)


def test_failed_demonstration_summary(tmp_path):
    node, _ = make_node(
        tmp_path, reason="Insertion incomplete"
    )

    node._approve_reviewed_recording("accept_failure")

    report = summary_data(tmp_path)
    assert report["saved_episodes"] == 1
    assert report["outcome_counts"]["accepted_failure"] == 1
    assert report["attempts"][0]["failure_reason"] == (
        "Insertion incomplete"
    )


def test_failed_write_then_retry_updates_same_attempt(
    tmp_path, monkeypatch
):
    import errno
    import os

    import robustlearn.teleop_demo_collection as collection

    node, _ = make_node(tmp_path)
    original = os.link
    calls = []

    def fail_once(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise OSError(errno.EACCES, "transient write failure")
        return original(src, dst)

    monkeypatch.setattr(collection.os, "link", fail_once)

    node._approve_reviewed_recording("accept_success")

    report = summary_data(tmp_path)
    assert report["saved_episodes"] == 0
    assert report["outcome_counts"]["persistence_failed"] == 1
    assert node._teleop_pending_approval is not None

    node._approve_reviewed_recording("accept_success")

    report = summary_data(tmp_path)
    assert report["total_attempts"] == 1
    assert report["saved_episodes"] == 1
    assert report["outcome_counts"]["persistence_failed"] == 0
    assert node._teleop_pending_approval is None


def test_saved_episode_not_replayed(tmp_path):
    node, logs = make_node(tmp_path)
    node._approve_reviewed_recording("accept_success")

    node._approve_reviewed_recording("accept_success")

    assert summary_data(tmp_path)["total_attempts"] == 1
    assert any(
        level == "error" and "Retry denied" in msg
        for level, msg in logs
    )


def test_missing_attempt_identity_fails_summary_only(tmp_path):
    node, logs = make_node(tmp_path)
    node._teleop_attempt_id = None

    node._approve_reviewed_recording("accept_success")

    files = list((tmp_path / "episodes").glob("*.json"))
    assert len(files) == 1
    assert not teleop_collection_summary_path(
        tmp_path / "episodes"
    ).exists()
    assert node._teleop_pending_summary is not None
    assert any(
        level == "error"
        and "collection summary update failed" in msg
        for level, msg in logs
    )


def test_summary_method_rejects_unconfigured_node(tmp_path):
    node, _ = make_node(tmp_path)
    node._teleop_collection_config = None

    with pytest.raises(ValueError):
        node._record_collection_outcome("rejected")



def test_abort_reports_without_exporting(tmp_path):
    node, _ = make_node(tmp_path)
    node._abort()
    assert node._aborted
    assert node._teleop_session.status == "aborted"
    report = summary_data(tmp_path)
    assert report["outcome_counts"]["aborted"] == 1
    assert report["saved_episodes"] == 0
    assert not list((tmp_path / "episodes").glob("*.json"))


def test_summary_only_retry_does_not_duplicate_episode(tmp_path):
    node, _ = make_node(tmp_path)
    real_reporter = node._record_collection_outcome

    def fail_once(*args, **kwargs):
        node._record_collection_outcome = real_reporter
        raise OSError("temporary summary failure")

    node._record_collection_outcome = fail_once
    node._approve_reviewed_recording("accept_success")

    files = list((tmp_path / "episodes").glob("*.json"))
    assert len(files) == 1
    assert node._teleop_pending_summary is not None

    node._retry_pending_summary()

    assert node._teleop_pending_summary is None
    assert summary_data(tmp_path)["saved_episodes"] == 1
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 1

    node._retry_pending_summary()
    assert summary_data(tmp_path)["total_attempts"] == 1


def test_abort_does_not_reclassify_saved_episode(tmp_path):
    node, _ = make_node(tmp_path)
    node._approve_reviewed_recording("accept_success")
    node._abort()
    report = summary_data(tmp_path)
    assert report["outcome_counts"]["accepted_success"] == 1
    assert report["outcome_counts"]["aborted"] == 0
    assert node._teleop_session.status == "accepted"
