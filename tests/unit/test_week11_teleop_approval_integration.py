"""Verify reviewed teleop approval and canonical persistence."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.rollout_dataset import RolloutFrame, read_episode
from robustlearn.teleop_demo_collection import (
    TeleopCollectionConfig,
    TeleopRecordingSession,
    apply_teleop_operator_decision,
    build_teleop_episode_result,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)


def load_approval_method():
    tree = ast.parse(
        SCRIPT.read_text(encoding="utf-8")
    )

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    method = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_approve_reviewed_recording"
    )

    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                method,
            ],
            type_ignores=[],
        )
    )

    namespace = {
        "build_teleop_episode_result": (
            build_teleop_episode_result
        ),
        "apply_teleop_operator_decision": (
            apply_teleop_operator_decision
        ),
    }

    exec(
        compile(module, str(SCRIPT), "exec"),
        namespace,
    )

    return namespace["_approve_reviewed_recording"]


def make_node(tmp_path, *, failure_reason=None):
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
            failure_reason=failure_reason,
        ),
        _review_mode=True,
        _review_ready_for_decision=lambda **kwargs: True,
        _record_collection_outcome=lambda *args, **kwargs: None,
        get_logger=lambda: SimpleNamespace(
            info=lambda message: logs.append(("info", message)),
            warning=lambda message: logs.append(
                ("warning", message)
            ),
            error=lambda message: logs.append(
                ("error", message)
            ),
        ),
    )

    node._approve_reviewed_recording = MethodType(
        load_approval_method(),
        node,
    )

    return node, session, logs


def output_files(tmp_path):
    return list((tmp_path / "episodes").glob("*.json"))


def test_success_approval_writes_canonical_episode(tmp_path):
    node, session, logs = make_node(tmp_path)

    node._approve_reviewed_recording("accept_success")

    paths = output_files(tmp_path)

    assert len(paths) == 1

    episode = read_episode(paths[0])

    assert session.status == "accepted"
    assert node._review_mode
    assert episode.source == "teleop"
    assert episode.task == "panda_insertion"
    assert episode.episode_index == 12
    assert episode.seed == 42
    assert episode.scenario == "nominal"
    assert episode.start_timestamp_s == 10.0
    assert episode.end_timestamp_s == 10.0
    assert episode.success is True
    assert episode.failure_reason is None
    assert episode.frames[-1].task_success
    assert episode.frames[-1].terminated
    assert any(level == "info" for level, _ in logs)


def test_failure_approval_preserves_explicit_reason(tmp_path):
    node, session, _ = make_node(
        tmp_path,
        failure_reason="Operator observed incomplete insertion",
    )

    node._approve_reviewed_recording("accept_failure")

    paths = output_files(tmp_path)

    assert len(paths) == 1

    episode = read_episode(paths[0])

    assert session.status == "accepted"
    assert episode.source == "teleop"
    assert episode.success is False
    assert episode.failure_reason == (
        "Operator observed incomplete insertion"
    )
    assert episode.frames[-1].task_failure


def test_failure_without_reason_is_not_accepted(tmp_path):
    node, session, logs = make_node(tmp_path)

    node._approve_reviewed_recording("accept_failure")

    assert session.status == "recording"
    assert output_files(tmp_path) == []
    assert any(level == "error" for level, _ in logs)


def test_unsettled_controller_blocks_approval(tmp_path):
    node, session, _ = make_node(tmp_path)

    node._review_ready_for_decision = lambda: False
    node._approve_reviewed_recording("accept_success")

    assert session.status == "recording"
    assert output_files(tmp_path) == []


def test_missing_collection_configuration_blocks_approval(tmp_path):
    node, session, _ = make_node(tmp_path)

    node._teleop_collection_config = None
    node._approve_reviewed_recording("accept_success")

    assert session.status == "recording"
    assert output_files(tmp_path) == []


def test_collision_never_overwrites_existing_episode(tmp_path):
    node, session, logs = make_node(tmp_path)

    directory = tmp_path / "episodes"
    directory.mkdir()

    path = directory / "episode_000012__seed_42__nominal.json"
    path.write_text("existing data", encoding="utf-8")

    node._approve_reviewed_recording("accept_success")

    assert path.read_text(encoding="utf-8") == "existing data"
    assert len(output_files(tmp_path)) == 1
    assert node._review_mode
    assert session.status == "recording"
    assert any(level == "error" for level, _ in logs)
    assert not any(level == "info" for level, _ in logs)


def test_invalid_approval_action_is_rejected(tmp_path):
    import pytest

    node, session, _ = make_node(tmp_path)

    with pytest.raises(ValueError):
        node._approve_reviewed_recording("accept_automatically")

    assert session.status == "recording"
    assert output_files(tmp_path) == []


def test_rejected_session_cannot_be_approved(tmp_path):
    node, session, logs = make_node(tmp_path)
    session.reject()

    node._approve_reviewed_recording("accept_success")

    assert session.status == "rejected"
    assert output_files(tmp_path) == []
    assert any(level == "error" for level, _ in logs)



def test_operator_retry_after_failed_write(tmp_path, monkeypatch):
    import errno
    import os

    import robustlearn.teleop_demo_collection as collection

    node, session, logs = make_node(tmp_path)

    original_link = os.link
    attempts = []

    def fail_once(source, destination):
        attempts.append(destination)
        if len(attempts) == 1:
            raise OSError(errno.EACCES, "simulated write failure")
        return original_link(source, destination)

    monkeypatch.setattr(collection.os, "link", fail_once)

    node._approve_reviewed_recording("accept_success")

    assert session.status == "accepted"
    assert node._review_mode
    assert node._teleop_pending_approval is not None
    assert output_files(tmp_path) == []
    assert not any(level == "info" for level, _ in logs)

    node._approve_reviewed_recording("accept_success")

    assert len(output_files(tmp_path)) == 1
    assert read_episode(output_files(tmp_path)[0]).success
    assert node._teleop_pending_approval is None
    assert any(level == "info" for level, _ in logs)


def test_retry_cannot_change_original_approval(tmp_path, monkeypatch):
    import errno
    import os

    import robustlearn.teleop_demo_collection as collection

    node, session, _ = make_node(tmp_path)
    original_link = os.link
    attempts = []

    def fail_once(source, destination):
        attempts.append(destination)
        if len(attempts) == 1:
            raise OSError(errno.EACCES, "temporary failure")
        return original_link(source, destination)

    monkeypatch.setattr(collection.os, "link", fail_once)

    node._approve_reviewed_recording("accept_success")

    assert session.status == "accepted"

    node._approve_reviewed_recording("accept_failure")

    assert output_files(tmp_path) == []
    assert node._teleop_pending_approval[0] == "accept_success"

    node._approve_reviewed_recording("accept_success")

    assert len(output_files(tmp_path)) == 1
    assert read_episode(output_files(tmp_path)[0]).success


def test_successful_save_cannot_be_replayed(tmp_path):
    node, session, logs = make_node(tmp_path)

    node._approve_reviewed_recording("accept_success")

    saved = output_files(tmp_path)
    assert len(saved) == 1
    original = saved[0].read_bytes()

    node._approve_reviewed_recording("accept_success")

    assert session.status == "accepted"
    assert saved[0].read_bytes() == original
    assert len(output_files(tmp_path)) == 1
    assert any(
        level == "error" and "Retry denied" in message
        for level, message in logs
    )


def test_failed_write_retry_still_requires_readiness(
    tmp_path, monkeypatch
):
    import errno
    import os

    import robustlearn.teleop_demo_collection as collection

    node, session, _ = make_node(tmp_path)
    original_link = os.link
    attempts = []

    def fail_once(source, destination):
        attempts.append(destination)
        if len(attempts) == 1:
            raise OSError(errno.EACCES, "temporary failure")
        return original_link(source, destination)

    monkeypatch.setattr(collection.os, "link", fail_once)

    node._approve_reviewed_recording("accept_success")

    assert session.status == "accepted"

    node._review_ready_for_decision = lambda **kwargs: False
    node._approve_reviewed_recording("accept_success")

    assert output_files(tmp_path) == []
    assert node._teleop_pending_approval is not None
