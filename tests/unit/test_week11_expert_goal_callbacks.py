"""Isolated tests for the ROS2 expert goal-response callback."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from robustlearn.scripted_demo_collection import ExpertFrameBuffer

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws"
    / "src"
    / "robustlearn_control"
    / "scripts"
    / "local_insertion_expert.py"
)


def load_goal_response_callback():
    """Load the real callback without importing the ROS2 runtime."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT_PATH))

    node_class = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "LocalInsertionExpertNode"
    )

    method = next(
        item
        for item in node_class.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_goal_response_callback"
    )

    module = ast.fix_missing_locations(
        ast.Module(body=[method], type_ignores=[])
    )

    namespace = {}
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)

    return namespace["_goal_response_callback"]


def make_pending_buffer():
    buffer = ExpertFrameBuffer()

    buffer.stage_submitted_command(
        timestamp_s=100.0,
        joint_position=(0.0,) * 7,
        joint_velocity=None,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        submitted_joint_target=(0.1,) * 7,
    )

    return buffer


def make_fake_node(buffer, *, terminal=False):
    failures = []
    registered_callbacks = []

    node = SimpleNamespace(
        _terminal=terminal,
        _goal_in_flight=True,
        _goal_handle=None,
        _expert_frame_buffer=buffer,
        _abort=lambda reason: failures.append(reason),
        _goal_result_callback=lambda future: None,
    )

    return node, failures, registered_callbacks


def test_accepted_goal_commits_pending_expert_frame():
    callback = load_goal_response_callback()
    buffer = make_pending_buffer()
    node, failures, callbacks = make_fake_node(buffer)

    result_future = SimpleNamespace(
        add_done_callback=lambda fn: callbacks.append(fn)
    )

    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
    )

    callback(
        node,
        SimpleNamespace(result=lambda: goal_handle),
    )

    assert len(buffer.frames) == 1
    assert buffer.frames[0].timestamp_s == 100.0
    assert buffer.frames[0].expert_action == (0.1,) * 7
    assert node._goal_handle is goal_handle
    assert len(callbacks) == 1
    assert failures == []


def test_rejected_goal_discards_pending_expert_frame():
    callback = load_goal_response_callback()
    buffer = make_pending_buffer()
    node, failures, _ = make_fake_node(buffer)

    callback(
        node,
        SimpleNamespace(
            result=lambda: SimpleNamespace(accepted=False)
        ),
    )

    assert buffer.frames == ()
    assert node._goal_in_flight is False
    assert len(failures) == 1
    assert "rejected" in failures[0]

    with pytest.raises(ValueError, match="pending"):
        buffer.accept_pending_command()


def test_failed_goal_request_discards_pending_frame():
    callback = load_goal_response_callback()
    buffer = make_pending_buffer()
    node, failures, _ = make_fake_node(buffer)

    def failed_request():
        raise RuntimeError("simulated action transport failure")

    callback(
        node,
        SimpleNamespace(result=failed_request),
    )

    assert buffer.frames == ()
    assert node._goal_in_flight is False
    assert len(failures) == 1
    assert "simulated action transport failure" in failures[0]

    with pytest.raises(ValueError, match="pending"):
        buffer.accept_pending_command()


def test_late_response_after_terminal_discards_pending_frame():
    callback = load_goal_response_callback()
    buffer = make_pending_buffer()

    node, failures, _ = make_fake_node(
        buffer,
        terminal=True,
    )

    callback(node, SimpleNamespace())

    assert node._goal_in_flight is False
    assert buffer.frames == ()
    assert failures == []

    with pytest.raises(ValueError, match="pending"):
        buffer.accept_pending_command()


def load_send_joint_target():
    """Load the real goal-submission method without starting ROS2."""
    import numpy as np

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT_PATH))

    node_class = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "LocalInsertionExpertNode"
    )

    method = next(
        item
        for item in node_class.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_send_joint_target"
    )

    module = ast.fix_missing_locations(
        ast.Module(body=[method], type_ignores=[])
    )

    class FakeGoal:
        def __init__(self):
            self.trajectory = SimpleNamespace(
                joint_names=[],
                points=[],
            )

    namespace = {
        "FloatArray": object,
        "np": np,
        "FollowJointTrajectory": SimpleNamespace(Goal=FakeGoal),
        "JointTrajectoryPoint": lambda: SimpleNamespace(
            positions=[],
            time_from_start=None,
        ),
        "Duration": lambda *, seconds: SimpleNamespace(
            to_msg=lambda: seconds,
        ),
        "ROS_PANDA_ARM_JOINT_NAMES": tuple(
            f"panda_joint_{index}"
            for index in range(1, 8)
        ),
    }

    exec(
        compile(module, str(SCRIPT_PATH), "exec"),
        namespace,
    )

    return namespace["_send_joint_target"]


def test_synchronous_goal_submission_failure_cleans_pending_state():
    """A failed send must leave no in-flight or pending command."""

    send_joint_target = load_send_joint_target()
    buffer = make_pending_buffer()

    def failed_send(goal):
        raise RuntimeError(
            "simulated synchronous goal submission failure"
        )

    client = SimpleNamespace(
        server_is_ready=lambda: True,
        send_goal_async=failed_send,
    )

    node = SimpleNamespace(
        _terminal=False,
        _goal_in_flight=False,
        _goal_started_ns=None,
        _trajectory_duration_s=0.1,
        _trajectory_client=client,
        _expert_frame_buffer=buffer,
        _goal_response_callback=lambda future: None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(
                nanoseconds=123456789,
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="simulated synchronous goal submission failure",
    ):
        send_joint_target(node, (0.1,) * 7)

    assert node._goal_in_flight is False
    assert node._goal_started_ns is None

    # The rejected submission must not enter the dataset.
    assert buffer.frames == ()

    # No pending observation-action pair may remain.
    with pytest.raises(ValueError, match="pending"):
        buffer.accept_pending_command()
