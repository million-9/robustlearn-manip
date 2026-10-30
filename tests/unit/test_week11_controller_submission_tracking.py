"""Test ROS2 submission tracking without running a controller."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.teleop_demo_collection import (
    TeleopControllerCompletionTracker,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)


class FakeGoal:
    def __init__(self):
        self.trajectory = SimpleNamespace(
            joint_names=[],
            points=[],
        )


class FakePoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = None


class FakeDuration:
    def __init__(self, *, seconds):
        self.seconds = seconds

    def to_msg(self):
        return self.seconds


class FakeFuture:
    def add_done_callback(self, callback):
        self.callback = callback


def load_send_target():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    method = next(
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_send_target"
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
        "FollowJointTrajectory": SimpleNamespace(
            Goal=FakeGoal,
        ),
        "JointTrajectoryPoint": FakePoint,
        "Duration": FakeDuration,
        "PANDA_JOINT_NAMES": tuple(
            f"panda_joint{i}" for i in range(1, 8)
        ),
    }

    exec(compile(module, str(SCRIPT), "exec"), namespace)

    return namespace["_send_target"]


def make_node(*, ready=True, stage=True, raise_on_send=False):
    tracker = TeleopControllerCompletionTracker()
    observed = []
    rejected = []
    errors = []

    node = SimpleNamespace(
        _controller_completion_tracker=tracker,
        _request_sequence=0,
        _active_request_id=None,
        _goal_pending=False,
        _trajectory_duration_s=0.1,
        _stage_capture_for_target=lambda target: stage,
        _resolve_pending_capture=lambda **kwargs: (
            rejected.append(kwargs)
        ),
        get_logger=lambda: SimpleNamespace(
            warning=lambda message: None,
            error=lambda message: errors.append(message),
        ),
    )

    def send_goal(goal):
        observed.append(
            (
                node._active_request_id,
                tracker.outstanding_request_ids,
                tuple(goal.trajectory.joint_names),
            )
        )

        if raise_on_send:
            raise RuntimeError("submission failed")

        return FakeFuture()

    node._trajectory_client = SimpleNamespace(
        server_is_ready=lambda: ready,
        send_goal_async=send_goal,
    )

    node._send_target = MethodType(
        load_send_target(), node
    )

    return node, tracker, observed, rejected, errors


def test_request_is_tracked_before_ros_submission():
    node, tracker, observed, _, _ = make_node()

    node._send_target([0.0] * 7)

    assert observed[0][0] == 1
    assert observed[0][1] == (1,)
    assert len(observed[0][2]) == 7
    assert node._goal_pending
    assert not tracker.settled


def test_failed_submission_blocks_review():
    node, tracker, _, rejected, errors = make_node(
        raise_on_send=True
    )

    node._send_target([0.0] * 7)

    assert tracker.uncertain
    assert not tracker.settled
    assert not node._goal_pending
    assert node._active_request_id is None
    assert rejected == [{"accepted": False}]
    assert errors


def test_unavailable_server_does_not_register_request():
    node, tracker, observed, _, _ = make_node(
        ready=False
    )

    node._send_target([0.0] * 7)

    assert observed == []
    assert tracker.settled
    assert node._request_sequence == 0


def test_capture_failure_does_not_submit_request():
    node, tracker, observed, _, _ = make_node(
        stage=False
    )

    node._send_target([0.0] * 7)

    assert observed == []
    assert tracker.settled
    assert node._request_sequence == 0
