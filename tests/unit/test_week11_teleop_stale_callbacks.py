"""Regression tests for stale asynchronous teleoperation callbacks."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import TeleopRecordingSession

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

NAMES = (
    "_send_target",
    "_resolve_pending_capture",
    "_on_goal_response",
    "_on_goal_result",
    "_cancel_active_goal",
)

JOINTS = (0.1,) * 7


class PendingFuture:
    def __init__(self):
        self.callback = None
        self.outcome = None

    def add_done_callback(self, callback):
        self.callback = callback

    def resolve(self, outcome):
        assert self.callback is not None
        self.outcome = outcome
        self.callback(self)

    def result(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0
        self.result_future = PendingFuture()

    def cancel_goal_async(self):
        self.cancel_count += 1

    def get_result_async(self):
        return self.result_future


class FakeClient:
    def __init__(self):
        self.futures = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        future = PendingFuture()
        self.futures.append(future)
        return future


class FakeGoal:
    def __init__(self):
        self.trajectory = SimpleNamespace(
            joint_names=None,
            points=None,
        )


class FakePoint:
    def __init__(self):
        self.positions = None
        self.time_from_start = None


class FakeDuration:
    def __init__(self, *, seconds):
        self.seconds = seconds

    def to_msg(self):
        return self.seconds


def frame(index):
    return RolloutFrame(
        frame_index=index,
        timestamp_s=10.0 + index,
        joint_position=JOINTS,
        joint_velocity=(0.0,) * 7,
        task_relative_state=(0.01, 0.0, 0.02),
        wrist_wrench=(0.0,) * 6,
        expert_action=JOINTS,
        task_success=False,
        task_failure=False,
        terminated=False,
    )


def load_methods():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    methods = {
        node.name: node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in NAMES
    }

    assert set(methods) == set(NAMES)

    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *methods.values(),
            ],
            type_ignores=[],
        )
    )

    namespace = {
        "FollowJointTrajectory": SimpleNamespace(
            Goal=FakeGoal
        ),
        "JointTrajectoryPoint": FakePoint,
        "Duration": FakeDuration,
        "PANDA_JOINT_NAMES": tuple(
            f"panda_joint{i}" for i in range(1, 8)
        ),
    }

    exec(compile(module, str(SCRIPT), "exec"), namespace)

    return {
        name: namespace[name]
        for name in NAMES
    }


def make_node():
    session = TeleopRecordingSession()
    client = FakeClient()
    logs = []

    logger = SimpleNamespace(
        info=lambda message: logs.append(message),
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )

    node = SimpleNamespace(
        _teleop_session=session,
        _trajectory_client=client,
        _trajectory_duration_s=0.1,
        _goal_pending=False,
        _goal_handle=None,
        _aborted=False,
        get_logger=lambda: logger,
    )

    def stage_target(positions):
        session.stage_frame(frame(len(session.frames)))
        return True

    node._stage_capture_for_target = stage_target

    for name, method in load_methods().items():
        setattr(node, name, MethodType(method, node))

    return node, session, client


def submit_two_with_first_canceled():
    node, session, client = make_node()

    node._send_target(list(JOINTS))

    assert len(client.futures) == 1
    assert session.has_pending_goal

    first_future = client.futures[0]

    node._cancel_active_goal()

    assert not session.has_pending_goal

    node._send_target(list(JOINTS))

    assert len(client.futures) == 2
    assert session.has_pending_goal

    return node, session, client, first_future


@pytest.mark.parametrize(
    "old_response",
    [
        "accepted",
        "rejected",
        "error",
    ],
)
def test_stale_response_does_not_modify_new_command(old_response):
    node, session, client, old_future = (
        submit_two_with_first_canceled()
    )

    old_handle = FakeGoalHandle()

    if old_response == "accepted":
        old_future.resolve(old_handle)
        assert old_handle.cancel_count == 1
    elif old_response == "rejected":
        old_future.resolve(
            SimpleNamespace(accepted=False)
        )
    else:
        old_future.resolve(
            RuntimeError("old controller request failed")
        )

    # The newer command must remain untouched.
    assert node._goal_pending
    assert session.has_pending_goal
    assert session.frames == ()
    assert node._goal_handle is None

    # The correct response can now commit the new frame.
    new_handle = FakeGoalHandle()
    client.futures[1].resolve(new_handle)

    assert session.frames == (frame(0),)
    assert node._goal_handle is new_handle


def test_stale_trajectory_result_does_not_clear_new_goal():
    node, session, client = make_node()

    node._send_target(list(JOINTS))

    first_handle = FakeGoalHandle()
    client.futures[0].resolve(first_handle)

    assert session.frames == (frame(0),)

    node._cancel_active_goal()

    node._send_target(list(JOINTS))

    assert session.has_pending_goal
    assert node._goal_pending

    # Old trajectory finishes after the newer command is sent.
    first_handle.result_future.resolve(None)

    assert session.frames == (frame(0),)
    assert session.has_pending_goal
    assert node._goal_pending

    second_handle = FakeGoalHandle()
    client.futures[1].resolve(second_handle)

    assert session.frames == (
        frame(0),
        frame(1),
    )
    assert node._goal_handle is second_handle
