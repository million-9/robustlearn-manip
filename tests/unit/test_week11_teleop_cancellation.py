"""Regression tests for teleoperation cancellation races."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import TeleopRecordingSession

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

CALLBACK_NAMES = (
    "_cancel_active_goal",
    "_stop",
    "_resolve_pending_capture",
    "_on_goal_response",
)


def load_callbacks():
    """Load real ROS2 callback methods without importing ROS2."""
    tree = ast.parse(
        SCRIPT_PATH.read_text(encoding="utf-8")
    )

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    methods = {
        node.name: node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in CALLBACK_NAMES
    }

    assert set(methods) == set(CALLBACK_NAMES)

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

    namespace = {}

    exec(
        compile(module, str(SCRIPT_PATH), "exec"),
        namespace,
    )

    return {
        name: namespace[name]
        for name in CALLBACK_NAMES
    }


def make_frame():
    return RolloutFrame(
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


class FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.cancel_count = 0

    def cancel_goal_async(self):
        self.cancel_count += 1

    def get_result_async(self):
        return SimpleNamespace(
            add_done_callback=lambda callback: None
        )


def goal_response(handle):
    return SimpleNamespace(
        result=lambda: handle
    )


def make_node():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())

    logs = []

    logger = SimpleNamespace(
        info=lambda message: logs.append(message),
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )

    node = SimpleNamespace(
        _teleop_session=session,
        _goal_pending=True,
        _goal_handle=None,
        _aborted=False,
        _last_deadman_refresh_s=10.0,
        _deadman_was_active=True,
        _on_goal_result=lambda future: None,
        get_logger=lambda: logger,
    )

    callbacks = load_callbacks()

    for name in CALLBACK_NAMES:
        setattr(
            node,
            name,
            MethodType(callbacks[name], node),
        )

    return node, session


def test_normal_goal_acceptance_records_frame():
    node, session = make_node()
    handle = FakeGoalHandle()

    node._on_goal_response(goal_response(handle))

    assert session.frames == (make_frame(),)
    assert not session.has_pending_goal
    assert handle.cancel_count == 0


def test_stop_prevents_late_goal_acceptance_from_recording():
    node, session = make_node()

    # Operator stops before the async controller reply.
    node._stop()

    handle = FakeGoalHandle()
    node._on_goal_response(goal_response(handle))

    assert session.frames == ()
    assert not session.has_pending_goal
    assert handle.cancel_count == 1
    assert not node._goal_pending


def test_deadman_cancel_prevents_late_recording():
    node, session = make_node()

    # This is the cancellation path used by deadman expiry.
    node._cancel_active_goal()

    handle = FakeGoalHandle()
    node._on_goal_response(goal_response(handle))

    assert session.frames == ()
    assert not session.has_pending_goal
    assert handle.cancel_count == 1
    assert not node._goal_pending


def test_aborted_recording_cancels_late_accepted_goal():
    node, session = make_node()

    session.abort()
    node._aborted = True

    handle = FakeGoalHandle()
    node._on_goal_response(goal_response(handle))

    assert session.status == "aborted"
    assert session.frames == ()
    assert handle.cancel_count == 1


def test_stop_cancels_existing_active_goal():
    node, _ = make_node()
    handle = FakeGoalHandle()

    node._goal_handle = handle

    node._stop()

    assert handle.cancel_count == 1
    assert node._goal_handle is None
    assert not node._goal_pending
