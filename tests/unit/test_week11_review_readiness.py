"""Verify that operator review requires safe controller completion."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.teleop_demo_collection import (
    TeleopControllerCompletionTracker,
    TeleopRecordingSession,
    TeleopReviewMotionGuard,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

STATIONARY = (0.0,) * 7


def load_readiness_method():
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
        and node.name == "_review_ready_for_decision"
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

    namespace = {}

    exec(
        compile(module, str(SCRIPT), "exec"),
        namespace,
    )

    return namespace["_review_ready_for_decision"]


def make_node():
    tracker = TeleopControllerCompletionTracker()
    guard = TeleopReviewMotionGuard()

    for timestamp in (10.0, 10.15, 10.30):
        guard.observe(
            timestamp_s=timestamp,
            joint_velocities=STATIONARY,
        )

    node = SimpleNamespace(
        _teleop_session=TeleopRecordingSession(),
        _controller_completion_tracker=tracker,
        _review_motion_guard=guard,
        _review_mode=True,
        _aborted=False,
        _goal_pending=False,
        _goal_handle=None,
        _active_request_id=None,
        _now_s=lambda: 10.31,
    )

    node._review_ready_for_decision = MethodType(
        load_readiness_method(),
        node,
    )

    return node, tracker, guard


def test_stationary_completed_controller_is_ready():
    node, tracker, _ = make_node()

    assert tracker.settled
    assert node._review_ready_for_decision()


def test_outstanding_submission_blocks_review():
    node, tracker, _ = make_node()

    tracker.submitted(1)

    assert not node._review_ready_for_decision()


def test_accepted_goal_blocks_review_until_terminal():
    node, tracker, _ = make_node()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)

    assert not node._review_ready_for_decision()

    tracker.terminal_result(1, confirmed=True)

    assert node._review_ready_for_decision()


def test_cancel_acknowledgment_does_not_unlock_review():
    node, tracker, _ = make_node()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)
    tracker.cancellation_requested(1)
    tracker.cancellation_response(
        1,
        acknowledged=True,
    )

    assert not node._review_ready_for_decision()

    tracker.terminal_result(1, confirmed=True)

    assert node._review_ready_for_decision()


def test_stale_outstanding_goal_blocks_review():
    node, tracker, _ = make_node()

    tracker.submitted(1)
    tracker.submitted(2)

    tracker.goal_response(1, accepted=True)
    tracker.goal_response(2, accepted=False)

    assert not node._review_ready_for_decision()

    tracker.terminal_result(1, confirmed=True)

    assert node._review_ready_for_decision()


def test_ambiguous_controller_result_blocks_review():
    node, tracker, _ = make_node()

    tracker.invalidate()

    assert not node._review_ready_for_decision()


def test_pending_goal_flag_blocks_review():
    node, _, _ = make_node()

    node._goal_pending = True

    assert not node._review_ready_for_decision()


def test_active_goal_handle_blocks_review():
    node, _, _ = make_node()

    node._goal_handle = object()

    assert not node._review_ready_for_decision()


def test_active_request_id_blocks_review():
    node, _, _ = make_node()

    node._active_request_id = 1

    assert not node._review_ready_for_decision()


def test_stale_velocity_feedback_blocks_review():
    node, _, _ = make_node()

    node._now_s = lambda: 11.0

    assert not node._review_ready_for_decision()


def test_motion_guard_reset_blocks_review():
    node, _, guard = make_node()

    guard.reset()

    assert not node._review_ready_for_decision()


def test_review_mode_is_mandatory():
    node, _, _ = make_node()

    node._review_mode = False

    assert not node._review_ready_for_decision()


def test_aborted_teleoperation_cannot_approve():
    node, _, _ = make_node()

    node._aborted = True

    assert not node._review_ready_for_decision()


def test_rejected_session_cannot_approve():
    node, _, _ = make_node()

    node._teleop_session.reject()

    assert not node._review_ready_for_decision()


def test_disabled_recording_cannot_approve():
    node, _, _ = make_node()

    node._teleop_session = None

    assert not node._review_ready_for_decision()


def test_missing_completion_tracker_fails_closed():
    node, _, _ = make_node()

    node._controller_completion_tracker = None

    assert not node._review_ready_for_decision()


def test_missing_motion_guard_fails_closed():
    node, _, _ = make_node()

    node._review_motion_guard = None

    assert not node._review_ready_for_decision()


def test_moving_feedback_blocks_review():
    node, _, guard = make_node()

    guard.observe(
        timestamp_s=10.35,
        joint_velocities=(
            0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ),
    )

    node._now_s = lambda: 10.36

    assert not node._review_ready_for_decision()
