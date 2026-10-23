"""Verify ROS2 goal-response and terminal-result accounting."""

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

STATUSES = SimpleNamespace(
    STATUS_SUCCEEDED=4,
    STATUS_CANCELED=5,
    STATUS_ABORTED=6,
)


class FakeFuture:
    def __init__(self, result=None, error=None):
        self.value = result
        self.error = error
        self.callback = None

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, callback):
        self.callback = callback

    def complete(self):
        assert self.callback is not None
        self.callback(self)


class FakeHandle:
    def __init__(self, accepted=True, status=4):
        self.accepted = accepted
        self.result_future = FakeFuture(
            SimpleNamespace(status=status)
        )
        self.cancel_count = 0
        self.cancel_future = FakeFuture(
            SimpleNamespace(
                return_code=0,
                goals_canceling=[object()],
            )
        )

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        return self.cancel_future


def load_callbacks():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    selected = [
        node
        for node in node_class.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in (
                "_on_goal_response",
                "_on_goal_result",
                "_on_cancel_response",
            )
        )
    ]

    assert len(selected) == 3

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
        "GoalStatus": STATUSES,
    }

    exec(
        compile(module, str(SCRIPT), "exec"),
        namespace,
    )

    return namespace


def make_node(*, active_id=1):
    tracker = TeleopControllerCompletionTracker()
    tracker.submitted(1)

    logs = []
    captures = []

    node = SimpleNamespace(
        _controller_completion_tracker=tracker,
        _active_request_id=active_id,
        _goal_pending=True,
        _goal_handle=None,
        _teleop_session=None,
        _aborted=False,
        _resolve_pending_capture=lambda **kwargs: (
            captures.append(kwargs)
        ),
        get_logger=lambda: SimpleNamespace(
            warning=lambda message: logs.append(message),
            error=lambda message: logs.append(message),
        ),
    )

    callbacks = load_callbacks()

    for name, callback in callbacks.items():
        if name in (
            "_on_goal_response",
            "_on_goal_result",
            "_on_cancel_response",
        ):
            setattr(node, name, MethodType(callback, node))

    return node, tracker, captures


def test_rejected_goal_completes_submission():
    node, tracker, captures = make_node()

    node._on_goal_response(
        FakeFuture(FakeHandle(accepted=False)),
        request_id=1,
    )

    assert tracker.settled
    assert captures == [{"accepted": False}]
    assert not node._goal_pending


def test_accepted_goal_requires_terminal_result():
    node, tracker, captures = make_node()
    handle = FakeHandle()

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    assert tracker.outstanding_request_ids == (1,)
    assert not tracker.settled
    assert captures == [{"accepted": True}]

    handle.result_future.complete()

    assert tracker.settled
    assert node._goal_handle is None
    assert not node._goal_pending


def test_stale_accepted_goal_retains_result_listener():
    node, tracker, _ = make_node(active_id=2)
    tracker.submitted(2)

    current_handle = object()
    node._goal_handle = current_handle

    handle = FakeHandle(status=5)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    assert handle.cancel_count == 1
    assert tracker.outstanding_request_ids == (1, 2)
    assert node._goal_handle is current_handle
    assert node._active_request_id == 2

    handle.result_future.complete()

    assert tracker.outstanding_request_ids == (2,)
    assert node._goal_handle is current_handle
    assert node._goal_pending


def test_stale_rejection_does_not_modify_current_goal():
    node, tracker, captures = make_node(active_id=2)
    tracker.submitted(2)

    node._on_goal_response(
        FakeFuture(FakeHandle(accepted=False)),
        request_id=1,
    )

    assert tracker.outstanding_request_ids == (2,)
    assert node._active_request_id == 2
    assert node._goal_pending
    assert captures == []


def test_aborted_trajectory_status_is_terminal():
    node, tracker, _ = make_node()
    handle = FakeHandle(status=6)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )
    handle.result_future.complete()

    assert tracker.settled


def test_nonterminal_status_fails_closed():
    node, tracker, _ = make_node()
    handle = FakeHandle(status=2)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )
    handle.result_future.complete()

    assert tracker.uncertain
    assert not tracker.settled


def test_goal_response_exception_fails_closed():
    node, tracker, _ = make_node()

    node._on_goal_response(
        FakeFuture(error=RuntimeError("unavailable")),
        request_id=1,
    )

    assert tracker.uncertain
    assert not tracker.settled


def test_terminal_result_exception_fails_closed():
    node, tracker, _ = make_node()

    handle = FakeHandle()
    handle.result_future.error = RuntimeError("lost result")

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )
    handle.result_future.complete()

    assert tracker.uncertain
    assert not tracker.settled


def test_late_accepted_goal_after_finalization_is_tracked():
    node, tracker, _ = make_node()
    node._teleop_session = SimpleNamespace(
        status="accepted"
    )

    handle = FakeHandle(status=5)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    assert handle.cancel_count == 1
    assert not tracker.settled

    handle.result_future.complete()

    assert tracker.settled



def test_stale_goal_cancel_ack_requires_terminal_result():
    node, tracker, _ = make_node(active_id=2)
    tracker.submitted(2)

    handle = FakeHandle(status=5)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    assert handle.cancel_count == 1
    assert tracker.outstanding_request_ids == (1, 2)

    handle.cancel_future.complete()

    # Cancellation acknowledgment must not release the goal.
    assert tracker.outstanding_request_ids == (1, 2)
    assert not tracker.settled

    handle.result_future.complete()

    assert tracker.outstanding_request_ids == (2,)
    assert not tracker.settled

    tracker.goal_response(2, accepted=False)

    assert tracker.settled


def test_late_goal_cancel_ack_requires_terminal_result():
    node, tracker, _ = make_node()

    node._teleop_session = SimpleNamespace(
        status="accepted"
    )

    handle = FakeHandle(status=5)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    assert handle.cancel_count == 1
    assert not tracker.settled

    handle.cancel_future.complete()

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)

    handle.result_future.complete()

    assert tracker.settled


def test_stale_cancel_rejection_requires_terminal_result():
    node, tracker, _ = make_node(active_id=2)
    tracker.submitted(2)

    handle = FakeHandle(status=4)

    handle.cancel_future.response = None
    handle.cancel_future.value = SimpleNamespace(
        return_code=1,
        goals_canceling=[],
    )

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    handle.cancel_future.complete()

    assert tracker.cancellation_rejected_ids == (1,)
    assert tracker.outstanding_request_ids == (1, 2)

    handle.result_future.complete()

    assert tracker.outstanding_request_ids == (2,)


def test_cancel_reply_after_terminal_result_is_safe():
    node, tracker, _ = make_node()

    node._teleop_session = SimpleNamespace(
        status="accepted"
    )

    handle = FakeHandle(status=5)

    node._on_goal_response(
        FakeFuture(handle),
        request_id=1,
    )

    # Terminal result arrives before cancellation reply.
    handle.result_future.complete()

    assert tracker.settled

    handle.cancel_future.complete()

    assert tracker.settled
    assert not tracker.uncertain
