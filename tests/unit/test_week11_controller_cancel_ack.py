"""Verify cancellation callbacks cannot prematurely settle trajectories."""

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


class FakeFuture:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.callback = None

    def result(self):
        if self.error is not None:
            raise self.error
        return self.response

    def add_done_callback(self, callback):
        self.callback = callback

    def complete(self):
        assert self.callback is not None
        self.callback(self)


class FakeHandle:
    def __init__(self, future=None, raise_on_cancel=False):
        self.cancel_future = future or FakeFuture(
            SimpleNamespace(
                return_code=0,
                goals_canceling=[object()],
            )
        )
        self.raise_on_cancel = raise_on_cancel
        self.cancel_count = 0

    def cancel_goal_async(self):
        self.cancel_count += 1

        if self.raise_on_cancel:
            raise RuntimeError("cancel transport failed")

        return self.cancel_future


def load_callbacks():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    names = {
        "_cancel_active_goal",
        "_on_cancel_response",
    }

    methods = [
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    assert len(methods) == 2

    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *methods,
            ],
            type_ignores=[],
        )
    )

    namespace = {}

    exec(
        compile(module, str(SCRIPT), "exec"),
        namespace,
    )

    return namespace


def make_node(*, accepted=True, handle=None):
    tracker = TeleopControllerCompletionTracker()
    tracker.submitted(1)

    if accepted:
        tracker.goal_response(1, accepted=True)

    logs = []
    captures = []

    node = SimpleNamespace(
        _controller_completion_tracker=tracker,
        _active_request_id=1,
        _goal_handle=handle,
        _goal_pending=True,
        _resolve_pending_capture=lambda **kwargs: (
            captures.append(kwargs)
        ),
        get_logger=lambda: SimpleNamespace(
            error=lambda message: logs.append(message),
        ),
    )

    callbacks = load_callbacks()

    for name in (
        "_cancel_active_goal",
        "_on_cancel_response",
    ):
        setattr(
            node,
            name,
            MethodType(callbacks[name], node),
        )

    return node, tracker, captures, logs


def test_pending_submission_remains_outstanding_after_stop():
    node, tracker, captures, _ = make_node(
        accepted=False,
        handle=None,
    )

    node._cancel_active_goal()

    assert tracker.outstanding_request_ids == (1,)
    assert not tracker.settled
    assert node._active_request_id is None
    assert not node._goal_pending
    assert captures == [{"accepted": False}]


def test_acknowledgment_does_not_complete_accepted_goal():
    handle = FakeHandle()
    node, tracker, _, _ = make_node(handle=handle)

    node._cancel_active_goal()

    assert handle.cancel_count == 1
    assert not tracker.settled

    handle.cancel_future.complete()

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_rejected_cancellation_requires_terminal_result():
    future = FakeFuture(
        SimpleNamespace(
            return_code=1,
            goals_canceling=[],
        )
    )
    handle = FakeHandle(future=future)
    node, tracker, _, _ = make_node(handle=handle)

    node._cancel_active_goal()
    future.complete()

    assert not tracker.settled
    assert tracker.cancellation_rejected_ids == (1,)

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_malformed_response_fails_closed():
    future = FakeFuture(
        SimpleNamespace(return_code=None)
    )
    node, tracker, _, _ = make_node(
        handle=FakeHandle(future=future)
    )

    node._cancel_active_goal()
    future.complete()

    assert tracker.uncertain
    assert not tracker.settled


def test_cancellation_future_exception_fails_closed():
    future = FakeFuture(
        error=RuntimeError("cancel reply unavailable")
    )
    node, tracker, _, logs = make_node(
        handle=FakeHandle(future=future)
    )

    node._cancel_active_goal()
    future.complete()

    assert tracker.uncertain
    assert not tracker.settled
    assert logs


def test_cancel_request_exception_fails_closed():
    handle = FakeHandle(raise_on_cancel=True)
    node, tracker, _, logs = make_node(handle=handle)

    node._cancel_active_goal()

    assert tracker.uncertain
    assert not tracker.settled
    assert logs


def test_reply_after_terminal_result_does_not_reopen_goal():
    handle = FakeHandle()
    node, tracker, _, _ = make_node(handle=handle)

    node._cancel_active_goal()

    tracker.terminal_result(1, confirmed=True)
    handle.cancel_future.complete()

    assert tracker.settled


def test_existing_non_recording_node_remains_compatible():
    handle = FakeHandle()
    node, _, captures, _ = make_node(handle=handle)

    delattr(node, "_controller_completion_tracker")

    node._cancel_active_goal()

    assert handle.cancel_count == 1
    assert captures == [{"accepted": False}]


def test_unknown_request_id_fails_closed():
    handle = FakeHandle()
    node, tracker, _, _ = make_node(handle=handle)

    node._active_request_id = 999
    node._cancel_active_goal()

    assert tracker.uncertain
    assert not tracker.settled
