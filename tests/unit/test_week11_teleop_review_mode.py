"""Test keyboard review-mode gating without starting ROS2."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.teleop import (
    TeleopConfig,
    bounded_joint_target,
    command_due,
    deadman_active,
    keyboard_joint_delta,
)
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    teleop_review_key_action,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

JOINTS = (
    0.1, -0.1, 0.08, -1.7, 0.1, 1.7, -0.7
)


def load_methods():
    tree = ast.parse(
        SCRIPT.read_text(encoding="utf-8")
    )

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    names = {"_poll_keyboard", "_stop"}

    methods = [
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    assert len(methods) == 2

    module = ast.fix_missing_locations(
        ast.Module(
            body=methods,
            type_ignores=[],
        )
    )

    namespace = {
        "deadman_active": deadman_active,
        "keyboard_joint_delta": keyboard_joint_delta,
        "bounded_joint_target": bounded_joint_target,
        "command_due": command_due,
        "teleop_review_key_action": teleop_review_key_action,
    }

    exec(compile(module, str(SCRIPT), "exec"), namespace)

    return namespace


def make_node(*, recording=True):
    session = (
        TeleopRecordingSession()
        if recording
        else None
    )

    keys = []
    commands = []
    cancellations = []
    logs = []

    logger = SimpleNamespace(
        info=lambda message: logs.append(message),
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )

    node = SimpleNamespace(
        _teleop_session=session,
        _review_ready_for_decision=lambda: False,
        _review_mode=False,
        _aborted=False,
        _last_operator_status_s=12.0,
        _operator_status_period_s=0.5,
        _last_deadman_refresh_s=None,
        _deadman_was_active=False,
        _last_command_s=None,
        _goal_pending=True,
        _joint_positions=list(JOINTS),
        _config=TeleopConfig(),
        _now_s=lambda: 12.0,
        _read_key=lambda: keys.pop(0) if keys else None,
        _log_operator_status=lambda now: None,
        _send_target=lambda target: commands.append(target),
        get_logger=lambda: logger,
    )
    node._record_collection_outcome = lambda *a, **k: None
    node._retry_pending_summary = lambda: None
    node._review_approval_calls = []
    node._approve_reviewed_recording = (
        node._review_approval_calls.append
    )

    def cancel_goal():
        cancellations.append(True)
        node._goal_pending = False

        if session is not None and session.has_pending_goal:
            session.reject_pending_frame()

    node._cancel_active_goal = cancel_goal
    node._abort = lambda: setattr(node, "_aborted", True)

    methods = load_methods()

    for name in ("_poll_keyboard", "_stop"):
        setattr(
            node,
            name,
            MethodType(methods[name], node),
        )

    def press(key):
        keys.append(key)
        node._poll_keyboard()

    return node, session, press, commands, cancellations


def test_enter_review_requests_stop_and_locks_motion():
    node, session, press, commands, cancellations = (
        make_node()
    )

    press("v")

    assert session.status == "recording"
    assert node._review_mode
    assert cancellations == [True]
    assert not node._goal_pending
    assert commands == []


def test_review_blocks_jog_deadman_and_approval_keys():
    node, session, press, commands, cancellations = (
        make_node()
    )

    press("v")

    for key in ("q", "a", " ", "1", "2", "3", "x"):
        press(key)

    assert node._review_mode
    assert session.status == "recording"
    assert node._last_deadman_refresh_s is None
    assert commands == []
    assert cancellations == [True]


def test_resume_requires_fresh_deadman():
    node, _, press, commands, _ = make_node()

    press("v")
    node._review_ready_for_decision = lambda: True
    press("0")

    assert not node._review_mode
    assert node._last_deadman_refresh_s is None

    # A jog without a new deadman refresh is prohibited.
    press("q")
    assert commands == []

    press(" ")
    press("q")

    assert len(commands) == 1


def test_escape_remains_available_during_review():
    node, _, press, commands, _ = make_node()

    press("v")
    press("\x1b")

    assert node._aborted
    assert commands == []


def test_disabled_recording_cannot_enter_review():
    node, session, press, commands, cancellations = (
        make_node(recording=False)
    )

    press("v")

    assert session is None
    assert not node._review_mode
    assert cancellations == []
    assert commands == []



def test_unsettled_review_cannot_resume():
    node, session, press, commands, _ = make_node()

    press("v")
    press("0")

    assert node._review_mode
    assert session.status == "recording"
    assert commands == []

    node._review_ready_for_decision = lambda: True
    press("0")

    assert not node._review_mode


def test_rejection_requires_controller_and_motion_readiness():
    node, session, press, commands, _ = make_node()

    press("v")
    press("3")

    assert session.status == "recording"
    assert node._review_mode

    node._review_ready_for_decision = lambda: True
    press("3")

    assert session.status == "rejected"
    assert node._review_mode
    assert commands == []


def test_rejected_recording_cannot_resume_or_jog():
    node, session, press, commands, _ = make_node()

    press("v")
    node._review_ready_for_decision = lambda: True
    press("3")

    assert session.status == "rejected"

    press("0")
    press(" ")
    press("q")

    assert node._review_mode
    assert node._last_deadman_refresh_s is None
    assert commands == []


def test_acceptance_keys_dispatch_to_approval_handler():
    node, session, press, commands, _ = make_node()

    press("v")
    node._review_ready_for_decision = lambda: True

    press("1")
    press("2")

    assert session.status == "recording"
    assert node._review_mode
    assert commands == []
    assert node._review_approval_calls == [
        "accept_success",
        "accept_failure",
    ]



def test_review_rejection_reports_outcome():
    node, session, press, _, _ = make_node()
    reports = []
    node._record_collection_outcome = reports.append
    press("v")
    node._review_ready_for_decision = lambda: True
    press("3")
    assert session.status == "rejected"
    assert reports == ["rejected"]


def test_review_summary_retry_key():
    node, _, press, _, _ = make_node()
    retries = []
    node._retry_pending_summary = lambda: retries.append(True)
    press("v")
    press("s")
    assert retries == [True]
