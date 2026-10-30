"""Test keyboard teleop controller responses without launching ROS2."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.rollout_dataset import RolloutFrame
from robustlearn.teleop_demo_collection import TeleopRecordingSession

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)


def load_callbacks():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    names = {
        "_resolve_pending_capture",
        "_on_goal_response",
    }

    methods = [
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    assert len(methods) == len(names)

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

    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)

    return namespace


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


def make_node(session, *, aborted=False):
    callbacks = load_callbacks()

    logger = SimpleNamespace(
        error=lambda message: None,
        warning=lambda message: None,
    )

    node = SimpleNamespace(
        _teleop_session=session,
        _aborted=aborted,
        _goal_pending=True,
        _goal_handle=None,
        _on_goal_result=lambda future: None,
        get_logger=lambda: logger,
    )

    node._resolve_pending_capture = MethodType(
        callbacks["_resolve_pending_capture"], node
    )

    node._on_goal_response = MethodType(
        callbacks["_on_goal_response"], node
    )

    return node


def make_future(*, accepted):
    result_future = SimpleNamespace(
        add_done_callback=lambda callback: None
    )

    goal_handle = SimpleNamespace(
        accepted=accepted,
        get_result_async=lambda: result_future,
    )

    return SimpleNamespace(result=lambda: goal_handle)


def test_accepted_controller_goal_records_pending_frame():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())

    node = make_node(session)
    node._on_goal_response(make_future(accepted=True))

    assert session.frames == (make_frame(),)
    assert not session.has_pending_goal


def test_rejected_controller_goal_discards_pending_frame():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())

    node = make_node(session)
    node._on_goal_response(make_future(accepted=False))

    assert session.frames == ()
    assert not session.has_pending_goal
    assert not node._goal_pending


def test_failed_goal_submission_discards_pending_frame():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())

    def fail():
        raise RuntimeError("controller unavailable")

    node = make_node(session)
    node._on_goal_response(SimpleNamespace(result=fail))

    assert session.frames == ()
    assert not session.has_pending_goal
    assert not node._goal_pending


def test_late_response_does_not_record_after_abort():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())
    session.abort()

    node = make_node(session, aborted=True)
    node._on_goal_response(make_future(accepted=True))

    assert session.status == "aborted"
    assert session.frames == ()


def test_aborted_node_cannot_commit_pending_frame():
    session = TeleopRecordingSession()
    session.stage_frame(make_frame())

    node = make_node(session, aborted=True)
    node._on_goal_response(make_future(accepted=True))

    assert session.frames == ()
    assert not session.has_pending_goal


def test_recording_disabled_preserves_controller_callback():
    node = make_node(None)

    node._on_goal_response(make_future(accepted=True))

    assert node._goal_handle is not None
