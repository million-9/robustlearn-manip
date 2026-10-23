"""Isolated tests for ROS2 teleoperation command staging."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.sim.shadow_kinematics import ShadowInsertionKinematics
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    build_teleop_observation_frame,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

JOINTS = (
    0.1, -0.1, 0.08, -1.7, 0.1, 1.7, -0.7
)


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


class FakeClient:
    def __init__(self, *, ready=True, fail=False):
        self.ready = ready
        self.fail = fail
        self.calls = 0
        self.session = None
        self.pending_at_submission = None

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        self.calls += 1

        if self.session is not None:
            self.pending_at_submission = (
                self.session.has_pending_goal
            )

        if self.fail:
            raise RuntimeError("submission failed")

        return SimpleNamespace(
            add_done_callback=lambda callback: None,
        )


def load_methods():
    tree = ast.parse(
        SCRIPT_PATH.read_text(encoding="utf-8")
    )

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    names = {
        "_stage_capture_for_target",
        "_send_target",
        "_resolve_pending_capture",
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

    namespace = {
        "build_teleop_observation_frame": (
            build_teleop_observation_frame
        ),
        "FollowJointTrajectory": SimpleNamespace(
            Goal=FakeGoal
        ),
        "JointTrajectoryPoint": FakePoint,
        "Duration": FakeDuration,
        "PANDA_JOINT_NAMES": tuple(
            f"panda_joint{i}" for i in range(1, 8)
        ),
    }

    exec(
        compile(module, str(SCRIPT_PATH), "exec"),
        namespace,
    )

    return namespace


def make_node(*, recording=True, ready=True, fail=False):
    methods = load_methods()

    session = (
        TeleopRecordingSession()
        if recording
        else None
    )

    client = FakeClient(
        ready=ready,
        fail=fail,
    )
    client.session = session

    logs = []

    logger = SimpleNamespace(
        warning=lambda message: logs.append(message),
        error=lambda message: logs.append(message),
    )

    node = SimpleNamespace(
        _teleop_session=session,
        _capture_kinematics=(
            ShadowInsertionKinematics()
            if recording
            else None
        ),
        _joint_positions=list(JOINTS),
        _joint_velocities=[0.01] * 7,
        _latest_wrist_force=(1.0, 2.0, 3.0),
        _latest_wrist_torque=(0.1, 0.2, 0.3),
        _trajectory_client=client,
        _trajectory_duration_s=0.1,
        _goal_pending=False,
        _aborted=False,
        _on_goal_response=lambda future: None,
        _now_s=lambda: 12.0,
        get_logger=lambda: logger,
    )

    for name in (
        "_stage_capture_for_target",
        "_send_target",
        "_resolve_pending_capture",
    ):
        setattr(
            node,
            name,
            MethodType(methods[name], node),
        )

    return node, client, session, logs


def test_command_staged_before_ros2_submission():
    node, client, session, _ = make_node()

    node._send_target(list(JOINTS))

    assert client.calls == 1
    assert client.pending_at_submission is True
    assert session.frames == ()
    assert session.has_pending_goal
    assert node._goal_pending


def test_accepted_frame_contains_real_observations():
    node, client, session, _ = make_node()

    node._send_target(list(JOINTS))
    session.accept_pending_frame()

    assert client.calls == 1
    assert len(session.frames) == 1

    frame = session.frames[0]

    assert frame.joint_position == JOINTS
    assert frame.joint_velocity == (0.01,) * 7
    assert frame.expert_action == JOINTS
    assert len(frame.task_relative_state) == 3
    assert len(frame.wrist_wrench) == 6
    assert frame.timestamp_s == 12.0


def test_missing_required_feedback_blocks_submission():
    node, client, session, logs = make_node()

    node._latest_wrist_force = None
    node._send_target(list(JOINTS))

    assert client.calls == 0
    assert session.frames == ()
    assert not session.has_pending_goal
    assert not node._goal_pending
    assert logs


def test_immediate_submission_error_discards_frame():
    node, client, session, logs = make_node(
        fail=True
    )

    node._send_target(list(JOINTS))

    assert client.calls == 1
    assert session.frames == ()
    assert not session.has_pending_goal
    assert not node._goal_pending
    assert logs


def test_controller_unavailable_does_not_stage():
    node, client, session, _ = make_node(
        ready=False
    )

    node._send_target(list(JOINTS))

    assert client.calls == 0
    assert not session.has_pending_goal


def test_recording_disabled_preserves_command_submission():
    node, client, session, _ = make_node(
        recording=False
    )

    node._send_target(list(JOINTS))

    assert session is None
    assert client.calls == 1
    assert node._goal_pending


def test_finalized_recording_blocks_new_commands():
    node, client, session, _ = make_node()

    session.reject()
    node._send_target(list(JOINTS))

    assert client.calls == 0
    assert session.status == "rejected"
