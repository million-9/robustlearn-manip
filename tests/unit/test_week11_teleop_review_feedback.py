"""Verify real ROS joint feedback drives review stationary evidence."""

import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

from robustlearn.teleop_demo_collection import (
    TeleopReviewMotionGuard,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

JOINT_NAMES = tuple(
    f"panda_joint{i}" for i in range(1, 8)
)

JOINTS = (
    0.1, -0.1, 0.08, -1.7, 0.1, 1.7, -0.7
)


def load_joint_callback():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    callback = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_on_joint_state"
    )

    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                callback,
            ],
            type_ignores=[],
        )
    )

    namespace = {
        "math": math,
        "PANDA_JOINT_NAMES": JOINT_NAMES,
    }

    exec(
        compile(module, str(SCRIPT), "exec"),
        namespace,
    )

    return namespace["_on_joint_state"]


def make_node():
    guard = TeleopReviewMotionGuard()

    times = iter((10.0, 10.15, 10.30, 10.35, 10.40))

    node = SimpleNamespace(
        _joint_positions=None,
        _joint_velocities=None,
        _review_motion_guard=guard,
        _now_s=lambda: next(times),
    )

    node._on_joint_state = MethodType(
        load_joint_callback(), node
    )

    return node, guard


def publish(node, velocities, *, names=JOINT_NAMES):
    positions = dict(zip(JOINT_NAMES, JOINTS, strict=True))

    message = SimpleNamespace(
        name=names,
        position=[positions[name] for name in names],
        velocity=velocities,
    )

    node._on_joint_state(message)


def is_ready(guard, *, now=10.31):
    return guard.ready(
        now_s=now,
        in_review=True,
        controller_request_pending=False,
        controller_goal_active=False,
        cancellation_confirmed=True,
    )


def test_live_stationary_samples_build_readiness_evidence():
    node, guard = make_node()

    for _ in range(3):
        publish(node, (0.0,) * 7)

    assert node._joint_velocities == [0.0] * 7
    assert is_ready(guard)


def test_moving_feedback_invalidates_readiness():
    node, guard = make_node()

    for _ in range(3):
        publish(node, (0.0,) * 7)

    assert is_ready(guard)

    publish(
        node,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    assert not is_ready(guard, now=10.36)


def test_missing_velocity_feedback_invalidates_readiness():
    node, guard = make_node()

    for _ in range(3):
        publish(node, (0.0,) * 7)

    publish(node, ())

    assert node._joint_velocities is None
    assert not is_ready(guard, now=10.36)


def test_missing_arm_joint_invalidates_readiness():
    node, guard = make_node()

    for _ in range(3):
        publish(node, (0.0,) * 7)

    publish(
        node,
        (0.0,) * 6,
        names=JOINT_NAMES[:-1],
    )

    assert not is_ready(guard, now=10.36)
