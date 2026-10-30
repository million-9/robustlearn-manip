"""Tests for joint velocity feedback in keyboard teleoperation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)

JOINT_NAMES = tuple(
    f"panda_joint{i}" for i in range(1, 8)
)


def load_callback():
    """Load the real joint state callback without importing ROS2."""
    tree = ast.parse(
        SCRIPT_PATH.read_text(encoding="utf-8")
    )

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    method = next(
        node for node in node_class.body
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
                method,
            ],
            type_ignores=[],
        )
    )

    namespace = {
        "PANDA_JOINT_NAMES": JOINT_NAMES,
        "math": math,
    }

    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)

    return namespace["_on_joint_state"]


def make_node():
    return SimpleNamespace(
        _joint_positions=None,
        _joint_velocities=None,
    )


def test_velocities_follow_canonical_joint_order():
    callback = load_callback()
    node = make_node()

    callback(
        node,
        SimpleNamespace(
            name=tuple(reversed(JOINT_NAMES)),
            position=tuple(range(7, 0, -1)),
            velocity=tuple(i / 10 for i in range(7, 0, -1)),
        ),
    )

    assert node._joint_positions == pytest.approx(
        [1, 2, 3, 4, 5, 6, 7]
    )
    assert node._joint_velocities == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )


def test_missing_velocities_clear_previous_sample():
    callback = load_callback()
    node = make_node()

    callback(
        node,
        SimpleNamespace(
            name=JOINT_NAMES,
            position=(0.1,) * 7,
            velocity=(0.2,) * 7,
        ),
    )

    assert node._joint_velocities is not None

    callback(
        node,
        SimpleNamespace(
            name=JOINT_NAMES,
            position=(0.3,) * 7,
            velocity=(),
        ),
    )

    assert node._joint_positions == pytest.approx(
        (0.3,) * 7
    )
    assert node._joint_velocities is None


@pytest.mark.parametrize(
    "velocities",
    [
        (0.1,) * 6,
        (math.nan,) + (0.1,) * 6,
        (math.inf,) + (0.1,) * 6,
    ],
)
def test_invalid_velocities_are_not_retained(velocities):
    callback = load_callback()
    node = make_node()

    node._joint_velocities = [0.2] * 7

    callback(
        node,
        SimpleNamespace(
            name=JOINT_NAMES,
            position=(0.1,) * 7,
            velocity=velocities,
        ),
    )

    assert node._joint_velocities is None
