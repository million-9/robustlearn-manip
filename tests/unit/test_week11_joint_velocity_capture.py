"""Tests for Panda joint velocity extraction in expert capture."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts"
    / "local_insertion_expert.py"
)

JOINT_NAMES = tuple(
    f"panda_joint_{index}" for index in range(1, 8)
)


def load_joint_state_callback():
    """Load the actual callback without starting ROS2."""
    tree = ast.parse(
        SCRIPT_PATH.read_text(encoding="utf-8")
    )

    node_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LocalInsertionExpertNode"
    )

    method = next(
        node for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_joint_state_callback"
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
        "np": np,
        "ROS_PANDA_ARM_JOINT_NAMES": JOINT_NAMES,
    }

    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["_joint_state_callback"]


def make_node():
    def unexpected_abort(reason):
        raise AssertionError(reason)

    return SimpleNamespace(
        _latest_joint_positions=None,
        _latest_joint_velocities=None,
        _abort=unexpected_abort,
    )


def test_joint_velocities_follow_canonical_joint_order():
    callback = load_joint_state_callback()
    node = make_node()

    callback(
        node,
        SimpleNamespace(
            name=tuple(reversed(JOINT_NAMES)),
            position=tuple(float(i) for i in range(7, 0, -1)),
            velocity=tuple(i / 10 for i in range(7, 0, -1)),
        ),
    )

    np.testing.assert_allclose(
        node._latest_joint_positions,
        (1., 2., 3., 4., 5., 6., 7.),
    )

    np.testing.assert_allclose(
        node._latest_joint_velocities,
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    )


def test_missing_joint_velocities_clear_previous_sample():
    callback = load_joint_state_callback()
    node = make_node()

    callback(
        node,
        SimpleNamespace(
            name=JOINT_NAMES,
            position=(0.0,) * 7,
            velocity=(0.1,) * 7,
        ),
    )

    assert node._latest_joint_velocities is not None

    callback(
        node,
        SimpleNamespace(
            name=JOINT_NAMES,
            position=(0.2,) * 7,
            velocity=(),
        ),
    )

    assert node._latest_joint_velocities is None
    np.testing.assert_allclose(
        node._latest_joint_positions,
        (0.2,) * 7,
    )
