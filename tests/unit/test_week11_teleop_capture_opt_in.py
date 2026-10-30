"""Tests for optional teleoperation recording initialization."""

import ast
from pathlib import Path

import pytest

import robustlearn.teleop_demo_collection as collection
from robustlearn.teleop_demo_collection import (
    TeleopRecordingSession,
    create_teleop_recording_components,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
)


def test_recording_disabled_does_not_initialize_kinematics(
    monkeypatch,
):
    def unexpected_initialization():
        raise AssertionError(
            "Disabled recording must not load MuJoCo kinematics."
        )

    monkeypatch.setattr(
        collection,
        "ShadowInsertionKinematics",
        unexpected_initialization,
    )

    session, kinematics = create_teleop_recording_components(
        False
    )

    assert session is None
    assert kinematics is None


def test_recording_enabled_creates_both_components(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        collection,
        "ShadowInsertionKinematics",
        lambda: sentinel,
    )

    session, kinematics = create_teleop_recording_components(
        True
    )

    assert isinstance(session, TeleopRecordingSession)
    assert session.status == "recording"
    assert session.frames == ()
    assert kinematics is sentinel


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_recording_flag_must_be_boolean(invalid):
    with pytest.raises(TypeError):
        create_teleop_recording_components(invalid)


def test_keyboard_node_declares_recording_disabled_by_default():
    tree = ast.parse(
        SCRIPT_PATH.read_text(encoding="utf-8")
    )

    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    constructor = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )

    declarations = [
        call
        for call in ast.walk(constructor)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "declare_parameter"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "record_demonstrations"
    ]

    assert len(declarations) == 1
    assert isinstance(declarations[0].args[1], ast.Constant)
    assert declarations[0].args[1].value is False
