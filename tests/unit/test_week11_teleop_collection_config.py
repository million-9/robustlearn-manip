"""Test explicit collection provenance and ROS2 parameter declarations."""

import ast
from pathlib import Path

import pytest

from robustlearn.teleop_demo_collection import TeleopCollectionConfig


def configuration(tmp_path, **overrides):
    values = {
        "output_root": tmp_path / "episodes",
        "episode_index": 12,
        "seed": 42,
        "scenario": "nominal",
        "git_commit": "abc123",
        "config_id": "keyboard-teleop-v1",
        "task": "panda_insertion",
        "failure_reason": None,
    }

    values.update(overrides)

    return TeleopCollectionConfig(**values)


def test_valid_configuration_preserves_provenance(tmp_path):
    config = configuration(tmp_path)

    assert config.output_root == tmp_path / "episodes"
    assert config.episode_index == 12
    assert config.seed == 42
    assert config.scenario == "nominal"
    assert config.git_commit == "abc123"
    assert config.config_id == "keyboard-teleop-v1"
    assert config.task == "panda_insertion"
    assert config.failure_reason is None


def test_configuration_does_not_create_dataset_files(tmp_path):
    configuration(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failure_reason_is_optional_at_startup(tmp_path):
    config = configuration(
        tmp_path,
        failure_reason="Insertion unsuccessful",
    )

    assert config.failure_reason == "Insertion unsuccessful"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("episode_index", -1),
        ("episode_index", True),
        ("episode_index", "12"),
        ("seed", -1),
        ("seed", True),
        ("seed", "42"),
        ("scenario", ""),
        ("scenario", "   "),
        ("git_commit", ""),
        ("config_id", ""),
        ("task", ""),
        ("failure_reason", "   "),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, field, invalid):
    with pytest.raises(ValueError):
        configuration(
            tmp_path,
            **{field: invalid},
        )


def test_output_path_must_be_absolute(tmp_path):
    with pytest.raises(ValueError):
        configuration(
            tmp_path,
            output_root=Path("relative/episodes"),
        )


def test_output_path_must_be_path_object(tmp_path):
    with pytest.raises(ValueError):
        configuration(
            tmp_path,
            output_root=str(tmp_path),
        )


def test_ros2_node_declares_collection_parameters():
    script = (
        Path(__file__).resolve().parents[2]
        / "ros2_ws/src/robustlearn_control/scripts/keyboard_teleop.py"
    )

    tree = ast.parse(script.read_text(encoding="utf-8"))

    keyboard = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KeyboardTeleop"
    )

    initializer = next(
        node
        for node in keyboard.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )

    declared = {
        node.args[0].value
        for node in ast.walk(initializer)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "declare_parameter"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
    }

    expected = {
        "record_demonstrations",
        "teleop_output_root",
        "teleop_episode_index",
        "teleop_seed",
        "teleop_scenario",
        "teleop_git_commit",
        "teleop_config_id",
        "teleop_task",
        "teleop_failure_reason",
    }

    assert expected.issubset(declared)
