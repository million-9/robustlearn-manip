"""Issue 101: deterministic dataset validation and corruption detection."""

import json

import pytest

from robustlearn.dataset_validation import (
    DatasetValidationError,
    main,
    validate_dataset,
)
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    write_episode,
)


def make_episode(index=1, source="scripted", success=True):
    frames = tuple(
        RolloutFrame(
            frame_index=i,
            timestamp_s=float(i + 1),
            joint_position=(0.1,) * 7,
            joint_velocity=(0.0,) * 7,
            task_relative_state=(0.01, 0.02, 0.03),
            wrist_wrench=(0.0,) * 6,
            expert_action=(0.2,) * 7,
            task_success=success if i == 1 else False,
            task_failure=(not success) if i == 1 else False,
            terminated=i == 1,
        )
        for i in range(2)
    )

    return RolloutEpisode(
        episode_index=index,
        seed=index,
        scenario="nominal",
        source=source,
        task="panda_insertion",
        success=success,
        failure_reason=None if success else "Insertion incomplete",
        start_timestamp_s=1.0,
        end_timestamp_s=2.0,
        git_commit="test-commit",
        config_id="test-config",
        frames=frames,
    )


def write_raw(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_mixed_dataset_statistics_are_deterministic(tmp_path):
    root = tmp_path / "episodes"

    write_episode(root, make_episode(1, "scripted", True))
    write_episode(root, make_episode(2, "teleop", False))

    first = validate_dataset(root)
    second = validate_dataset(root)

    assert first == second
    assert first["total_episodes"] == 2
    assert first["total_frames"] == 4
    assert first["scripted_episodes"] == 1
    assert first["teleop_episodes"] == 1
    assert first["successes"] == 1
    assert first["failures"] == 1
    assert first["success_rate"] == 0.5
    assert first["failure_reasons"] == {"Insertion incomplete": 1}
    assert first["scenarios"] == {"nominal": 2}
    assert first["tasks"] == {"panda_insertion": 2}
    assert first["action_dimension"] == 7

    output = tmp_path / "stats.json"
    assert main([str(root), "--json-output", str(output)]) == 0
    contents = output.read_bytes()

    assert main([str(root), "--json-output", str(output)]) == 0
    assert output.read_bytes() == contents

    loaded = json.loads(contents)
    assert loaded == first
    assert loaded["numeric_statistics"]["expert_action"]["mean"] == [0.2] * 7


@pytest.mark.parametrize(
    ("corruption", "diagnostic"),
    [
        ("missing_task", "missing required episode fields"),
        ("missing_scenario", "scenario must be non-empty"),
        ("unknown_source", "invalid demonstration source"),
        ("empty_frames", "episode has no frames"),
        ("missing_frame_field", "missing required fields"),
        ("duplicate_frame_index", "frame indices must be contiguous"),
        ("invalid_frame_index", "frame indices must be contiguous"),
        ("reversed_timestamp", "frame timestamp"),
        ("nan_observation", "finite"),
        ("bad_observation_dimension", "joint_position must have shape"),
        ("bad_action_dimension", "expert_action must have shape"),
        ("missing_terminal", "last frame must be terminated"),
        ("early_terminal", "non-final frame cannot be terminated"),
        ("invalid_failure_metadata", "successful episode"),
        ("invalid_terminal_label", "terminal success flag"),
    ],
)
def test_corrupted_episode_is_rejected(tmp_path, corruption, diagnostic):
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    data = json.loads(path.read_text())

    if corruption == "missing_task":
        del data["task"]
    elif corruption == "missing_scenario":
        data["scenario"] = " "
    elif corruption == "unknown_source":
        data["source"] = "unverified"
    elif corruption == "empty_frames":
        data["frames"] = []
    elif corruption == "missing_frame_field":
        del data["frames"][0]["joint_position"]
    elif corruption == "duplicate_frame_index":
        data["frames"][1]["frame_index"] = 0
    elif corruption == "invalid_frame_index":
        data["frames"][1]["frame_index"] = 5
    elif corruption == "reversed_timestamp":
        data["frames"][1]["timestamp_s"] = 0.5
    elif corruption == "nan_observation":
        data["frames"][0]["wrist_wrench"][0] = float("nan")
    elif corruption == "bad_observation_dimension":
        data["frames"][0]["joint_position"] = [0.1] * 6
    elif corruption == "bad_action_dimension":
        data["frames"][0]["expert_action"] = [0.2] * 8
    elif corruption == "missing_terminal":
        data["frames"][1]["terminated"] = False
        data["frames"][1]["task_success"] = False
    elif corruption == "early_terminal":
        data["frames"][0]["terminated"] = True
    elif corruption == "invalid_failure_metadata":
        data["failure_reason"] = "should not be set"
    elif corruption == "invalid_terminal_label":
        data["frames"][1]["task_success"] = False

    write_raw(path, data)

    with pytest.raises(DatasetValidationError) as exc:
        validate_dataset(root)

    assert diagnostic in str(exc.value)
    assert path.name in str(exc.value)


def test_duplicate_episode_ids_rejected(tmp_path):
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    duplicate = root / "duplicate.json"
    duplicate.write_bytes(path.read_bytes())

    with pytest.raises(DatasetValidationError, match="duplicate episode ID"):
        validate_dataset(root)


def test_malformed_json_rejected(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()
    (root / "broken.json").write_text("{invalid JSON")

    with pytest.raises(DatasetValidationError, match="broken.json"):
        validate_dataset(root)


def test_empty_dataset_rejected(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()

    with pytest.raises(DatasetValidationError, match="no episode"):
        validate_dataset(root)


def test_cli_invalid_dataset_has_nonzero_exit(tmp_path, capsys):
    root = tmp_path / "episodes"
    root.mkdir()
    (root / "broken.json").write_text("not-json")

    output = tmp_path / "stats.json"
    assert main([str(root), "--json-output", str(output)]) == 1
    assert "DATASET VALIDATION FAILED" in capsys.readouterr().err
    assert not output.exists()


def test_cli_refuses_report_inside_episode_directory(tmp_path):
    root = tmp_path / "episodes"
    write_episode(root, make_episode())

    assert main([
        str(root),
        "--json-output",
        str(root / "stats.json"),
    ]) == 2
    assert not (root / "stats.json").exists()


def test_out_of_order_timestamps_within_episode_bounds(tmp_path):
    """Reject reversed timestamps even when both lie within episode bounds."""
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    data = json.loads(path.read_text(encoding="utf-8"))

    data["frames"][0]["timestamp_s"] = 1.8
    data["frames"][1]["timestamp_s"] = 1.7

    write_raw(path, data)

    with pytest.raises(
        DatasetValidationError,
        match="frame timestamps must be non-decreasing",
    ):
        validate_dataset(root)


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf")],
    ids=["positive_infinity", "negative_infinity"],
)
def test_infinite_action_values_rejected(tmp_path, value):
    """Reject positive and negative infinity in canonical action data."""
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    data = json.loads(path.read_text(encoding="utf-8"))
    data["frames"][0]["expert_action"][3] = value

    write_raw(path, data)

    with pytest.raises(DatasetValidationError, match="finite"):
        validate_dataset(root)


@pytest.mark.parametrize("field", ["scenario", "task"])
def test_blank_required_metadata_rejected(tmp_path, field):
    """Whitespace-only scenario and task metadata must fail."""
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = "   "

    write_raw(path, data)

    with pytest.raises(
        DatasetValidationError,
        match=f"{field} must be non-empty",
    ):
        validate_dataset(root)


def test_nonfinite_timestamp_rejected(tmp_path):
    """Reject infinite timestamps before statistics generation."""
    root = tmp_path / "episodes"
    path = write_episode(root, make_episode())

    data = json.loads(path.read_text(encoding="utf-8"))
    data["frames"][0]["timestamp_s"] = float("inf")

    write_raw(path, data)

    with pytest.raises(DatasetValidationError, match="finite"):
        validate_dataset(root)


def test_multiple_corrupted_files_reported(tmp_path):
    """Report separate corrupted episodes instead of stopping at the first."""
    root = tmp_path / "episodes"
    root.mkdir()

    (root / "broken_a.json").write_text("{invalid", encoding="utf-8")
    (root / "broken_b.json").write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetValidationError) as exc:
        validate_dataset(root)

    message = str(exc.value)

    assert "broken_a.json" in message
    assert "broken_b.json" in message
    assert len(exc.value.diagnostics) == 2
