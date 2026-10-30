"""Test atomic and isolated collection-summary file output."""

import json

import pytest

import robustlearn.teleop_collection_summary as summary
from robustlearn.teleop_collection_summary import (
    TeleopAttemptSummary,
    teleop_collection_summary_path,
    write_teleop_collection_summary,
)


def attempt(outcome="rejected"):
    return TeleopAttemptSummary(
        episode_index=12,
        seed=42,
        scenario="nominal",
        task="panda_insertion",
        git_commit="test-commit",
        config_id="keyboard-teleop-test",
        outcome=outcome,
        frame_count=1,
        start_timestamp_s=10.0,
        end_timestamp_s=10.0,
        episode_file=None,
        failure_reason=None,
        error=None,
    )


def test_summary_written_outside_dataset(tmp_path):
    output_root = tmp_path / "collection" / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    assert path.parent == output_root.parent
    assert path.name == (
        "episodes__teleop_collection_summary.json"
    )

    assert not output_root.exists()
    assert path.exists()


def test_written_summary_is_valid_json(tmp_path):
    output_root = tmp_path / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["source"] == "teleop"
    assert data["total_attempts"] == 1
    assert data["saved_episodes"] == 0
    assert data["outcome_counts"]["rejected"] == 1


def test_empty_collection_is_valid(tmp_path):
    path = write_teleop_collection_summary(
        [],
        output_root=tmp_path / "episodes",
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_attempts"] == 0
    assert data["saved_episodes"] == 0


def test_existing_summary_is_replaced_atomically(tmp_path):
    output_root = tmp_path / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    write_teleop_collection_summary(
        [attempt(), attempt()],
        output_root=output_root,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_attempts"] == 2
    assert data["outcome_counts"]["rejected"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_replace_preserves_previous_summary(
    tmp_path,
    monkeypatch,
):
    output_root = tmp_path / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    original_content = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(
        summary.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(OSError):
        write_teleop_collection_summary(
            [attempt(), attempt()],
            output_root=output_root,
        )

    assert path.read_bytes() == original_content
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_report_does_not_modify_existing_summary(
    tmp_path,
):
    output_root = tmp_path / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    original_content = path.read_bytes()

    with pytest.raises(TypeError):
        write_teleop_collection_summary(
            [{"outcome": "accepted_success"}],
            output_root=output_root,
        )

    assert path.read_bytes() == original_content


def test_replacement_removes_temporary_file(tmp_path):
    output_root = tmp_path / "episodes"

    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output_root,
    )

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_relative_output_root_is_rejected():
    with pytest.raises(ValueError):
        teleop_collection_summary_path(
            __import__("pathlib").Path("relative/episodes")
        )


def test_dataset_path_is_never_used_for_summary(tmp_path):
    output_root = tmp_path / "episodes"

    path = teleop_collection_summary_path(output_root)

    assert path != output_root
    assert output_root not in path.parents


def test_summary_write_does_not_create_episode_files(tmp_path):
    output_root = tmp_path / "episodes"

    write_teleop_collection_summary(
        [attempt("rejected")],
        output_root=output_root,
    )

    assert not list(output_root.glob("*.json"))
