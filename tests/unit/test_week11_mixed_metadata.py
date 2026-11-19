"""Regression tests for Week 11 mixed-source metadata generation."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from robustlearn.dataset_freeze import verify_dataset_snapshot
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    write_episode,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "build_week11_mixed_metadata.py"
)

spec = importlib.util.spec_from_file_location(
    "week11_mixed_metadata",
    SCRIPT,
)
assert spec is not None and spec.loader is not None

metadata_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata_tool)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def candidate_template(tmp_path_factory):
    """Build a complete synthetic 94-episode mixed dataset."""
    root = tmp_path_factory.mktemp("week11_mixed_template")
    episodes_dir = root / "episodes"
    episodes_dir.mkdir()

    for index in range(94):
        scripted = index < 93
        frame_count = 40 if scripted else 7

        frames = tuple(
            RolloutFrame(
                frame_index=frame_index,
                timestamp_s=float(frame_index + 1),
                joint_position=(0.001 * (index + 1),) * 7,
                joint_velocity=(0.0,) * 7,
                task_relative_state=(0.0, 0.0, 0.012),
                wrist_wrench=(0.0,) * 6,
                expert_action=(0.2,) * 7,
                task_success=frame_index == frame_count - 1,
                task_failure=False,
                terminated=frame_index == frame_count - 1,
            )
            for frame_index in range(frame_count)
        )

        episode = RolloutEpisode(
            episode_index=1000 + index if scripted else 2004,
            seed=1000 + index if scripted else 2004,
            scenario=f"test_scenario_{index:03d}",
            source="scripted" if scripted else "teleop",
            task="peg_insertion" if scripted else "panda_insertion",
            success=True,
            failure_reason=None,
            start_timestamp_s=0.0,
            end_timestamp_s=float(frame_count + 1),
            git_commit="test-revision",
            config_id="test-collection",
            frames=frames,
        )

        write_episode(episodes_dir, episode)

    paths = sorted(episodes_dir.glob("*.json"))
    assert len(paths) == 94

    records = []

    for path in paths:
        episode = json.loads(path.read_text())
        checksum = _sha256(path)

        records.append({
            "curated_filename": path.name,
            "curated_sha256": checksum,
            "curated_source": episode["source"],
            "episode_index": episode["episode_index"],
            "frame_count": len(episode["frames"]),
            "original_source": episode["source"],
            "scenario": episode["scenario"],
            "seed": episode["seed"],
            "source_config_id": episode["config_id"],
            "source_filename": path.name,
            "source_git_commit": episode["git_commit"],
            "source_sha256": checksum,
            "success": episode["success"],
            "task": episode["task"],
            "transformation": "synthetic-test-fixture",
        })

    _write_json(
        root / "provenance.json",
        {
            "manifest_version": 1,
            "episode_count": len(records),
            "episodes": records,
            "status": "CURATED_CANDIDATE_NOT_FROZEN",
        },
    )

    _write_json(
        root / "dataset_manifest.draft.json",
        {
            "manifest_version": 1,
            "dataset_version": "v0.1",
            "status": "DRAFT_NOT_FROZEN",
        },
    )

    return root


@pytest.fixture
def candidate(candidate_template, tmp_path):
    root = tmp_path / "mixed_candidate"
    shutil.copytree(candidate_template, root)
    return root


def test_regeneration_produces_valid_snapshot(candidate):
    metadata_tool.regenerate(candidate)

    result = verify_dataset_snapshot(candidate)

    assert result["episodes"] == 94
    assert result["frames"] == 3727
    assert result["verified_episode_checksums"] == 94
    assert result["status"] == "DRAFT_NOT_FROZEN"


def test_regeneration_is_deterministic(candidate):
    metadata_tool.regenerate(candidate)

    report_names = (
        "statistics.json",
        "splits.json",
        "provenance.json",
        "dataset_manifest.draft.json",
    )

    before = {
        name: _sha256(candidate / name)
        for name in report_names
    }

    metadata_tool.regenerate(candidate)

    after = {
        name: _sha256(candidate / name)
        for name in report_names
    }

    assert before == after


def test_corrupted_provenance_is_rejected(candidate):
    path = candidate / "provenance.json"
    provenance = json.loads(path.read_text())

    provenance["episodes"][0]["curated_sha256"] = "0" * 64
    _write_json(path, provenance)

    with pytest.raises(ValueError, match="Provenance checksum"):
        metadata_tool.regenerate(candidate)


def test_missing_episode_is_rejected(candidate):
    episode = sorted((candidate / "episodes").glob("*.json"))[0]
    episode.unlink()

    with pytest.raises(ValueError, match="Expected 94 episodes"):
        metadata_tool.regenerate(candidate)


def test_modified_episode_is_rejected(candidate):
    episode = sorted((candidate / "episodes").glob("*.json"))[0]
    episode.write_bytes(episode.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Provenance checksum"):
        metadata_tool.regenerate(candidate)


def test_missing_provenance_record_is_rejected(candidate):
    path = candidate / "provenance.json"
    provenance = json.loads(path.read_text())

    provenance["episodes"].pop()
    provenance["episode_count"] = 93
    _write_json(path, provenance)

    with pytest.raises(ValueError, match="Expected 94 provenance"):
        metadata_tool.regenerate(candidate)
