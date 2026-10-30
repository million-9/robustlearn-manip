"""Tests for provenance-preserving scripted dataset curation."""

import hashlib
import json

import pytest

from robustlearn.dataset_curation import (
    CurationError,
    curate_expert_episodes,
)
from robustlearn.dataset_validation import validate_dataset
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    write_episode,
)


def raw_episode(index=0, source="expert", scenario="nominal"):
    frames = (
        RolloutFrame(
            frame_index=0,
            timestamp_s=1.0,
            joint_position=(0.1,) * 7,
            joint_velocity=(0.0,) * 7,
            task_relative_state=(0.0, 0.0, 0.01),
            wrist_wrench=(0.0,) * 6,
            expert_action=(0.2,) * 7,
            task_success=True,
            task_failure=False,
            terminated=True,
        ),
    )

    return RolloutEpisode(
        episode_index=index,
        seed=index,
        scenario=scenario,
        source=source,
        task="peg_insertion",
        success=True,
        failure_reason=None,
        start_timestamp_s=0.0,
        end_timestamp_s=2.0,
        git_commit="recorded-commit-dirty",
        config_id="test-configuration",
        frames=frames,
    )


def test_curation_preserves_original_data_and_provenance(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    original = write_episode(source, raw_episode())
    original_bytes = original.read_bytes()

    stats = curate_expert_episodes(source, output)

    assert stats["total_episodes"] == 1
    assert stats["scripted_episodes"] == 1
    assert original.read_bytes() == original_bytes

    curated = output / "episodes" / original.name

    raw_data = json.loads(original_bytes)
    curated_data = json.loads(curated.read_bytes())

    assert raw_data["source"] == "expert"
    assert curated_data["source"] == "scripted"

    curated_data["source"] = "expert"
    assert curated_data == raw_data

    provenance = json.loads(
        (output / "provenance.json").read_text()
    )

    record = provenance["episodes"][0]

    assert record["source_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()

    assert record["curated_sha256"] == hashlib.sha256(
        curated.read_bytes()
    ).hexdigest()

    assert record["source_git_commit"] == "recorded-commit-dirty"
    assert record["transformation"] == "source-label-only"

    assert validate_dataset(output / "episodes") == stats


def test_curation_preserves_multiple_episode_identities(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    write_episode(source, raw_episode(1, scenario="scenario_a"))
    write_episode(source, raw_episode(2, scenario="scenario_b"))

    stats = curate_expert_episodes(source, output)

    assert stats["total_episodes"] == 2
    assert len(list((output / "episodes").glob("*.json"))) == 2


def test_duplicate_episode_ids_are_rejected(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    write_episode(source, raw_episode(1, scenario="scenario_a"))
    write_episode(source, raw_episode(1, scenario="scenario_b"))

    with pytest.raises(CurationError, match="Duplicate episode ID"):
        curate_expert_episodes(source, output)

    assert not output.exists()


def test_nonexpert_source_is_rejected(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    write_episode(source, raw_episode(source="teleop"))

    with pytest.raises(CurationError, match="source='expert'"):
        curate_expert_episodes(source, output)

    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    write_episode(source, raw_episode())

    curate_expert_episodes(source, output)

    before = (output / "provenance.json").read_bytes()

    with pytest.raises(CurationError, match="already exists"):
        curate_expert_episodes(source, output)

    assert (output / "provenance.json").read_bytes() == before


def test_corrupted_source_cannot_publish_candidate(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "curated"

    path = write_episode(source, raw_episode())

    data = json.loads(path.read_text())
    data["frames"][0]["expert_action"] = [float("nan")] * 7
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        curate_expert_episodes(source, output)

    assert not output.exists()


def test_output_inside_source_is_rejected(tmp_path):
    source = tmp_path / "raw"
    write_episode(source, raw_episode())

    with pytest.raises(CurationError, match="outside"):
        curate_expert_episodes(source, source / "curated")
