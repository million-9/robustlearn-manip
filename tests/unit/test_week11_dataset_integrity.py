"""Regression tests for independent candidate integrity verification."""

import hashlib
import json

import pytest

from robustlearn.dataset_curation import curate_expert_episodes
from robustlearn.dataset_integrity import (
    IntegrityError,
    verify_curated_candidate,
)
from robustlearn.dataset_splits import build_split_manifest
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    write_episode,
)


@pytest.fixture
def candidate(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "candidate"

    for index in range(3):
        frame = RolloutFrame(
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
        )

        episode = RolloutEpisode(
            episode_index=index,
            seed=index,
            scenario=f"scenario_{index}",
            source="expert",
            task="peg_insertion",
            success=True,
            failure_reason=None,
            start_timestamp_s=0.0,
            end_timestamp_s=2.0,
            git_commit="test-revision",
            config_id="test-config",
            frames=(frame,),
        )

        write_episode(source, episode)

    curate_expert_episodes(source, output)

    splits = build_split_manifest(
        output / "episodes",
        dataset_reference="integrity-test",
    )

    (output / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True) + "\n"
    )

    digest = hashlib.sha256(
        (output / "provenance.json").read_bytes()
    ).hexdigest()

    return source, output, digest


def test_valid_candidate_passes(candidate):
    source, output, digest = candidate

    result = verify_curated_candidate(source, output, digest)

    assert result["episodes"] == 3
    assert result["frames"] == 3
    assert result["scripted"] == 3
    assert result["status"] == "VERIFIED_CANDIDATE_NOT_FROZEN"


def test_wrong_trusted_digest_is_rejected(candidate):
    source, output, _ = candidate

    with pytest.raises(IntegrityError, match="provenance SHA256"):
        verify_curated_candidate(source, output, "0" * 64)


def test_modified_original_is_detected(candidate):
    source, output, digest = candidate
    path = sorted(source.glob("*.json"))[0]

    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(IntegrityError, match="Original checksum"):
        verify_curated_candidate(source, output, digest)


def test_modified_curated_episode_is_detected(candidate):
    source, output, digest = candidate
    path = sorted((output / "episodes").glob("*.json"))[0]

    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(IntegrityError, match="Curated checksum"):
        verify_curated_candidate(source, output, digest)


def test_modified_provenance_is_detected(candidate):
    source, output, digest = candidate
    path = output / "provenance.json"

    data = json.loads(path.read_text())
    data["status"] = "FROZEN"
    path.write_text(json.dumps(data))

    with pytest.raises(IntegrityError, match="provenance SHA256"):
        verify_curated_candidate(source, output, digest)


def test_modified_statistics_are_detected(candidate):
    source, output, digest = candidate

    (output / "statistics.json").write_text("{}")

    with pytest.raises(IntegrityError, match="statistics mismatch"):
        verify_curated_candidate(source, output, digest)


def test_modified_split_manifest_is_detected(candidate):
    source, output, digest = candidate
    path = output / "splits.json"

    data = json.loads(path.read_text())
    data["seed"] += 1
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="manifest does not match"):
        verify_curated_candidate(source, output, digest)


def test_unlisted_episode_is_detected(candidate):
    source, output, digest = candidate

    path = sorted((output / "episodes").glob("*.json"))[0]
    (output / "episodes" / "extra.json").write_bytes(path.read_bytes())

    with pytest.raises(IntegrityError, match="inventory mismatch"):
        verify_curated_candidate(source, output, digest)
