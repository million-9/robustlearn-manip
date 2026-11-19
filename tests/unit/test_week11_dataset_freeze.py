"""Regression tests for dataset snapshot verification."""

import hashlib
import json

import pytest

from robustlearn import dataset_freeze
from robustlearn.dataset_freeze import (
    FreezeVerificationError,
    verify_dataset_snapshot,
)
from robustlearn.dataset_splits import SplitError, _dataset_fingerprint


def _write_json(path, data):
    path.write_text(
        json.dumps(data, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Construct a small, isolated draft dataset snapshot."""
    root = tmp_path / "v0.1_staging"
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True)

    records = []
    hashes = {}

    for index in range(2):
        name = f"episode_{index:06d}__seed_{index}__test.json"
        path = episodes_dir / name

        _write_json(
            path,
            {
                "episode_index": index,
                "source": "scripted",
                "seed": index,
            },
        )

        digest = _sha256(path)
        hashes[name] = digest

        records.append(
            {
                "curated_filename": name,
                "curated_sha256": digest,
                "source_config_id": "test-config",
                "source_git_commit": "test-commit",
                "seed": index,
            }
        )

    fingerprint = _dataset_fingerprint(
        sorted(episodes_dir.glob("episode_*.json"))
    )

    stats = {
        "schema_version": 1,
        "total_episodes": 2,
        "total_frames": 2,
        "scripted_episodes": 2,
        "teleop_episodes": 0,
        "successes": 2,
        "failures": 0,
        "scenarios": {"test": 2},
        "tasks": {"peg_insertion": 2},
        "sources": {"scripted": 2},
    }

    splits = {
        "dataset_reference": "test-candidate",
        "dataset_sha256": fingerprint,
    }

    provenance = {
        "episode_count": 2,
        "episodes": records,
    }

    reports = {
        "statistics.json": stats,
        "splits.json": splits,
        "provenance.json": provenance,
    }

    for name, data in reports.items():
        _write_json(root / name, data)

    manifest = {
        "manifest_version": 1,
        "dataset_version": "v0.1",
        "status": "DRAFT_NOT_FROZEN",
        "schema_version": 1,
        "episode_count": 2,
        "frame_count": 2,
        "scripted_episode_count": 2,
        "teleop_episode_count": 0,
        "success_count": 2,
        "failure_count": 0,
        "scenario_distribution": {"test": 2},
        "task_distribution": {"peg_insertion": 2},
        "source_distribution": {"scripted": 2},
        "source_config_ids": ["test-config"],
        "source_git_commits": ["test-commit"],
        "seeds": [0, 1],
        "canonical_split_manifest": "splits.json",
        "split_dataset_reference": "test-candidate",
        "dataset_sha256": fingerprint,
        "artifact_sha256": {
            name: _sha256(root / name)
            for name in reports
        },
        "episode_sha256": hashes,
    }

    _write_json(
        root / "dataset_manifest.draft.json",
        manifest,
    )

    # These unit tests isolate the freeze verifier's inventory,
    # checksum, and metadata logic. Canonical dataset validation
    # is exercised separately by its own tests.
    monkeypatch.setattr(
        dataset_freeze,
        "validate_dataset",
        lambda _: stats,
    )

    # This synthetic fixture has a minimal split report.
    # The canonical split algorithm has its own regression tests.
    def check_split_fixture(dataset_root, manifest):
        assert dataset_root == episodes_dir
        assert manifest == splits

    monkeypatch.setattr(
        dataset_freeze,
        "validate_split_manifest",
        check_split_fixture,
    )

    return root


def test_valid_draft_snapshot(snapshot):
    result = verify_dataset_snapshot(snapshot)

    assert result["status"] == "DRAFT_NOT_FROZEN"
    assert result["episodes"] == 2
    assert result["frames"] == 2
    assert result["verified_episode_checksums"] == 2


def test_modified_episode_rejected(snapshot):
    path = next((snapshot / "episodes").glob("*.json"))
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        FreezeVerificationError,
        match="Episode checksum mismatch",
    ):
        verify_dataset_snapshot(snapshot)


def test_missing_episode_rejected(snapshot):
    path = next((snapshot / "episodes").glob("*.json"))
    path.unlink()

    with pytest.raises(
        FreezeVerificationError,
        match="Episode inventory differs",
    ):
        verify_dataset_snapshot(snapshot)


def test_extra_episode_rejected(snapshot):
    path = snapshot / "episodes" / "episode_999999.json"
    _write_json(path, {"unexpected": True})

    with pytest.raises(
        FreezeVerificationError,
        match="Episode inventory differs",
    ):
        verify_dataset_snapshot(snapshot)


def test_modified_statistics_rejected(snapshot):
    path = snapshot / "statistics.json"
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        FreezeVerificationError,
        match="Report checksum mismatch",
    ):
        verify_dataset_snapshot(snapshot)


def test_modified_split_report_rejected(snapshot):
    path = snapshot / "splits.json"
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        FreezeVerificationError,
        match="Report checksum mismatch",
    ):
        verify_dataset_snapshot(snapshot)


def test_manifest_metadata_mismatch_rejected(snapshot):
    path = snapshot / "dataset_manifest.draft.json"
    manifest = json.loads(path.read_text())
    manifest["frame_count"] = 999
    _write_json(path, manifest)

    with pytest.raises(
        FreezeVerificationError,
        match="Metadata mismatch: frame_count",
    ):
        verify_dataset_snapshot(snapshot)


def test_unexpected_root_artifact_rejected(snapshot):
    (snapshot / "unexpected.txt").write_text("extra")

    with pytest.raises(
        FreezeVerificationError,
        match="Unexpected root artifact inventory",
    ):
        verify_dataset_snapshot(snapshot)


def test_manifest_status_mismatch_rejected(snapshot):
    path = snapshot / "dataset_manifest.draft.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "FROZEN"
    _write_json(path, manifest)

    with pytest.raises(
        FreezeVerificationError,
        match="Status mismatch",
    ):
        verify_dataset_snapshot(snapshot)


def test_canonical_split_failure_rejected(snapshot, monkeypatch):
    """The freeze verifier must reject canonical split failures."""

    def reject_invalid_splits(_root, _manifest):
        raise SplitError("deliberate split assignment mismatch")

    monkeypatch.setattr(
        dataset_freeze,
        "validate_split_manifest",
        reject_invalid_splits,
    )

    with pytest.raises(
        FreezeVerificationError,
        match="Canonical split validation failed",
    ):
        verify_dataset_snapshot(snapshot)


def test_valid_frozen_snapshot(snapshot):
    """A properly promoted frozen snapshot must verify."""
    draft = snapshot / "dataset_manifest.draft.json"
    frozen = snapshot / "dataset_manifest.json"

    manifest = json.loads(draft.read_text())
    manifest["status"] = "FROZEN"

    _write_json(frozen, manifest)
    draft.unlink()

    result = verify_dataset_snapshot(snapshot)

    assert result["dataset_version"] == "v0.1"
    assert result["status"] == "FROZEN"
    assert result["episodes"] == 2


def test_duplicate_manifests_rejected(snapshot):
    """Draft and frozen manifests must not coexist."""
    draft = snapshot / "dataset_manifest.draft.json"
    frozen = snapshot / "dataset_manifest.json"

    frozen.write_bytes(draft.read_bytes())

    with pytest.raises(
        FreezeVerificationError,
        match="Expected exactly one dataset manifest",
    ):
        verify_dataset_snapshot(snapshot)


def test_frozen_manifest_wrong_status_rejected(snapshot):
    """Renaming a draft alone must not complete a freeze."""
    draft = snapshot / "dataset_manifest.draft.json"
    frozen = snapshot / "dataset_manifest.json"

    draft.rename(frozen)

    with pytest.raises(
        FreezeVerificationError,
        match="Status mismatch",
    ):
        verify_dataset_snapshot(snapshot)
