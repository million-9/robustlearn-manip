"""Verify a versioned dataset snapshot against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robustlearn.dataset_splits import (
    SplitError,
    _dataset_fingerprint,
    validate_split_manifest,
)
from robustlearn.dataset_validation import validate_dataset


class FreezeVerificationError(ValueError):
    """Dataset snapshot does not match its manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeVerificationError(message)


def _sha256(path: Path) -> str:
    _require(
        path.is_file() and not path.is_symlink(),
        f"Missing or unsafe file: {path}",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    _sha256(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), f"Invalid JSON object: {path}")
    return data


def verify_dataset_snapshot(root: Path) -> dict:
    """Verify the exact episode and report inventory of a snapshot."""
    _require(
        root.is_dir() and not root.is_symlink(),
        "Dataset directory missing or unsafe",
    )

    drafts = root / "dataset_manifest.draft.json"
    frozen = root / "dataset_manifest.json"

    manifests = [p for p in (drafts, frozen) if p.exists()]
    _require(len(manifests) == 1, "Expected exactly one dataset manifest")

    manifest_path = manifests[0]
    manifest = _load_json(manifest_path)

    _require(manifest.get("dataset_version") == "v0.1", "Version mismatch")
    _require(manifest.get("manifest_version") == 1, "Manifest version mismatch")

    expected_status = (
        "DRAFT_NOT_FROZEN"
        if manifest_path == drafts
        else "FROZEN"
    )
    _require(manifest.get("status") == expected_status, "Status mismatch")

    reports = ("statistics.json", "splits.json", "provenance.json")

    expected_root = {"episodes", manifest_path.name, *reports}
    actual_root = {p.name for p in root.iterdir()}
    _require(actual_root == expected_root, "Unexpected root artifact inventory")

    episodes_dir = root / "episodes"
    _require(
        episodes_dir.is_dir() and not episodes_dir.is_symlink(),
        "Episode directory missing or unsafe",
    )

    episode_files = sorted(episodes_dir.iterdir())

    for path in episode_files:
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.name.startswith("episode_")
            and path.suffix == ".json",
            f"Unexpected episode artifact: {path.name}",
        )

    declared = manifest.get("episode_sha256")
    _require(isinstance(declared, dict), "Missing episode checksums")

    actual_names = {p.name for p in episode_files}
    _require(
        actual_names == set(declared),
        "Episode inventory differs from manifest",
    )

    for path in episode_files:
        _require(
            _sha256(path) == declared[path.name],
            f"Episode checksum mismatch: {path.name}",
        )

    report_hashes = manifest.get("artifact_sha256")
    _require(isinstance(report_hashes, dict), "Missing report checksums")
    _require(set(report_hashes) == set(reports), "Report inventory mismatch")

    for name in reports:
        _require(
            _sha256(root / name) == report_hashes[name],
            f"Report checksum mismatch: {name}",
        )

    fingerprint = _dataset_fingerprint(episode_files)

    _require(
        fingerprint == manifest.get("dataset_sha256"),
        "Dataset fingerprint mismatch",
    )

    stats = _load_json(root / "statistics.json")
    splits = _load_json(root / "splits.json")
    provenance = _load_json(root / "provenance.json")

    _require(
        validate_dataset(episodes_dir) == stats,
        "Saved statistics do not match dataset",
    )

    expected_fields = {
        "schema_version": stats["schema_version"],
        "episode_count": stats["total_episodes"],
        "frame_count": stats["total_frames"],
        "scripted_episode_count": stats["scripted_episodes"],
        "teleop_episode_count": stats["teleop_episodes"],
        "success_count": stats["successes"],
        "failure_count": stats["failures"],
        "scenario_distribution": stats["scenarios"],
        "task_distribution": stats["tasks"],
        "source_distribution": stats["sources"],
    }

    for key, expected in expected_fields.items():
        _require(manifest.get(key) == expected, f"Metadata mismatch: {key}")

    _require(
        splits.get("dataset_sha256") == fingerprint,
        "Split dataset fingerprint mismatch",
    )
    _require(
        manifest.get("canonical_split_manifest") == "splits.json",
        "Split manifest reference mismatch",
    )
    _require(
        manifest.get("split_dataset_reference")
        == splits.get("dataset_reference"),
        "Split dataset reference mismatch",
    )

    try:
        validate_split_manifest(episodes_dir, splits)
    except SplitError as exc:
        raise FreezeVerificationError(
            f"Canonical split validation failed: {exc}"
        ) from exc

    records = provenance.get("episodes")
    _require(isinstance(records, list), "Missing provenance records")
    _require(
        len(records) == len(episode_files),
        "Provenance episode count mismatch",
    )
    _require(
        provenance.get("episode_count") == len(records),
        "Provenance count mismatch",
    )

    provenance_names = set()
    for record in records:
        _require(isinstance(record, dict), "Invalid provenance record")
        name = record.get("curated_filename")
        _require(name in actual_names, "Unknown provenance episode")
        _require(name not in provenance_names, "Duplicate provenance episode")
        _require(
            record.get("curated_sha256") == declared[name],
            f"Provenance checksum mismatch: {name}",
        )
        provenance_names.add(name)

    _require(
        provenance_names == actual_names,
        "Incomplete provenance coverage",
    )

    expected_configs = sorted({r["source_config_id"] for r in records})
    expected_commits = sorted({r["source_git_commit"] for r in records})
    expected_seeds = sorted({r["seed"] for r in records})

    _require(
        manifest.get("source_config_ids") == expected_configs,
        "Source configurations mismatch",
    )
    _require(
        manifest.get("source_git_commits") == expected_commits,
        "Source commits mismatch",
    )
    _require(
        manifest.get("seeds") == expected_seeds,
        "Recorded seeds mismatch",
    )

    return {
        "dataset_version": manifest["dataset_version"],
        "status": manifest["status"],
        "episodes": len(episode_files),
        "frames": stats["total_frames"],
        "dataset_sha256": fingerprint,
        "verified_episode_checksums": len(episode_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a versioned dataset snapshot.",
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    try:
        result = verify_dataset_snapshot(args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"DATASET SNAPSHOT VERIFICATION FAILED: {exc}")
        return 1

    print("DATASET SNAPSHOT VERIFICATION PASSED")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
