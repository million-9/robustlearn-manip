"""Regenerate metadata for an already-curated mixed v0.1 snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

from robustlearn.dataset_freeze import verify_dataset_snapshot
from robustlearn.dataset_splits import (
    _dataset_fingerprint,
    build_split_manifest,
    validate_split_manifest,
)
from robustlearn.dataset_validation import validate_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def regenerate(root: Path) -> None:
    root = root.resolve()
    episodes_dir = root / "episodes"

    if not episodes_dir.is_dir():
        raise ValueError("Episode directory missing")

    paths = sorted(episodes_dir.glob("*.json"))

    if len(paths) != 94:
        raise ValueError(f"Expected 94 episodes, found {len(paths)}")

    manifest_path = root / "dataset_manifest.draft.json"
    manifest = load(manifest_path)
    provenance = load(root / "provenance.json")

    print("=== VERIFY EXISTING PROVENANCE ===")

    records = provenance["episodes"]

    if len(records) != 94:
        raise ValueError("Expected 94 provenance records")

    hashes = {path.name: sha256(path) for path in paths}
    record_names = [r["curated_filename"] for r in records]

    if len(set(record_names)) != len(record_names):
        raise ValueError("Duplicate provenance records")

    if set(record_names) != set(hashes):
        raise ValueError("Provenance inventory mismatch")

    for record in records:
        name = record["curated_filename"]

        if record["curated_sha256"] != hashes[name]:
            raise ValueError(f"Provenance checksum mismatch: {name}")

    print("PASS: All 94 provenance records verified")

    print("\n=== REGENERATE STATISTICS ===")

    stats = validate_dataset(episodes_dir)

    expected = {
        "total_episodes": 94,
        "scripted_episodes": 93,
        "teleop_episodes": 1,
        "total_frames": 3727,
        "successes": 94,
        "failures": 0,
    }

    for key, value in expected.items():
        if stats[key] != value:
            raise ValueError(
                f"Unexpected {key}: {stats[key]}, expected {value}"
            )

    print("PASS: Statistics")

    print("\n=== REGENERATE SPLITS ===")

    splits = build_split_manifest(
        episodes_dir,
        dataset_reference="week11-v01-94-mixed-candidate-not-frozen",
        seed=42,
        group_by="scenario_initial",
    )

    validate_split_manifest(episodes_dir, splits)
    print("PASS: Deterministic splits")

    print("\n=== REGENERATE MANIFEST ===")

    reports = {
        "statistics.json": stats,
        "splits.json": splits,
        "provenance.json": provenance,
    }

    report_bytes = {
        name: serialize(data)
        for name, data in reports.items()
    }

    manifest.update({
        "schema_version": stats["schema_version"],
        "dataset_sha256": _dataset_fingerprint(paths),
        "episode_sha256": hashes,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in report_bytes.items()
        },
        "episode_count": stats["total_episodes"],
        "frame_count": stats["total_frames"],
        "scripted_episode_count": stats["scripted_episodes"],
        "teleop_episode_count": stats["teleop_episodes"],
        "success_count": stats["successes"],
        "failure_count": stats["failures"],
        "scenario_distribution": stats["scenarios"],
        "task_distribution": stats["tasks"],
        "source_distribution": stats["sources"],
        "source_config_ids": sorted({
            r["source_config_id"] for r in records
        }),
        "source_git_commits": sorted({
            r["source_git_commit"] for r in records
        }),
        "seeds": sorted({r["seed"] for r in records}),
        "split_dataset_reference": splits["dataset_reference"],
        "canonical_split_manifest": "splits.json",
        "status": "DRAFT_NOT_FROZEN",
    })

    for name, payload in report_bytes.items():
        (root / name).write_bytes(payload)

    manifest_path.write_bytes(serialize(manifest))

    result = verify_dataset_snapshot(root)

    print("\n=== VERIFIED MIXED DATASET ===")
    print(json.dumps(result, indent=2))
    print("PASS: Self-contained metadata regeneration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Existing curated mixed-source dataset directory",
    )
    args = parser.parse_args()
    regenerate(args.root)


if __name__ == "__main__":
    main()
