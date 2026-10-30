"""Independent integrity verification for curated dataset candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from robustlearn.dataset_splits import validate_split_manifest
from robustlearn.dataset_validation import validate_dataset
from robustlearn.rollout_dataset import episode_filename, read_episode


class IntegrityError(ValueError):
    """Candidate dataset or provenance integrity failure."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"Missing or symlinked artifact: {path}")

    return path.read_bytes()


def verify_curated_candidate(
    source_root: Path,
    candidate_root: Path,
    expected_provenance_sha256: str,
) -> dict[str, object]:
    """Verify a candidate against a separately trusted provenance digest."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_provenance_sha256) is None:
        raise IntegrityError("Expected provenance SHA256 must be 64 hex digits")

    if not source_root.is_dir() or not candidate_root.is_dir():
        raise IntegrityError("Source and candidate directories must exist")

    provenance_path = candidate_root / "provenance.json"
    provenance_bytes = _require_file(provenance_path)

    if _sha256(provenance_bytes) != expected_provenance_sha256:
        raise IntegrityError("Trusted provenance SHA256 mismatch")

    manifest = json.loads(provenance_bytes)

    if not isinstance(manifest, dict):
        raise IntegrityError("Provenance must be a JSON object")

    if manifest.get("status") != "CURATED_CANDIDATE_NOT_FROZEN":
        raise IntegrityError("Unexpected candidate status")

    records = manifest.get("episodes")

    if not isinstance(records, list) or not records:
        raise IntegrityError("Provenance must contain episode records")

    if manifest.get("episode_count") != len(records):
        raise IntegrityError("Provenance episode count mismatch")

    episode_root = candidate_root / "episodes"

    if not episode_root.is_dir() or episode_root.is_symlink():
        raise IntegrityError("Curated episode directory missing")

    source_names: set[str] = set()
    curated_names: set[str] = set()
    episode_ids: set[int] = set()

    for record in records:
        if not isinstance(record, dict):
            raise IntegrityError("Invalid episode provenance record")

        source_name = record.get("source_filename")
        curated_name = record.get("curated_filename")
        episode_id = record.get("episode_index")

        if (
            not isinstance(source_name, str)
            or not isinstance(curated_name, str)
            or Path(source_name).name != source_name
            or Path(curated_name).name != curated_name
            or source_name != curated_name
        ):
            raise IntegrityError("Unsafe or inconsistent episode filename")

        if type(episode_id) is not int:
            raise IntegrityError("Invalid episode ID in provenance")

        if (
            source_name in source_names
            or curated_name in curated_names
            or episode_id in episode_ids
        ):
            raise IntegrityError("Duplicate provenance episode identity")

        source_names.add(source_name)
        curated_names.add(curated_name)
        episode_ids.add(episode_id)

        original_path = source_root / source_name
        curated_path = episode_root / curated_name

        original_bytes = _require_file(original_path)
        curated_bytes = _require_file(curated_path)

        if _sha256(original_bytes) != record.get("source_sha256"):
            raise IntegrityError(f"Original checksum mismatch: {source_name}")

        if _sha256(curated_bytes) != record.get("curated_sha256"):
            raise IntegrityError(f"Curated checksum mismatch: {curated_name}")

        original = read_episode(original_path)

        if episode_filename(original) != source_name:
            raise IntegrityError("Original filename does not match episode")

        if original.source != "expert":
            raise IntegrityError("Original episode source is not expert")

        if original.episode_index != episode_id:
            raise IntegrityError("Original episode ID mismatch")

        expected_fields = {
            "seed": original.seed,
            "scenario": original.scenario,
            "task": original.task,
            "success": original.success,
            "frame_count": len(original.frames),
            "source_git_commit": original.git_commit,
            "source_config_id": original.config_id,
            "original_source": "expert",
            "curated_source": "scripted",
            "transformation": "source-label-only",
        }

        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise IntegrityError(
                    f"Provenance mismatch for {source_name}: {field}"
                )

        raw_original = json.loads(original_bytes)
        raw_curated = json.loads(curated_bytes)

        if not isinstance(raw_original, dict):
            raise IntegrityError("Original episode is not an object")

        expected_curated = dict(raw_original)
        expected_curated["source"] = "scripted"

        if raw_curated != expected_curated:
            raise IntegrityError(
                f"Unexpected transformation: {curated_name}"
            )

    actual_source_names = {
        path.name for path in source_root.glob("*.json")
    }
    actual_curated_names = {
        path.name for path in episode_root.glob("*.json")
    }

    if actual_source_names != source_names:
        raise IntegrityError("Original source inventory mismatch")

    if actual_curated_names != curated_names:
        raise IntegrityError("Curated episode inventory mismatch")

    statistics = validate_dataset(episode_root)

    saved_statistics = json.loads(
        _require_file(candidate_root / "statistics.json")
    )

    if saved_statistics != statistics:
        raise IntegrityError("Dataset statistics mismatch")

    split_path = candidate_root / "splits.json"

    if split_path.exists() or split_path.is_symlink():
        splits = json.loads(_require_file(split_path))
        validate_split_manifest(episode_root, splits)

    return {
        "status": "VERIFIED_CANDIDATE_NOT_FROZEN",
        "episodes": statistics["total_episodes"],
        "frames": statistics["total_frames"],
        "scripted": statistics["scripted_episodes"],
        "teleop": statistics["teleop_episodes"],
        "provenance_sha256": expected_provenance_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    """Verify integrity of one curated candidate."""

    parser = argparse.ArgumentParser(
        description="Verify curated episodes against trusted provenance."
    )

    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--expected-provenance-sha256",
        required=True,
    )

    args = parser.parse_args(argv)

    try:
        result = verify_curated_candidate(
            args.source_root,
            args.candidate_root,
            args.expected_provenance_sha256,
        )
    except (ValueError, TypeError, OSError) as exc:
        print(f"INTEGRITY ERROR: {exc}", file=sys.stderr)
        return 1

    print("DATASET CANDIDATE INTEGRITY VERIFIED")
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
