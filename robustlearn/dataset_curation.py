"""Auditable conversion of raw scripted-expert episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from robustlearn.dataset_validation import validate_dataset
from robustlearn.rollout_dataset import episode_filename, read_episode


class CurationError(ValueError):
    """Invalid or unsafe dataset curation request."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _serialize(data: object) -> bytes:
    return (
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def curate_expert_episodes(
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Curate validated raw expert episodes without modifying originals."""

    if not source_root.is_dir():
        raise CurationError(f"Source directory not found: {source_root}")

    if output_root.exists() or output_root.is_symlink():
        raise CurationError(
            f"Output already exists; refusing overwrite: {output_root}"
        )

    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()

    if output_resolved.is_relative_to(source_resolved):
        raise CurationError("Output must be outside the source directory")

    paths = sorted(source_root.glob("*.json"))

    if not paths:
        raise CurationError("No source episode files found")

    records: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    seen_ids: set[int] = set()

    # Verify all source episodes before writing anything.
    for path in paths:
        if path.is_symlink():
            raise CurationError(f"Symlinked source is not allowed: {path}")

        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)

        if not isinstance(raw, dict):
            raise CurationError(f"Not an episode object: {path}")

        episode = read_episode(path)

        if path.name != episode_filename(episode):
            raise CurationError(f"Noncanonical episode filename: {path.name}")

        if episode.source != "expert":
            raise CurationError(
                f"{path.name}: expected raw source='expert', "
                f"found {episode.source!r}"
            )

        if episode.episode_index in seen_ids:
            raise CurationError(
                f"Duplicate episode ID: {episode.episode_index}"
            )

        seen_ids.add(episode.episode_index)

        # Preserve all original JSON fields and numerical values.
        # Change only the source classification.
        curated = dict(raw)
        curated["source"] = "scripted"

        curated_bytes = _serialize(curated)

        payloads.append((path.name, curated_bytes))

        records.append(
            {
                "episode_index": episode.episode_index,
                "seed": episode.seed,
                "scenario": episode.scenario,
                "task": episode.task,
                "success": episode.success,
                "frame_count": len(episode.frames),
                "source_filename": path.name,
                "source_sha256": _sha256(raw_bytes),
                "source_git_commit": episode.git_commit,
                "source_config_id": episode.config_id,
                "original_source": "expert",
                "curated_source": "scripted",
                "curated_filename": path.name,
                "curated_sha256": _sha256(curated_bytes),
                "transformation": "source-label-only",
            }
        )

    provenance: dict[str, object] = {
        "manifest_version": 1,
        "status": "CURATED_CANDIDATE_NOT_FROZEN",
        "source_format": "week11_scripted_expert",
        "curated_format": "canonical_scripted",
        "episode_count": len(records),
        "episodes": records,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Stage the entire candidate before publishing it.
    with tempfile.TemporaryDirectory(
        prefix=".week11-curation-",
        dir=output_root.parent,
    ) as temporary:
        staging = Path(temporary) / "candidate"
        episodes_dir = staging / "episodes"
        episodes_dir.mkdir(parents=True)

        for name, payload in payloads:
            (episodes_dir / name).write_bytes(payload)

        statistics = validate_dataset(episodes_dir)

        if statistics["total_episodes"] != len(records):
            raise CurationError("Curated episode count mismatch")

        (staging / "statistics.json").write_bytes(
            _serialize(statistics)
        )

        (staging / "provenance.json").write_bytes(
            _serialize(provenance)
        )

        if output_root.exists() or output_root.is_symlink():
            raise CurationError("Output appeared during curation")

        staging.rename(output_root)

    return statistics


def main(argv: list[str] | None = None) -> int:
    """Run the raw expert -> canonical scripted conversion."""

    parser = argparse.ArgumentParser(
        description="Curate raw scripted-expert episodes with provenance."
    )

    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        stats = curate_expert_episodes(
            args.source_root,
            args.output_root,
        )
    except (ValueError, TypeError, OSError) as exc:
        print(f"CURATION ERROR: {exc}", file=sys.stderr)
        return 1

    print("DATASET CURATION PASSED")
    print(f"Candidate: {args.output_root}")
    print(f"Episodes: {stats['total_episodes']}")
    print(f"Frames: {stats['total_frames']}")
    print(f"Scripted: {stats['scripted_episodes']}")
    print("Status: CURATED_CANDIDATE_NOT_FROZEN")

    return 0


if __name__ == "__main__":
    sys.exit(main())
