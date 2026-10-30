"""Validate canonical rollout datasets and report reproducible statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean, median, pstdev

from robustlearn.rollout_dataset import (
    ROLLOUT_SCHEMA_VERSION,
    RolloutEpisode,
    episode_from_dict,
)

SOURCES = ("scripted", "teleop")

EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_index",
        "seed",
        "scenario",
        "source",
        "task",
        "success",
        "failure_reason",
        "start_timestamp_s",
        "end_timestamp_s",
        "git_commit",
        "config_id",
        "frames",
    }
)

FRAME_FIELDS = frozenset(
    {
        "frame_index",
        "timestamp_s",
        "joint_position",
        "joint_velocity",
        "task_relative_state",
        "wrist_wrench",
        "expert_action",
        "task_success",
        "task_failure",
        "terminated",
    }
)

VECTOR_SIZES = {
    "joint_position": 7,
    "joint_velocity": 7,
    "task_relative_state": 3,
    "wrist_wrench": 6,
    "expert_action": 7,
}


class DatasetValidationError(ValueError):
    """Dataset validation failure containing actionable diagnostics."""

    def __init__(self, diagnostics: list[str]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(diagnostics))


def _load_episode(path: Path) -> RolloutEpisode:
    """Parse and validate one complete canonical episode."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("episode must be a JSON object")

    missing = sorted(EPISODE_FIELDS - raw.keys())
    if missing:
        raise ValueError(f"missing required episode fields: {missing}")

    source = raw["source"]
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError(f"invalid demonstration source: {source!r}")

    frames_raw = raw["frames"]
    if not isinstance(frames_raw, list):
        raise ValueError("frames must be an array")

    if not frames_raw:
        raise ValueError("episode has no frames")

    for index, record in enumerate(frames_raw):
        if not isinstance(record, dict):
            raise ValueError(f"frame[{index}] must be an object")

        missing_frame = sorted(FRAME_FIELDS - record.keys())
        if missing_frame:
            raise ValueError(
                f"frame[{index}] missing required fields: {missing_frame}"
            )

    episode = episode_from_dict(raw)

    if any(frame.terminated for frame in episode.frames[:-1]):
        raise ValueError("non-final frame cannot be terminated")

    final = episode.frames[-1]

    if not final.terminated:
        raise ValueError("last frame must be terminated")

    if final.task_success != episode.success:
        raise ValueError("terminal success flag disagrees with episode")

    if final.task_failure != (not episode.success):
        raise ValueError("terminal failure flag disagrees with episode")

    return episode


def _counts(values: Iterable[str]) -> dict[str, int]:
    """Return stable, sorted frequency counts."""
    counter = Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def _vector_statistics(
    episodes: list[RolloutEpisode],
    field: str,
    dimension: int,
) -> dict[str, object]:
    """Calculate per-component statistics for later normalization."""
    samples: list[tuple[float, ...]] = []

    for episode in episodes:
        for frame in episode.frames:
            values = getattr(frame, field)
            if values is not None:
                samples.append(values)

    if not samples:
        return {
            "dimension": dimension,
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    columns = list(zip(*samples, strict=True))

    return {
        "dimension": dimension,
        "count": len(samples),
        "min": [min(column) for column in columns],
        "max": [max(column) for column in columns],
        "mean": [fmean(column) for column in columns],
        "std": [pstdev(column) for column in columns],
    }


def validate_dataset(root: Path) -> dict[str, object]:
    """Validate all episode JSON files and return deterministic statistics."""
    if not root.is_dir():
        raise DatasetValidationError(
            [f"dataset directory does not exist: {root}"]
        )

    paths = sorted(root.glob("*.json"), key=lambda path: path.name)

    if not paths:
        raise DatasetValidationError(
            ["dataset contains no episode JSON files"]
        )

    episodes: list[RolloutEpisode] = []
    seen_ids: dict[int, str] = {}
    diagnostics: list[str] = []

    for path in paths:
        try:
            episode = _load_episode(path)

            previous = seen_ids.get(episode.episode_index)
            if previous is not None:
                raise ValueError(
                    f"duplicate episode ID {episode.episode_index}; "
                    f"also found in {previous}"
                )

            seen_ids[episode.episode_index] = path.name
            episodes.append(episode)

        except (OSError, ValueError, TypeError) as exc:
            diagnostics.append(f"{path.name}: {exc}")

    if diagnostics:
        raise DatasetValidationError(diagnostics)

    total = len(episodes)
    lengths = [len(episode.frames) for episode in episodes]
    successes = sum(episode.success for episode in episodes)

    source_counts = _counts(episode.source for episode in episodes)

    numeric = {
        name: _vector_statistics(episodes, name, dimension)
        for name, dimension in VECTOR_SIZES.items()
    }

    report: dict[str, object] = {
        "report_version": 1,
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "total_episodes": total,
        "total_frames": sum(lengths),
        "scripted_episodes": source_counts.get("scripted", 0),
        "teleop_episodes": source_counts.get("teleop", 0),
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total,
        "failure_reasons": _counts(
            episode.failure_reason
            for episode in episodes
            if episode.failure_reason is not None
        ),
        "scenarios": _counts(episode.scenario for episode in episodes),
        "tasks": _counts(episode.task for episode in episodes),
        "sources": source_counts,
        "episode_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": fmean(lengths),
            "median": median(lengths),
        },
        "observation_dimensions": {
            "joint_position": 7,
            "joint_velocity": 7,
            "task_relative_state": 3,
            "wrist_wrench": 6,
        },
        "action_dimension": 7,
        "numeric_statistics": numeric,
    }

    return report


def main(argv: list[str] | None = None) -> int:
    """Run dataset validation from a single CLI command."""
    parser = argparse.ArgumentParser(
        description="Validate RobustLearn episodes and generate statistics."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing canonical episode JSON files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for deterministic statistics JSON.",
    )

    args = parser.parse_args(argv)

    if (
        args.json_output is not None
        and args.json_output.resolve().parent == args.dataset.resolve()
    ):
        print(
            "ERROR: Statistics output must be outside the episode directory.",
            file=sys.stderr,
        )
        return 2

    try:
        report = validate_dataset(args.dataset)
    except DatasetValidationError as exc:
        print("DATASET VALIDATION FAILED", file=sys.stderr)
        for diagnostic in exc.diagnostics:
            print(f"  {diagnostic}", file=sys.stderr)
        return 1

    print("DATASET VALIDATION PASSED")
    print(f"Episodes: {report['total_episodes']}")
    print(f"Frames: {report['total_frames']}")
    print(f"Scripted: {report['scripted_episodes']}")
    print(f"Teleop: {report['teleop_episodes']}")
    print(f"Successes: {report['successes']}")
    print(f"Failures: {report['failures']}")
    print(f"Success rate: {report['success_rate']:.4f}")

    for field in (
        "failure_reasons",
        "scenarios",
        "tasks",
        "episode_lengths",
        "observation_dimensions",
        "action_dimension",
    ):
        print(f"{field}: {json.dumps(report[field], sort_keys=True)}")

    print("Numeric ranges and normalization statistics:")
    numeric = report["numeric_statistics"]

    if isinstance(numeric, dict):
        for name in sorted(numeric):
            print(f"  {name}: {json.dumps(numeric[name], sort_keys=True)}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        args.json_output.write_text(payload, encoding="utf-8")
        print(f"Statistics JSON: {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
