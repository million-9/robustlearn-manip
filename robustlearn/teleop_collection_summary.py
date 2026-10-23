"""Machine-readable reporting for reviewed teleoperation attempts.

Summaries are separate from canonical training episodes.
Rejected, aborted, and failed-write attempts never contribute
episode files to the training dataset.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from robustlearn.rollout_dataset import read_episode
from robustlearn.teleop_demo_collection import (
    TeleopCollectionConfig,
    TeleopRecordingSession,
)

SUMMARY_SCHEMA_VERSION = 1

OUTCOMES = (
    "accepted_success",
    "accepted_failure",
    "rejected",
    "aborted",
    "persistence_failed",
)

SAVED_OUTCOMES = (
    "accepted_success",
    "accepted_failure",
)


@dataclass(frozen=True)
class TeleopAttemptSummary:
    """One reviewed collection attempt, not a training episode."""

    episode_index: int
    seed: int
    scenario: str
    task: str
    git_commit: str
    config_id: str
    outcome: str
    frame_count: int
    start_timestamp_s: float | None
    end_timestamp_s: float | None
    episode_file: str | None
    failure_reason: str | None
    error: str | None
    source: str = "teleop"
    schema_version: int = SUMMARY_SCHEMA_VERSION
    attempt_id: str | None = None


def summarize_teleop_attempt(
    *,
    config: TeleopCollectionConfig,
    session: TeleopRecordingSession,
    outcome: str,
    episode_path: Path | None = None,
    error: str | None = None,
) -> TeleopAttemptSummary:
    """Validate one attempt against its recording and saved artifact."""
    if not isinstance(config, TeleopCollectionConfig):
        raise TypeError("config must be TeleopCollectionConfig")

    if not isinstance(session, TeleopRecordingSession):
        raise TypeError("session must be TeleopRecordingSession")

    if outcome not in OUTCOMES:
        raise ValueError("unsupported teleop collection outcome")

    saved = outcome in SAVED_OUTCOMES

    if saved:
        if session.status != "accepted":
            raise ValueError("saved outcome requires accepted session")

        if episode_path is None:
            raise ValueError("saved outcome requires episode path")

        if error is not None:
            raise ValueError("saved outcome cannot include write error")

        if (
            not isinstance(episode_path, Path)
            or episode_path.parent != config.output_root
        ):
            raise ValueError("episode path must be in output_root")

        # Reading the dataset verifies its canonical schema.
        episode = read_episode(episode_path)

        expected_success = outcome == "accepted_success"

        if (
            episode.source != "teleop"
            or episode.success != expected_success
            or episode.episode_index != config.episode_index
            or episode.seed != config.seed
            or episode.scenario != config.scenario
            or episode.task != config.task
            or episode.git_commit != config.git_commit
            or episode.config_id != config.config_id
            or len(episode.frames) != len(session.frames)
        ):
            raise ValueError(
                "saved episode does not match collection metadata"
            )

        if episode.frames != session.frames:
            raise ValueError(
                "saved episode frames do not match accepted recording"
            )

        if (
            not expected_success
            and episode.failure_reason != config.failure_reason
        ):
            raise ValueError(
                "saved failure reason does not match configuration"
            )

    else:
        if episode_path is not None:
            raise ValueError(
                "unsaved attempt cannot reference a training episode"
            )

        if outcome == "rejected" and session.status != "rejected":
            raise ValueError("rejected outcome requires rejected session")

        if outcome == "aborted" and session.status != "aborted":
            raise ValueError("aborted outcome requires aborted session")

        if outcome == "persistence_failed":
            if session.status not in ("recording", "accepted"):
                raise ValueError(
                    "write failure requires an unfinished save"
                )

            if not isinstance(error, str) or not error.strip():
                raise ValueError(
                    "persistence failure requires an error message"
                )

        elif error is not None:
            raise ValueError(
                "only persistence failures can include write errors"
            )

    frames = session.frames

    return TeleopAttemptSummary(
        episode_index=config.episode_index,
        seed=config.seed,
        scenario=config.scenario,
        task=config.task,
        git_commit=config.git_commit,
        config_id=config.config_id,
        outcome=outcome,
        frame_count=len(frames),
        start_timestamp_s=(
            frames[0].timestamp_s if frames else None
        ),
        end_timestamp_s=(
            frames[-1].timestamp_s if frames else None
        ),
        episode_file=(
            episode_path.name
            if saved and episode_path is not None
            else None
        ),
        failure_reason=(
            config.failure_reason
            if outcome == "accepted_failure"
            else None
        ),
        error=error,
    )


def build_teleop_collection_summary(
    attempts: list[TeleopAttemptSummary],
) -> dict[str, object]:
    """Aggregate attempts without mixing reports into the dataset."""
    if not all(
        isinstance(attempt, TeleopAttemptSummary)
        for attempt in attempts
    ):
        raise TypeError(
            "all attempts must be TeleopAttemptSummary instances"
        )

    counts = {
        outcome: sum(
            attempt.outcome == outcome
            for attempt in attempts
        )
        for outcome in OUTCOMES
    }

    report = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": "teleop",
        "total_attempts": len(attempts),
        "saved_episodes": (
            counts["accepted_success"]
            + counts["accepted_failure"]
        ),
        "outcome_counts": counts,
        "attempts": [
            asdict(attempt)
            for attempt in attempts
        ],
    }

    # Validate JSON compatibility without writing anything.
    json.dumps(report, allow_nan=False)

    return report

def teleop_collection_summary_path(output_root: Path) -> Path:
    """Keep summary metadata outside the canonical episode directory."""
    if (
        not isinstance(output_root, Path)
        or not output_root.is_absolute()
        or output_root == output_root.parent
    ):
        raise ValueError(
            "output_root must be an absolute episode directory"
        )

    return (
        output_root.parent
        / f"{output_root.name}__teleop_collection_summary.json"
    )


def write_teleop_collection_summary(
    attempts: list[TeleopAttemptSummary],
    *,
    output_root: Path,
) -> Path:
    """Atomically persist one validated collection-summary snapshot.

    The caller supplies the complete attempt history. Replacing a
    snapshot never means that an individual attempt was accepted.
    Episode files are created by their separate exclusive writer.
    """
    report = build_teleop_collection_summary(attempts)

    path = teleop_collection_summary_path(output_root)

    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return path
