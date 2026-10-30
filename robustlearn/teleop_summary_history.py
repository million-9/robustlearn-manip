"""Preserve teleoperation collection history across sequential runs.

The episode dataset and collection summary have separate writers.
This module expects one collection writer at a time. Concurrent
cross-process updates require an additional locking mechanism.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from robustlearn.teleop_collection_summary import (
    OUTCOMES,
    SUMMARY_SCHEMA_VERSION,
    TeleopAttemptSummary,
    build_teleop_collection_summary,
    teleop_collection_summary_path,
    write_teleop_collection_summary,
)


def read_teleop_summary_history(
    *,
    output_root: Path,
) -> list[TeleopAttemptSummary]:
    """Read and validate the complete previous summary snapshot."""
    path = teleop_collection_summary_path(output_root)

    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("collection summary must be an object")

    if (
        raw.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or raw.get("source") != "teleop"
    ):
        raise ValueError("incompatible collection summary")

    records = raw.get("attempts")

    if not isinstance(records, list):
        raise ValueError("collection attempts must be an array")

    allowed_fields = {field.name for field in fields(
        TeleopAttemptSummary
    )}

    attempts = []

    for record in records:
        if (
            not isinstance(record, dict)
            or not set(record).issubset(allowed_fields)
        ):
            raise ValueError("invalid collection attempt entry")

        try:
            attempt = TeleopAttemptSummary(**record)
        except TypeError as exc:
            raise ValueError("invalid collection attempt fields") from exc

        if (
            attempt.outcome not in OUTCOMES
            or attempt.source != "teleop"
            or attempt.schema_version != SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("invalid collection attempt metadata")

        if attempt.attempt_id is not None and (
            not isinstance(attempt.attempt_id, str)
            or not attempt.attempt_id.strip()
        ):
            raise ValueError("invalid collection attempt ID")

        attempts.append(attempt)

    recomputed = build_teleop_collection_summary(attempts)

    for key in (
        "total_attempts",
        "saved_episodes",
        "outcome_counts",
    ):
        if raw.get(key) != recomputed[key]:
            raise ValueError(
                f"inconsistent collection summary: {key}"
            )

    ids = [
        attempt.attempt_id
        for attempt in attempts
        if attempt.attempt_id is not None
    ]

    if len(ids) != len(set(ids)):
        raise ValueError("duplicate collection attempt IDs")

    return attempts


def update_teleop_summary_history(
    attempt: TeleopAttemptSummary,
    *,
    output_root: Path,
) -> Path:
    """Append an attempt or replace its own failed-write record.

    A finalized attempt is immutable. A persistence-failed attempt
    may advance to a final outcome using the same attempt ID.
    No previous history is discarded during sequential updates.
    """
    if not isinstance(attempt, TeleopAttemptSummary):
        raise TypeError("attempt must be TeleopAttemptSummary")

    if (
        not isinstance(attempt.attempt_id, str)
        or not attempt.attempt_id.strip()
    ):
        raise ValueError("attempt_id must be non-empty")

    if (
        attempt.source != "teleop"
        or attempt.schema_version != SUMMARY_SCHEMA_VERSION
        or attempt.outcome not in OUTCOMES
    ):
        raise ValueError("invalid teleoperation attempt")

    attempts = read_teleop_summary_history(
        output_root=output_root
    )

    for index, existing in enumerate(attempts):
        if existing.attempt_id != attempt.attempt_id:
            continue

        identity_fields = (
            "episode_index",
            "seed",
            "scenario",
            "task",
            "git_commit",
            "config_id",
            "frame_count",
            "start_timestamp_s",
            "end_timestamp_s",
        )

        if any(
            getattr(existing, field) != getattr(attempt, field)
            for field in identity_fields
        ):
            raise ValueError(
                "attempt ID reused with different recording metadata"
            )

        if existing == attempt:
            return teleop_collection_summary_path(output_root)

        if existing.outcome != "persistence_failed":
            raise ValueError(
                "finalized collection attempt cannot be changed"
            )

        attempts[index] = attempt
        break

    else:
        attempts.append(attempt)

    return write_teleop_collection_summary(
        attempts,
        output_root=output_root,
    )
