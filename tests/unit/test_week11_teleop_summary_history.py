"""Test durable accumulation and retry-safe summary accounting."""

import json
from dataclasses import replace

import pytest

from robustlearn.teleop_collection_summary import (
    TeleopAttemptSummary,
    teleop_collection_summary_path,
    write_teleop_collection_summary,
)
from robustlearn.teleop_summary_history import (
    read_teleop_summary_history,
    update_teleop_summary_history,
)


def attempt(
    identifier="attempt-1",
    *,
    outcome="rejected",
    start=10.0,
    error=None,
    episode_file=None,
):
    return TeleopAttemptSummary(
        episode_index=12,
        seed=42,
        scenario="nominal",
        task="panda_insertion",
        git_commit="test-commit",
        config_id="keyboard-teleop-test",
        outcome=outcome,
        frame_count=1,
        start_timestamp_s=start,
        end_timestamp_s=start,
        episode_file=episode_file,
        failure_reason=None,
        error=error,
        attempt_id=identifier,
    )


def output(tmp_path):
    return tmp_path / "episodes"


def report(tmp_path):
    path = teleop_collection_summary_path(output(tmp_path))
    return json.loads(path.read_text(encoding="utf-8"))


def test_first_attempt_creates_summary(tmp_path):
    update_teleop_summary_history(
        attempt(),
        output_root=output(tmp_path),
    )

    assert report(tmp_path)["total_attempts"] == 1
    assert report(tmp_path)["saved_episodes"] == 0


def test_second_session_preserves_first_attempt(tmp_path):
    update_teleop_summary_history(
        attempt("first"),
        output_root=output(tmp_path),
    )

    update_teleop_summary_history(
        attempt("second", outcome="aborted", start=20.0),
        output_root=output(tmp_path),
    )

    history = read_teleop_summary_history(
        output_root=output(tmp_path)
    )

    assert [item.attempt_id for item in history] == [
        "first",
        "second",
    ]
    assert report(tmp_path)["total_attempts"] == 2


def test_failed_write_then_success_is_one_attempt(tmp_path):
    failed = attempt(
        outcome="persistence_failed",
        error="temporary disk error",
    )

    update_teleop_summary_history(
        failed,
        output_root=output(tmp_path),
    )

    accepted = replace(
        failed,
        outcome="accepted_success",
        episode_file="episode_000012__seed_42__nominal.json",
        error=None,
    )

    update_teleop_summary_history(
        accepted,
        output_root=output(tmp_path),
    )

    result = report(tmp_path)

    assert result["total_attempts"] == 1
    assert result["saved_episodes"] == 1
    assert result["outcome_counts"]["persistence_failed"] == 0
    assert result["outcome_counts"]["accepted_success"] == 1


def test_identical_update_is_idempotent(tmp_path):
    record = attempt()

    path = update_teleop_summary_history(
        record,
        output_root=output(tmp_path),
    )

    before = path.read_bytes()

    update_teleop_summary_history(
        record,
        output_root=output(tmp_path),
    )

    assert path.read_bytes() == before
    assert report(tmp_path)["total_attempts"] == 1


def test_finalized_attempt_cannot_change_outcome(tmp_path):
    record = attempt()

    update_teleop_summary_history(
        record,
        output_root=output(tmp_path),
    )

    with pytest.raises(ValueError):
        update_teleop_summary_history(
            replace(record, outcome="aborted"),
            output_root=output(tmp_path),
        )

    assert report(tmp_path)["outcome_counts"]["rejected"] == 1


def test_attempt_id_cannot_change_recording_identity(tmp_path):
    record = attempt()

    update_teleop_summary_history(
        record,
        output_root=output(tmp_path),
    )

    with pytest.raises(ValueError):
        update_teleop_summary_history(
            replace(record, seed=99),
            output_root=output(tmp_path),
        )


def test_reused_episode_identity_gets_distinct_attempts(tmp_path):
    update_teleop_summary_history(
        attempt("first"),
        output_root=output(tmp_path),
    )

    update_teleop_summary_history(
        attempt("second", start=30.0),
        output_root=output(tmp_path),
    )

    assert report(tmp_path)["total_attempts"] == 2


def test_missing_attempt_id_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        update_teleop_summary_history(
            attempt(identifier=None),
            output_root=output(tmp_path),
        )


def test_corrupt_existing_summary_is_not_overwritten(tmp_path):
    path = teleop_collection_summary_path(output(tmp_path))
    path.write_text("{invalid json", encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError):
        update_teleop_summary_history(
            attempt(),
            output_root=output(tmp_path),
        )

    assert path.read_bytes() == original


def test_inconsistent_prior_counts_are_not_overwritten(tmp_path):
    path = write_teleop_collection_summary(
        [attempt()],
        output_root=output(tmp_path),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    data["saved_episodes"] = 100
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError):
        update_teleop_summary_history(
            attempt("second"),
            output_root=output(tmp_path),
        )

    assert path.read_bytes() == original


def test_prior_summary_without_attempt_ids_is_preserved(tmp_path):
    old_record = replace(attempt(), attempt_id=None)

    path = write_teleop_collection_summary(
        [old_record],
        output_root=output(tmp_path),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    data["attempts"][0].pop("attempt_id")
    path.write_text(json.dumps(data), encoding="utf-8")

    update_teleop_summary_history(
        attempt("new-attempt", start=50.0),
        output_root=output(tmp_path),
    )

    assert report(tmp_path)["total_attempts"] == 2
