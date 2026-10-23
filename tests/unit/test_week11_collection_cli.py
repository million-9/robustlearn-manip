"""Tests for the Week 11 scripted demonstration CLI."""

from pathlib import Path

from tools import collect_scripted_demos as collector


def test_cli_parses_collection_configuration(tmp_path):
    """CLI arguments must preserve explicit collection settings."""

    args = collector.parse_args([
        "--count", "300",
        "--start-seed", "100",
        "--timeout-s", "90",
        "--dataset-root", str(tmp_path / "dataset"),
        "--capture-root", str(tmp_path / "captures"),
        "--summary-path", str(tmp_path / "summary.json"),
        "--config-id", "week11-config-v1",
        "--git-commit", "test-commit",
    ])

    assert args.count == 300
    assert args.start_seed == 100
    assert args.timeout_s == 90.0

    assert args.dataset_root == tmp_path / "dataset"
    assert args.capture_root == tmp_path / "captures"
    assert args.summary_path == tmp_path / "summary.json"

    assert args.config_id == "week11-config-v1"
    assert args.git_commit == "test-commit"
    assert args.task == "peg_insertion"


def test_cli_dry_run_does_not_execute_ros2(
    tmp_path, monkeypatch, capsys
):
    """A dry run must inspect the schedule without collection."""

    def unexpected_collection(*args, **kwargs):
        raise AssertionError(
            "Dry run must never launch collection"
        )

    monkeypatch.setattr(
        collector,
        "collect_episodes",
        unexpected_collection,
    )

    result = collector.main([
        "--count", "3",
        "--start-seed", "18",
        "--dataset-root", str(tmp_path / "dataset"),
        "--capture-root", str(tmp_path / "captures"),
        "--summary-path", str(tmp_path / "summary.json"),
        "--config-id", "week11-config-v1",
        "--git-commit", "test-commit",
        "--dry-run",
    ])

    assert result == 0

    output = capsys.readouterr().out

    assert "week9_case_19" in output
    assert "week9_case_20" in output
    assert "week9_case_01" in output

    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "captures").exists()
    assert not (tmp_path / "summary.json").exists()


def test_cli_forwards_configuration_to_batch_collector(
    tmp_path, monkeypatch
):
    """Normal CLI execution must forward the requested settings."""

    calls = {}

    def fake_collect_episodes(specs, **kwargs):
        calls["specs"] = specs
        calls["kwargs"] = kwargs

        return {
            "requested_count": len(specs),
            "skipped_count": 0,
            "collected_success_count": len(specs),
            "collected_failure_count": 0,
            "collection_error_count": 0,
            "episodes": [],
        }

    monkeypatch.setattr(
        collector,
        "collect_episodes",
        fake_collect_episodes,
    )

    result = collector.main([
        "--count", "2",
        "--start-seed", "40",
        "--dataset-root", str(tmp_path / "dataset"),
        "--capture-root", str(tmp_path / "captures"),
        "--summary-path", str(tmp_path / "summary.json"),
        "--config-id", "week11-config-v1",
        "--git-commit", "test-commit",
        "--base-domain-id", "190",
        "--domain-pool-size", "40",
    ])

    assert result == 0

    assert [spec.seed for spec in calls["specs"]] == [40, 41]
    assert [
        spec.scenario for spec in calls["specs"]
    ] == [
        "week9_case_01",
        "week9_case_02",
    ]

    assert calls["kwargs"]["dataset_root"] == (
        tmp_path / "dataset"
    )
    assert calls["kwargs"]["capture_root"] == (
        tmp_path / "captures"
    )
    assert calls["kwargs"]["config_id"] == "week11-config-v1"
    assert calls["kwargs"]["git_commit"] == "test-commit"
    assert calls["kwargs"]["base_domain_id"] == 190
    assert calls["kwargs"]["domain_pool_size"] == 40
