"""Regression tests for opt-in Week 11 expert capture forwarding."""

import pytest

from tools import run_week9_acceptance as runner


def prepare_mock_runner(tmp_path, monkeypatch):
    """Prevent real ROS subprocess execution."""
    test_path = tmp_path / "scenario_launch.py"
    test_path.write_text("# test fixture\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "scenario_test_path",
        lambda: test_path,
    )

    captured = {}

    def fake_run(*args, **kwargs):
        captured["environment"] = kwargs["env"]
        raise RuntimeError("test stopped before subprocess")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    return captured


def test_capture_path_forwarded(tmp_path, monkeypatch):
    captured = prepare_mock_runner(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="test stopped"):
        runner.run_scenario(
            "week9_case_01",
            domain_id=190,
            timeout_s=90.0,
            model_path="/tmp/seeded_model.xml",
            initial_keyframe="seed_42",
            expert_capture_path="/tmp/expert_capture.json",
        )

    environment = captured["environment"]

    assert (
        environment["ROBUSTLEARN_WEEK11_EXPERT_CAPTURE_PATH"]
        == "/tmp/expert_capture.json"
    )
    assert (
        environment["ROBUSTLEARN_WEEK10_MODEL_PATH"]
        == "/tmp/seeded_model.xml"
    )
    assert (
        environment["ROBUSTLEARN_WEEK10_INITIAL_KEYFRAME"]
        == "seed_42"
    )


def test_capture_disabled_for_existing_callers(
    tmp_path, monkeypatch
):
    captured = prepare_mock_runner(tmp_path, monkeypatch)

    monkeypatch.setenv(
        "ROBUSTLEARN_WEEK11_EXPERT_CAPTURE_PATH",
        "/tmp/stale_capture.json",
    )

    with pytest.raises(RuntimeError, match="test stopped"):
        runner.run_scenario(
            "week9_case_01",
            domain_id=190,
            timeout_s=90.0,
        )

    assert (
        "ROBUSTLEARN_WEEK11_EXPERT_CAPTURE_PATH"
        not in captured["environment"]
    )


def test_blank_capture_path_rejected(tmp_path, monkeypatch):
    prepare_mock_runner(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="non-empty"):
        runner.run_scenario(
            "week9_case_01",
            domain_id=190,
            timeout_s=90.0,
            expert_capture_path="   ",
        )


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_seeded_runner_forwards_optional_capture(
    tmp_path, monkeypatch, capture_enabled
):
    """Verify seeded execution forwards optional expert recording."""

    from robustlearn.episode_manager import EpisodeSpec
    from tools import run_week10_benchmark as benchmark_runner

    spec = EpisodeSpec(
        episode_index=2,
        seed=42,
        scenario="week9_case_03",
        timeout_s=90.0,
    )

    capture_path = (
        tmp_path / "expert_capture.json"
        if capture_enabled
        else None
    )

    expected_record = object()
    calls = {}

    monkeypatch.setattr(
        benchmark_runner,
        "load_benchmark_base_joint_positions",
        lambda **kwargs: (0.0,) * 7,
    )

    def fake_start_state(**kwargs):
        calls["start_state"] = kwargs
        return "seeded_start_state"

    monkeypatch.setattr(
        benchmark_runner,
        "build_benchmark_start_state",
        fake_start_state,
    )

    monkeypatch.setattr(
        benchmark_runner,
        "benchmark_keyframe_name",
        lambda seed: f"seed_{seed}",
    )

    monkeypatch.setattr(
        benchmark_runner,
        "build_seeded_benchmark_mjcf",
        lambda **kwargs: None,
    )

    def fake_run_scenario(scenario, **kwargs):
        calls["scenario"] = scenario
        calls["runner"] = kwargs
        return {"success": True}

    monkeypatch.setattr(
        benchmark_runner,
        "run_scenario",
        fake_run_scenario,
    )

    monkeypatch.setattr(
        benchmark_runner,
        "benchmark_record_from_week9_result",
        lambda **kwargs: expected_record,
    )

    result = benchmark_runner.run_episode(
        spec,
        base_domain_id=190,
        domain_pool_size=40,
        expert_capture_path=capture_path,
    )

    assert result is expected_record

    assert calls["start_state"]["seed"] == 42
    assert calls["start_state"]["scenario"] == "week9_case_03"

    assert calls["scenario"] == "week9_case_03"
    assert calls["runner"]["domain_id"] == 192
    assert calls["runner"]["initial_keyframe"] == "seed_42"
    assert calls["runner"]["timeout_s"] == 90.0

    assert calls["runner"]["expert_capture_path"] == (
        str(capture_path)
        if capture_enabled
        else None
    )
