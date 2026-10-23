"""Tests for deterministic Week 11 collection schedules."""

import pytest

from tools import collect_scripted_demos as collector


def test_collection_schedule_is_deterministic():
    """Identical inputs must generate identical episode specs."""

    first = collector.build_collection_specs(
        count=3,
        start_seed=18,
        timeout_s=90.0,
    )

    second = collector.build_collection_specs(
        count=3,
        start_seed=18,
        timeout_s=90.0,
    )

    assert first == second
    assert len(first) == 3

    assert [
        (
            spec.episode_index,
            spec.seed,
            spec.scenario,
        )
        for spec in first
    ] == [
        (18, 18, "week9_case_19"),
        (19, 19, "week9_case_20"),
        (20, 20, "week9_case_01"),
    ]

    assert all(spec.timeout_s == 90.0 for spec in first)


def test_collection_schedule_covers_all_scenarios():
    """A 300-episode schedule must cover all Week 9 scenarios."""

    specs = collector.build_collection_specs(
        count=300,
        start_seed=0,
        timeout_s=90.0,
    )

    assert len(specs) == 300

    assert len({spec.seed for spec in specs}) == 300

    assert len({
        spec.episode_index for spec in specs
    }) == 300

    scenarios = {spec.scenario for spec in specs}

    assert scenarios == {
        f"week9_case_{index:02d}"
        for index in range(1, 21)
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"count": 0, "start_seed": 0, "timeout_s": 90.0},
        {"count": -1, "start_seed": 0, "timeout_s": 90.0},
        {"count": 10, "start_seed": -1, "timeout_s": 90.0},
        {"count": 10, "start_seed": 0, "timeout_s": 0.0},
    ],
)
def test_collection_schedule_rejects_invalid_inputs(arguments):
    """Reject invalid collection configurations."""

    with pytest.raises(ValueError):
        collector.build_collection_specs(**arguments)
