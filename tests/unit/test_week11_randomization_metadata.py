"""Tests for reproducible initial-condition metadata."""

from dataclasses import replace
import math

import pytest

from robustlearn.rollout_dataset import (
    RolloutEpisode,
    episode_from_dict,
    episode_to_dict,
)


def make_episode():
    return RolloutEpisode(
        episode_index=0,
        seed=42,
        scenario="week9_case_03",
        source="expert",
        task="peg_insertion",
        success=False,
        failure_reason="test failure",
        start_timestamp_s=100.0,
        end_timestamp_s=103.0,
        git_commit="test-commit",
        config_id="test-config",
        frames=(),
    )


BASE = (0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.7)
DELTA = (0.01, 0.02, 0.0, -0.01, 0.0, 0.03, -0.02)
START = tuple(
    base + delta
    for base, delta in zip(BASE, DELTA, strict=True)
)


def test_randomization_metadata_round_trip():
    """All starting-state vectors must survive serialization."""

    episode = replace(
        make_episode(),
        start_base_joint_positions_rad=BASE,
        start_perturbation_rad=DELTA,
        start_joint_positions_rad=START,
    )

    payload = episode_to_dict(episode)

    assert payload["start_base_joint_positions_rad"] == list(BASE)
    assert payload["start_perturbation_rad"] == list(DELTA)
    assert payload["start_joint_positions_rad"] == list(START)

    restored = episode_from_dict(payload)

    assert restored == episode


def test_legacy_episode_without_randomization_metadata():
    """Existing version-1 datasets must remain readable."""

    episode = make_episode()
    payload = episode_to_dict(episode)

    for key in (
        "start_base_joint_positions_rad",
        "start_perturbation_rad",
        "start_joint_positions_rad",
    ):
        payload.pop(key, None)

    restored = episode_from_dict(payload)

    assert restored == episode
    assert restored.start_base_joint_positions_rad is None
    assert restored.start_perturbation_rad is None
    assert restored.start_joint_positions_rad is None


@pytest.mark.parametrize(
    "changes",
    [
        {"start_base_joint_positions_rad": BASE},
        {"start_perturbation_rad": DELTA},
        {"start_joint_positions_rad": START},
        {
            "start_base_joint_positions_rad": BASE[:-1],
            "start_perturbation_rad": DELTA,
            "start_joint_positions_rad": START,
        },
        {
            "start_base_joint_positions_rad": BASE,
            "start_perturbation_rad": DELTA,
            "start_joint_positions_rad": (math.nan,) + START[1:],
        },
        {
            "start_base_joint_positions_rad": BASE,
            "start_perturbation_rad": DELTA,
            "start_joint_positions_rad": (0.0,) * 7,
        },
    ],
)
def test_invalid_randomization_metadata_rejected(changes):
    """Prevent incomplete or inconsistent initial conditions."""

    with pytest.raises(ValueError):
        replace(make_episode(), **changes)
