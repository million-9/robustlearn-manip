"""Week 6 integrated acceptance checks."""

import numpy as np

from robustlearn.config import (
    FloatRange,
    PandaInsertionRandomizationConfig,
    PandaInsertionTaskConfig,
)
from robustlearn.evaluation import evaluate_scripted_episodes
from robustlearn.policy import (
    POLICY_ACTION_HIGH,
    POLICY_ACTION_LOW,
    POLICY_ACTION_SIZE,
    POLICY_OBSERVATION_SIZE,
    build_policy_observation,
    policy_action,
)
from robustlearn.sim.simulation import MuJoCoSimulation


def test_week6_clean_configuration_and_policy_contract() -> None:
    config = PandaInsertionTaskConfig()

    assert config.randomization.enabled is False

    simulation = MuJoCoSimulation()
    simulation.reset(
        seed=2026,
        randomization=config.randomization,
    )

    previous_action = policy_action(
        np.zeros(POLICY_ACTION_SIZE, dtype=np.float64)
    )
    observation = build_policy_observation(
        simulation,
        previous_action,
    )

    assert observation.shape == (POLICY_OBSERVATION_SIZE,)
    assert observation.dtype == np.float64
    assert np.all(np.isfinite(observation))

    assert previous_action.shape == (POLICY_ACTION_SIZE,)
    assert np.all(previous_action >= POLICY_ACTION_LOW)
    assert np.all(previous_action <= POLICY_ACTION_HIGH)


def test_week6_ci_clean_evaluation_is_reproducible() -> None:
    first = evaluate_scripted_episodes(
        base_seed=2026,
        episode_count=3,
    )
    second = evaluate_scripted_episodes(
        base_seed=2026,
        episode_count=3,
    )

    assert first.to_dict() == second.to_dict()

    assert first.summary.episode_count == 3
    assert first.summary.success_count == 3
    assert first.summary.failure_count == 0
    assert first.summary.truncated_count == 0
    assert first.summary.success_rate == 1.0

    for episode in first.episodes:
        assert np.isfinite(episode.simulation_duration_s)
        assert np.isfinite(episode.final_lateral_error_m)
        assert np.isfinite(episode.final_insertion_depth_m)


def test_week6_randomized_evaluation_is_seeded_and_bounded() -> None:
    randomization = PandaInsertionRandomizationConfig(
        enabled=True,
        receptacle_x_offset_m=FloatRange(-0.0002, 0.0002),
        receptacle_y_offset_m=FloatRange(-0.0002, 0.0002),
        receptacle_yaw_offset_rad=FloatRange(-0.01, 0.01),
    )
    config = PandaInsertionTaskConfig(
        randomization=randomization,
    )

    first = evaluate_scripted_episodes(
        seeds=[2026, 2027],
        config=config,
    )
    second = evaluate_scripted_episodes(
        seeds=[2026, 2027],
        config=config,
    )

    assert first.to_dict() == second.to_dict()

    for episode in first.episodes:
        sample = episode.randomization_sample

        assert (
            randomization.receptacle_x_offset_m.low
            <= sample["receptacle_x_offset_m"]
            <= randomization.receptacle_x_offset_m.high
        )
        assert (
            randomization.receptacle_y_offset_m.low
            <= sample["receptacle_y_offset_m"]
            <= randomization.receptacle_y_offset_m.high
        )
        assert (
            randomization.receptacle_yaw_offset_rad.low
            <= sample["receptacle_yaw_offset_rad"]
            <= randomization.receptacle_yaw_offset_rad.high
        )

        assert all(
            np.isfinite(value)
            for value in sample.values()
        )
