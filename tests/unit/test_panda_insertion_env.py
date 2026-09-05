"""Tests for the Gymnasium Panda insertion environment scaffold."""

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from robustlearn.envs import PandaInsertionEnv


def valid_action(env: PandaInsertionEnv) -> np.ndarray:
    """Return a deterministic action guaranteed to belong to the action space."""
    return np.asarray(
        (env.action_space.low + env.action_space.high) / 2.0,
        dtype=env.action_space.dtype,
    )


def test_environment_checker_passes() -> None:
    env = PandaInsertionEnv()

    check_env(
        env,
        skip_render_check=True,
    )

    env.close()


def test_environment_defines_explicit_spaces() -> None:
    env = PandaInsertionEnv()

    assert isinstance(env.action_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)

    assert env.action_space.shape == (8,)
    assert env.observation_space.shape == (30,)

    assert env.action_space.dtype == np.dtype(np.float64)
    assert env.observation_space.dtype == np.dtype(np.float64)

    env.close()


def test_reset_returns_valid_observation_and_info() -> None:
    env = PandaInsertionEnv()

    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (30,)
    assert observation.dtype == np.float64
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)

    assert isinstance(info, dict)
    assert info["simulation_time"] == 0.0
    assert info["elapsed_steps"] == 0

    env.close()


def test_step_returns_gymnasium_five_element_result() -> None:
    env = PandaInsertionEnv()

    env.reset(seed=42)

    result = env.step(valid_action(env))

    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert env.observation_space.contains(observation)
    assert np.all(np.isfinite(observation))

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["elapsed_steps"] == 1

    env.close()


def test_step_advances_defined_number_of_physics_steps() -> None:
    env = PandaInsertionEnv(
        frame_skip=5,
    )

    env.reset(seed=42)

    initial_time = env.simulation.data.time

    env.step(valid_action(env))

    expected_time = (
        initial_time
        + env.frame_skip * env.simulation.timestep
    )

    assert env.simulation.data.time == pytest.approx(
        expected_time
    )

    env.close()


def test_same_seed_reset_matches_across_independent_environments() -> None:
    env_a = PandaInsertionEnv()
    env_b = PandaInsertionEnv()

    observation_a, info_a = env_a.reset(seed=2026)
    observation_b, info_b = env_b.reset(seed=2026)

    np.testing.assert_array_equal(
        observation_a,
        observation_b,
    )

    assert info_a == info_b

    env_a.close()
    env_b.close()


def test_same_action_produces_same_transition_after_same_seed_reset() -> None:
    env_a = PandaInsertionEnv()
    env_b = PandaInsertionEnv()

    env_a.reset(seed=42)
    env_b.reset(seed=42)

    action = valid_action(env_a)

    result_a = env_a.step(action)
    result_b = env_b.step(action)

    np.testing.assert_array_equal(
        result_a[0],
        result_b[0],
    )

    assert result_a[1] == result_b[1]
    assert result_a[2] == result_b[2]
    assert result_a[3] == result_b[3]
    assert result_a[4] == result_b[4]

    env_a.close()
    env_b.close()


def test_reset_after_rollout_reproduces_initial_observation() -> None:
    env = PandaInsertionEnv()

    initial_observation, initial_info = env.reset(
        seed=123,
    )

    action = valid_action(env)

    for _ in range(20):
        env.step(action)

    restored_observation, restored_info = env.reset(
        seed=123,
    )

    np.testing.assert_array_equal(
        initial_observation,
        restored_observation,
    )

    assert initial_info == restored_info

    env.close()


def test_episode_truncates_at_configured_step_limit() -> None:
    env = PandaInsertionEnv(
        max_episode_steps=3,
    )

    env.reset(seed=42)

    action = valid_action(env)

    first = env.step(action)
    second = env.step(action)
    third = env.step(action)

    assert first[3] is False
    assert second[3] is False
    assert third[3] is True

    assert first[4]["elapsed_steps"] == 1
    assert second[4]["elapsed_steps"] == 2
    assert third[4]["elapsed_steps"] == 3

    env.close()


def test_valid_action_belongs_to_action_space() -> None:
    env = PandaInsertionEnv()

    action = valid_action(env)

    assert action.shape == (8,)
    assert action.dtype == np.float64
    assert np.all(np.isfinite(action))
    assert env.action_space.contains(action)

    env.close()


def test_invalid_action_is_rejected() -> None:
    env = PandaInsertionEnv()

    env.reset(seed=42)

    invalid_action = env.action_space.high.copy()
    invalid_action[0] += 1.0

    with pytest.raises(
        ValueError,
        match="action is outside the environment action space",
    ):
        env.step(invalid_action)

    env.close()


def test_invalid_frame_skip_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="frame_skip must be at least 1",
    ):
        PandaInsertionEnv(
            frame_skip=0,
        )


def test_invalid_episode_length_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_episode_steps must be at least 1",
    ):
        PandaInsertionEnv(
            max_episode_steps=0,
        )