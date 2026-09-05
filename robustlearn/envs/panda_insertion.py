"""Gymnasium environment scaffold for the RobustLearn Panda insertion task."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from robustlearn.sim import MuJoCoSimulation

FloatArray = NDArray[np.float64]


class PandaInsertionEnv(gym.Env[FloatArray, FloatArray]):
    """Minimal deterministic Gymnasium interface for Panda insertion."""

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        *,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
    ) -> None:
        """Create a headless Panda insertion environment."""
        super().__init__()

        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.simulation = MuJoCoSimulation()

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        ctrl_limited = np.asarray(
            self.simulation.model.actuator_ctrllimited,
            dtype=bool,
        )

        if not np.all(ctrl_limited):
            raise RuntimeError(
                "Week 4 environment requires finite control limits "
                "for every actuator"
            )

        ctrl_range = np.asarray(
            self.simulation.model.actuator_ctrlrange,
            dtype=np.float64,
        )

        action_low = ctrl_range[:, 0].copy()
        action_high = ctrl_range[:, 1].copy()

        if not (
            np.all(np.isfinite(action_low))
            and np.all(np.isfinite(action_high))
        ):
            raise RuntimeError(
                "Week 4 environment requires finite actuator control ranges"
            )

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float64,
        )

        initial_observation = self._observation()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_observation.shape,
            dtype=np.float64,
        )

    def _observation(self) -> FloatArray:
        """Return the current minimal Week 4 observation."""
        snapshot = self.simulation.snapshot()

        return np.concatenate(
            (
                snapshot.qpos,
                snapshot.qvel,
                snapshot.task_site_xpos.reshape(-1),
            ),
            dtype=np.float64,
        )

    def _info(self) -> dict[str, Any]:
        """Return diagnostic information for the current episode."""
        return {
            "simulation_time": float(self.simulation.data.time),
            "elapsed_steps": self._elapsed_steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the environment through the deterministic simulation layer."""
        super().reset(seed=seed)

        # Reserved for later task configuration without committing to the
        # final randomization/options schema during Week 4.
        _ = options

        self._elapsed_steps = 0

        self.simulation.reset(
            seed=seed,
        )

        observation = self._observation()

        return observation, self._info()

    def step(
        self,
        action: FloatArray,
    ) -> tuple[
        FloatArray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply an actuator command and advance the MuJoCo simulation."""
        action_array = np.asarray(
            action,
            dtype=np.float64,
        )

        if not self.action_space.contains(action_array):
            raise ValueError(
                "action is outside the environment action space"
            )

        self.simulation.data.ctrl[:] = action_array

        self.simulation.step(
            self.frame_skip,
        )

        self._elapsed_steps += 1

        observation = self._observation()

        # Week 4 intentionally does not define the final insertion reward or
        # task-success conditions.
        reward = 0.0
        terminated = False
        truncated = self._elapsed_steps >= self.max_episode_steps

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._info(),
        )