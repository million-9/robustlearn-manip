"""Deterministic scripted control for the clean Panda insertion task."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.sim.control import (
    current_control_targets,
    panda_arm_actuator_ids,
    panda_site_spatial_jacobian,
)
from robustlearn.sim.simulation import TASK_SITE_NAMES

FloatArray = NDArray[np.float64]


class InsertionPhase(StrEnum):
    """Discrete phases of the classical local insertion expert."""

    ALIGN = "align"
    INSERT = "insert"
    HOLD = "hold"
    ABORT = "abort"


@dataclass(frozen=True)
class ClassicalInsertionExpertConfig:
    """Validated tuning parameters for the classical insertion expert."""

    cartesian_step_m: float = 0.001
    damping: float = 0.05
    max_joint_step_rad: float = 0.03
    alignment_tolerance_m: float = 0.0005

    def __post_init__(self) -> None:
        """Validate finite, positive expert tuning parameters."""
        values = {
            "cartesian_step_m": self.cartesian_step_m,
            "damping": self.damping,
            "max_joint_step_rad": self.max_joint_step_rad,
            "alignment_tolerance_m": self.alignment_tolerance_m,
        }

        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ClassicalInsertionExpert:
    """Stateful phase-based classical insertion expert."""

    config: ClassicalInsertionExpertConfig = ClassicalInsertionExpertConfig()
    phase: InsertionPhase = InsertionPhase.ALIGN

    def update_phase(
        self,
        *,
        lateral_error_m: float,
        insertion_depth_m: float,
        success_insertion_depth_m: float,
        failed: bool = False,
    ) -> InsertionPhase:
        """Update and return the expert phase from current task state."""
        values = np.asarray(
            [
                lateral_error_m,
                insertion_depth_m,
                success_insertion_depth_m,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            self.phase = InsertionPhase.ABORT
            return self.phase

        if failed:
            self.phase = InsertionPhase.ABORT
        elif insertion_depth_m >= success_insertion_depth_m:
            self.phase = InsertionPhase.HOLD
        elif lateral_error_m > self.config.alignment_tolerance_m:
            self.phase = InsertionPhase.ALIGN
        else:
            self.phase = InsertionPhase.INSERT

        return self.phase

    def action(self, env: PandaInsertionEnv) -> FloatArray:
        """Return one deterministic bounded expert action."""
        simulation = env.simulation
        snapshot = simulation.snapshot()

        site_positions = {
            name: position
            for name, position in zip(
                TASK_SITE_NAMES,
                snapshot.task_site_xpos,
                strict=True,
            )
        }

        peg_tip = site_positions["peg_tip"]
        receptacle_center = site_positions["receptacle_center"]
        insertion_axis = site_positions["insertion_axis"]

        axis_vector = insertion_axis - receptacle_center
        axis_norm = float(np.linalg.norm(axis_vector))

        if axis_norm <= np.finfo(np.float64).eps:
            self.phase = InsertionPhase.ABORT
            raise RuntimeError("Insertion axis must differ from receptacle center")

        axis_direction = axis_vector / axis_norm

        delta = peg_tip - receptacle_center
        axial_offset = float(np.dot(delta, axis_direction))

        lateral_vector = delta - axial_offset * axis_direction

        insertion_depth = -axial_offset
        lateral_error = float(np.linalg.norm(lateral_vector))

        phase = self.update_phase(
            lateral_error_m=lateral_error,
            insertion_depth_m=insertion_depth,
            success_insertion_depth_m=env.config.success_insertion_depth_m,
        )

        if phase is InsertionPhase.ABORT:
            raise RuntimeError("Classical insertion expert entered abort phase")

        remaining_depth = env.config.success_insertion_depth_m - insertion_depth

        axial_step = 0.0
        lateral_correction = np.zeros(
            3,
            dtype=np.float64,
        )

        if phase is InsertionPhase.ALIGN:
            if lateral_error > np.finfo(np.float64).eps:
                correction_magnitude = min(
                    self.config.cartesian_step_m,
                    lateral_error,
                )

                lateral_correction = -lateral_vector * correction_magnitude / lateral_error

        elif phase is InsertionPhase.INSERT:
            axial_step = max(
                0.0,
                min(
                    self.config.cartesian_step_m,
                    remaining_depth,
                ),
            )

        desired_translation = -axial_step * axis_direction + lateral_correction

        jacobian = panda_site_spatial_jacobian(
            simulation.model,
            simulation.data,
            "peg_tip",
        )

        desired_twist = np.concatenate(
            (
                desired_translation,
                np.zeros(
                    3,
                    dtype=np.float64,
                ),
            )
        )

        regularized = jacobian @ jacobian.T + (self.config.damping**2) * np.eye(
            6,
            dtype=np.float64,
        )

        joint_delta = jacobian.T @ np.linalg.solve(
            regularized,
            desired_twist,
        )

        joint_delta = np.clip(
            joint_delta,
            -self.config.max_joint_step_rad,
            self.config.max_joint_step_rad,
        )

        action = current_control_targets(simulation.data)

        arm_actuator_ids = panda_arm_actuator_ids(simulation.model)

        arm_targets = (
            action[
                np.asarray(
                    arm_actuator_ids,
                    dtype=np.int64,
                )
            ]
            + joint_delta
        )

        for actuator_index, target in zip(
            arm_actuator_ids,
            arm_targets,
            strict=True,
        ):
            action[actuator_index] = target

        if not isinstance(
            env.action_space,
            gym.spaces.Box,
        ):
            raise RuntimeError("Panda insertion action space must be a Box")

        action = np.clip(
            action,
            np.asarray(
                env.action_space.low,
                dtype=np.float64,
            ),
            np.asarray(
                env.action_space.high,
                dtype=np.float64,
            ),
        )

        if not np.all(np.isfinite(action)):
            self.phase = InsertionPhase.ABORT
            raise RuntimeError("Classical insertion expert produced a non-finite action")

        return np.asarray(
            action,
            dtype=np.float64,
        )

    def reset(self) -> None:
        """Reset the expert to its initial phase."""
        self.phase = InsertionPhase.ALIGN


DEFAULT_CARTESIAN_STEP = 0.001
DEFAULT_DAMPING = 0.05
DEFAULT_MAX_JOINT_STEP = 0.03


def scripted_insertion_action(
    env: PandaInsertionEnv,
    *,
    cartesian_step: float = DEFAULT_CARTESIAN_STEP,
    damping: float = DEFAULT_DAMPING,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> FloatArray:
    """Return one deterministic joint-target action toward insertion success."""
    expert = ClassicalInsertionExpert(
        config=ClassicalInsertionExpertConfig(
            cartesian_step_m=cartesian_step,
            damping=damping,
            max_joint_step_rad=max_joint_step,
        )
    )

    return expert.action(env)


def run_scripted_insertion(
    *,
    seed: int = 2026,
    max_episode_steps: int = 100,
) -> list[dict[str, object]]:
    """Run the deterministic scripted insertion sequence and return diagnostics."""
    env = PandaInsertionEnv(
        max_episode_steps=max_episode_steps,
    )

    observation, _ = env.reset(
        seed=seed,
    )

    expert = ClassicalInsertionExpert()

    if not np.all(np.isfinite(observation)):
        raise RuntimeError("Scripted insertion reset produced a non-finite observation")

    diagnostics: list[dict[str, object]] = []

    for step_index in range(
        1,
        max_episode_steps + 1,
    ):
        action = expert.action(env)

        if not np.all(np.isfinite(action)):
            raise RuntimeError("Scripted insertion produced a non-finite action")

        (
            observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Scripted insertion produced a non-finite observation")

        sensors = env.simulation.sensor_snapshot()

        diagnostic: dict[str, object] = {
            "step": step_index,
            "simulation_time": info["simulation_time"],
            "phase": expert.phase.value,
            "lateral_error": info["task_lateral_error"],
            "axial_offset": info["task_axial_offset"],
            "insertion_depth": info["task_insertion_depth"],
            "wrist_force": sensors.wrist_force.copy(),
            "wrist_torque": sensors.wrist_torque.copy(),
            "success": info["task_success"],
            "failure": info["task_failure"],
            "truncated": truncated,
        }

        diagnostics.append(diagnostic)

        if terminated or truncated:
            break

    return diagnostics
