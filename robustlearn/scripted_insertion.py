"""Deterministic scripted control for the clean Panda insertion task."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.sim.control import (
    current_control_targets,
    panda_arm_actuator_ids,
    panda_site_spatial_jacobian,
)
from robustlearn.sim.simulation import TASK_SITE_NAMES

FloatArray = NDArray[np.float64]

PANDA_ARM_DOF = 7


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

    # Conservative contact limits for the local insertion controller.
    #
    # The force supplied to ``joint_targets`` must be expressed in the same
    # world/task frame as the insertion geometry.
    max_axial_force_n: float = 10.0
    max_lateral_force_n: float = 5.0

    def __post_init__(self) -> None:
        """Validate finite, positive expert tuning parameters."""
        values = {
            "cartesian_step_m": self.cartesian_step_m,
            "damping": self.damping,
            "max_joint_step_rad": self.max_joint_step_rad,
            "alignment_tolerance_m": self.alignment_tolerance_m,
            "max_axial_force_n": self.max_axial_force_n,
            "max_lateral_force_n": self.max_lateral_force_n,
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

    def joint_targets(
        self,
        *,
        current_joint_targets: ArrayLike,
        peg_tip_position: ArrayLike,
        receptacle_center_position: ArrayLike,
        insertion_axis_position: ArrayLike,
        peg_tip_jacobian: ArrayLike,
        success_insertion_depth_m: float,
        wrist_force_world_n: ArrayLike | None = None,
        failed: bool = False,
    ) -> FloatArray:
        """Return bounded seven-joint targets from explicit insertion state.

        This method contains the environment-independent expert control law.
        Callers provide the current seven Panda joint targets, task geometry,
        and the 6x7 peg-tip spatial Jacobian.

        ``wrist_force_world_n`` must be expressed in the same coordinate frame
        as the task positions. When omitted, zero contact force is assumed.

        Excessive axial contact stops insertion progression for the current
        control step. Excessive lateral contact is treated as unsafe and moves
        the expert into the ABORT phase.

        The method performs no simulation stepping and does not access MuJoCo
        actuator state directly. This makes it suitable for both the existing
        simulation adapter and ROS-side execution using externally observed
        robot state.
        """
        current = np.asarray(
            current_joint_targets,
            dtype=np.float64,
        )

        peg_tip = np.asarray(
            peg_tip_position,
            dtype=np.float64,
        )

        receptacle_center = np.asarray(
            receptacle_center_position,
            dtype=np.float64,
        )

        insertion_axis = np.asarray(
            insertion_axis_position,
            dtype=np.float64,
        )

        jacobian = np.asarray(
            peg_tip_jacobian,
            dtype=np.float64,
        )

        wrist_force = (
            np.zeros(
                3,
                dtype=np.float64,
            )
            if wrist_force_world_n is None
            else np.asarray(
                wrist_force_world_n,
                dtype=np.float64,
            )
        )

        if current.shape != (PANDA_ARM_DOF,):
            self.phase = InsertionPhase.ABORT
            raise ValueError(
                "current_joint_targets must have shape "
                f"({PANDA_ARM_DOF},); got {current.shape}"
            )

        positions = {
            "peg_tip_position": peg_tip,
            "receptacle_center_position": receptacle_center,
            "insertion_axis_position": insertion_axis,
        }

        for name, position in positions.items():
            if position.shape != (3,):
                self.phase = InsertionPhase.ABORT
                raise ValueError(
                    f"{name} must have shape (3,); got {position.shape}"
                )

        if jacobian.shape != (6, PANDA_ARM_DOF):
            self.phase = InsertionPhase.ABORT
            raise ValueError(
                "peg_tip_jacobian must have shape "
                f"(6, {PANDA_ARM_DOF}); got {jacobian.shape}"
            )

        if wrist_force.shape != (3,):
            self.phase = InsertionPhase.ABORT
            raise ValueError(
                "wrist_force_world_n must have shape "
                f"(3,); got {wrist_force.shape}"
            )

        scalar_values = np.asarray(
            [success_insertion_depth_m],
            dtype=np.float64,
        )

        finite_inputs = (
            np.all(np.isfinite(current))
            and all(
                np.all(np.isfinite(position))
                for position in positions.values()
            )
            and np.all(np.isfinite(jacobian))
            and np.all(np.isfinite(wrist_force))
            and np.all(np.isfinite(scalar_values))
        )

        if not finite_inputs:
            self.phase = InsertionPhase.ABORT
            raise ValueError(
                "classical insertion expert inputs must contain only "
                "finite values"
            )

        if success_insertion_depth_m <= 0.0:
            self.phase = InsertionPhase.ABORT
            raise ValueError(
                "success_insertion_depth_m must be positive"
            )

        axis_vector = insertion_axis - receptacle_center
        axis_norm = float(np.linalg.norm(axis_vector))

        if axis_norm <= np.finfo(np.float64).eps:
            self.phase = InsertionPhase.ABORT
            raise RuntimeError(
                "Insertion axis must differ from receptacle center"
            )

        axis_direction = axis_vector / axis_norm

        delta = peg_tip - receptacle_center
        axial_offset = float(
            np.dot(
                delta,
                axis_direction,
            )
        )

        lateral_vector = (
            delta
            - axial_offset * axis_direction
        )

        insertion_depth = -axial_offset
        lateral_error = float(
            np.linalg.norm(lateral_vector)
        )

        signed_axial_force = float(
            np.dot(
                wrist_force,
                axis_direction,
            )
        )

        axial_force = abs(signed_axial_force)

        lateral_force_vector = (
            wrist_force
            - signed_axial_force * axis_direction
        )

        lateral_force = float(
            np.linalg.norm(
                lateral_force_vector
            )
        )

        if lateral_force >= self.config.max_lateral_force_n:
            self.update_phase(
                lateral_error_m=lateral_error,
                insertion_depth_m=insertion_depth,
                success_insertion_depth_m=success_insertion_depth_m,
                failed=True,
            )

            raise RuntimeError(
                "Classical insertion expert aborted because lateral "
                f"contact force {lateral_force:.6f} N reached the "
                f"configured limit {self.config.max_lateral_force_n:.6f} N"
            )

        phase = self.update_phase(
            lateral_error_m=lateral_error,
            insertion_depth_m=insertion_depth,
            success_insertion_depth_m=success_insertion_depth_m,
            failed=failed,
        )

        if phase is InsertionPhase.ABORT:
            raise RuntimeError(
                "Classical insertion expert entered abort phase"
            )

        # Excessive axial force prevents further insertion progression.
        # Returning the current targets gives the controller an explicit hold
        # command without introducing a separate task-success state.
        if (
            phase is InsertionPhase.INSERT
            and axial_force >= self.config.max_axial_force_n
        ):
            return current.copy()

        remaining_depth = (
            success_insertion_depth_m
            - insertion_depth
        )

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

                lateral_correction = (
                    -lateral_vector
                    * correction_magnitude
                    / lateral_error
                )

        elif phase is InsertionPhase.INSERT:
            axial_step = max(
                0.0,
                min(
                    self.config.cartesian_step_m,
                    remaining_depth,
                ),
            )

        desired_translation = (
            -axial_step * axis_direction
            + lateral_correction
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

        regularized = (
            jacobian @ jacobian.T
            + (self.config.damping**2)
            * np.eye(
                6,
                dtype=np.float64,
            )
        )

        try:
            joint_delta = (
                jacobian.T
                @ np.linalg.solve(
                    regularized,
                    desired_twist,
                )
            )
        except np.linalg.LinAlgError as exc:
            self.phase = InsertionPhase.ABORT
            raise RuntimeError(
                "Classical insertion expert Jacobian solve failed"
            ) from exc

        joint_delta = np.clip(
            joint_delta,
            -self.config.max_joint_step_rad,
            self.config.max_joint_step_rad,
        )

        targets = current + joint_delta

        if not np.all(np.isfinite(targets)):
            self.phase = InsertionPhase.ABORT
            raise RuntimeError(
                "Classical insertion expert produced non-finite "
                "joint targets"
            )

        return np.asarray(
            targets,
            dtype=np.float64,
        )

    def action(self, env: PandaInsertionEnv) -> FloatArray:
        """Return one deterministic bounded environment action."""
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

        action = current_control_targets(
            simulation.data
        )

        arm_actuator_ids = panda_arm_actuator_ids(
            simulation.model
        )

        current_arm_targets = np.asarray(
            [
                action[actuator_index]
                for actuator_index in arm_actuator_ids
            ],
            dtype=np.float64,
        )

        task_status = simulation.task_status(
            config=env.config
        )

        arm_targets = self.joint_targets(
            current_joint_targets=current_arm_targets,
            peg_tip_position=site_positions["peg_tip"],
            receptacle_center_position=(
                site_positions["receptacle_center"]
            ),
            insertion_axis_position=(
                site_positions["insertion_axis"]
            ),
            peg_tip_jacobian=panda_site_spatial_jacobian(
                simulation.model,
                simulation.data,
                "peg_tip",
            ),
            success_insertion_depth_m=(
                env.config.success_insertion_depth_m
            ),
            failed=task_status.failure,
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
            self.phase = InsertionPhase.ABORT
            raise RuntimeError(
                "Panda insertion action space must be a Box"
            )

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
            raise RuntimeError(
                "Classical insertion expert produced a non-finite action"
            )

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
        raise RuntimeError(
            "Scripted insertion reset produced a non-finite observation"
        )

    diagnostics: list[dict[str, object]] = []

    for step_index in range(
        1,
        max_episode_steps + 1,
    ):
        action = expert.action(env)

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                "Scripted insertion produced a non-finite action"
            )

        (
            observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "Scripted insertion produced a non-finite observation"
            )

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
