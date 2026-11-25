"""First checkpoint-controlled, closed-loop MuJoCo BC evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from robustlearn.bc_inference import BCInference
from robustlearn.bc_sim_adapter import (
    bc_action_to_env,
    bc_observation_from_env,
)
from robustlearn.envs.panda_insertion import PandaInsertionEnv


def run_bc_rollout(
    checkpoint: str | Path,
    dataset_sha256: str,
    *,
    seed: int = 2026,
    max_steps: int = 100,
    max_force_n: float = 50.0,
) -> dict[str, Any]:
    """Run a real policy-controlled simulation and record its outcome."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not np.isfinite(max_force_n) or max_force_n <= 0:
        raise ValueError("max_force_n must be finite and positive")

    checkpoint_path = Path(checkpoint)
    policy = BCInference(
        checkpoint_path,
        expected_dataset_sha256=dataset_sha256,
    )
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    env = PandaInsertionEnv(max_episode_steps=max_steps)
    trajectory: list[dict[str, Any]] = []
    outcome = "timeout"
    failure_reason: str | None = "episode_step_limit"
    rejected_action: dict[str, Any] | None = None

    try:
        env.reset(seed=seed)

        for step in range(max_steps):
            try:
                observation = bc_observation_from_env(env)
                sensors_before = env.simulation.sensor_snapshot()
                pre_force = float(
                    np.linalg.norm(sensors_before.wrist_force)
                )
            except (ValueError, RuntimeError) as exc:
                outcome = "sensor_failure"
                failure_reason = str(exc)
                break

            if not np.isfinite(pre_force):
                outcome = "sensor_failure"
                failure_reason = "nonfinite_pre_action_force"
                break

            if pre_force > max_force_n:
                outcome = "excessive_force"
                failure_reason = "pre_action_force_limit"
                break

            try:
                prediction = policy.predict(observation)
            except (ValueError, RuntimeError) as exc:
                outcome = "unsafe_action"
                failure_reason = str(exc)
                break

            try:
                action = bc_action_to_env(env, prediction)
            except (ValueError, RuntimeError) as exc:
                outcome = "unsafe_action"
                failure_reason = str(exc)

                from robustlearn.sim.control import panda_arm_actuator_ids

                arm_ids = panda_arm_actuator_ids(env.simulation.model)
                action_space = env.action_space

                if not isinstance(action_space, spaces.Box):
                    raise RuntimeError(
                        "Expected Box actuator action space"
                    ) from exc

                lower = action_space.low[list(arm_ids)]
                upper = action_space.high[list(arm_ids)]

                rejected_action = {
                    "step": step + 1,
                    "observation": observation.tolist(),
                    "policy_targets_rad": prediction.tolist(),
                    "joint_lower_limits_rad": lower.tolist(),
                    "joint_upper_limits_rad": upper.tolist(),
                }
                break

            _, _, terminated, truncated, info = env.step(action)
            sensors = env.simulation.sensor_snapshot()
            force_n = float(np.linalg.norm(sensors.wrist_force))

            record = {
                "step": step + 1,
                "time_s": float(info["simulation_time"]),
                "observation": observation.tolist(),
                "policy_targets_rad": prediction.tolist(),
                "applied_actuator_controls": action.tolist(),
                "lateral_error_m": float(info["task_lateral_error"]),
                "insertion_depth_m": float(info["task_insertion_depth"]),
                "wrist_force_n": sensors.wrist_force.tolist(),
                "wrist_torque_nm": sensors.wrist_torque.tolist(),
                "force_magnitude_n": force_n,
                "task_success": bool(info["task_success"]),
                "task_failure": bool(info["task_failure"]),
            }
            trajectory.append(record)

            if not np.isfinite(force_n):
                outcome = "sensor_failure"
                failure_reason = "nonfinite_post_action_force"
                break

            if force_n > max_force_n:
                outcome = "excessive_force"
                failure_reason = "post_action_force_limit"
                break

            if bool(info["task_success"]):
                outcome = "success"
                failure_reason = None
                break

            if bool(info["task_failure"]):
                outcome = "task_failure"
                failure_reason = "geometric_insertion_failure"
                break

            if terminated:
                outcome = "task_failure"
                failure_reason = "environment_termination"
                break

            if truncated:
                outcome = "timeout"
                failure_reason = "episode_step_limit"
                break

        controls = [
            np.asarray(row["policy_targets_rad"], dtype=float)
            for row in trajectory
        ]
        smoothness = (
            float(np.mean([
                np.linalg.norm(current - previous)
                for previous, current in zip(
                    controls[:-1], controls[1:], strict=False
                )
            ]))
            if len(controls) > 1
            else None
        )

        return {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_sha256": dataset_sha256,
            "seed": seed,
            "max_steps": max_steps,
            "max_force_n": max_force_n,
            "outcome": outcome,
            "success": outcome == "success",
            "failure_reason": failure_reason,
            "steps_executed": len(trajectory),
            "completion_time_s": (
                float(env.simulation.data.time)
                if outcome == "success"
                else None
            ),
            "elapsed_simulation_time_s": float(env.simulation.data.time),
            "max_observed_force_n": max(
                (row["force_magnitude_n"] for row in trajectory),
                default=0.0,
            ),
            "mean_action_change_rad": smoothness,
            "rejected_action": rejected_action,
            "trajectory": trajectory,
        }
    finally:
        env.close()


def main() -> None:
    """CLI entry point for one deterministic evaluation rollout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-force-n", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        args.dataset_manifest.read_text(encoding="utf-8")
    )

    result = run_bc_rollout(
        args.checkpoint,
        manifest["dataset_sha256"],
        seed=args.seed,
        max_steps=args.max_steps,
        max_force_n=args.max_force_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        key: value
        for key, value in result.items()
        if key != "trajectory"
    }, indent=2))


if __name__ == "__main__":
    main()
