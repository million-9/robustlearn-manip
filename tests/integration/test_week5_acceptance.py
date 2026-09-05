"""Week 5 sensing and insertion milestone acceptance tests."""

import mujoco
import numpy as np
import pytest

from robustlearn.envs.panda_insertion import PandaInsertionEnv
from robustlearn.scripted_insertion import run_scripted_insertion, scripted_insertion_action
from robustlearn.sim.rendering import (
    WORKCELL_CAMERA_HEIGHT,
    WORKCELL_CAMERA_NAME,
    WORKCELL_CAMERA_WIDTH,
    CameraRenderer,
    CameraRenderingUnavailableError,
)

ACCEPTANCE_SEED = 2026


def test_week5_seeded_reset_sensing_camera_and_task_state() -> None:
    """Verify the complete Week 5 reset-state sensing contract."""
    env = PandaInsertionEnv()

    observation, info = env.reset(
        seed=ACCEPTANCE_SEED,
    )

    assert np.all(np.isfinite(observation))

    sensors = env.simulation.sensor_snapshot()

    assert sensors.joint_positions.shape == (7,)
    assert sensors.joint_velocities.shape == (7,)
    assert sensors.wrist_force.shape == (3,)
    assert sensors.wrist_torque.shape == (3,)

    assert np.all(np.isfinite(sensors.joint_positions))
    assert np.all(np.isfinite(sensors.joint_velocities))
    assert np.all(np.isfinite(sensors.wrist_force))
    assert np.all(np.isfinite(sensors.wrist_torque))

    np.testing.assert_array_equal(
        sensors.joint_positions,
        np.asarray(
            env.simulation.data.qpos[:7],
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        sensors.joint_velocities,
        np.asarray(
            env.simulation.data.qvel[:7],
            dtype=np.float64,
        ),
    )

    assert info["task_success"] is False
    assert info["task_failure"] is False

    renderer = CameraRenderer(
        env.simulation.model,
    )

    assert renderer.camera_name == WORKCELL_CAMERA_NAME
    assert renderer.width == WORKCELL_CAMERA_WIDTH
    assert renderer.height == WORKCELL_CAMERA_HEIGHT
    assert renderer.renderer_initialized is False



def test_week5_wrist_wrench_responds_to_tool_loading() -> None:
    """Verify that controlled tool loading produces a measurable wrist wrench."""
    env = PandaInsertionEnv()
    env.reset(seed=ACCEPTANCE_SEED)

    simulation = env.simulation
    baseline = simulation.sensor_snapshot()

    tool_body_id = int(
        mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "peg_tool",
        )
    )

    assert tool_body_id >= 0

    simulation.data.xfrc_applied[tool_body_id, :3] = np.asarray(
        [12.0, -7.0, 5.0],
        dtype=np.float64,
    )
    simulation.data.xfrc_applied[tool_body_id, 3:] = np.asarray(
        [1.2, -0.7, 0.4],
        dtype=np.float64,
    )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    loaded = simulation.sensor_snapshot()

    force_delta = (
        loaded.wrist_force
        - baseline.wrist_force
    )
    torque_delta = (
        loaded.wrist_torque
        - baseline.wrist_torque
    )

    assert np.all(np.isfinite(force_delta))
    assert np.all(np.isfinite(torque_delta))

    assert np.linalg.norm(force_delta) > 0.1
    assert np.linalg.norm(torque_delta) > 0.01


def test_week5_scripted_insertion_reaches_success_with_finite_state() -> None:
    """Verify the canonical scripted rollout reaches task-evaluator success."""
    env = PandaInsertionEnv(
        max_episode_steps=100,
    )

    observation, info = env.reset(
        seed=ACCEPTANCE_SEED,
    )

    assert np.all(np.isfinite(observation))
    assert info["task_success"] is False
    assert info["task_failure"] is False

    final_info = info
    terminated = False
    truncated = False

    for _ in range(100):
        action = scripted_insertion_action(env)

        assert np.all(np.isfinite(action))

        (
            observation,
            _,
            terminated,
            truncated,
            final_info,
        ) = env.step(action)

        assert np.all(np.isfinite(observation))

        state = env.simulation.snapshot()

        assert np.isfinite(state.time)
        assert np.all(np.isfinite(state.qpos))
        assert np.all(np.isfinite(state.qvel))
        assert np.all(np.isfinite(state.act))
        assert np.all(np.isfinite(state.ctrl))
        assert np.all(np.isfinite(state.task_site_xpos))

        sensors = env.simulation.sensor_snapshot()

        assert np.all(np.isfinite(sensors.joint_positions))
        assert np.all(np.isfinite(sensors.joint_velocities))
        assert np.all(np.isfinite(sensors.wrist_force))
        assert np.all(np.isfinite(sensors.wrist_torque))

        assert np.isfinite(final_info["task_lateral_error"])
        assert np.isfinite(final_info["task_axial_offset"])
        assert np.isfinite(final_info["task_insertion_depth"])

        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert final_info["task_success"] is True
    assert final_info["task_failure"] is False


def test_week5_same_seed_reproduces_scripted_outcome() -> None:
    """Verify that the canonical Week 5 scripted outcome is reproducible."""
    first = run_scripted_insertion(
        seed=ACCEPTANCE_SEED,
        max_episode_steps=100,
    )
    second = run_scripted_insertion(
        seed=ACCEPTANCE_SEED,
        max_episode_steps=100,
    )

    assert first
    assert second

    first_trace = np.asarray(
        [
            [
                record["step"],
                record["simulation_time"],
                record["lateral_error"],
                record["axial_offset"],
                record["insertion_depth"],
                record["success"],
                record["failure"],
                record["truncated"],
            ]
            for record in first
        ],
        dtype=np.float64,
    )

    second_trace = np.asarray(
        [
            [
                record["step"],
                record["simulation_time"],
                record["lateral_error"],
                record["axial_offset"],
                record["insertion_depth"],
                record["success"],
                record["failure"],
                record["truncated"],
            ]
            for record in second
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        first_trace,
        second_trace,
    )

    assert first[-1]["success"] is True
    assert first[-1]["failure"] is False
    assert first[-1]["truncated"] is False


def test_week5_rgb_depth_output_structure_when_available() -> None:
    """Verify RGB/depth output structure when an offscreen backend is available."""
    env = PandaInsertionEnv()
    env.reset(seed=ACCEPTANCE_SEED)

    renderer = CameraRenderer(
        env.simulation.model,
    )

    try:
        frame = renderer.render(
            env.simulation.data,
        )
    except CameraRenderingUnavailableError as exc:
        pytest.skip(str(exc))
    finally:
        renderer.close()

    assert frame.rgb.shape == (
        WORKCELL_CAMERA_HEIGHT,
        WORKCELL_CAMERA_WIDTH,
        3,
    )
    assert frame.rgb.dtype == np.uint8

    assert frame.depth.shape == (
        WORKCELL_CAMERA_HEIGHT,
        WORKCELL_CAMERA_WIDTH,
    )
    assert frame.depth.dtype == np.float32

    assert np.all(np.isfinite(frame.depth))
