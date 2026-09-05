"""Week 4 deterministic MuJoCo acceptance workflow."""

import mujoco
import numpy as np

from robustlearn.envs import PandaInsertionEnv
from robustlearn.sim import (
    SimulationSnapshot,
    fixed_peg_model_path,
    panda_model_path,
    workcell_model_path,
)

REQUIRED_BODY_NAMES: tuple[str, ...] = (
    "hand",
    "peg_tool",
    "workstation",
    "insertion_fixture",
    "receptacle",
)

REQUIRED_SITE_NAMES: tuple[str, ...] = (
    "peg_tip",
    "receptacle_center",
    "insertion_axis",
    "pre_insertion",
)


def assert_snapshots_equal(
    first: SimulationSnapshot,
    second: SimulationSnapshot,
) -> None:
    """Assert exact equality for the Week 4 controlled reset state."""
    assert first.time == second.time

    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.qvel, second.qvel)
    np.testing.assert_array_equal(first.act, second.act)
    np.testing.assert_array_equal(first.ctrl, second.ctrl)
    np.testing.assert_array_equal(first.mocap_pos, second.mocap_pos)
    np.testing.assert_array_equal(first.mocap_quat, second.mocap_quat)
    np.testing.assert_array_equal(
        first.qfrc_applied,
        second.qfrc_applied,
    )
    np.testing.assert_array_equal(
        first.xfrc_applied,
        second.xfrc_applied,
    )
    np.testing.assert_array_equal(
        first.task_site_xpos,
        second.task_site_xpos,
    )


def fixed_action_sequence(
    env: PandaInsertionEnv,
) -> tuple[np.ndarray, ...]:
    """Return a deterministic sequence of valid Week 4 actuator commands."""
    low = env.action_space.low
    high = env.action_space.high

    return tuple(
        np.asarray(
            low + alpha * (high - low),
            dtype=env.action_space.dtype,
        )
        for alpha in (
            0.48,
            0.50,
            0.52,
            0.50,
            0.48,
        )
    )


def test_week4_deterministic_mujoco_acceptance_workflow() -> None:
    """Verify the complete Week 4 deterministic environment pipeline."""
    assert panda_model_path().is_file()
    assert workcell_model_path().is_file()
    assert fixed_peg_model_path().is_file()

    env_a = PandaInsertionEnv()
    env_b = PandaInsertionEnv()

    try:
        model = env_a.simulation.model

        for name in REQUIRED_BODY_NAMES:
            object_id = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
            )
            assert object_id >= 0

        for name in REQUIRED_SITE_NAMES:
            object_id = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    name,
                )
            )
            assert object_id >= 0

        seed = 2026

        observation_a, info_a = env_a.reset(seed=seed)
        observation_b, info_b = env_b.reset(seed=seed)

        snapshot_a = env_a.simulation.snapshot()
        snapshot_b = env_b.simulation.snapshot()

        assert_snapshots_equal(
            snapshot_a,
            snapshot_b,
        )

        np.testing.assert_array_equal(
            observation_a,
            observation_b,
        )

        assert info_a == info_b
        assert env_a.observation_space.contains(observation_a)
        assert env_b.observation_space.contains(observation_b)

        assert np.all(np.isfinite(observation_a))
        assert np.all(np.isfinite(observation_b))

        actions = fixed_action_sequence(env_a)

        for action in actions:
            assert env_a.action_space.contains(action)
            assert env_b.action_space.contains(action)

            result_a = env_a.step(action)
            result_b = env_b.step(action)

            observation_a = result_a[0]
            observation_b = result_b[0]

            np.testing.assert_array_equal(
                observation_a,
                observation_b,
            )

            assert result_a[1] == result_b[1]
            assert result_a[2] == result_b[2]
            assert result_a[3] == result_b[3]
            assert result_a[4] == result_b[4]

            assert np.all(np.isfinite(observation_a))
            assert np.all(np.isfinite(observation_b))

            assert np.all(
                np.isfinite(
                    env_a.simulation.data.qpos
                )
            )
            assert np.all(
                np.isfinite(
                    env_a.simulation.data.qvel
                )
            )
            assert np.all(
                np.isfinite(
                    env_a.simulation.data.ctrl
                )
            )

            assert np.all(
                np.isfinite(
                    env_b.simulation.data.qpos
                )
            )
            assert np.all(
                np.isfinite(
                    env_b.simulation.data.qvel
                )
            )
            assert np.all(
                np.isfinite(
                    env_b.simulation.data.ctrl
                )
            )

        final_snapshot_a = env_a.simulation.snapshot()
        final_snapshot_b = env_b.simulation.snapshot()

        assert_snapshots_equal(
            final_snapshot_a,
            final_snapshot_b,
        )

        assert env_a.simulation.data.time > 0.0
        assert env_a.simulation.data.time == env_b.simulation.data.time

        reset_observation_a, reset_info_a = env_a.reset(
            seed=seed,
        )
        reset_snapshot_a = env_a.simulation.snapshot()

        assert_snapshots_equal(
            snapshot_a,
            reset_snapshot_a,
        )

        reset_observation_b, reset_info_b = env_b.reset(
            seed=seed,
        )

        np.testing.assert_array_equal(
            reset_observation_a,
            reset_observation_b,
        )

        assert reset_info_a == reset_info_b

        assert reset_info_a["simulation_time"] == 0.0
        assert reset_info_a["elapsed_steps"] == 0

        assert np.all(np.isfinite(observation_a))
        assert np.all(np.isfinite(reset_observation_a))

    finally:
        env_a.close()
        env_b.close()