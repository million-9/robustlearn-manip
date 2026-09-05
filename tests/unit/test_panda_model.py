from math import isfinite

import mujoco

from robustlearn.sim import PANDA_ARM_JOINT_NAMES, load_panda_model, panda_model_path


def test_panda_model_file_exists() -> None:
    assert panda_model_path().is_file()


def test_panda_model_compiles() -> None:
    model = load_panda_model()

    assert model.nq > 0
    assert model.nv > 0
    assert model.nbody > 0


def test_expected_panda_arm_joints_exist() -> None:
    model = load_panda_model()

    actual_joint_names = {
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        for joint_id in range(model.njnt)
    }

    assert set(PANDA_ARM_JOINT_NAMES).issubset(actual_joint_names)



def test_panda_has_exactly_seven_expected_arm_joints() -> None:
    assert len(PANDA_ARM_JOINT_NAMES) == 7
    assert PANDA_ARM_JOINT_NAMES == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    )


def test_panda_arm_joint_limits_are_finite_and_ordered() -> None:
    model = load_panda_model()

    for joint_name in PANDA_ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        assert joint_id >= 0

        lower, upper = model.jnt_range[joint_id]

        assert isfinite(float(lower))
        assert isfinite(float(upper))
        assert lower < upper


def test_expected_panda_bodies_exist() -> None:
    model = load_panda_model()

    expected_bodies = (
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
    )

    for body_name in expected_bodies:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        assert body_id >= 0


def test_panda_model_has_expected_arm_actuators() -> None:
    model = load_panda_model()

    expected_actuators = tuple(f"actuator{i}" for i in range(1, 8))

    actual_actuator_names = {
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        for actuator_id in range(model.nu)
    }

    assert set(expected_actuators).issubset(actual_actuator_names)

def test_panda_simulation_steps_with_finite_state() -> None:
    model = load_panda_model()
    data = mujoco.MjData(model)

    home_keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    assert home_keyframe_id >= 0

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_keyframe_id,
    )

    initial_time = data.time

    for _ in range(100):
        mujoco.mj_step(model, data)

    assert data.time > initial_time
    assert all(isfinite(float(value)) for value in data.qpos)
    assert all(isfinite(float(value)) for value in data.qvel)
    assert all(isfinite(float(value)) for value in data.qacc)
    assert all(isfinite(float(value)) for value in data.ctrl)
