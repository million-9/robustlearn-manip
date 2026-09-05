"""Tests for the composed Panda insertion MuJoCo model."""

from math import hypot, isfinite, sqrt

import mujoco

from robustlearn.sim import load_insertion_model

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

FIXTURE_COLLISION_GEOM_NAMES: tuple[str, ...] = (
    "fixture_left_collision",
    "fixture_right_collision",
    "fixture_front_collision",
    "fixture_rear_collision",
)


def object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """Return a named MuJoCo object id and assert that it exists."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"Missing MuJoCo object: {name}"
    return object_id


def home_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Return model data initialized at the Panda home keyframe."""
    data = mujoco.MjData(model)

    home_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    return data


def test_insertion_model_compiles() -> None:
    model = load_insertion_model()

    assert model.nbody > 1
    assert model.ngeom > 0
    assert model.nsite >= len(REQUIRED_SITE_NAMES)
    assert model.nu == 8


def test_required_task_bodies_exist() -> None:
    model = load_insertion_model()

    for name in REQUIRED_BODY_NAMES:
        object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )


def test_required_task_sites_exist() -> None:
    model = load_insertion_model()

    for name in REQUIRED_SITE_NAMES:
        object_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )


def test_contact_critical_geometry_uses_simple_boxes() -> None:
    model = load_insertion_model()

    geom_names = (
        "peg_collision",
        *FIXTURE_COLLISION_GEOM_NAMES,
    )

    for name in geom_names:
        geom_id = object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX


def test_home_pose_is_aligned_with_receptacle() -> None:
    model = load_insertion_model()
    data = home_data(model)

    peg_tip_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "peg_tip",
    )
    receptacle_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "receptacle_center",
    )
    pre_insertion_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "pre_insertion",
    )

    peg = data.site_xpos[peg_tip_id]
    receptacle = data.site_xpos[receptacle_id]
    pre_insertion = data.site_xpos[pre_insertion_id]

    horizontal_error = hypot(
        float(peg[0] - receptacle[0]),
        float(peg[1] - receptacle[1]),
    )

    vertical_offset = float(peg[2] - receptacle[2])

    pre_insertion_error = sqrt(
        sum(
            float(peg[index] - pre_insertion[index]) ** 2
            for index in range(3)
        )
    )

    assert horizontal_error < 1e-4
    assert 0.015 < vertical_offset < 0.025
    assert pre_insertion_error < 1e-4


def test_peg_axis_is_aligned_with_insertion_axis() -> None:
    model = load_insertion_model()
    data = home_data(model)

    peg_tip_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "peg_tip",
    )
    insertion_axis_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "insertion_axis",
    )

    peg_matrix = data.site_xmat[peg_tip_id].reshape(3, 3)
    insertion_matrix = data.site_xmat[insertion_axis_id].reshape(3, 3)

    peg_z = peg_matrix[:, 2]
    insertion_z = insertion_matrix[:, 2]

    dot_product = float(peg_z @ insertion_z)

    assert dot_product < -0.999


def test_home_pose_has_no_initial_contacts() -> None:
    model = load_insertion_model()
    data = home_data(model)

    assert data.ncon == 0

def test_nominal_scene_remains_stable_under_gravity() -> None:
    model = load_insertion_model()
    data = home_data(model)

    peg_tip_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "peg_tip",
    )
    receptacle_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "receptacle_center",
    )

    initial_peg = tuple(float(v) for v in data.site_xpos[peg_tip_id])
    initial_receptacle = tuple(
        float(v) for v in data.site_xpos[receptacle_id]
    )

    duration = 5.0
    steps = int(duration / model.opt.timestep)

    maximum_contacts = data.ncon

    for _ in range(steps):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, data.ncon)

    final_peg = tuple(float(v) for v in data.site_xpos[peg_tip_id])
    final_receptacle = tuple(
        float(v) for v in data.site_xpos[receptacle_id]
    )

    peg_drift = sqrt(
        sum(
            (final_peg[index] - initial_peg[index]) ** 2
            for index in range(3)
        )
    )

    receptacle_drift = sqrt(
        sum(
            (
                final_receptacle[index]
                - initial_receptacle[index]
            )
            ** 2
            for index in range(3)
        )
    )

    assert all(
        isfinite(float(value))
        for array in (
            data.qpos,
            data.qvel,
            data.qacc,
            data.ctrl,
        )
        for value in array
    )

    # The world-fixed workcell must not move.
    assert receptacle_drift < 1e-12

    # Some compliant settling of the Panda position-controlled arm is
    # expected under gravity, but the nominal pose must remain nearby.
    assert peg_drift < 0.02

    # The nominal pre-insertion configuration must stay contact-free.
    assert maximum_contacts == 0