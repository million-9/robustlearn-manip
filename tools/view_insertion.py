"""Launch the RobustLearn Panda insertion scene in the MuJoCo viewer."""

import time

import mujoco
import mujoco.viewer

from robustlearn.sim import load_insertion_model


def main() -> None:
    """Launch an interactive viewer for the insertion task."""
    model = load_insertion_model()
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id < 0:
        raise RuntimeError("Insertion model does not contain keyframe 'home'")

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Start with the camera focused on the insertion workcell.
        viewer.cam.lookat[:] = (0.55, 0.0, 0.45)
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()