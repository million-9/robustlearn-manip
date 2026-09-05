"""MuJoCo model composition for the RobustLearn insertion task."""

from pathlib import Path

import mujoco

from robustlearn.sim.panda import panda_model_path, project_root


def insertion_asset_directory() -> Path:
    """Return the directory containing insertion-task MJCF assets."""
    return project_root() / "robot_description" / "mjcf" / "insertion"


def workcell_model_path() -> Path:
    """Return the path to the insertion workcell MJCF."""
    path = insertion_asset_directory() / "workcell.xml"

    if not path.is_file():
        raise FileNotFoundError(f"Insertion workcell MJCF not found: {path}")

    return path


def fixed_peg_model_path() -> Path:
    """Return the path to the fixed peg MJCF."""
    path = insertion_asset_directory() / "fixed_peg.xml"

    if not path.is_file():
        raise FileNotFoundError(f"Fixed peg MJCF not found: {path}")

    return path


def load_insertion_model() -> mujoco.MjModel:
    """Compose and compile the Panda insertion-task MuJoCo model."""
    panda_spec = mujoco.MjSpec.from_file(str(panda_model_path()))
    tool_spec = mujoco.MjSpec.from_file(str(fixed_peg_model_path()))
    workcell_spec = mujoco.MjSpec.from_file(str(workcell_model_path()))

    hand = panda_spec.body("hand")
    tool_mount = tool_spec.frame("tool_mount")
    workcell_root = workcell_spec.frame("workcell_root")

    if hand is None:
        raise RuntimeError("Panda model does not contain body 'hand'")

    if tool_mount is None:
        raise RuntimeError("Fixed peg model does not contain frame 'tool_mount'")

    if workcell_root is None:
        raise RuntimeError("Workcell model does not contain frame 'workcell_root'")

    hand.attach_frame(
        tool_mount,
        "",
        "",
    )

    panda_spec.worldbody.attach_frame(
        workcell_root,
        "",
        "",
    )

    panda_spec.modelname = "robustlearn panda insertion task"

    return panda_spec.compile()