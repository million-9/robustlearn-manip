"""Utilities for loading the vendored Franka Panda MuJoCo model."""

from pathlib import Path

import mujoco

PANDA_ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)


def project_root() -> Path:
    """Return the RobustLearn-Manip repository root."""
    return Path(__file__).resolve().parents[2]


def panda_model_path() -> Path:
    """Return the path to the vendored Panda MJCF model."""
    path = (
        project_root()
        / "robot_description"
        / "mjcf"
        / "franka_emika_panda"
        / "panda.xml"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Panda MJCF model not found: {path}")

    return path


def load_panda_model() -> mujoco.MjModel:
    """Compile and return the vendored Franka Panda MuJoCo model."""
    return mujoco.MjModel.from_xml_path(str(panda_model_path()))
