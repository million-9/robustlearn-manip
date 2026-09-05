"""Deterministic MuJoCo simulation wrapper for RobustLearn-Manip."""

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from robustlearn.sim.insertion import load_insertion_model

FloatArray = NDArray[np.float64]

TASK_SITE_NAMES: tuple[str, ...] = (
    "peg_tip",
    "receptacle_center",
    "insertion_axis",
    "pre_insertion",
)


@dataclass(frozen=True)
class SimulationSnapshot:
    """Copy of the controlled MuJoCo state used for determinism tests."""

    time: float
    qpos: FloatArray
    qvel: FloatArray
    act: FloatArray
    ctrl: FloatArray
    mocap_pos: FloatArray
    mocap_quat: FloatArray
    qfrc_applied: FloatArray
    xfrc_applied: FloatArray
    task_site_xpos: FloatArray


class MuJoCoSimulation:
    """Own the MuJoCo model, data, RNG, reset, and stepping workflow."""

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        """Create a simulation around the RobustLearn insertion model."""
        self.model = model if model is not None else load_insertion_model()
        self.data = mujoco.MjData(self.model)

        self.rng: np.random.Generator = np.random.default_rng()
        self.last_seed: int | None = None

        self._home_keyframe_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                "home",
            )
        )

        if self._home_keyframe_id < 0:
            raise RuntimeError(
                "Insertion model does not contain keyframe 'home'"
            )

        self._task_site_ids = tuple(
            self._site_id(name)
            for name in TASK_SITE_NAMES
        )

        self.reset()

    @property
    def timestep(self) -> float:
        """Return the MuJoCo physics timestep in seconds."""
        return float(self.model.opt.timestep)

    def _site_id(self, name: str) -> int:
        """Resolve a required task site by name."""
        site_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
        )

        if site_id < 0:
            raise RuntimeError(
                f"Insertion model does not contain required site {name!r}"
            )

        return site_id

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> SimulationSnapshot:
        """Reset all controlled simulator state to the canonical start state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.last_seed = seed

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self._home_keyframe_id,
        )

        # These inputs can be modified during an episode and must never leak
        # into the next episode.
        self.data.qfrc_applied.fill(0.0)
        self.data.xfrc_applied.fill(0.0)

        # Solver warm-start state can depend on previous simulation history.
        # Clear it explicitly so reset does not inherit previous-episode state.
        self.data.qacc_warmstart.fill(0.0)

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.snapshot()

    def step(self, steps: int = 1) -> None:
        """Advance the MuJoCo simulation by one or more physics steps."""
        if steps < 1:
            raise ValueError("steps must be at least 1")

        for _ in range(steps):
            mujoco.mj_step(
                self.model,
                self.data,
            )

    def snapshot(self) -> SimulationSnapshot:
        """Return an independent copy of the determinism-relevant state."""
        task_site_xpos = np.stack(
            [
                np.asarray(
                    self.data.site_xpos[site_id],
                    dtype=np.float64,
                )
                for site_id in self._task_site_ids
            ]
        )

        return SimulationSnapshot(
            time=float(self.data.time),
            qpos=np.asarray(
                self.data.qpos,
                dtype=np.float64,
            ).copy(),
            qvel=np.asarray(
                self.data.qvel,
                dtype=np.float64,
            ).copy(),
            act=np.asarray(
                self.data.act,
                dtype=np.float64,
            ).copy(),
            ctrl=np.asarray(
                self.data.ctrl,
                dtype=np.float64,
            ).copy(),
            mocap_pos=np.asarray(
                self.data.mocap_pos,
                dtype=np.float64,
            ).copy(),
            mocap_quat=np.asarray(
                self.data.mocap_quat,
                dtype=np.float64,
            ).copy(),
            qfrc_applied=np.asarray(
                self.data.qfrc_applied,
                dtype=np.float64,
            ).copy(),
            xfrc_applied=np.asarray(
                self.data.xfrc_applied,
                dtype=np.float64,
            ).copy(),
            task_site_xpos=task_site_xpos.copy(),
        )