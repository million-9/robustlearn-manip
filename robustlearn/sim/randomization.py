"""Deterministic episode randomization for Panda insertion."""

from dataclasses import dataclass

import numpy as np

from robustlearn.config import (
    FloatRange,
    PandaInsertionRandomizationConfig,
)


@dataclass(frozen=True)
class PandaInsertionRandomizationSample:
    """Sampled episode-level Panda insertion perturbations."""

    receptacle_x_offset_m: float
    receptacle_y_offset_m: float
    receptacle_yaw_offset_rad: float

    def to_dict(self) -> dict[str, float]:
        """Return a stable serializable representation."""
        return {
            "receptacle_x_offset_m": self.receptacle_x_offset_m,
            "receptacle_y_offset_m": self.receptacle_y_offset_m,
            "receptacle_yaw_offset_rad": self.receptacle_yaw_offset_rad,
        }


def _sample_range(
    rng: np.random.Generator,
    value_range: FloatRange,
) -> float:
    """Sample one finite value from an inclusive configuration range."""
    if value_range.low == value_range.high:
        return value_range.low

    return float(
        rng.uniform(
            value_range.low,
            value_range.high,
        )
    )


def sample_panda_insertion_randomization(
    rng: np.random.Generator,
    config: PandaInsertionRandomizationConfig,
) -> PandaInsertionRandomizationSample:
    """Sample deterministic episode perturbations from the supplied RNG."""
    if not config.enabled:
        return PandaInsertionRandomizationSample(
            receptacle_x_offset_m=0.0,
            receptacle_y_offset_m=0.0,
            receptacle_yaw_offset_rad=0.0,
        )

    return PandaInsertionRandomizationSample(
        receptacle_x_offset_m=_sample_range(
            rng,
            config.receptacle_x_offset_m,
        ),
        receptacle_y_offset_m=_sample_range(
            rng,
            config.receptacle_y_offset_m,
        ),
        receptacle_yaw_offset_rad=_sample_range(
            rng,
            config.receptacle_yaw_offset_rad,
        ),
    )


class PandaInsertionRandomizer:
    """Apply sampled receptacle perturbations to the named workcell fixture."""

    def __init__(self, model: object) -> None:
        """Cache the canonical insertion-fixture pose."""
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")

        self.model = model

        body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "insertion_fixture",
            )
        )

        if body_id < 0:
            raise RuntimeError(
                "Insertion model does not contain body 'insertion_fixture'"
            )

        self._fixture_body_id = body_id
        self._base_position = np.asarray(
            model.body_pos[body_id],
            dtype=np.float64,
        ).copy()
        self._base_quaternion = np.asarray(
            model.body_quat[body_id],
            dtype=np.float64,
        ).copy()

    def apply(
        self,
        sample: PandaInsertionRandomizationSample,
    ) -> None:
        """Restore the canonical fixture pose, then apply one episode sample."""
        import mujoco

        position = self._base_position.copy()
        position[0] += sample.receptacle_x_offset_m
        position[1] += sample.receptacle_y_offset_m

        half_yaw = sample.receptacle_yaw_offset_rad / 2.0
        yaw_quaternion = np.asarray(
            [
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
            ],
            dtype=np.float64,
        )

        quaternion = np.empty(4, dtype=np.float64)

        mujoco.mju_mulQuat(
            quaternion,
            yaw_quaternion,
            self._base_quaternion,
        )

        self.model.body_pos[self._fixture_body_id] = position
        self.model.body_quat[self._fixture_body_id] = quaternion
