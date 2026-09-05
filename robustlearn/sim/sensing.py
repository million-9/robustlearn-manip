"""MuJoCo sensing definitions for the RobustLearn Panda insertion task."""

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from robustlearn.sim.panda import PANDA_ARM_JOINT_NAMES

FloatArray = NDArray[np.float64]

WRIST_FT_SITE_NAME = "panda_wrist_ft"

JOINT_POSITION_SENSOR_NAMES: tuple[str, ...] = tuple(
    f"panda_{joint_name}_position"
    for joint_name in PANDA_ARM_JOINT_NAMES
)

JOINT_VELOCITY_SENSOR_NAMES: tuple[str, ...] = tuple(
    f"panda_{joint_name}_velocity"
    for joint_name in PANDA_ARM_JOINT_NAMES
)

WRIST_FORCE_SENSOR_NAME = "panda_wrist_force"
WRIST_TORQUE_SENSOR_NAME = "panda_wrist_torque"

PANDA_SENSOR_NAMES: tuple[str, ...] = (
    *JOINT_POSITION_SENSOR_NAMES,
    *JOINT_VELOCITY_SENSOR_NAMES,
    WRIST_FORCE_SENSOR_NAME,
    WRIST_TORQUE_SENSOR_NAME,
)


@dataclass(frozen=True)
class PandaSensorSnapshot:
    """Independent copy of the Panda sensing state."""

    joint_positions: FloatArray
    joint_velocities: FloatArray
    wrist_force: FloatArray
    wrist_torque: FloatArray


@dataclass(frozen=True)
class _SensorSlice:
    """Resolved segment of MuJoCo sensordata."""

    address: int
    dimension: int


def add_panda_sensors(spec: mujoco.MjSpec) -> None:
    """Add project-owned Panda joint-state and wrist-wrench sensors."""
    for joint_name, sensor_name in zip(
        PANDA_ARM_JOINT_NAMES,
        JOINT_POSITION_SENSOR_NAMES,
        strict=True,
    ):
        spec.add_sensor(
            name=sensor_name,
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint_name,
        )

    for joint_name, sensor_name in zip(
        PANDA_ARM_JOINT_NAMES,
        JOINT_VELOCITY_SENSOR_NAMES,
        strict=True,
    ):
        spec.add_sensor(
            name=sensor_name,
            type=mujoco.mjtSensor.mjSENS_JOINTVEL,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint_name,
        )

    spec.add_sensor(
        name=WRIST_FORCE_SENSOR_NAME,
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=WRIST_FT_SITE_NAME,
    )

    spec.add_sensor(
        name=WRIST_TORQUE_SENSOR_NAME,
        type=mujoco.mjtSensor.mjSENS_TORQUE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=WRIST_FT_SITE_NAME,
    )


class PandaSensorReader:
    """Resolve and read Panda sensors without exposing raw MuJoCo indexing."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Resolve all required Panda sensors in the compiled model."""
        self._joint_position_slices = tuple(
            self._resolve_sensor(
                model,
                sensor_name,
                expected_dimension=1,
            )
            for sensor_name in JOINT_POSITION_SENSOR_NAMES
        )

        self._joint_velocity_slices = tuple(
            self._resolve_sensor(
                model,
                sensor_name,
                expected_dimension=1,
            )
            for sensor_name in JOINT_VELOCITY_SENSOR_NAMES
        )

        self._wrist_force_slice = self._resolve_sensor(
            model,
            WRIST_FORCE_SENSOR_NAME,
            expected_dimension=3,
        )

        self._wrist_torque_slice = self._resolve_sensor(
            model,
            WRIST_TORQUE_SENSOR_NAME,
            expected_dimension=3,
        )

    @staticmethod
    def _resolve_sensor(
        model: mujoco.MjModel,
        name: str,
        *,
        expected_dimension: int,
    ) -> _SensorSlice:
        """Resolve one required sensor and validate its output dimension."""
        sensor_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                name,
            )
        )

        if sensor_id < 0:
            raise RuntimeError(
                f"MuJoCo model does not contain required sensor {name!r}"
            )

        dimension = int(model.sensor_dim[sensor_id])

        if dimension != expected_dimension:
            raise RuntimeError(
                f"Sensor {name!r} has dimension {dimension}; "
                f"expected {expected_dimension}"
            )

        return _SensorSlice(
            address=int(model.sensor_adr[sensor_id]),
            dimension=dimension,
        )

    @staticmethod
    def _read_slice(
        data: mujoco.MjData,
        sensor_slice: _SensorSlice,
    ) -> FloatArray:
        """Copy one resolved sensor segment from MuJoCo sensordata."""
        start = sensor_slice.address
        stop = start + sensor_slice.dimension

        return np.asarray(
            data.sensordata[start:stop],
            dtype=np.float64,
        ).copy()

    def snapshot(
        self,
        data: mujoco.MjData,
    ) -> PandaSensorSnapshot:
        """Return an independent snapshot of all Panda sensing outputs."""
        joint_positions = np.asarray(
            [
                self._read_slice(
                    data,
                    sensor_slice,
                )[0]
                for sensor_slice in self._joint_position_slices
            ],
            dtype=np.float64,
        )

        joint_velocities = np.asarray(
            [
                self._read_slice(
                    data,
                    sensor_slice,
                )[0]
                for sensor_slice in self._joint_velocity_slices
            ],
            dtype=np.float64,
        )

        return PandaSensorSnapshot(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            wrist_force=self._read_slice(
                data,
                self._wrist_force_slice,
            ),
            wrist_torque=self._read_slice(
                data,
                self._wrist_torque_slice,
            ),
        )