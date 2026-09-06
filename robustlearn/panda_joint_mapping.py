"""Validation of the Panda ROS/URDF-to-MuJoCo joint mapping contract."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mujoco


class MappingValidationError(ValueError):
    """Raised when the Panda cross-model mapping contract is invalid."""


@dataclass(frozen=True)
class JointMapping:
    """One ROS joint to MuJoCo joint/actuator mapping."""

    ros_joint: str
    mujoco_joint: str
    mujoco_actuator: str
    position_sign: float
    position_offset_rad: float


@dataclass(frozen=True)
class LimitComparison:
    """Joint-limit comparison policy."""

    policy: str
    absolute_tolerance_rad: float


@dataclass(frozen=True)
class HomeReference:
    """MuJoCo keyframe and expected ROS-side joint positions."""

    mujoco_keyframe: str
    positions_rad_by_ros_joint: dict[str, float]


@dataclass(frozen=True)
class CrossModelFk:
    """Frames reserved for later cross-model FK comparison."""

    ros_base_frame: str
    mujoco_base_body: str
    ros_tip_frame: str
    mujoco_tip_body: str
    include_all_mapped_joints: bool


@dataclass(frozen=True)
class MappingContract:
    """Complete Panda cross-model joint mapping contract."""

    schema_version: int
    joint_count: int
    position_transform: str
    limit_comparison: LimitComparison
    home_reference: HomeReference
    cross_model_fk: CrossModelFk
    joints: tuple[JointMapping, ...]


@dataclass(frozen=True)
class MappingValidationReport:
    """Summary returned after successful validation."""

    joint_count: int
    ros_joint_order: tuple[str, ...]
    mujoco_joint_order: tuple[str, ...]
    mujoco_qpos_addresses: tuple[int, ...]
    home_positions_rad: tuple[float, ...]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "robot_description"
    / "panda_joint_mapping.json"
)

DEFAULT_URDF_PATH = (
    REPOSITORY_ROOT
    / "ros2_ws"
    / "src"
    / "robustlearn_description"
    / "urdf"
    / "panda.urdf.xacro"
)

DEFAULT_MJCF_PATH = (
    REPOSITORY_ROOT
    / "robot_description"
    / "mjcf"
    / "insertion"
    / "panda_insertion.xml"
)


def _require_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MappingValidationError(
            f"{context} must be a JSON object"
        )

    result: dict[str, object] = {}

    for key, item in value.items():
        if not isinstance(key, str):
            raise MappingValidationError(
                f"{context} contains a non-string key"
            )

        result[key] = item

    return result


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise MappingValidationError(
            f"{context} must be a JSON array"
        )

    return cast(list[object], value)


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MappingValidationError(
            f"{context} must be a non-empty string"
        )

    return value


def _require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MappingValidationError(
            f"{context} must be an integer"
        )

    return value


def _require_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise MappingValidationError(
            f"{context} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise MappingValidationError(
            f"{context} must be finite"
        )

    return result


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise MappingValidationError(
            f"{context} must be boolean"
        )

    return value


def _required_field(
    mapping: dict[str, object],
    key: str,
    context: str,
) -> object:
    if key not in mapping:
        raise MappingValidationError(
            f"{context} is missing required field '{key}'"
        )

    return mapping[key]


def load_mapping_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> MappingContract:
    """Load and structurally validate the mapping contract."""

    if not path.is_file():
        raise MappingValidationError(
            f"Mapping contract does not exist: {path}"
        )

    try:
        raw: object = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise MappingValidationError(
            f"Mapping contract is not valid JSON: {error}"
        ) from error

    root = _require_dict(raw, "mapping contract")

    schema_version = _require_int(
        _required_field(
            root,
            "schema_version",
            "mapping contract",
        ),
        "schema_version",
    )

    if schema_version != 1:
        raise MappingValidationError(
            f"Unsupported mapping schema version: {schema_version}"
        )

    joint_count = _require_int(
        _required_field(
            root,
            "joint_count",
            "mapping contract",
        ),
        "joint_count",
    )

    position_transform = _require_str(
        _required_field(
            root,
            "position_transform",
            "mapping contract",
        ),
        "position_transform",
    )

    limit_raw = _require_dict(
        _required_field(
            root,
            "limit_comparison",
            "mapping contract",
        ),
        "limit_comparison",
    )

    limit_comparison = LimitComparison(
        policy=_require_str(
            _required_field(
                limit_raw,
                "policy",
                "limit_comparison",
            ),
            "limit_comparison.policy",
        ),
        absolute_tolerance_rad=_require_float(
            _required_field(
                limit_raw,
                "absolute_tolerance_rad",
                "limit_comparison",
            ),
            "limit_comparison.absolute_tolerance_rad",
        ),
    )

    if limit_comparison.policy != "equal":
        raise MappingValidationError(
            "Only the 'equal' joint-limit policy is supported"
        )

    if limit_comparison.absolute_tolerance_rad < 0.0:
        raise MappingValidationError(
            "Joint-limit tolerance must be non-negative"
        )

    home_raw = _require_dict(
        _required_field(
            root,
            "home_reference",
            "mapping contract",
        ),
        "home_reference",
    )

    home_positions_raw = _require_dict(
        _required_field(
            home_raw,
            "positions_rad_by_ros_joint",
            "home_reference",
        ),
        "home_reference.positions_rad_by_ros_joint",
    )

    home_positions: dict[str, float] = {}

    for joint_name, value in home_positions_raw.items():
        home_positions[joint_name] = _require_float(
            value,
            (
                "home_reference.positions_rad_by_ros_joint"
                f".{joint_name}"
            ),
        )

    home_reference = HomeReference(
        mujoco_keyframe=_require_str(
            _required_field(
                home_raw,
                "mujoco_keyframe",
                "home_reference",
            ),
            "home_reference.mujoco_keyframe",
        ),
        positions_rad_by_ros_joint=home_positions,
    )

    fk_raw = _require_dict(
        _required_field(
            root,
            "cross_model_fk",
            "mapping contract",
        ),
        "cross_model_fk",
    )

    cross_model_fk = CrossModelFk(
        ros_base_frame=_require_str(
            _required_field(
                fk_raw,
                "ros_base_frame",
                "cross_model_fk",
            ),
            "cross_model_fk.ros_base_frame",
        ),
        mujoco_base_body=_require_str(
            _required_field(
                fk_raw,
                "mujoco_base_body",
                "cross_model_fk",
            ),
            "cross_model_fk.mujoco_base_body",
        ),
        ros_tip_frame=_require_str(
            _required_field(
                fk_raw,
                "ros_tip_frame",
                "cross_model_fk",
            ),
            "cross_model_fk.ros_tip_frame",
        ),
        mujoco_tip_body=_require_str(
            _required_field(
                fk_raw,
                "mujoco_tip_body",
                "cross_model_fk",
            ),
            "cross_model_fk.mujoco_tip_body",
        ),
        include_all_mapped_joints=_require_bool(
            _required_field(
                fk_raw,
                "include_all_mapped_joints",
                "cross_model_fk",
            ),
            "cross_model_fk.include_all_mapped_joints",
        ),
    )

    joint_entries = _require_list(
        _required_field(
            root,
            "joints",
            "mapping contract",
        ),
        "joints",
    )

    joints: list[JointMapping] = []

    for index, raw_joint in enumerate(joint_entries):
        context = f"joints[{index}]"
        joint_raw = _require_dict(raw_joint, context)

        joints.append(
            JointMapping(
                ros_joint=_require_str(
                    _required_field(
                        joint_raw,
                        "ros_joint",
                        context,
                    ),
                    f"{context}.ros_joint",
                ),
                mujoco_joint=_require_str(
                    _required_field(
                        joint_raw,
                        "mujoco_joint",
                        context,
                    ),
                    f"{context}.mujoco_joint",
                ),
                mujoco_actuator=_require_str(
                    _required_field(
                        joint_raw,
                        "mujoco_actuator",
                        context,
                    ),
                    f"{context}.mujoco_actuator",
                ),
                position_sign=_require_float(
                    _required_field(
                        joint_raw,
                        "position_sign",
                        context,
                    ),
                    f"{context}.position_sign",
                ),
                position_offset_rad=_require_float(
                    _required_field(
                        joint_raw,
                        "position_offset_rad",
                        context,
                    ),
                    f"{context}.position_offset_rad",
                ),
            )
        )

    contract = MappingContract(
        schema_version=schema_version,
        joint_count=joint_count,
        position_transform=position_transform,
        limit_comparison=limit_comparison,
        home_reference=home_reference,
        cross_model_fk=cross_model_fk,
        joints=tuple(joints),
    )

    _validate_contract_identity(contract)

    return contract


def _validate_contract_identity(
    contract: MappingContract,
) -> None:
    if contract.joint_count != len(contract.joints):
        raise MappingValidationError(
            "joint_count does not match the number of joint mappings"
        )

    if contract.joint_count != 7:
        raise MappingValidationError(
            "Panda arm mapping must contain exactly seven joints"
        )

    ros_names = [
        mapping.ros_joint
        for mapping in contract.joints
    ]

    mujoco_names = [
        mapping.mujoco_joint
        for mapping in contract.joints
    ]

    actuator_names = [
        mapping.mujoco_actuator
        for mapping in contract.joints
    ]

    if len(set(ros_names)) != len(ros_names):
        raise MappingValidationError(
            "ROS joint mapping contains duplicate joint names"
        )

    if len(set(mujoco_names)) != len(mujoco_names):
        raise MappingValidationError(
            "MuJoCo joint mapping contains duplicate joint names"
        )

    if len(set(actuator_names)) != len(actuator_names):
        raise MappingValidationError(
            "MuJoCo actuator mapping contains duplicate actuator names"
        )

    home_names = set(
        contract.home_reference.positions_rad_by_ros_joint
    )

    if home_names != set(ros_names):
        raise MappingValidationError(
            "Home-reference joint names must exactly match "
            "the mapped ROS joint names"
        )

    for mapping in contract.joints:
        if mapping.position_sign not in (-1.0, 1.0):
            raise MappingValidationError(

                    f"Joint '{mapping.ros_joint}' position_sign "
                    "must be either -1 or +1"

            )


def _parse_vector(
    value: str,
    context: str,
) -> tuple[float, float, float]:
    parts = value.split()

    if len(parts) != 3:
        raise MappingValidationError(
            f"{context} must contain exactly three values"
        )

    try:
        vector = tuple(float(part) for part in parts)
    except ValueError as error:
        raise MappingValidationError(
            f"{context} contains a non-numeric value"
        ) from error

    if len(vector) != 3:
        raise MappingValidationError(
            f"{context} must contain exactly three values"
        )

    if not all(math.isfinite(item) for item in vector):
        raise MappingValidationError(
            f"{context} contains a non-finite value"
        )

    return vector


def _close(
    first: float,
    second: float,
    tolerance: float,
) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def validate_mapping_contract(
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    urdf_path: Path = DEFAULT_URDF_PATH,
    mjcf_path: Path = DEFAULT_MJCF_PATH,
) -> MappingValidationReport:
    """Validate the Panda mapping against URDF and compiled MuJoCo."""

    contract = load_mapping_contract(contract_path)

    if not urdf_path.is_file():
        raise MappingValidationError(
            f"URDF/Xacro file does not exist: {urdf_path}"
        )

    if not mjcf_path.is_file():
        raise MappingValidationError(
            f"MuJoCo model does not exist: {mjcf_path}"
        )

    try:
        urdf_root = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise MappingValidationError(
            f"URDF/Xacro XML could not be parsed: {error}"
        ) from error

    ros_order = tuple(
        mapping.ros_joint
        for mapping in contract.joints
    )

    ros_name_set = set(ros_order)

    matching_urdf_joints = [
        joint
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("name") in ros_name_set
    ]

    actual_urdf_order = tuple(
        joint.attrib["name"]
        for joint in matching_urdf_joints
    )

    if actual_urdf_order != ros_order:
        raise MappingValidationError(

                "URDF Panda joint ordering does not match the mapping "
                f"contract: expected {ros_order}, "
                f"got {actual_urdf_order}"

        )

    if len(matching_urdf_joints) != contract.joint_count:
        raise MappingValidationError(
            "URDF does not contain exactly one instance of every "
            "mapped Panda joint"
        )

    urdf_joint_by_name = {
        joint.attrib["name"]: joint
        for joint in matching_urdf_joints
    }

    if len(urdf_joint_by_name) != contract.joint_count:
        raise MappingValidationError(
            "URDF contains a duplicate mapped Panda joint"
        )

    urdf_links = {
        link.attrib["name"]
        for link in urdf_root.findall("link")
        if "name" in link.attrib
    }

    if contract.cross_model_fk.ros_base_frame not in urdf_links:
        raise MappingValidationError(

                "ROS FK base frame is missing from the URDF: "
                f"{contract.cross_model_fk.ros_base_frame}"

        )

    if contract.cross_model_fk.ros_tip_frame not in urdf_links:
        raise MappingValidationError(

                "ROS FK tip frame is missing from the URDF: "
                f"{contract.cross_model_fk.ros_tip_frame}"

        )

    try:
        model = mujoco.MjModel.from_xml_path(
            str(mjcf_path)
        )
    except Exception as error:
        raise MappingValidationError(
            f"MuJoCo model failed to compile: {error}"
        ) from error

    if (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            contract.cross_model_fk.mujoco_base_body,
        )
        < 0
    ):
        raise MappingValidationError(

                "MuJoCo FK base body is missing: "
                f"{contract.cross_model_fk.mujoco_base_body}"

        )

    if (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            contract.cross_model_fk.mujoco_tip_body,
        )
        < 0
    ):
        raise MappingValidationError(

                "MuJoCo FK tip body is missing: "
                f"{contract.cross_model_fk.mujoco_tip_body}"

        )

    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        contract.home_reference.mujoco_keyframe,
    )

    if keyframe_id < 0:
        raise MappingValidationError(

                "Required MuJoCo home keyframe is missing: "
                f"{contract.home_reference.mujoco_keyframe}"

        )

    tolerance = (
        contract.limit_comparison.absolute_tolerance_rad
    )

    qpos_addresses: list[int] = []
    home_positions: list[float] = []

    for mapping in contract.joints:
        urdf_joint = urdf_joint_by_name[mapping.ros_joint]

        if urdf_joint.attrib.get("type") != "revolute":
            raise MappingValidationError(

                    f"ROS joint '{mapping.ros_joint}' must be "
                    "revolute"

            )

        axis_element = urdf_joint.find("axis")

        if axis_element is None:
            raise MappingValidationError(
                f"ROS joint '{mapping.ros_joint}' has no axis"
            )

        axis_text = axis_element.attrib.get("xyz")

        if axis_text is None:
            raise MappingValidationError(

                    f"ROS joint '{mapping.ros_joint}' axis "
                    "has no xyz value"

            )

        ros_axis = _parse_vector(
            axis_text,
            f"ROS joint '{mapping.ros_joint}' axis",
        )

        limit_element = urdf_joint.find("limit")

        if limit_element is None:
            raise MappingValidationError(
                f"ROS joint '{mapping.ros_joint}' has no limit"
            )

        lower_text = limit_element.attrib.get("lower")
        upper_text = limit_element.attrib.get("upper")

        if lower_text is None or upper_text is None:
            raise MappingValidationError(

                    f"ROS joint '{mapping.ros_joint}' must "
                    "define lower and upper limits"

            )

        try:
            ros_lower = float(lower_text)
            ros_upper = float(upper_text)
        except ValueError as error:
            raise MappingValidationError(

                    f"ROS joint '{mapping.ros_joint}' has "
                    "non-numeric limits"

            ) from error

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            mapping.mujoco_joint,
        )

        if joint_id < 0:
            raise MappingValidationError(

                    "Mapped MuJoCo joint is missing: "
                    f"{mapping.mujoco_joint}"

            )

        if (
            int(model.jnt_type[joint_id])
            != int(mujoco.mjtJoint.mjJNT_HINGE)
        ):
            raise MappingValidationError(

                    f"MuJoCo joint '{mapping.mujoco_joint}' "
                    "must be a hinge joint"

            )

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        qpos_addresses.append(qpos_address)

        mujoco_axis = tuple(
            float(value)
            for value in model.jnt_axis[joint_id]
        )

        if len(mujoco_axis) != 3:
            raise MappingValidationError(

                    f"MuJoCo joint '{mapping.mujoco_joint}' "
                    "has an invalid axis"

            )

        for axis_index in range(3):
            expected_ros_axis = (
                mapping.position_sign
                * mujoco_axis[axis_index]
            )

            if not _close(
                ros_axis[axis_index],
                expected_ros_axis,
                tolerance,
            ):
                raise MappingValidationError(

                        "Position sign/axis convention mismatch for "
                        f"'{mapping.ros_joint}' -> "
                        f"'{mapping.mujoco_joint}'"

                )

        if not bool(model.jnt_limited[joint_id]):
            raise MappingValidationError(

                    f"MuJoCo joint '{mapping.mujoco_joint}' "
                    "must be range-limited"

            )

        mujoco_lower = float(
            model.jnt_range[joint_id, 0]
        )

        mujoco_upper = float(
            model.jnt_range[joint_id, 1]
        )

        transformed_first = (
            mapping.position_sign * mujoco_lower
            + mapping.position_offset_rad
        )

        transformed_second = (
            mapping.position_sign * mujoco_upper
            + mapping.position_offset_rad
        )

        expected_ros_lower = min(
            transformed_first,
            transformed_second,
        )

        expected_ros_upper = max(
            transformed_first,
            transformed_second,
        )

        if not _close(
            ros_lower,
            expected_ros_lower,
            tolerance,
        ):
            raise MappingValidationError(

                    f"Lower-limit mismatch for '{mapping.ros_joint}': "
                    f"URDF={ros_lower}, "
                    f"mapped MuJoCo={expected_ros_lower}"

            )

        if not _close(
            ros_upper,
            expected_ros_upper,
            tolerance,
        ):
            raise MappingValidationError(

                    f"Upper-limit mismatch for '{mapping.ros_joint}': "
                    f"URDF={ros_upper}, "
                    f"mapped MuJoCo={expected_ros_upper}"

            )

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            mapping.mujoco_actuator,
        )

        if actuator_id < 0:
            raise MappingValidationError(

                    "Mapped MuJoCo actuator is missing: "
                    f"{mapping.mujoco_actuator}"

            )

        if (
            int(model.actuator_trntype[actuator_id])
            != int(mujoco.mjtTrn.mjTRN_JOINT)
        ):
            raise MappingValidationError(

                    f"MuJoCo actuator '{mapping.mujoco_actuator}' "
                    "must use a joint transmission"

            )

        targeted_joint_id = int(
            model.actuator_trnid[actuator_id, 0]
        )

        if targeted_joint_id != joint_id:
            raise MappingValidationError(

                    f"MuJoCo actuator '{mapping.mujoco_actuator}' "
                    f"does not target '{mapping.mujoco_joint}'"

            )

        home_mujoco = float(
            model.key_qpos[
                keyframe_id,
                qpos_address,
            ]
        )

        if not math.isfinite(home_mujoco):
            raise MappingValidationError(

                    f"MuJoCo home position for "
                    f"'{mapping.mujoco_joint}' is non-finite"

            )

        home_ros = (
            mapping.position_sign * home_mujoco
            + mapping.position_offset_rad
        )

        expected_home_ros = (
            contract.home_reference
            .positions_rad_by_ros_joint[
                mapping.ros_joint
            ]
        )

        if not _close(
            home_ros,
            expected_home_ros,
            tolerance,
        ):
            raise MappingValidationError(

                    f"Home-reference mismatch for "
                    f"'{mapping.ros_joint}': "
                    f"contract={expected_home_ros}, "
                    f"mapped MuJoCo={home_ros}"

            )

        if (
            home_ros < ros_lower - tolerance
            or home_ros > ros_upper + tolerance
        ):
            raise MappingValidationError(

                    f"Home position for '{mapping.ros_joint}' "
                    "lies outside its URDF joint limits"

            )

        home_positions.append(home_ros)

    if qpos_addresses != sorted(qpos_addresses):
        raise MappingValidationError(

                "MuJoCo joint qpos ordering does not match "
                "the mapping contract"

        )

    if len(set(qpos_addresses)) != contract.joint_count:
        raise MappingValidationError(
            "Mapped MuJoCo joints do not have unique qpos addresses"
        )

    return MappingValidationReport(
        joint_count=contract.joint_count,
        ros_joint_order=ros_order,
        mujoco_joint_order=tuple(
            mapping.mujoco_joint
            for mapping in contract.joints
        ),
        mujoco_qpos_addresses=tuple(qpos_addresses),
        home_positions_rad=tuple(home_positions),
    )