"""Tests for the Panda ROS/URDF-to-MuJoCo joint mapping contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustlearn.panda_joint_mapping import (
    DEFAULT_CONTRACT_PATH,
    MappingValidationError,
    load_mapping_contract,
    validate_mapping_contract,
)

EXPECTED_ROS_JOINTS = tuple(
    f"panda_joint{index}"
    for index in range(1, 8)
)

EXPECTED_MUJOCO_JOINTS = tuple(
    f"joint{index}"
    for index in range(1, 8)
)

EXPECTED_HOME = (
    0.0,
    0.0,
    0.0,
    -1.57079,
    0.0,
    1.57079,
    -0.7853,
)


def _load_raw_contract() -> dict[str, object]:
    raw = json.loads(DEFAULT_CONTRACT_PATH.read_text())

    assert isinstance(raw, dict)

    return raw


def _write_contract(
    tmp_path: Path,
    contract: dict[str, object],
) -> Path:
    path = tmp_path / "panda_joint_mapping.json"

    path.write_text(
        json.dumps(
            contract,
            indent=2,
        )
        + "\n"
    )

    return path


def _joint_entries(
    contract: dict[str, object],
) -> list[dict[str, object]]:
    joints = contract["joints"]

    assert isinstance(joints, list)
    assert all(
        isinstance(joint, dict)
        for joint in joints
    )

    return joints


def test_real_mapping_contract_validates() -> None:
    report = validate_mapping_contract()

    assert report.joint_count == 7
    assert report.ros_joint_order == EXPECTED_ROS_JOINTS
    assert report.mujoco_joint_order == EXPECTED_MUJOCO_JOINTS
    assert report.mujoco_qpos_addresses == tuple(range(7))
    assert report.home_positions_rad == EXPECTED_HOME


def test_mapping_contract_declares_identity_position_transform() -> None:
    contract = load_mapping_contract()

    assert tuple(
        mapping.ros_joint
        for mapping in contract.joints
    ) == EXPECTED_ROS_JOINTS

    assert tuple(
        mapping.mujoco_joint
        for mapping in contract.joints
    ) == EXPECTED_MUJOCO_JOINTS

    assert all(
        mapping.position_sign == 1.0
        for mapping in contract.joints
    )

    assert all(
        mapping.position_offset_rad == 0.0
        for mapping in contract.joints
    )


def test_duplicate_mujoco_joint_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[1]["mujoco_joint"] = joints[0]["mujoco_joint"]

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="duplicate joint names",
    ):
        load_mapping_contract(path)


def test_duplicate_ros_joint_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[1]["ros_joint"] = joints[0]["ros_joint"]

    home_reference = contract["home_reference"]
    assert isinstance(home_reference, dict)

    home_positions = home_reference[
        "positions_rad_by_ros_joint"
    ]
    assert isinstance(home_positions, dict)

    home_positions.pop("panda_joint2")

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="duplicate joint names",
    ):
        load_mapping_contract(path)


def test_duplicate_actuator_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[1]["mujoco_actuator"] = (
        joints[0]["mujoco_actuator"]
    )

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="duplicate actuator names",
    ):
        load_mapping_contract(path)


def test_invalid_position_sign_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[0]["position_sign"] = 0.0

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="position_sign",
    ):
        load_mapping_contract(path)


def test_missing_mujoco_joint_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[0]["mujoco_joint"] = "missing_joint"

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="Mapped MuJoCo joint is missing",
    ):
        validate_mapping_contract(
            contract_path=path,
        )


def test_wrong_joint_order_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()
    joints = _joint_entries(contract)

    joints[0], joints[1] = joints[1], joints[0]

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="URDF Panda joint ordering",
    ):
        validate_mapping_contract(
            contract_path=path,
        )


def test_home_reference_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()

    home_reference = contract["home_reference"]
    assert isinstance(home_reference, dict)

    home_positions = home_reference[
        "positions_rad_by_ros_joint"
    ]
    assert isinstance(home_positions, dict)

    home_positions["panda_joint4"] = -1.0

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="Home-reference mismatch",
    ):
        validate_mapping_contract(
            contract_path=path,
        )


def test_missing_home_joint_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _load_raw_contract()

    home_reference = contract["home_reference"]
    assert isinstance(home_reference, dict)

    home_positions = home_reference[
        "positions_rad_by_ros_joint"
    ]
    assert isinstance(home_positions, dict)

    home_positions.pop("panda_joint7")

    path = _write_contract(
        tmp_path,
        contract,
    )

    with pytest.raises(
        MappingValidationError,
        match="Home-reference joint names",
    ):
        load_mapping_contract(path)