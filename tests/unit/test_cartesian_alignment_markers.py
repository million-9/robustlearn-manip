"""Regression tests for read-only ROS Cartesian alignment geometry."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("visualization_msgs")

from builtin_interfaces.msg import Time  # noqa: E402

from robustlearn.sim.shadow_kinematics import (  # noqa: E402
    ShadowInsertionKinematics,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ros2_ws/src/robustlearn_control/scripts/cartesian_alignment_markers.py"
)

spec = importlib.util.spec_from_file_location("alignment_under_test", SCRIPT)
assert spec is not None and spec.loader is not None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _snapshot():
    model = ShadowInsertionKinematics()
    positions = np.array([
        0.10, -0.10, 0.10, -1.70, 0.10, 1.70, -0.70
    ])
    return model.update(positions)


def test_marker_geometry_matches_measured_shadow_fk():
    snapshot = _snapshot()
    markers = module.alignment_markers(snapshot, Time()).markers

    assert len(markers) == 5
    assert [marker.id for marker in markers] == list(range(5))
    assert all(marker.header.frame_id == "world" for marker in markers)
    assert all(
        marker.lifetime.nanosec == 300_000_000
        for marker in markers
    )

    def xyz(value):
        return np.array([value.x, value.y, value.z])

    np.testing.assert_allclose(
        xyz(markers[0].pose.position),
        snapshot.peg_tip_position,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        xyz(markers[1].pose.position),
        snapshot.receptacle_center_position,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        xyz(markers[2].points[0]),
        snapshot.peg_tip_position,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        xyz(markers[2].points[1]),
        snapshot.receptacle_center_position,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        xyz(markers[3].points[1]),
        snapshot.insertion_axis_position,
        atol=1e-9,
    )
    assert "Peg-to-target XYZ (mm)" in markers[4].text


def test_nonfinite_geometry_is_rejected():
    snapshot = replace(
        _snapshot(),
        peg_tip_position=np.array([np.nan, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="geometry"):
        module.alignment_markers(snapshot, Time())


def test_invalid_point_shape_is_rejected():
    with pytest.raises(ValueError, match="finite XYZ"):
        module.point([0.0, 1.0])
