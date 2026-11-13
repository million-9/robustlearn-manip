#!/usr/bin/env python3
"""Publish read-only Panda insertion alignment markers for RViz."""
# ruff: noqa: I001

from __future__ import annotations

import math
import time

from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.node import Node
from robustlearn.sim.shadow_kinematics import ShadowInsertionKinematics
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray


JOINT_NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))
MARKER_TOPIC = '/robustlearn/cartesian_alignment_markers'
MARKER_NAMESPACE = 'robustlearn_insertion'


def point(values):
    """Convert one finite XYZ position to a ROS Point."""
    coordinates = np.asarray(values, dtype=float)

    if (
        coordinates.shape != (3,)
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError('Marker position must be finite XYZ')

    result = Point()
    result.x, result.y, result.z = map(float, coordinates)
    return result


def make_marker(identifier, marker_type, stamp):
    """Construct a marker with a common world-frame contract."""
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.header.stamp = stamp
    marker.ns = MARKER_NAMESPACE
    marker.id = identifier
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 300_000_000
    return marker


def alignment_markers(snapshot, stamp):
    """Build marker geometry from one consistent FK snapshot."""
    peg = np.asarray(snapshot.peg_tip_position, dtype=float)
    target = np.asarray(
        snapshot.receptacle_center_position, dtype=float
    )
    axis = np.asarray(snapshot.insertion_axis_position, dtype=float)

    if not all(
        values.shape == (3,) and np.all(np.isfinite(values))
        for values in (peg, target, axis)
    ):
        raise ValueError('Invalid insertion-task geometry')

    error_mm = (target - peg) * 1000.0
    distance_mm = float(np.linalg.norm(error_mm))

    tip_marker = make_marker(0, Marker.SPHERE, stamp)
    tip_marker.pose.position = point(peg)
    tip_marker.scale.x = 0.012
    tip_marker.scale.y = 0.012
    tip_marker.scale.z = 0.012
    tip_marker.color.r = 0.0
    tip_marker.color.g = 0.85
    tip_marker.color.b = 1.0

    target_marker = make_marker(1, Marker.SPHERE, stamp)
    target_marker.pose.position = point(target)
    target_marker.scale.x = 0.012
    target_marker.scale.y = 0.012
    target_marker.scale.z = 0.012
    target_marker.color.r = 0.1
    target_marker.color.g = 1.0
    target_marker.color.b = 0.1

    gap_marker = make_marker(2, Marker.LINE_STRIP, stamp)
    gap_marker.points = [point(peg), point(target)]
    gap_marker.scale.x = 0.003
    gap_marker.color.r = 1.0
    gap_marker.color.g = 0.6
    gap_marker.color.b = 0.0

    axis_marker = make_marker(3, Marker.LINE_STRIP, stamp)
    axis_marker.points = [point(target), point(axis)]
    axis_marker.scale.x = 0.003
    axis_marker.color.r = 0.2
    axis_marker.color.g = 0.5
    axis_marker.color.b = 1.0

    text_marker = make_marker(4, Marker.TEXT_VIEW_FACING, stamp)
    label_position = (peg + target) / 2.0
    label_position[2] += 0.045
    text_marker.pose.position = point(label_position)
    text_marker.scale.z = 0.025
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 1.0
    text_marker.text = (
        f'Peg-to-target XYZ (mm): '
        f'{error_mm[0]:+.2f}, '
        f'{error_mm[1]:+.2f}, '
        f'{error_mm[2]:+.2f} '
        f'| norm={distance_mm:.2f}'
    )

    result = MarkerArray()
    result.markers = [
        tip_marker,
        target_marker,
        gap_marker,
        axis_marker,
        text_marker,
    ]
    return result


class CartesianAlignmentMarkers(Node):
    """Visualize measured geometry without sending robot commands."""

    def __init__(self):
        super().__init__('cartesian_alignment_markers')

        self._shadow = ShadowInsertionKinematics()
        self._positions = None
        self._last_joint_time = None

        self._publisher = self.create_publisher(
            MarkerArray, MARKER_TOPIC, 10
        )
        self._subscription = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10
        )
        self._timer = self.create_timer(0.1, self._publish)

        self.get_logger().info(
            'Read-only Cartesian alignment markers available on '
            + MARKER_TOPIC
        )

    def _on_joint_state(self, message):
        if (
            len(message.name) != len(message.position)
            or len(set(message.name)) != len(message.name)
        ):
            self._positions = None
            return

        values = dict(zip(
            message.name, message.position, strict=True
        ))

        if not all(name in values for name in JOINT_NAMES):
            self._positions = None
            return

        positions = np.asarray(
            [values[name] for name in JOINT_NAMES],
            dtype=float,
        )

        if not np.all(np.isfinite(positions)):
            self._positions = None
            return

        self._positions = positions
        self._last_joint_time = time.monotonic()

    def _publish(self):
        if (
            self._positions is None
            or self._last_joint_time is None
        ):
            return

        age = time.monotonic() - self._last_joint_time

        if not math.isfinite(age) or not 0 <= age <= 0.25:
            return

        try:
            snapshot = self._shadow.update(self._positions)
            message = alignment_markers(
                snapshot,
                self.get_clock().now().to_msg(),
            )
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warning(
                f'Alignment geometry rejected: {exc}'
            )
            return

        self._publisher.publish(message)


def main(args=None):
    """Run the read-only alignment publisher."""
    rclpy.init(args=args)
    node = None

    try:
        node = CartesianAlignmentMarkers()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
