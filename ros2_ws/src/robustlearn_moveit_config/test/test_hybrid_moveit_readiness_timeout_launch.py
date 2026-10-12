# Copyright 2026 Mohamed Musthafa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: I001

import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg


os.environ['ROS_DOMAIN_ID'] = '183'


PANDA_ARM_JOINT_NAMES = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
)


VALID_ARM_POSITIONS = (
    0.10,
    -0.10,
    0.10,
    -1.70,
    0.10,
    1.70,
    -0.70,
)


MOVEIT_READY_TIMEOUT_S = 0.5


@pytest.mark.launch_test
def generate_test_description():
    coordinator = launch_ros.actions.Node(
        package='robustlearn_moveit_config',
        executable='hybrid_insertion_coordinator.py',
        name='hybrid_insertion_coordinator',
        output='screen',
        parameters=[
            {
                'handoff_tolerance_rad': 0.002,
                'moveit_ready_timeout_s': MOVEIT_READY_TIMEOUT_S,
                'planning_timeout_s': 2.0,
                'execution_timeout_s': 2.0,
                'handoff_settle_s': 0.1,
            }
        ],
    )

    return (
        launch.LaunchDescription(
            [
                coordinator,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'coordinator': coordinator,
        },
    )


class TestHybridMoveItReadinessTimeout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(
            'hybrid_moveit_readiness_timeout_test'
        )

        self.publisher = self.node.create_publisher(
            sensor_msgs.msg.JointState,
            '/joint_states',
            qos_profile_sensor_data,
        )

    def tearDown(self):
        self.node.destroy_publisher(
            self.publisher
        )

        self.node.destroy_node()

    def _publish_valid_joint_state(self):
        message = sensor_msgs.msg.JointState()

        message.header.stamp = (
            self.node.get_clock().now().to_msg()
        )

        message.name = list(
            PANDA_ARM_JOINT_NAMES
        )

        message.position = list(
            VALID_ARM_POSITIONS
        )

        message.velocity = [
            0.0
            for _ in PANDA_ARM_JOINT_NAMES
        ]

        message.effort = [
            0.0
            for _ in PANDA_ARM_JOINT_NAMES
        ]

        self.publisher.publish(
            message
        )

    def test_missing_moveit_service_fails_boundedly(
        self,
        proc_output,
        coordinator,
    ):
        publish_deadline = (
            time.monotonic()
            + 1.5
        )

        while time.monotonic() < publish_deadline:
            self._publish_valid_joint_state()

            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

            time.sleep(0.05)

        proc_output.assertWaitFor(
            (
                'Hybrid insertion failed: '
                'MoveIt planning service did not become ready within '
                '0.500 s'
            ),
            process=coordinator,
            timeout=5.0,
        )
