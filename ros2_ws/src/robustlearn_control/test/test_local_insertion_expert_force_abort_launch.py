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

import threading
import time
import unittest

from geometry_msgs.msg import WrenchStamped
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import String


PANDA_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]


CANONICAL_PRE_INSERTION_JOINT_POSITIONS = [
    0.0,
    0.013358,
    -0.000005,
    -1.583575,
    -0.000669,
    1.568475,
    -0.785300,
]


EXPECTED_FAILURE_PREFIX = (
    'failed|Classical insertion expert aborted because lateral '
    'contact force '
)


@pytest.mark.launch_test
def generate_test_description():
    local_expert = launch_ros.actions.Node(
        package='robustlearn_control',
        executable='local_insertion_expert.py',
        name='local_insertion_expert',
        output='screen',
        prefix='uv run',
        parameters=[
            {
                'command_enabled': False,
                'start_armed': True,
                'control_period_s': 0.02,
                'trajectory_duration_s': 0.10,
                'goal_timeout_s': 1.0,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'warn',
            '--log-level',
            'local_insertion_expert:=debug',
        ],
    )

    return (
        launch.LaunchDescription(
            [
                local_expert,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'local_expert': local_expert,
        },
    )


class TestLocalInsertionExpertForceAbort(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.node = rclpy.create_node(
            'local_insertion_expert_force_abort_test'
        )

        cls.executor = MultiThreadedExecutor(
            num_threads=2
        )

        cls.executor.add_node(
            cls.node
        )

        cls.executor_thread = threading.Thread(
            target=cls.executor.spin,
            daemon=True,
        )

        cls.executor_thread.start()

        cls.joint_state_publisher = cls.node.create_publisher(
            JointState,
            '/joint_states',
            10,
        )

        cls.wrench_publisher = cls.node.create_publisher(
            WrenchStamped,
            '/wrist_ft/wrench',
            10,
        )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        cls.status_lock = threading.Lock()
        cls.status_history = []

        cls.status_subscription = cls.node.create_subscription(
            String,
            '/local_insertion_expert/status',
            cls.status_callback,
            status_qos,
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_subscription(
            cls.status_subscription
        )

        cls.executor.shutdown()

        cls.executor_thread.join(
            timeout=2.0
        )

        cls.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    @classmethod
    def status_callback(
        cls,
        message: String,
    ):
        with cls.status_lock:
            cls.status_history.append(
                message.data
            )

    @classmethod
    def publish_joint_state(cls):
        message = JointState()

        message.header.stamp = (
            cls.node.get_clock().now().to_msg()
        )

        message.name = list(
            PANDA_JOINT_NAMES
        )

        message.position = list(
            CANONICAL_PRE_INSERTION_JOINT_POSITIONS
        )

        message.velocity = [
            0.0
            for _ in PANDA_JOINT_NAMES
        ]

        cls.joint_state_publisher.publish(
            message
        )

    @classmethod
    def publish_force(
        cls,
        force_xyz: tuple[float, float, float],
    ):
        message = WrenchStamped()

        message.header.stamp = (
            cls.node.get_clock().now().to_msg()
        )

        message.header.frame_id = (
            'panda_wrist_ft'
        )

        message.wrench.force.x = force_xyz[0]
        message.wrench.force.y = force_xyz[1]
        message.wrench.force.z = force_xyz[2]

        cls.wrench_publisher.publish(
            message
        )

    @classmethod
    def failure_status(cls) -> str | None:
        with cls.status_lock:
            for status in cls.status_history:
                if status.startswith(
                    EXPECTED_FAILURE_PREFIX
                ):
                    return status

        return None

    @classmethod
    def wait_for_failure(
        cls,
        timeout_sec: float,
    ) -> str | None:
        deadline = (
            time.monotonic()
            + timeout_sec
        )

        while time.monotonic() < deadline:
            status = cls.failure_status()

            if status is not None:
                return status

            time.sleep(0.01)

        return cls.failure_status()

    def test_excessive_lateral_contact_force_aborts(
        self,
        proc_output,
        local_expert,
    ):
        # The wrench arrives in the wrist sensor frame and is transformed by
        # the ROS expert into the world/task frame. Cycling strong forces
        # along all three orthogonal sensor axes guarantees that at least one
        # has a lateral component above the project-owned 5 N lateral limit,
        # independent of the wrist-frame orientation.
        sensor_forces = (
            (8.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        )

        deadline = (
            time.monotonic()
            + 3.0
        )

        index = 0

        while (
            time.monotonic() < deadline
            and self.failure_status() is None
        ):
            self.publish_joint_state()

            self.publish_force(
                sensor_forces[
                    index % len(sensor_forces)
                ]
            )

            index += 1
            time.sleep(0.05)

        failure_status = self.wait_for_failure(
            timeout_sec=2.0
        )

        self.assertIsNotNone(
            failure_status,
            (
                'Local insertion expert did not publish an excessive-force '
                f'failure; history={self.status_history}'
            ),
        )

        assert failure_status is not None

        self.assertIn(
            'configured limit 5.000000 N',
            failure_status,
        )

        proc_output.assertWaitFor(
            (
                'Local insertion expert aborted: '
                'Classical insertion expert aborted because lateral '
                'contact force'
            ),
            process=local_expert,
            timeout=5.0,
        )

        with self.status_lock:
            statuses = list(
                self.status_history
            )

        self.assertIn(
            'running',
            statuses,
            (
                'Expected running status before force failure; '
                f'history={statuses}'
            ),
        )

        self.assertFalse(
            any(
                status.startswith('success|')
                for status in statuses
            ),
            (
                'Excessive lateral force must never be reported as success; '
                f'history={statuses}'
            ),
        )
