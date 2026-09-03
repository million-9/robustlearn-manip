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

import time
import unittest

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_testing.actions import ReadyToTest
from pytest import mark
from rclpy import (
    create_node,
    init,
    ok,
    shutdown,
    spin_once,
    spin_until_future_complete,
)
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]

TARGET_POSITIONS = [
    0.15,
    -0.65,
    0.10,
    -2.10,
    0.10,
    1.70,
    0.65,
]

POSITION_TOLERANCE = 0.02


@mark.launch_test
def generate_test_description():
    package_share = get_package_share_directory(
        'robustlearn_moveit_config'
    )

    launch_file = (
        package_share
        + '/launch/moveit_demo.launch.py'
    )

    week3_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            'use_rviz': 'false',
        }.items(),
    )

    return LaunchDescription(
        [
            week3_stack,
            ReadyToTest(),
        ]
    )


class TestWeek3PandaAcceptance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init()

    @classmethod
    def tearDownClass(cls):
        if ok():
            shutdown()

    def setUp(self):
        self.node = create_node(
            'week3_panda_acceptance_test'
        )

        self.latest_joint_state = None

        self.joint_state_subscription = (
            self.node.create_subscription(
                JointState,
                '/joint_states',
                self._joint_state_callback,
                qos_profile_sensor_data,
            )
        )

    def tearDown(self):
        self.node.destroy_node()

    def _joint_state_callback(self, message):
        self.latest_joint_state = message

    def _spin_until(self, condition, timeout_sec):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            spin_once(
                self.node,
                timeout_sec=0.1,
            )

            if condition():
                return True

        return False

    def _has_all_arm_joints(self):
        if self.latest_joint_state is None:
            return False

        available_joints = set(
            self.latest_joint_state.name
        )

        return all(
            joint_name in available_joints
            for joint_name in JOINT_NAMES
        )

    def _target_reached(self):
        if not self._has_all_arm_joints():
            return False

        positions = dict(
            zip(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
                strict=True,
            )
        )

        return all(
            abs(
                positions[joint_name] - target_position
            ) <= POSITION_TOLERANCE
            for joint_name, target_position in zip(
                JOINT_NAMES,
                TARGET_POSITIONS,
                strict=True,
            )
        )

    def _wait_for_active_controllers(
        self,
        controller_client,
        timeout_sec,
    ):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            if not controller_client.service_is_ready():
                spin_once(
                    self.node,
                    timeout_sec=0.1,
                )
                continue

            future = controller_client.call_async(
                ListControllers.Request()
            )

            spin_until_future_complete(
                self.node,
                future,
                timeout_sec=2.0,
            )

            if not future.done():
                continue

            response = future.result()

            if response is None:
                continue

            controller_states = {
                controller.name: controller.state
                for controller in response.controller
            }

            joint_state_broadcaster_active = (
                controller_states.get(
                    'joint_state_broadcaster'
                )
                == 'active'
            )

            panda_arm_controller_active = (
                controller_states.get(
                    'panda_arm_controller'
                )
                == 'active'
            )

            if (
                joint_state_broadcaster_active
                and panda_arm_controller_active
            ):
                return True

            time.sleep(0.2)

        return False

    def test_week3_control_acceptance(self):
        controller_client = self.node.create_client(
            ListControllers,
            '/controller_manager/list_controllers',
        )

        self.assertTrue(
            controller_client.wait_for_service(
                timeout_sec=30.0
            ),
            (
                'controller_manager list_controllers '
                'service did not become available'
            ),
        )

        self.assertTrue(
            self._wait_for_active_controllers(
                controller_client,
                timeout_sec=30.0,
            ),
            (
                'joint_state_broadcaster and '
                'panda_arm_controller did not become active'
            ),
        )

        self.assertTrue(
            self._spin_until(
                self._has_all_arm_joints,
                timeout_sec=20.0,
            ),
            (
                'did not receive all seven Panda arm joints '
                'on /joint_states'
            ),
        )

        action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            (
                '/panda_arm_controller/'
                'follow_joint_trajectory'
            ),
        )

        self.assertTrue(
            action_client.wait_for_server(
                timeout_sec=20.0
            ),
            (
                'FollowJointTrajectory action server '
                'did not become available'
            ),
        )

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = TARGET_POSITIONS
        point.time_from_start = Duration(
            sec=3,
            nanosec=0,
        )

        goal.trajectory.points = [
            point,
        ]

        goal_future = action_client.send_goal_async(
            goal
        )

        spin_until_future_complete(
            self.node,
            goal_future,
            timeout_sec=10.0,
        )

        self.assertTrue(
            goal_future.done(),
            'trajectory goal request timed out',
        )

        goal_handle = goal_future.result()

        self.assertIsNotNone(
            goal_handle,
            'trajectory goal returned no goal handle',
        )

        self.assertTrue(
            goal_handle.accepted,
            'trajectory goal was rejected',
        )

        result_future = goal_handle.get_result_async()

        spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=15.0,
        )

        self.assertTrue(
            result_future.done(),
            'trajectory execution timed out',
        )

        wrapped_result = result_future.result()

        self.assertIsNotNone(
            wrapped_result,
            'trajectory action returned no result',
        )

        self.assertEqual(
            wrapped_result.status,
            GoalStatus.STATUS_SUCCEEDED,
            'trajectory action did not finish successfully',
        )

        self.assertEqual(
            wrapped_result.result.error_code,
            FollowJointTrajectory.Result.SUCCESSFUL,
            (
                'controller reported trajectory failure: '
                + wrapped_result.result.error_string
            ),
        )

        self.assertTrue(
            self._spin_until(
                self._target_reached,
                timeout_sec=5.0,
            ),
            (
                'final Panda joint state did not reach '
                'the commanded target'
            ),
        )

        final_positions = dict(
            zip(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
                strict=True,
            )
        )

        for joint_name, target_position in zip(
            JOINT_NAMES,
            TARGET_POSITIONS,
            strict=True,
        ):
            self.assertAlmostEqual(
                final_positions[joint_name],
                target_position,
                delta=POSITION_TOLERANCE,
                msg=(
                    f'{joint_name} did not reach '
                    f'target {target_position}'
                ),
            )

        action_client.destroy()
