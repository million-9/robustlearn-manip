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

import math
import os
import pathlib
import time
import unittest

from action_msgs.msg import GoalStatus
import ament_index_python.packages
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import controller_manager_msgs.srv
import launch
import launch.actions
import launch.launch_description_sources
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.action import ActionClient
import sensor_msgs.msg
from trajectory_msgs.msg import JointTrajectoryPoint


# Keep this test isolated from the Week 7 launch acceptance test and
# any other controller_manager instances running concurrently.
os.environ['ROS_DOMAIN_ID'] = '164'


PANDA_JOINTS = tuple(
    f'panda_joint{index}'
    for index in range(1, 8)
)

FINAL_POSITION_TOLERANCE = 0.01
JOINT1_DELTA = 0.03


@pytest.mark.launch_test
def generate_test_description():
    package_share = pathlib.Path(
        ament_index_python.packages.get_package_share_directory(
            'robustlearn_description'
        )
    )

    launch_file = (
        package_share
        / 'launch'
        / 'mujoco_panda_trajectory.launch.py'
    )

    repository_root = pathlib.Path(__file__).resolve().parents[4]
    model_path = (
        repository_root
        / 'robot_description'
        / 'mjcf'
        / 'insertion'
        / 'panda_insertion.xml'
    )

    if not model_path.is_file():
        raise RuntimeError(
            f'MuJoCo acceptance model not found: {model_path}'
        )

    mujoco_bringup = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            str(launch_file)
        ),
        launch_arguments={
            'model_path': str(model_path),
        }.items(),
    )

    return launch.LaunchDescription(
        [
            mujoco_bringup,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestMuJoCoTrajectoryBringup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node(
            'mujoco_trajectory_acceptance_test'
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _call_service(self, service_type, service_name):
        client = self.node.create_client(
            service_type,
            service_name,
        )

        self.assertTrue(
            client.wait_for_service(timeout_sec=10.0),
            f'Service did not become available: {service_name}',
        )

        future = client.call_async(service_type.Request())

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=5.0,
        )

        self.assertTrue(
            future.done(),
            f'Service call did not complete: {service_name}',
        )

        result = future.result()

        self.assertIsNotNone(
            result,
            f'Service returned no result: {service_name}',
        )

        self.node.destroy_client(client)

        return result

    def _wait_for_joint_state(self):
        received_messages = []

        def callback(message):
            received_messages.append(message)

        subscription = self.node.create_subscription(
            sensor_msgs.msg.JointState,
            '/joint_states',
            callback,
            10,
        )

        deadline = time.monotonic() + 5.0

        while (
            time.monotonic() < deadline
            and not received_messages
        ):
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

        self.node.destroy_subscription(subscription)

        self.assertTrue(
            received_messages,
            'No /joint_states message was received.',
        )

        message = received_messages[-1]

        self.assertEqual(
            tuple(message.name),
            PANDA_JOINTS,
        )

        self.assertEqual(
            len(message.position),
            len(PANDA_JOINTS),
        )

        self.assertEqual(
            len(message.velocity),
            len(PANDA_JOINTS),
        )

        self.assertTrue(
            all(math.isfinite(value) for value in message.position),
            'One or more Panda positions are non-finite.',
        )

        self.assertTrue(
            all(math.isfinite(value) for value in message.velocity),
            'One or more Panda velocities are non-finite.',
        )

        return message

    def test_direct_follow_joint_trajectory(self):
        hardware_deadline = time.monotonic() + 10.0
        panda_system = None
        last_hardware_components = []

        while time.monotonic() < hardware_deadline:
            hardware_response = self._call_service(
                controller_manager_msgs.srv.ListHardwareComponents,
                '/controller_manager/list_hardware_components',
            )

            last_hardware_components = [
                f'{component.name}:{component.state.label}'
                for component in hardware_response.component
            ]

            panda_components = [
                component
                for component in hardware_response.component
                if component.name == 'PandaSystem'
            ]

            if (
                len(panda_components) == 1
                and panda_components[0].state.id == 3
            ):
                panda_system = panda_components[0]
                break

            time.sleep(0.1)

        self.assertIsNotNone(
            panda_system,
            (
                'PandaSystem did not become active. '
                f'Last hardware components: {last_hardware_components}'
            ),
        )

        self.assertEqual(
            panda_system.plugin_name,
            'robustlearn_mujoco_hardware/MuJoCoSystem',
        )

        self.assertEqual(
            panda_system.state.id,
            3,
            'PandaSystem hardware is not active.',
        )

        self.assertEqual(
            panda_system.state.label,
            'active',
        )

        self.assertEqual(
            panda_system.rw_rate,
            500,
        )

        controller_deadline = time.monotonic() + 10.0
        controllers = {}

        while time.monotonic() < controller_deadline:
            response = self._call_service(
                controller_manager_msgs.srv.ListControllers,
                '/controller_manager/list_controllers',
            )

            controllers = {
                controller.name: controller
                for controller in response.controller
            }

            if (
                controllers.get('joint_state_broadcaster')
                and controllers.get('panda_arm_controller')
                and controllers['joint_state_broadcaster'].state == 'active'
                and controllers['panda_arm_controller'].state == 'active'
            ):
                break

            time.sleep(0.1)

        self.assertIn(
            'joint_state_broadcaster',
            controllers,
        )

        self.assertIn(
            'panda_arm_controller',
            controllers,
        )

        self.assertEqual(
            controllers['joint_state_broadcaster'].state,
            'active',
        )

        self.assertEqual(
            controllers['panda_arm_controller'].state,
            'active',
        )

        interface_response = self._call_service(
            controller_manager_msgs.srv.ListHardwareInterfaces,
            '/controller_manager/list_hardware_interfaces',
        )

        command_interfaces = {
            interface.name: interface
            for interface in interface_response.command_interfaces
        }

        for joint_name in PANDA_JOINTS:
            interface_name = f'{joint_name}/position'

            self.assertIn(
                interface_name,
                command_interfaces,
            )

            self.assertTrue(
                command_interfaces[interface_name].is_available,
                f'{interface_name} is not available.',
            )

            self.assertTrue(
                command_interfaces[interface_name].is_claimed,
                f'{interface_name} is not claimed.',
            )

        before = self._wait_for_joint_state()

        target_positions = list(before.position)
        target_positions[0] += JOINT1_DELTA

        action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
        )

        self.assertTrue(
            action_client.wait_for_server(timeout_sec=10.0),
            'FollowJointTrajectory action server did not become available.',
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(PANDA_JOINTS)

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = Duration(
            sec=2,
            nanosec=0,
        )

        goal.trajectory.points = [point]

        send_future = action_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self.node,
            send_future,
            timeout_sec=5.0,
        )

        self.assertTrue(
            send_future.done(),
            'Trajectory goal request did not complete.',
        )

        goal_handle = send_future.result()

        self.assertIsNotNone(
            goal_handle,
            'Trajectory goal returned no goal handle.',
        )

        self.assertTrue(
            goal_handle.accepted,
            'Trajectory goal was rejected.',
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=10.0,
        )

        self.assertTrue(
            result_future.done(),
            'Trajectory result did not complete.',
        )

        wrapped_result = result_future.result()

        self.assertIsNotNone(
            wrapped_result,
            'Trajectory action returned no result.',
        )

        self.assertEqual(
            wrapped_result.status,
            GoalStatus.STATUS_SUCCEEDED,
        )

        self.assertEqual(
            wrapped_result.result.error_code,
            FollowJointTrajectory.Result.SUCCESSFUL,
            wrapped_result.result.error_string,
        )

        after = self._wait_for_joint_state()

        self.assertGreater(
            abs(after.position[0] - before.position[0]),
            0.02,
            'panda_joint1 did not move meaningfully.',
        )

        final_errors = [
            abs(actual - target)
            for actual, target in zip(
                after.position,
                target_positions,
                strict=True,
            )
        ]

        self.assertTrue(
            all(
                error <= FINAL_POSITION_TOLERANCE
                for error in final_errors
            ),
            (
                'Final Panda joint error exceeded '
                f'{FINAL_POSITION_TOLERANCE} rad: {final_errors}'
            ),
        )

        action_client.destroy()
