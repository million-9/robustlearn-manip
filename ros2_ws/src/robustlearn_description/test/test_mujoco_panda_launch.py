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

import math
import os
import pathlib
import time
import unittest

import ament_index_python.packages
import controller_manager_msgs.srv
import launch
import launch.actions
import launch.launch_description_sources
import launch_testing
import launch_testing.actions
import pytest
import rclpy
import sensor_msgs.msg

# Isolate this launch test from other controller_manager instances that
# may be started concurrently by colcon test.
os.environ['ROS_DOMAIN_ID'] = '163'


PANDA_JOINTS = tuple(
    f'panda_joint{index}'
    for index in range(1, 8)
)


@pytest.mark.launch_test
def generate_test_description():
    package_share = pathlib.Path(
        ament_index_python.packages.get_package_share_directory('robustlearn_description')
    )

    launch_file = (
        package_share / 'launch' / 'mujoco_panda.launch.py'
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
        launch.launch_description_sources.PythonLaunchDescriptionSource(str(launch_file)),
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


class TestMuJoCoPandaBringup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node(
            'mujoco_panda_acceptance_test'
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

    def test_mujoco_hardware_acceptance(self):
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
        broadcaster = None
        last_controller_states = []

        while time.monotonic() < controller_deadline:
            controller_response = self._call_service(
                controller_manager_msgs.srv.ListControllers,
                '/controller_manager/list_controllers',
            )

            last_controller_states = [
                f'{controller.name}:{controller.state}'
                for controller in controller_response.controller
            ]

            broadcasters = [
                controller
                for controller in controller_response.controller
                if controller.name == 'joint_state_broadcaster'
            ]

            if (
                len(broadcasters) == 1
                and broadcasters[0].state == 'active'
            ):
                broadcaster = broadcasters[0]
                break

            time.sleep(0.1)

        self.assertIsNotNone(
            broadcaster,
            (
                'joint_state_broadcaster did not become active. '
                f'Last controller states: {last_controller_states}'
            ),
        )

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
            set(message.name),
            set(PANDA_JOINTS),
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
            all(
                math.isfinite(value)
                for value in message.position
            ),
            'One or more Panda positions are non-finite.',
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in message.velocity
            ),
            'One or more Panda velocities are non-finite.',
        )
