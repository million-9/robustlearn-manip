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
import geometry_msgs.msg
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

WRIST_FT_INTERFACES = (
    'wrist_ft/force.x',
    'wrist_ft/force.y',
    'wrist_ft/force.z',
    'wrist_ft/torque.x',
    'wrist_ft/torque.y',
    'wrist_ft/torque.z',
)


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
        / 'mujoco_panda.launch.py'
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

        return received_messages[-1]

    def _wait_for_wrist_wrench(self):
        received_messages = []

        def callback(message):
            received_messages.append(message)

        subscription = self.node.create_subscription(
            geometry_msgs.msg.WrenchStamped,
            '/wrist_ft/wrench',
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
            'No /wrist_ft/wrench message was received.',
        )

        return received_messages[-1]

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
        controllers = {}
        last_controller_states = []

        while time.monotonic() < controller_deadline:
            controller_response = self._call_service(
                controller_manager_msgs.srv.ListControllers,
                '/controller_manager/list_controllers',
            )

            controllers = {
                controller.name: controller
                for controller in controller_response.controller
            }

            last_controller_states = [
                f'{controller.name}:{controller.state}'
                for controller in controller_response.controller
            ]

            joint_state_broadcaster = controllers.get(
                'joint_state_broadcaster'
            )
            wrist_ft_broadcaster = controllers.get(
                'wrist_ft_broadcaster'
            )

            if (
                joint_state_broadcaster
                and wrist_ft_broadcaster
                and joint_state_broadcaster.state == 'active'
                and wrist_ft_broadcaster.state == 'active'
            ):
                break

            time.sleep(0.1)

        self.assertIn(
            'joint_state_broadcaster',
            controllers,
            (
                'joint_state_broadcaster was not loaded. '
                f'Last controller states: {last_controller_states}'
            ),
        )

        self.assertIn(
            'wrist_ft_broadcaster',
            controllers,
            (
                'wrist_ft_broadcaster was not loaded. '
                f'Last controller states: {last_controller_states}'
            ),
        )

        self.assertEqual(
            controllers['joint_state_broadcaster'].state,
            'active',
        )

        self.assertEqual(
            controllers['wrist_ft_broadcaster'].state,
            'active',
        )

        interface_response = self._call_service(
            controller_manager_msgs.srv.ListHardwareInterfaces,
            '/controller_manager/list_hardware_interfaces',
        )

        state_interfaces = {
            interface.name: interface
            for interface in interface_response.state_interfaces
        }

        for interface_name in WRIST_FT_INTERFACES:
            self.assertIn(
                interface_name,
                state_interfaces,
                f'Missing wrist F/T state interface: {interface_name}',
            )

            self.assertTrue(
                state_interfaces[interface_name].is_available,
                f'Wrist F/T state interface unavailable: {interface_name}',
            )

        joint_state = self._wait_for_joint_state()

        self.assertEqual(
            set(joint_state.name),
            set(PANDA_JOINTS),
        )

        self.assertEqual(
            len(joint_state.position),
            len(PANDA_JOINTS),
        )

        self.assertEqual(
            len(joint_state.velocity),
            len(PANDA_JOINTS),
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in joint_state.position
            ),
            'One or more Panda positions are non-finite.',
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in joint_state.velocity
            ),
            'One or more Panda velocities are non-finite.',
        )

        wrench = self._wait_for_wrist_wrench()

        self.assertEqual(
            wrench.header.frame_id,
            'panda_wrist_ft',
        )

        wrench_values = (
            wrench.wrench.force.x,
            wrench.wrench.force.y,
            wrench.wrench.force.z,
            wrench.wrench.torque.x,
            wrench.wrench.torque.y,
            wrench.wrench.torque.z,
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in wrench_values
            ),
            (
                'One or more wrist force/torque values are non-finite: '
                f'{wrench_values}'
            ),
        )
