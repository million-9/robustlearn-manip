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

import ament_index_python.packages
import controller_manager_msgs.srv
from geometry_msgs.msg import WrenchStamped
import launch
import launch.actions
import launch.launch_description_sources
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
import sensor_msgs.msg
from std_msgs.msg import String


os.environ['ROS_DOMAIN_ID'] = '186'


PANDA_JOINTS = tuple(
    f'panda_joint{index}'
    for index in range(1, 8)
)


PRE_INSERTION_MEASURED_JOINT_POSITIONS = (
    0.0,
    0.013358,
    -0.000005,
    -1.583575,
    -0.000669,
    1.568475,
    -0.785300,
)


EXPECTED_EXPERT_SUCCESS = (
    'success|canonical insertion task reached success'
)


MINIMUM_INITIAL_PREINSERTION_DISTANCE = 0.05
MINIMUM_OBSERVED_MOTION = 0.05


@pytest.mark.launch_test
def generate_test_description():
    package_share = pathlib.Path(
        ament_index_python.packages.get_package_share_directory(
            'robustlearn_moveit_config'
        )
    )

    launch_file = (
        package_share
        / 'launch'
        / 'hybrid_insertion.launch.py'
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
            f'MuJoCo hybrid acceptance model not found: {model_path}'
        )

    hybrid_insertion = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            str(launch_file)
        ),
        launch_arguments={
            'model_path': str(model_path),
            'use_rviz': 'false',
        }.items(),
    )

    return launch.LaunchDescription(
        [
            hybrid_insertion,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestHybridInsertionE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(
            'hybrid_insertion_e2e_test'
        )

        self.latest_joint_state = None
        self.first_arm_positions = None
        self.max_motion_from_start = 0.0

        self.latest_wrench = None
        self.finite_wrench_samples = 0

        self.status_history = []

        self.joint_subscription = self.node.create_subscription(
            sensor_msgs.msg.JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        self.wrench_subscription = self.node.create_subscription(
            WrenchStamped,
            '/wrist_ft/wrench',
            self._wrench_callback,
            qos_profile_sensor_data,
        )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.status_subscription = self.node.create_subscription(
            String,
            '/local_insertion_expert/status',
            self._status_callback,
            status_qos,
        )

    def tearDown(self):
        self.node.destroy_subscription(
            self.joint_subscription
        )

        self.node.destroy_subscription(
            self.wrench_subscription
        )

        self.node.destroy_subscription(
            self.status_subscription
        )

        self.node.destroy_node()

    def _status_callback(
        self,
        message: String,
    ):
        self.status_history.append(
            message.data
        )

    def _wrench_callback(
        self,
        message: WrenchStamped,
    ):
        values = (
            message.wrench.force.x,
            message.wrench.force.y,
            message.wrench.force.z,
            message.wrench.torque.x,
            message.wrench.torque.y,
            message.wrench.torque.z,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            return

        self.latest_wrench = message
        self.finite_wrench_samples += 1

    def _joint_state_callback(
        self,
        message: sensor_msgs.msg.JointState,
    ):
        self.latest_joint_state = message

        arm_positions = self._arm_positions(
            message
        )

        if arm_positions is None:
            return

        if self.first_arm_positions is None:
            self.first_arm_positions = tuple(
                arm_positions
            )
            return

        motion = max(
            abs(current - initial)
            for current, initial in zip(
                arm_positions,
                self.first_arm_positions,
                strict=True,
            )
        )

        self.max_motion_from_start = max(
            self.max_motion_from_start,
            motion,
        )

    def _arm_positions(
        self,
        message,
    ):
        if message is None:
            return None

        if len(message.name) != len(message.position):
            return None

        lookup = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        if not all(
            joint_name in lookup
            for joint_name in PANDA_JOINTS
        ):
            return None

        positions = tuple(
            float(
                lookup[joint_name]
            )
            for joint_name in PANDA_JOINTS
        )

        if not all(
            math.isfinite(value)
            for value in positions
        ):
            return None

        return positions

    def _spin_until(
        self,
        condition,
        timeout_sec,
    ):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

            if condition():
                return True

        return condition()

    def _call_service(
        self,
        service_type,
        service_name,
        timeout_sec=10.0,
    ):
        client = self.node.create_client(
            service_type,
            service_name,
        )

        self.assertTrue(
            client.wait_for_service(
                timeout_sec=timeout_sec
            ),
            f'Service unavailable: {service_name}',
        )

        future = client.call_async(
            service_type.Request()
        )

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=5.0,
        )

        self.assertTrue(
            future.done(),
            f'Service call timed out: {service_name}',
        )

        result = future.result()

        self.assertIsNotNone(
            result,
            f'Service returned no result: {service_name}',
        )

        self.node.destroy_client(
            client
        )

        return result

    def _hardware_is_active(self):
        response = self._call_service(
            controller_manager_msgs.srv.ListHardwareComponents,
            '/controller_manager/list_hardware_components',
        )

        panda_components = [
            component
            for component in response.component
            if component.name == 'PandaSystem'
        ]

        if len(panda_components) != 1:
            return False

        panda_system = panda_components[0]

        return (
            panda_system.state.label == 'active'
            and panda_system.plugin_name
            == 'robustlearn_mujoco_hardware/MuJoCoSystem'
        )

    def _controllers_are_active(self):
        response = self._call_service(
            controller_manager_msgs.srv.ListControllers,
            '/controller_manager/list_controllers',
        )

        controllers = {
            controller.name: controller.state
            for controller in response.controller
        }

        return (
            controllers.get('joint_state_broadcaster')
            == 'active'
            and controllers.get('panda_arm_controller')
            == 'active'
            and controllers.get('wrist_ft_broadcaster')
            == 'active'
        )

    def _expert_succeeded(self):
        return (
            EXPECTED_EXPERT_SUCCESS
            in self.status_history
        )

    def test_hybrid_moveit_to_local_expert_insertion_succeeds(
        self,
        proc_output,
    ):
        self.assertTrue(
            self._spin_until(
                self._hardware_is_active,
                timeout_sec=20.0,
            ),
            (
                'PandaSystem did not become active using '
                'robustlearn_mujoco_hardware/MuJoCoSystem.'
            ),
        )

        hardware_response = self._call_service(
            controller_manager_msgs.srv.ListHardwareComponents,
            '/controller_manager/list_hardware_components',
        )

        self.assertEqual(
            len(hardware_response.component),
            1,
            (
                'Hybrid workflow must expose exactly one hardware '
                f'component, got {len(hardware_response.component)}.'
            ),
        )

        panda_system = hardware_response.component[0]

        self.assertEqual(
            panda_system.name,
            'PandaSystem',
        )

        self.assertEqual(
            panda_system.plugin_name,
            'robustlearn_mujoco_hardware/MuJoCoSystem',
        )

        self.assertNotEqual(
            panda_system.plugin_name,
            'mock_components/GenericSystem',
        )

        self.assertTrue(
            self._spin_until(
                self._controllers_are_active,
                timeout_sec=20.0,
            ),
            (
                'Hybrid workflow controllers did not all become active: '
                'joint_state_broadcaster, panda_arm_controller, '
                'wrist_ft_broadcaster.'
            ),
        )

        controller_response = self._call_service(
            controller_manager_msgs.srv.ListControllers,
            '/controller_manager/list_controllers',
        )

        panda_arm_controllers = [
            controller
            for controller in controller_response.controller
            if controller.name == 'panda_arm_controller'
        ]

        self.assertEqual(
            len(panda_arm_controllers),
            1,
            (
                'Expected exactly one panda_arm_controller, '
                f'got {len(panda_arm_controllers)}.'
            ),
        )

        panda_arm_controller = panda_arm_controllers[0]

        expected_claimed_interfaces = {
            f'{joint_name}/position'
            for joint_name in PANDA_JOINTS
        }

        self.assertEqual(
            set(panda_arm_controller.claimed_interfaces),
            expected_claimed_interfaces,
            (
                'panda_arm_controller claimed unexpected command '
                f'interfaces: {panda_arm_controller.claimed_interfaces}'
            ),
        )

        self.assertTrue(
            self._spin_until(
                lambda: self.first_arm_positions is not None,
                timeout_sec=10.0,
            ),
            'No complete finite Panda arm joint state was observed.',
        )

        initial_distance_from_preinsertion = max(
            abs(actual - expected)
            for actual, expected in zip(
                self.first_arm_positions,
                PRE_INSERTION_MEASURED_JOINT_POSITIONS,
                strict=True,
            )
        )

        self.assertGreater(
            initial_distance_from_preinsertion,
            MINIMUM_INITIAL_PREINSERTION_DISTANCE,
            (
                'Hybrid workflow did not begin from a meaningful global '
                'approach state. Initial maximum distance from the '
                'pre-insertion contract was '
                f'{initial_distance_from_preinsertion:.6f} rad.'
            ),
        )

        self.assertTrue(
            self._spin_until(
                lambda: self.finite_wrench_samples > 0,
                timeout_sec=10.0,
            ),
            'No finite wrist F/T sample was observed.',
        )

        self.assertTrue(
            self._spin_until(
                self._expert_succeeded,
                timeout_sec=40.0,
            ),
            (
                'Local insertion expert did not report canonical '
                f'success; status history={self.status_history}'
            ),
        )

        self.assertIn(
            'disarmed',
            self.status_history,
            (
                'Expected expert to begin disarmed before MoveIt handoff; '
                f'history={self.status_history}'
            ),
        )

        self.assertIn(
            'running',
            self.status_history,
            (
                'Expected expert running status after MoveIt handoff; '
                f'history={self.status_history}'
            ),
        )

        self.assertIn(
            EXPECTED_EXPERT_SUCCESS,
            self.status_history,
        )

        self.assertGreater(
            self.max_motion_from_start,
            MINIMUM_OBSERVED_MOTION,
            (
                'Observed MuJoCo joint feedback did not show meaningful '
                'motion from the global approach state; '
                f'maximum motion={self.max_motion_from_start:.6f} rad.'
            ),
        )

        final_positions = self._arm_positions(
            self.latest_joint_state
        )

        self.assertIsNotNone(
            final_positions,
            'Final Panda arm state is invalid.',
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in final_positions
            ),
            'Final Panda arm state contains non-finite values.',
        )

        self.assertIsNotNone(
            self.latest_wrench,
            'No final finite wrist F/T sample is available.',
        )

        proc_output.assertWaitFor(
            'Applied RobustLearn workcell planning scene: '
            'workstation, four fixture walls, and attached fixed peg',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=planning',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=executing',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=verifying_handoff',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Handoff verification against measured pre-insertion contract',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'tolerance=0.002000',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=arming_expert',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion handoff completed: '
            'local insertion expert armed; '
            'waiting for terminal expert status',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=waiting_for_expert',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Local insertion expert completed: '
            'canonical insertion task reached success',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion completed: '
            'local insertion expert reported success: '
            'canonical insertion task reached success',
            timeout=5.0,
        )
