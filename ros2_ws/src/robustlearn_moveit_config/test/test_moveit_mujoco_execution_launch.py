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
import controller_manager_msgs.srv
import launch
import launch.actions
import launch.launch_description_sources
import launch_testing
import launch_testing.actions
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetMotionPlan
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg

os.environ['ROS_DOMAIN_ID'] = '172'


PANDA_JOINTS = tuple(
    f'panda_joint{index}'
    for index in range(1, 8)
)

PANDA_FINGER_JOINTS = (
    'panda_finger_joint1',
    'panda_finger_joint2',
)

TARGET_POSITIONS = (
    0.10,
    -0.10,
    0.10,
    -1.70,
    0.10,
    1.70,
    -0.70,
)

FINAL_POSITION_TOLERANCE = 0.02
MINIMUM_OBSERVED_MOTION = 0.02


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
        / 'moveit_mujoco.launch.py'
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

    moveit_mujoco = launch.actions.IncludeLaunchDescription(
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
            moveit_mujoco,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestMoveItMuJoCoExecution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(
            'moveit_mujoco_execution_acceptance_test'
        )

        self.latest_joint_state = None
        self.initial_arm_positions = None
        self.max_motion_from_start = 0.0

        self.subscription = self.node.create_subscription(
            sensor_msgs.msg.JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

    def tearDown(self):
        self.node.destroy_subscription(
            self.subscription
        )
        self.node.destroy_node()

    def _joint_state_callback(self, message):
        self.latest_joint_state = message

        arm_positions = self._arm_positions(message)

        if (
            arm_positions is None
            or self.initial_arm_positions is None
        ):
            return

        motion = max(
            abs(current - initial)
            for current, initial in zip(
                arm_positions,
                self.initial_arm_positions,
                strict=True,
            )
        )

        self.max_motion_from_start = max(
            self.max_motion_from_start,
            motion,
        )

    def _arm_positions(self, message):
        if message is None:
            return None

        if len(message.name) != len(message.position):
            return None

        positions = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        if not all(
            joint_name in positions
            for joint_name in PANDA_JOINTS
        ):
            return None

        arm_positions = tuple(
            positions[joint_name]
            for joint_name in PANDA_JOINTS
        )

        if not all(
            math.isfinite(value)
            for value in arm_positions
        ):
            return None

        return arm_positions

    def _spin_until(self, condition, timeout_sec):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

            if condition():
                return True

        return False

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

        self.node.destroy_client(client)

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
        )

    def _complete_joint_state_available(self):
        message = self.latest_joint_state

        if message is None:
            return False

        expected_joints = set(
            PANDA_JOINTS + PANDA_FINGER_JOINTS
        )

        if not expected_joints.issubset(
            set(message.name)
        ):
            return False

        if len(message.name) != len(message.position):
            return False

        if len(message.name) != len(message.velocity):
            return False

        position_lookup = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        velocity_lookup = dict(
            zip(
                message.name,
                message.velocity,
                strict=True,
            )
        )

        return all(
            math.isfinite(position_lookup[joint_name])
            and math.isfinite(velocity_lookup[joint_name])
            for joint_name in expected_joints
        )

    def _plan_trajectory(self):
        planning_client = self.node.create_client(
            GetMotionPlan,
            '/plan_kinematic_path',
        )

        self.assertTrue(
            planning_client.wait_for_service(
                timeout_sec=20.0
            ),
            'MoveIt planning service did not become available.',
        )

        request = GetMotionPlan.Request()

        motion_request = request.motion_plan_request

        motion_request.group_name = 'panda_arm'
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.2
        motion_request.max_acceleration_scaling_factor = 0.2

        motion_request.start_state.joint_state = (
            self.latest_joint_state
        )
        motion_request.start_state.is_diff = False

        goal_constraints = Constraints()
        goal_constraints.name = 'week8_execution_goal'

        for joint_name, target_position in zip(
            PANDA_JOINTS,
            TARGET_POSITIONS,
            strict=True,
        ):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = target_position
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0

            goal_constraints.joint_constraints.append(
                joint_constraint
            )

        motion_request.goal_constraints = [
            goal_constraints
        ]

        future = planning_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=15.0,
        )

        self.assertTrue(
            future.done(),
            'MoveIt planning request timed out.',
        )

        response = future.result()

        self.assertIsNotNone(
            response,
            'MoveIt planning returned no response.',
        )

        plan = response.motion_plan_response

        self.assertEqual(
            plan.error_code.val,
            MoveItErrorCodes.SUCCESS,
            (
                'MoveIt planning failed: '
                f'{plan.error_code.message}'
            ),
        )

        self.assertTrue(
            plan.trajectory.joint_trajectory.points,
            'MoveIt returned an empty trajectory.',
        )

        self.node.destroy_client(
            planning_client
        )

        return plan.trajectory

    def test_moveit_trajectory_executes_in_mujoco(self):
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
                'Expected exactly one hardware component, '
                f'got {len(hardware_response.component)}.'
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
                'joint_state_broadcaster and '
                'panda_arm_controller did not become active.'
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
            len(panda_arm_controller.claimed_interfaces),
            len(PANDA_JOINTS),
            (
                'panda_arm_controller did not claim exactly '
                'seven command interfaces: '
                f'{panda_arm_controller.claimed_interfaces}'
            ),
        )

        self.assertEqual(
            set(panda_arm_controller.claimed_interfaces),
            expected_claimed_interfaces,
            (
                'panda_arm_controller claimed unexpected '
                'command interfaces: '
                f'{panda_arm_controller.claimed_interfaces}'
            ),
        )

        self.assertTrue(
            self._spin_until(
                self._complete_joint_state_available,
                timeout_sec=20.0,
            ),
            (
                'MoveIt did not receive complete finite Panda '
                'arm and finger joint state.'
            ),
        )

        initial_positions = self._arm_positions(
            self.latest_joint_state
        )

        self.assertIsNotNone(
            initial_positions,
            'Initial Panda arm state is invalid.',
        )

        self.initial_arm_positions = tuple(
            initial_positions
        )

        trajectory = self._plan_trajectory()

        trajectory_joint_names = (
            trajectory.joint_trajectory.joint_names
        )

        planned_final_lookup = dict(
            zip(
                trajectory_joint_names,
                trajectory.joint_trajectory.points[-1].positions,
                strict=True,
            )
        )

        self.assertTrue(
            all(
                joint_name in planned_final_lookup
                for joint_name in PANDA_JOINTS
            ),
            (
                'MoveIt trajectory does not contain all '
                'seven Panda arm joints.'
            ),
        )

        planned_final_positions = tuple(
            planned_final_lookup[joint_name]
            for joint_name in PANDA_JOINTS
        )

        execute_client = ActionClient(
            self.node,
            ExecuteTrajectory,
            '/execute_trajectory',
        )

        self.assertTrue(
            execute_client.wait_for_server(
                timeout_sec=20.0
            ),
            (
                'MoveIt ExecuteTrajectory action server '
                'did not become available.'
            ),
        )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        goal.controller_names = [
            'panda_arm_controller'
        ]

        send_future = execute_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self.node,
            send_future,
            timeout_sec=10.0,
        )

        self.assertTrue(
            send_future.done(),
            'MoveIt execution goal request timed out.',
        )

        goal_handle = send_future.result()

        self.assertIsNotNone(
            goal_handle,
            'MoveIt execution returned no goal handle.',
        )

        self.assertTrue(
            goal_handle.accepted,
            'MoveIt execution goal was rejected.',
        )

        result_future = goal_handle.get_result_async()

        execution_deadline = time.monotonic() + 20.0

        while (
            time.monotonic() < execution_deadline
            and not result_future.done()
        ):
            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

        self.assertTrue(
            result_future.done(),
            'MoveIt trajectory execution timed out.',
        )

        wrapped_result = result_future.result()

        self.assertIsNotNone(
            wrapped_result,
            'MoveIt execution returned no result.',
        )

        self.assertEqual(
            wrapped_result.status,
            GoalStatus.STATUS_SUCCEEDED,
        )

        self.assertEqual(
            wrapped_result.result.error_code.val,
            MoveItErrorCodes.SUCCESS,
            (
                'MoveIt execution failed: '
                f'{wrapped_result.result.error_code.message}'
            ),
        )

        settle_deadline = time.monotonic() + 0.5

        while time.monotonic() < settle_deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

        final_positions = self._arm_positions(
            self.latest_joint_state
        )

        self.assertIsNotNone(
            final_positions,
            'Final Panda arm state is invalid.',
        )

        self.assertGreater(
            self.max_motion_from_start,
            MINIMUM_OBSERVED_MOTION,
            (
                'MuJoCo joint feedback did not show '
                'meaningful trajectory motion.'
            ),
        )

        final_errors = [
            abs(actual - planned)
            for actual, planned in zip(
                final_positions,
                planned_final_positions,
                strict=True,
            )
        ]

        self.assertLessEqual(
            max(final_errors),
            FINAL_POSITION_TOLERANCE,
            (
                'Final Panda joint error exceeded '
                f'{FINAL_POSITION_TOLERANCE} rad: '
                f'{final_errors}'
            ),
        )

        execute_client.destroy()
