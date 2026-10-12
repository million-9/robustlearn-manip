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
import threading
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
import pytest
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint


os.environ['ROS_DOMAIN_ID'] = '187'


PANDA_ARM_JOINT_NAMES = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
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


PRE_INSERTION_COMMAND_JOINT_POSITIONS = (
    -0.0009080421759644523,
    0.006943047572395109,
    0.00020979908022258667,
    -1.578042725295331,
    -0.0000265037733280852,
    1.5696296944064593,
    -0.7854804248365594,
)


FAR_FROM_HANDOFF_POSITIONS = (
    0.10,
    -0.10,
    0.10,
    -1.70,
    0.10,
    1.70,
    -0.70,
)


SCENARIOS = (
    'planning_failure',
    'execution_failure',
    'handoff_failure',
    'expert_timeout',
)


def _scenario_name(
    scenario: str,
    resource: str,
) -> str:
    return (
        f'/hybrid_failure_tests/{scenario}/{resource}'
    )


def _coordinator_node(
    scenario: str,
):
    remappings = [
        (
            '/joint_states',
            _scenario_name(
                scenario,
                'joint_states',
            ),
        ),
        (
            '/plan_kinematic_path',
            _scenario_name(
                scenario,
                'plan_kinematic_path',
            ),
        ),
        (
            '/apply_planning_scene',
            _scenario_name(
                scenario,
                'apply_planning_scene',
            ),
        ),
        (
            '/execute_trajectory',
            _scenario_name(
                scenario,
                'execute_trajectory',
            ),
        ),
        (
            '/local_insertion_expert/set_armed',
            _scenario_name(
                scenario,
                'set_armed',
            ),
        ),
        (
            '/local_insertion_expert/status',
            _scenario_name(
                scenario,
                'expert_status',
            ),
        ),
    ]

    return launch_ros.actions.Node(
        package='robustlearn_moveit_config',
        executable='hybrid_insertion_coordinator.py',
        name=(
            f'hybrid_insertion_coordinator_'
            f'{scenario}'
        ),
        output='screen',
        remappings=remappings,
        parameters=[
            {
                'handoff_tolerance_rad': 0.002,
                'moveit_ready_timeout_s': 2.0,
                'planning_scene_timeout_s': 2.0,
                'planning_timeout_s': 2.0,
                'execution_timeout_s': 2.0,
                'handoff_settle_s': 0.05,
                'expert_arm_timeout_s': 1.0,
                'expert_timeout_s': 0.25,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'info',
            '--log-level',
            (
                'hybrid_insertion_coordinator_'
                f'{scenario}:=debug'
            ),
        ],
    )


@pytest.mark.launch_test
def generate_test_description():
    planning_failure = _coordinator_node(
        'planning_failure'
    )

    execution_failure = _coordinator_node(
        'execution_failure'
    )

    handoff_failure = _coordinator_node(
        'handoff_failure'
    )

    expert_timeout = _coordinator_node(
        'expert_timeout'
    )

    return (
        launch.LaunchDescription(
            [
                planning_failure,
                execution_failure,
                handoff_failure,
                expert_timeout,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'planning_failure': planning_failure,
            'execution_failure': execution_failure,
            'handoff_failure': handoff_failure,
            'expert_timeout': expert_timeout,
        },
    )


class TestHybridCoordinatorFailures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.node = rclpy.create_node(
            'hybrid_coordinator_failure_test'
        )

        cls.executor = MultiThreadedExecutor(
            num_threads=4
        )

        cls.executor.add_node(
            cls.node
        )

        cls.executor_thread = threading.Thread(
            target=cls.executor.spin,
            daemon=True,
        )

        cls.executor_thread.start()

        cls.publishers = {
            scenario: cls.node.create_publisher(
                JointState,
                _scenario_name(
                    scenario,
                    'joint_states',
                ),
                qos_profile_sensor_data,
            )
            for scenario in SCENARIOS
        }

        cls.services = []

        for scenario in SCENARIOS:
            cls.services.append(
                cls.node.create_service(
                    ApplyPlanningScene,
                    _scenario_name(
                        scenario,
                        'apply_planning_scene',
                    ),
                    cls._apply_planning_scene,
                )
            )

        cls.services.append(
            cls.node.create_service(
                GetMotionPlan,
                _scenario_name(
                    'planning_failure',
                    'plan_kinematic_path',
                ),
                cls._planning_failure,
            )
        )

        for scenario in (
            'execution_failure',
            'handoff_failure',
            'expert_timeout',
        ):
            cls.services.append(
                cls.node.create_service(
                    GetMotionPlan,
                    _scenario_name(
                        scenario,
                        'plan_kinematic_path',
                    ),
                    cls._planning_success,
                )
            )

        cls.services.append(
            cls.node.create_service(
                SetBool,
                _scenario_name(
                    'expert_timeout',
                    'set_armed',
                ),
                cls._arm_expert_success,
            )
        )

        cls.action_servers = [
            ActionServer(
                cls.node,
                ExecuteTrajectory,
                _scenario_name(
                    'execution_failure',
                    'execute_trajectory',
                ),
                execute_callback=(
                    cls._execute_failure
                ),
            ),
            ActionServer(
                cls.node,
                ExecuteTrajectory,
                _scenario_name(
                    'handoff_failure',
                    'execute_trajectory',
                ),
                execute_callback=(
                    cls._execute_success
                ),
            ),
            ActionServer(
                cls.node,
                ExecuteTrajectory,
                _scenario_name(
                    'expert_timeout',
                    'execute_trajectory',
                ),
                execute_callback=(
                    cls._execute_success
                ),
            ),
        ]

    @classmethod
    def tearDownClass(cls):
        for action_server in cls.action_servers:
            action_server.destroy()

        for service in cls.services:
            cls.node.destroy_service(
                service
            )

        for publisher in cls.publishers.values():
            cls.node.destroy_publisher(
                publisher
            )

        cls.executor.shutdown()

        cls.executor_thread.join(
            timeout=2.0
        )

        cls.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    @staticmethod
    def _apply_planning_scene(
        _request,
        response,
    ):
        response.success = True
        return response

    @staticmethod
    def _planning_failure(
        _request,
        response,
    ):
        response.motion_plan_response.error_code.val = (
            MoveItErrorCodes.PLANNING_FAILED
        )

        response.motion_plan_response.error_code.message = (
            'mock planning failure'
        )

        return response

    @staticmethod
    def _planning_success(
        _request,
        response,
    ):
        motion_plan = (
            response.motion_plan_response
        )

        motion_plan.error_code.val = (
            MoveItErrorCodes.SUCCESS
        )

        trajectory = (
            motion_plan
            .trajectory
            .joint_trajectory
        )

        trajectory.joint_names = list(
            PANDA_ARM_JOINT_NAMES
        )

        point = JointTrajectoryPoint()

        point.positions = list(
            PRE_INSERTION_COMMAND_JOINT_POSITIONS
        )

        trajectory.points = [
            point
        ]

        return response

    @staticmethod
    def _arm_expert_success(
        request,
        response,
    ):
        if not request.data:
            response.success = False
            response.message = (
                'mock expert expected arm request'
            )
            return response

        response.success = True
        response.message = (
            'mock expert armed'
        )

        return response

    @staticmethod
    def _execute_failure(
        goal_handle,
    ):
        goal_handle.succeed()

        result = ExecuteTrajectory.Result()

        result.error_code.val = (
            MoveItErrorCodes.CONTROL_FAILED
        )

        result.error_code.message = (
            'mock trajectory execution failure'
        )

        return result

    @staticmethod
    def _execute_success(
        goal_handle,
    ):
        goal_handle.succeed()

        result = ExecuteTrajectory.Result()

        result.error_code.val = (
            MoveItErrorCodes.SUCCESS
        )

        return result

    @classmethod
    def _publish_joint_state(
        cls,
        scenario: str,
        positions: tuple[float, ...],
    ):
        message = JointState()

        message.header.stamp = (
            cls.node.get_clock().now().to_msg()
        )

        message.name = list(
            PANDA_ARM_JOINT_NAMES
        )

        message.position = list(
            positions
        )

        message.velocity = [
            0.0
            for _ in PANDA_ARM_JOINT_NAMES
        ]

        message.effort = [
            0.0
            for _ in PANDA_ARM_JOINT_NAMES
        ]

        cls.publishers[
            scenario
        ].publish(
            message
        )

    @classmethod
    def _publish_for_duration(
        cls,
        scenario: str,
        positions: tuple[float, ...],
        duration_sec: float = 0.75,
    ):
        deadline = (
            time.monotonic()
            + duration_sec
        )

        while time.monotonic() < deadline:
            cls._publish_joint_state(
                scenario,
                positions,
            )

            time.sleep(
                0.02
            )

    def test_planning_failure_is_terminal(
        self,
        proc_output,
        planning_failure,
    ):
        self._publish_for_duration(
            'planning_failure',
            PRE_INSERTION_MEASURED_JOINT_POSITIONS,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion failed: '
                'MoveIt planning failed with code'
            ),
            process=planning_failure,
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'mock planning failure',
            process=planning_failure,
            timeout=5.0,
        )

    def test_execution_failure_is_terminal(
        self,
        proc_output,
        execution_failure,
    ):
        self._publish_for_duration(
            'execution_failure',
            PRE_INSERTION_MEASURED_JOINT_POSITIONS,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion failed: '
                'MoveIt execution failed with code'
            ),
            process=execution_failure,
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'mock trajectory execution failure',
            process=execution_failure,
            timeout=5.0,
        )

    def test_handoff_tolerance_failure_is_terminal(
        self,
        proc_output,
        handoff_failure,
    ):
        self._publish_for_duration(
            'handoff_failure',
            FAR_FROM_HANDOFF_POSITIONS,
            duration_sec=1.0,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion phase='
                'verifying_handoff'
            ),
            process=handoff_failure,
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion failed: '
                'measured pre-insertion handoff tolerance '
                'was not satisfied'
            ),
            process=handoff_failure,
            timeout=5.0,
        )

    def test_expert_terminal_timeout_is_bounded(
        self,
        proc_output,
        expert_timeout,
    ):
        self._publish_for_duration(
            'expert_timeout',
            PRE_INSERTION_MEASURED_JOINT_POSITIONS,
            duration_sec=1.0,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion handoff completed: '
                'local insertion expert armed; '
                'waiting for terminal expert status'
            ),
            process=expert_timeout,
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            (
                'Hybrid insertion failed: '
                'local insertion expert did not report terminal status '
                'within 0.250 s'
            ),
            process=expert_timeout,
            timeout=5.0,
        )
