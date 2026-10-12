#!/usr/bin/env python3
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

"""Coordinate MoveIt pre-insertion approach and local expert handoff."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
import sensor_msgs.msg
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import SetBool


PANDA_ARM_JOINT_NAMES: tuple[str, ...] = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
)


# Project-owned measured pre-insertion contract. This is the configuration
# from which the local insertion expert has been validated to start safely.
PRE_INSERTION_MEASURED_JOINT_POSITIONS: tuple[float, ...] = (
    0.0,
    0.013358,
    -0.000005,
    -1.583575,
    -0.000669,
    1.568475,
    -0.785300,
)


# MuJoCo position-control equilibrium is offset from the commanded trajectory
# target under load. These command positions apply the measured first-order
# compensation so that the physical joint state settles near the project-owned
# measured pre-insertion contract above.
PRE_INSERTION_COMMAND_JOINT_POSITIONS: tuple[float, ...] = (
    -0.0009080421759644523,
    0.006943047572395109,
    0.00020979908022258667,
    -1.578042725295331,
    -0.0000265037733280852,
    1.5696296944064593,
    -0.7854804248365594,
)


# MoveIt world-collision geometry mirrors the project-owned MuJoCo workcell.
# MuJoCo box sizes are half-extents, so MoveIt dimensions are doubled.
WORKCELL_BOXES: tuple[
    tuple[
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    ...,
] = (
    (
        'workstation',
        (0.55, 0.0, 0.35),
        (0.90, 0.80, 0.05),
    ),
    (
        'fixture_left',
        (0.49975, 0.0, 0.405),
        (0.0905, 0.20, 0.06),
    ),
    (
        'fixture_right',
        (0.60925, 0.0, 0.405),
        (0.0905, 0.20, 0.06),
    ),
    (
        'fixture_front',
        (0.5545, -0.05475, 0.405),
        (0.019, 0.0905, 0.06),
    ),
    (
        'fixture_rear',
        (0.5545, 0.05475, 0.405),
        (0.019, 0.0905, 0.06),
    ),
)


PLANNING_SERVICE = '/plan_kinematic_path'
PLANNING_SCENE_SERVICE = '/apply_planning_scene'
EXECUTION_ACTION = '/execute_trajectory'
EXPERT_ARM_SERVICE = '/local_insertion_expert/set_armed'
EXPERT_STATUS_TOPIC = '/local_insertion_expert/status'

EXPERT_STATUS_DISARMED = 'disarmed'
EXPERT_STATUS_RUNNING = 'running'
EXPERT_STATUS_SUCCESS = 'success'
EXPERT_STATUS_FAILED = 'failed'


class HybridPhase(StrEnum):
    """Phases of the hybrid MoveIt-to-local-expert workflow."""

    WAITING_FOR_STATE = 'waiting_for_state'
    WAITING_FOR_MOVEIT = 'waiting_for_moveit'
    APPLYING_PLANNING_SCENE = 'applying_planning_scene'
    PLANNING = 'planning'
    EXECUTING = 'executing'
    VERIFYING_HANDOFF = 'verifying_handoff'
    ARMING_EXPERT = 'arming_expert'
    WAITING_FOR_EXPERT = 'waiting_for_expert'
    COMPLETE = 'complete'
    FAILED = 'failed'


class HybridInsertionCoordinator(Node):
    """Run the deterministic MoveIt-to-local-expert handoff."""

    def __init__(self) -> None:
        super().__init__(
            'hybrid_insertion_coordinator'
        )

        self.declare_parameter(
            'handoff_tolerance_rad',
            0.002,
        )

        self.declare_parameter(
            'moveit_ready_timeout_s',
            10.0,
        )

        self.declare_parameter(
            'planning_scene_timeout_s',
            5.0,
        )

        self.declare_parameter(
            'planning_timeout_s',
            15.0,
        )

        self.declare_parameter(
            'execution_timeout_s',
            20.0,
        )

        self.declare_parameter(
            'handoff_settle_s',
            0.5,
        )

        self.declare_parameter(
            'expert_arm_timeout_s',
            5.0,
        )

        self.declare_parameter(
            'expert_timeout_s',
            20.0,
        )

        self._handoff_tolerance_rad = float(
            self.get_parameter(
                'handoff_tolerance_rad'
            ).value
        )

        self._moveit_ready_timeout_s = float(
            self.get_parameter(
                'moveit_ready_timeout_s'
            ).value
        )

        self._planning_scene_timeout_s = float(
            self.get_parameter(
                'planning_scene_timeout_s'
            ).value
        )

        self._planning_timeout_s = float(
            self.get_parameter(
                'planning_timeout_s'
            ).value
        )

        self._execution_timeout_s = float(
            self.get_parameter(
                'execution_timeout_s'
            ).value
        )

        self._handoff_settle_s = float(
            self.get_parameter(
                'handoff_settle_s'
            ).value
        )

        self._expert_arm_timeout_s = float(
            self.get_parameter(
                'expert_arm_timeout_s'
            ).value
        )

        self._expert_timeout_s = float(
            self.get_parameter(
                'expert_timeout_s'
            ).value
        )

        numeric_parameters = {
            'handoff_tolerance_rad': self._handoff_tolerance_rad,
            'moveit_ready_timeout_s': self._moveit_ready_timeout_s,
            'planning_scene_timeout_s': self._planning_scene_timeout_s,
            'planning_timeout_s': self._planning_timeout_s,
            'execution_timeout_s': self._execution_timeout_s,
            'handoff_settle_s': self._handoff_settle_s,
            'expert_arm_timeout_s': self._expert_arm_timeout_s,
            'expert_timeout_s': self._expert_timeout_s,
        }

        for name, value in numeric_parameters.items():
            if (
                not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f'{name} must be finite and positive'
                )

        self._phase = HybridPhase.WAITING_FOR_STATE

        self._latest_joint_state: sensor_msgs.msg.JointState | None = None
        self._planned_final_positions: tuple[float, ...] | None = None
        self._latest_expert_status: str | None = None

        self._terminal = False
        self._started = False
        self._planning_scene_applied = False

        self._moveit_wait_started_ns: int | None = None
        self._planning_scene_started_ns: int | None = None
        self._planning_started_ns: int | None = None
        self._execution_started_ns: int | None = None
        self._handoff_started_ns: int | None = None
        self._expert_arm_started_ns: int | None = None
        self._expert_started_ns: int | None = None

        self._waiting_for_moveit_resource = (
            'planning service'
        )

        self._planning_client = self.create_client(
            GetMotionPlan,
            PLANNING_SERVICE,
        )

        self._planning_scene_client = self.create_client(
            ApplyPlanningScene,
            PLANNING_SCENE_SERVICE,
        )

        self._execution_client = ActionClient(
            self,
            ExecuteTrajectory,
            EXECUTION_ACTION,
        )

        self._expert_arm_client = self.create_client(
            SetBool,
            EXPERT_ARM_SERVICE,
        )

        self._joint_state_subscription = self.create_subscription(
            sensor_msgs.msg.JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        expert_status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._expert_status_subscription = self.create_subscription(
            String,
            EXPERT_STATUS_TOPIC,
            self._expert_status_callback,
            expert_status_qos,
        )

        self._timer = self.create_timer(
            0.05,
            self._tick,
        )

        self.get_logger().info(
            'Hybrid insertion coordinator started; '
            'waiting for finite Panda joint state'
        )

    def _set_phase(
        self,
        phase: HybridPhase,
    ) -> None:
        if self._phase is phase:
            return

        self._phase = phase

        self.get_logger().info(
            f'Hybrid insertion phase={phase.value}'
        )

    def _fail(
        self,
        reason: str,
    ) -> None:
        if self._terminal:
            return

        self._terminal = True
        self._phase = HybridPhase.FAILED

        self.get_logger().error(
            f'Hybrid insertion failed: {reason}'
        )

    def _complete(
        self,
        reason: str,
    ) -> None:
        if self._terminal:
            return

        self._terminal = True
        self._phase = HybridPhase.COMPLETE

        self.get_logger().info(
            'Hybrid insertion completed: '
            f'local insertion expert reported success: {reason}'
        )

    def _joint_state_callback(
        self,
        message: sensor_msgs.msg.JointState,
    ) -> None:
        if len(message.name) != len(message.position):
            return

        position_lookup = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        if not all(
            joint_name in position_lookup
            for joint_name in PANDA_ARM_JOINT_NAMES
        ):
            return

        arm_positions = tuple(
            float(
                position_lookup[joint_name]
            )
            for joint_name in PANDA_ARM_JOINT_NAMES
        )

        if not all(
            isfinite(value)
            for value in arm_positions
        ):
            self._fail(
                'received non-finite Panda joint feedback'
            )
            return

        self._latest_joint_state = message

    def _expert_status_callback(
        self,
        message: String,
    ) -> None:
        status = message.data.strip()

        if not status:
            if self._phase is HybridPhase.WAITING_FOR_EXPERT:
                self._fail(
                    'local insertion expert published an empty status'
                )
            return

        self._latest_expert_status = status

        if self._phase is HybridPhase.WAITING_FOR_EXPERT:
            self._process_expert_status(
                status
            )

    def _process_expert_status(
        self,
        status: str,
    ) -> None:
        if self._terminal:
            return

        state, separator, reason = status.partition('|')

        if state in (
            EXPERT_STATUS_DISARMED,
            EXPERT_STATUS_RUNNING,
        ):
            return

        if state == EXPERT_STATUS_SUCCESS:
            if not separator or not reason:
                self._fail(
                    'local insertion expert success status had no reason'
                )
                return

            self._complete(
                reason
            )
            return

        if state == EXPERT_STATUS_FAILED:
            if not separator or not reason:
                self._fail(
                    'local insertion expert failure status had no reason'
                )
                return

            self._fail(
                'local insertion expert reported failure: '
                f'{reason}'
            )
            return

        self._fail(
            'local insertion expert published unknown status '
            f'{status!r}'
        )

    def _current_arm_positions(
        self,
    ) -> tuple[float, ...] | None:
        message = self._latest_joint_state

        if message is None:
            return None

        if len(message.name) != len(message.position):
            return None

        position_lookup = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        if not all(
            joint_name in position_lookup
            for joint_name in PANDA_ARM_JOINT_NAMES
        ):
            return None

        positions = tuple(
            float(
                position_lookup[joint_name]
            )
            for joint_name in PANDA_ARM_JOINT_NAMES
        )

        if not all(
            isfinite(value)
            for value in positions
        ):
            return None

        return positions

    @staticmethod
    def _box_collision_object(
        object_id: str,
        center: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        frame_id: str,
    ) -> CollisionObject:
        collision_object = CollisionObject()

        collision_object.header.frame_id = frame_id
        collision_object.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(
            dimensions
        )

        pose = Pose()

        pose.position.x = center[0]
        pose.position.y = center[1]
        pose.position.z = center[2]

        pose.orientation.w = 1.0

        collision_object.primitives = [
            primitive
        ]

        collision_object.primitive_poses = [
            pose
        ]

        collision_object.operation = CollisionObject.ADD

        return collision_object

    def _planning_scene(
        self,
    ) -> PlanningScene:
        scene = PlanningScene()
        scene.is_diff = True

        scene.world.collision_objects = [
            self._box_collision_object(
                object_id,
                center,
                dimensions,
                'world',
            )
            for object_id, center, dimensions in WORKCELL_BOXES
        ]

        peg_collision = self._box_collision_object(
            'fixed_insertion_peg',
            (0.0, 0.0, 0.140),
            (0.016, 0.016, 0.060),
            'panda_hand',
        )

        attached_peg = AttachedCollisionObject()
        attached_peg.link_name = 'panda_hand'
        attached_peg.object = peg_collision
        attached_peg.touch_links = [
            'panda_hand',
            'panda_leftfinger',
            'panda_rightfinger',
        ]

        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [
            attached_peg
        ]

        return scene

    def _request_planning_scene(
        self,
    ) -> None:
        if not self._planning_scene_client.service_is_ready():
            self._waiting_for_moveit_resource = (
                'planning scene service'
            )

            if self._moveit_wait_started_ns is None:
                self._moveit_wait_started_ns = (
                    self.get_clock().now().nanoseconds
                )

            self._set_phase(
                HybridPhase.WAITING_FOR_MOVEIT
            )
            return

        self._set_phase(
            HybridPhase.APPLYING_PLANNING_SCENE
        )

        self._planning_scene_started_ns = (
            self.get_clock().now().nanoseconds
        )

        request = ApplyPlanningScene.Request()
        request.scene = self._planning_scene()

        future = self._planning_scene_client.call_async(
            request
        )

        future.add_done_callback(
            self._planning_scene_response_callback
        )

    def _planning_scene_response_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            return

        try:
            response = future.result()
        except Exception as exc:
            self._fail(
                f'MoveIt planning scene request failed: {exc}'
            )
            return

        if response is None:
            self._fail(
                'MoveIt planning scene returned no response'
            )
            return

        if not response.success:
            self._fail(
                'MoveIt rejected the RobustLearn planning scene'
            )
            return

        self._planning_scene_applied = True
        self._planning_scene_started_ns = None

        self.get_logger().info(
            'Applied RobustLearn workcell planning scene: '
            'workstation, four fixture walls, and attached fixed peg'
        )

        self._request_plan()

    def _request_plan(
        self,
    ) -> None:
        if self._latest_joint_state is None:
            return

        if not self._planning_client.service_is_ready():
            self._waiting_for_moveit_resource = (
                'planning service'
            )

            if self._moveit_wait_started_ns is None:
                self._moveit_wait_started_ns = (
                    self.get_clock().now().nanoseconds
                )

            self._set_phase(
                HybridPhase.WAITING_FOR_MOVEIT
            )
            return

        if not self._planning_scene_applied:
            self._request_planning_scene()
            return

        self._moveit_wait_started_ns = None

        self._set_phase(
            HybridPhase.PLANNING
        )

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = 'panda_arm'
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.2
        motion_request.max_acceleration_scaling_factor = 0.2
        motion_request.start_state.joint_state = (
            self._latest_joint_state
        )
        motion_request.start_state.is_diff = False

        goal_constraints = Constraints()
        goal_constraints.name = (
            'week9_pre_insertion_goal'
        )

        for joint_name, target_position in zip(
            PANDA_ARM_JOINT_NAMES,
            PRE_INSERTION_COMMAND_JOINT_POSITIONS,
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

        self._planning_started_ns = (
            self.get_clock().now().nanoseconds
        )

        future = self._planning_client.call_async(
            request
        )

        future.add_done_callback(
            self._plan_response_callback
        )

    def _plan_response_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            return

        try:
            response = future.result()
        except Exception as exc:
            self._fail(
                f'MoveIt planning request failed: {exc}'
            )
            return

        if response is None:
            self._fail(
                'MoveIt planning returned no response'
            )
            return

        plan = response.motion_plan_response

        if plan.error_code.val != MoveItErrorCodes.SUCCESS:
            self._fail(
                'MoveIt planning failed with code '
                f'{plan.error_code.val}: '
                f'{plan.error_code.message}'
            )
            return

        trajectory = plan.trajectory
        points = trajectory.joint_trajectory.points

        if not points:
            self._fail(
                'MoveIt planning returned an empty trajectory'
            )
            return

        joint_names = tuple(
            trajectory.joint_trajectory.joint_names
        )

        final_positions = tuple(
            float(value)
            for value in points[-1].positions
        )

        if len(joint_names) != len(final_positions):
            self._fail(
                'MoveIt trajectory joint names and final positions '
                'have different lengths'
            )
            return

        final_lookup = dict(
            zip(
                joint_names,
                final_positions,
                strict=True,
            )
        )

        if not all(
            joint_name in final_lookup
            for joint_name in PANDA_ARM_JOINT_NAMES
        ):
            self._fail(
                'MoveIt trajectory does not contain all Panda arm joints'
            )
            return

        self._planned_final_positions = tuple(
            final_lookup[joint_name]
            for joint_name in PANDA_ARM_JOINT_NAMES
        )

        if not all(
            isfinite(value)
            for value in self._planned_final_positions
        ):
            self._fail(
                'MoveIt trajectory final target is non-finite'
            )
            return

        self._send_execution_goal(
            trajectory
        )

    def _send_execution_goal(
        self,
        trajectory,
    ) -> None:
        if not self._execution_client.server_is_ready():
            self._fail(
                'MoveIt ExecuteTrajectory action server is unavailable'
            )
            return

        self._set_phase(
            HybridPhase.EXECUTING
        )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        goal.controller_names = [
            'panda_arm_controller'
        ]

        self._execution_started_ns = (
            self.get_clock().now().nanoseconds
        )

        future = self._execution_client.send_goal_async(
            goal
        )

        future.add_done_callback(
            self._execution_goal_response_callback
        )

    def _execution_goal_response_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            return

        try:
            goal_handle = future.result()
        except Exception as exc:
            self._fail(
                f'MoveIt execution goal request failed: {exc}'
            )
            return

        if goal_handle is None:
            self._fail(
                'MoveIt execution returned no goal handle'
            )
            return

        if not goal_handle.accepted:
            self._fail(
                'MoveIt execution goal was rejected'
            )
            return

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self._execution_result_callback
        )

    def _execution_result_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            return

        try:
            wrapped_result = future.result()
        except Exception as exc:
            self._fail(
                f'MoveIt execution result failed: {exc}'
            )
            return

        if wrapped_result is None:
            self._fail(
                'MoveIt execution returned no result'
            )
            return

        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self._fail(
                'MoveIt execution ended with action status '
                f'{wrapped_result.status}'
            )
            return

        result = wrapped_result.result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._fail(
                'MoveIt execution failed with code '
                f'{result.error_code.val}: '
                f'{result.error_code.message}'
            )
            return

        self._set_phase(
            HybridPhase.VERIFYING_HANDOFF
        )

        self._handoff_started_ns = (
            self.get_clock().now().nanoseconds
        )

    def _handoff_ready(
        self,
    ) -> bool:
        current_positions = self._current_arm_positions()

        if current_positions is None:
            return False

        errors = tuple(
            abs(current - expected)
            for current, expected in zip(
                current_positions,
                PRE_INSERTION_MEASURED_JOINT_POSITIONS,
                strict=True,
            )
        )

        maximum_error = max(errors)

        self.get_logger().debug(
            'Handoff verification against measured pre-insertion contract '
            f'max_joint_error={maximum_error:.6f} '
            f'tolerance={self._handoff_tolerance_rad:.6f}'
        )

        return (
            maximum_error
            <= self._handoff_tolerance_rad
        )

    def _request_expert_arm(
        self,
    ) -> None:
        if not self._expert_arm_client.service_is_ready():
            self._fail(
                'Local insertion expert arming service is unavailable'
            )
            return

        self._set_phase(
            HybridPhase.ARMING_EXPERT
        )

        self._expert_arm_started_ns = (
            self.get_clock().now().nanoseconds
        )

        request = SetBool.Request()
        request.data = True

        future = self._expert_arm_client.call_async(
            request
        )

        future.add_done_callback(
            self._expert_arm_response_callback
        )

    def _expert_arm_response_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            return

        try:
            response = future.result()
        except Exception as exc:
            self._fail(
                f'Local insertion expert arming request failed: {exc}'
            )
            return

        if response is None:
            self._fail(
                'Local insertion expert arming returned no response'
            )
            return

        if not response.success:
            self._fail(
                'Local insertion expert rejected handoff: '
                f'{response.message}'
            )
            return

        self._expert_arm_started_ns = None
        self._expert_started_ns = (
            self.get_clock().now().nanoseconds
        )

        self.get_logger().info(
            'Hybrid insertion handoff completed: '
            'local insertion expert armed; '
            'waiting for terminal expert status'
        )

        self._set_phase(
            HybridPhase.WAITING_FOR_EXPERT
        )

        if self._latest_expert_status is not None:
            self._process_expert_status(
                self._latest_expert_status
            )

    def _elapsed_s(
        self,
        started_ns: int | None,
    ) -> float:
        if started_ns is None:
            return 0.0

        return (
            self.get_clock().now().nanoseconds
            - started_ns
        ) / 1.0e9

    def _moveit_readiness_failure_reason(
        self,
    ) -> str:
        if (
            self._waiting_for_moveit_resource
            == 'planning service'
        ):
            return (
                'MoveIt planning service did not become ready within '
                f'{self._moveit_ready_timeout_s:.3f} s'
            )

        return (
            'MoveIt planning scene service did not become ready within '
            f'{self._moveit_ready_timeout_s:.3f} s'
        )

    def _tick(self) -> None:
        if self._terminal:
            return

        if not self._started:
            if self._latest_joint_state is None:
                return

            self._started = True
            self._request_plan()
            return

        if self._phase is HybridPhase.WAITING_FOR_MOVEIT:
            if (
                self._elapsed_s(
                    self._moveit_wait_started_ns
                )
                >= self._moveit_ready_timeout_s
            ):
                self._fail(
                    self._moveit_readiness_failure_reason()
                )
                return

            self._request_plan()
            return

        if self._phase is HybridPhase.APPLYING_PLANNING_SCENE:
            if (
                self._elapsed_s(
                    self._planning_scene_started_ns
                )
                >= self._planning_scene_timeout_s
            ):
                self._fail(
                    'MoveIt planning scene application exceeded '
                    f'{self._planning_scene_timeout_s:.3f} s timeout'
                )
            return

        if self._phase is HybridPhase.PLANNING:
            if (
                self._elapsed_s(
                    self._planning_started_ns
                )
                >= self._planning_timeout_s
            ):
                self._fail(
                    'MoveIt planning exceeded '
                    f'{self._planning_timeout_s:.3f} s timeout'
                )
            return

        if self._phase is HybridPhase.EXECUTING:
            if (
                self._elapsed_s(
                    self._execution_started_ns
                )
                >= self._execution_timeout_s
            ):
                self._fail(
                    'MoveIt execution exceeded '
                    f'{self._execution_timeout_s:.3f} s timeout'
                )
            return

        if self._phase is HybridPhase.VERIFYING_HANDOFF:
            if (
                self._elapsed_s(
                    self._handoff_started_ns
                )
                < self._handoff_settle_s
            ):
                return

            if not self._handoff_ready():
                self._fail(
                    'measured pre-insertion handoff tolerance '
                    'was not satisfied'
                )
                return

            self._request_expert_arm()
            return

        if self._phase is HybridPhase.ARMING_EXPERT:
            if (
                self._elapsed_s(
                    self._expert_arm_started_ns
                )
                >= self._expert_arm_timeout_s
            ):
                self._fail(
                    'Local insertion expert arming exceeded '
                    f'{self._expert_arm_timeout_s:.3f} s timeout'
                )
            return

        if self._phase is HybridPhase.WAITING_FOR_EXPERT:
            if (
                self._elapsed_s(
                    self._expert_started_ns
                )
                >= self._expert_timeout_s
            ):
                self._fail(
                    'local insertion expert did not report terminal status '
                    f'within {self._expert_timeout_s:.3f} s'
                )


def main(
    args: list[str] | None = None,
) -> None:
    """Run the hybrid insertion coordinator."""
    rclpy.init(
        args=args
    )

    node = HybridInsertionCoordinator()

    try:
        rclpy.spin(
            node
        )
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
