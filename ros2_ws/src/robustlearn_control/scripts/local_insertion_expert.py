#!/usr/bin/env python3
"""ROS-side classical local insertion expert for RobustLearn-Manip."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic

from control_msgs.action import FollowJointTrajectory
import control_msgs.msg
import geometry_msgs.msg
import mujoco
import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from robustlearn.config import PandaInsertionTaskConfig
from robustlearn.scripted_demo_collection import (
    ExpertFrameBuffer,
    write_expert_frame_capture,
)
from robustlearn.scripted_insertion import (
    ClassicalInsertionExpert,
    InsertionPhase,
)
from robustlearn.sim.control import panda_site_spatial_jacobian
from robustlearn.sim.insertion import load_insertion_model
from robustlearn.sim.task import evaluate_insertion_task
import sensor_msgs.msg
from std_msgs.msg import String
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint


FloatArray = NDArray[np.float64]


ROS_PANDA_ARM_JOINT_NAMES: tuple[str, ...] = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
)


MUJOCO_PANDA_ARM_JOINT_NAMES: tuple[str, ...] = (
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
)


PEG_TIP_SITE_NAME = 'peg_tip'
RECEPTACLE_CENTER_SITE_NAME = 'receptacle_center'
INSERTION_AXIS_SITE_NAME = 'insertion_axis'
WRIST_FT_SITE_NAME = 'panda_wrist_ft'


FOLLOW_JOINT_TRAJECTORY_ACTION = (
    '/panda_arm_controller/follow_joint_trajectory'
)


CONTROLLER_STATE_TOPIC = (
    '/panda_arm_controller/controller_state'
)


STATUS_TOPIC = (
    '/local_insertion_expert/status'
)


STATUS_DISARMED = 'disarmed'
STATUS_RUNNING = 'running'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'


@dataclass(frozen=True)
class ShadowKinematicsSnapshot:
    """Finite kinematic state reconstructed from ROS joint feedback."""

    joint_positions: FloatArray
    peg_tip_position: FloatArray
    receptacle_center_position: FloatArray
    insertion_axis_position: FloatArray
    peg_tip_jacobian: FloatArray


class ShadowInsertionKinematics:
    """Project-owned MuJoCo model used only for FK/Jacobian reconstruction."""

    def __init__(self) -> None:
        self.model = load_insertion_model()
        self.data = mujoco.MjData(self.model)

        self._joint_qpos_addresses = tuple(
            self._joint_qpos_address(name)
            for name in MUJOCO_PANDA_ARM_JOINT_NAMES
        )

        self._peg_tip_site_id = self._site_id(
            PEG_TIP_SITE_NAME
        )

        self._receptacle_center_site_id = self._site_id(
            RECEPTACLE_CENTER_SITE_NAME
        )

        self._insertion_axis_site_id = self._site_id(
            INSERTION_AXIS_SITE_NAME
        )

        self._wrist_ft_site_id = self._site_id(
            WRIST_FT_SITE_NAME
        )

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
        )

        if joint_id < 0:
            raise RuntimeError(
                f'Shadow insertion model is missing joint {name!r}'
            )

        if int(self.model.jnt_type[joint_id]) != int(
            mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise RuntimeError(
                f'Shadow insertion joint {name!r} is not a hinge'
            )

        return int(
            self.model.jnt_qposadr[joint_id]
        )

    def _site_id(self, name: str) -> int:
        site_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
        )

        if site_id < 0:
            raise RuntimeError(
                f'Shadow insertion model is missing site {name!r}'
            )

        return site_id

    def update(
        self,
        joint_positions: FloatArray,
    ) -> ShadowKinematicsSnapshot:
        """Reconstruct task geometry from observed Panda joint positions."""
        positions = np.asarray(
            joint_positions,
            dtype=np.float64,
        )

        if positions.shape != (7,):
            raise ValueError(
                f'joint_positions must have shape (7,), got {positions.shape}'
            )

        if not np.all(np.isfinite(positions)):
            raise ValueError(
                'joint_positions must contain only finite values'
            )

        for qpos_address, position in zip(
            self._joint_qpos_addresses,
            positions,
            strict=True,
        ):
            self.data.qpos[qpos_address] = position

        self.data.qvel.fill(0.0)

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        peg_tip_position = np.asarray(
            self.data.site_xpos[self._peg_tip_site_id],
            dtype=np.float64,
        ).copy()

        receptacle_center_position = np.asarray(
            self.data.site_xpos[
                self._receptacle_center_site_id
            ],
            dtype=np.float64,
        ).copy()

        insertion_axis_position = np.asarray(
            self.data.site_xpos[
                self._insertion_axis_site_id
            ],
            dtype=np.float64,
        ).copy()

        peg_tip_jacobian = panda_site_spatial_jacobian(
            self.model,
            self.data,
            PEG_TIP_SITE_NAME,
        )

        arrays = (
            peg_tip_position,
            receptacle_center_position,
            insertion_axis_position,
            peg_tip_jacobian,
        )

        if not all(
            np.all(np.isfinite(array))
            for array in arrays
        ):
            raise RuntimeError(
                'Shadow insertion kinematics produced non-finite state'
            )

        return ShadowKinematicsSnapshot(
            joint_positions=positions.copy(),
            peg_tip_position=peg_tip_position,
            receptacle_center_position=receptacle_center_position,
            insertion_axis_position=insertion_axis_position,
            peg_tip_jacobian=peg_tip_jacobian,
        )

    def wrist_rotation_world(self) -> FloatArray:
        """Return the world rotation matrix of the wrist F/T site."""
        rotation = np.asarray(
            self.data.site_xmat[self._wrist_ft_site_id],
            dtype=np.float64,
        ).reshape(3, 3).copy()

        if not np.all(np.isfinite(rotation)):
            raise RuntimeError(
                'Shadow wrist rotation contains non-finite values'
            )

        return rotation


class LocalInsertionExpertNode(Node):
    """Execute the classical insertion expert through ros2_control."""

    def __init__(self) -> None:
        super().__init__(
            'local_insertion_expert'
        )

        self.declare_parameter(
            'command_enabled',
            False,
        )

        self.declare_parameter(
            'start_armed',
            True,
        )

        self.declare_parameter(
            'control_period_s',
            0.02,
        )

        self.declare_parameter(
            'trajectory_duration_s',
            0.10,
        )

        self.declare_parameter(
            'goal_timeout_s',
            1.0,
        )

        self.declare_parameter(
            'expert_capture_path',
            '',
        )

        self._command_enabled = bool(
            self.get_parameter(
                'command_enabled'
            ).value
        )

        self._armed = bool(
            self.get_parameter(
                'start_armed'
            ).value
        )

        self._control_period_s = float(
            self.get_parameter(
                'control_period_s'
            ).value
        )

        self._trajectory_duration_s = float(
            self.get_parameter(
                'trajectory_duration_s'
            ).value
        )

        self._goal_timeout_s = float(
            self.get_parameter(
                'goal_timeout_s'
            ).value
        )

        self._expert_capture_path = str(
            self.get_parameter(
                'expert_capture_path'
            ).value
        ).strip()

        self._expert_frame_buffer = (
            ExpertFrameBuffer()
            if self._expert_capture_path
            else None
        )

        timing_values = {
            'control_period_s': self._control_period_s,
            'trajectory_duration_s': self._trajectory_duration_s,
            'goal_timeout_s': self._goal_timeout_s,
        }

        for name, value in timing_values.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f'{name} must be finite and positive'
                )

        self._task_config = PandaInsertionTaskConfig()
        self._kinematics = ShadowInsertionKinematics()
        self._expert = ClassicalInsertionExpert()

        self._latest_joint_positions: FloatArray | None = None
        self._latest_joint_velocities: FloatArray | None = None
        self._latest_controller_command: FloatArray | None = None
        self._latest_wrist_force_sensor: FloatArray | None = None
        self._latest_wrist_torque_sensor: FloatArray | None = None

        self._goal_in_flight = False
        self._goal_handle = None
        self._goal_started_ns: int | None = None

        self._terminal = False
        self._terminal_reason: str | None = None

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._status_publisher = self.create_publisher(
            String,
            STATUS_TOPIC,
            status_qos,
        )

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            FOLLOW_JOINT_TRAJECTORY_ACTION,
        )

        self._joint_subscription = self.create_subscription(
            sensor_msgs.msg.JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        self._controller_state_subscription = self.create_subscription(
            control_msgs.msg.JointTrajectoryControllerState,
            CONTROLLER_STATE_TOPIC,
            self._controller_state_callback,
            10,
        )

        self._wrench_subscription = self.create_subscription(
            geometry_msgs.msg.WrenchStamped,
            '/wrist_ft/wrench',
            self._wrench_callback,
            qos_profile_sensor_data,
        )

        self._set_armed_service = self.create_service(
            SetBool,
            '~/set_armed',
            self._set_armed_callback,
        )

        self._timer = self.create_timer(
            self._control_period_s,
            self._control_tick,
        )

        mode = (
            'command'
            if self._command_enabled
            else 'preview'
        )

        arm_state = (
            'armed'
            if self._armed
            else 'disarmed'
        )

        if self._armed:
            self._publish_status(
                STATUS_RUNNING
            )
        else:
            self._publish_status(
                STATUS_DISARMED
            )

        self.get_logger().info(
            'Local insertion expert started in '
            f'{mode} mode ({arm_state}); waiting for joint, controller, '
            'and wrist F/T feedback'
        )

    def _publish_status(
        self,
        state: str,
        reason: str | None = None,
    ) -> None:
        message = String()

        if reason is None:
            message.data = state
        else:
            message.data = (
                f'{state}|{reason}'
            )

        self._status_publisher.publish(
            message
        )

    def _set_armed_callback(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        requested_armed = bool(request.data)

        if self._terminal:
            response.success = False
            response.message = (
                'local insertion expert is already terminal: '
                f'{self._terminal_reason}'
            )
            return response

        if not requested_armed and self._goal_in_flight:
            response.success = False
            response.message = (
                'cannot disarm local insertion expert while a '
                'trajectory goal is in flight'
            )
            return response

        self._armed = requested_armed

        if self._armed:
            response.message = 'local insertion expert armed'

            self._publish_status(
                STATUS_RUNNING
            )
        else:
            response.message = 'local insertion expert disarmed'

            self._publish_status(
                STATUS_DISARMED
            )

        response.success = True

        self.get_logger().info(
            response.message
        )

        return response

    def _joint_state_callback(
        self,
        message: sensor_msgs.msg.JointState,
    ) -> None:
        name_to_position = dict(
            zip(
                message.name,
                message.position,
                strict=False,
            )
        )

        try:
            positions = np.asarray(
                [
                    name_to_position[name]
                    for name in ROS_PANDA_ARM_JOINT_NAMES
                ],
                dtype=np.float64,
            )
        except KeyError:
            return

        if positions.shape != (7,):
            return

        if not np.all(np.isfinite(positions)):
            self._abort(
                'received non-finite Panda joint feedback'
            )
            return

        velocities: FloatArray | None = None

        if len(message.velocity) > 0:
            name_to_velocity = dict(
                zip(
                    message.name,
                    message.velocity,
                    strict=False,
                )
            )

            try:
                candidate = np.asarray(
                    [
                        name_to_velocity[name]
                        for name in ROS_PANDA_ARM_JOINT_NAMES
                    ],
                    dtype=np.float64,
                )
            except KeyError:
                # Incomplete velocity feedback is represented
                # as unavailable rather than using stale data.
                candidate = None

            if candidate is not None:
                if not np.all(np.isfinite(candidate)):
                    self._abort(
                        'received non-finite Panda joint velocities'
                    )
                    return

                velocities = candidate

        self._latest_joint_positions = positions
        self._latest_joint_velocities = velocities

    def _controller_state_callback(
        self,
        message: control_msgs.msg.JointTrajectoryControllerState,
    ) -> None:
        name_to_command = dict(
            zip(
                message.joint_names,
                message.output.positions,
                strict=False,
            )
        )

        try:
            commands = np.asarray(
                [
                    name_to_command[name]
                    for name in ROS_PANDA_ARM_JOINT_NAMES
                ],
                dtype=np.float64,
            )
        except KeyError:
            return

        if commands.shape != (7,):
            return

        if not np.all(np.isfinite(commands)):
            self._abort(
                'received non-finite Panda controller output'
            )
            return

        self._latest_controller_command = commands

    def _wrench_callback(
        self,
        message: geometry_msgs.msg.WrenchStamped,
    ) -> None:
        force = np.asarray(
            [
                message.wrench.force.x,
                message.wrench.force.y,
                message.wrench.force.z,
            ],
            dtype=np.float64,
        )

        torque = np.asarray(
            [
                message.wrench.torque.x,
                message.wrench.torque.y,
                message.wrench.torque.z,
            ],
            dtype=np.float64,
        )

        if not (
            np.all(np.isfinite(force))
            and np.all(np.isfinite(torque))
        ):
            self._abort(
                'received non-finite wrist wrench'
            )
            return

        self._latest_wrist_force_sensor = force
        self._latest_wrist_torque_sensor = torque

    def _persist_expert_capture(self) -> str | None:
        if self._expert_frame_buffer is None:
            return None

        try:
            write_expert_frame_capture(
                Path(self._expert_capture_path),
                self._expert_frame_buffer.frames,
            )
        except (OSError, ValueError, TypeError) as exc:
            return (
                'expert capture persistence failed: '
                f'{type(exc).__name__}: {exc}'
            )

        return None

    def _abort(
        self,
        reason: str,
    ) -> None:
        if self._terminal:
            return

        if self._expert_frame_buffer is not None:
            try:
                self._expert_frame_buffer.mark_terminal(
                    success=False
                )
            except (TypeError, ValueError) as exc:
                reason = (
                    f'{reason}; expert capture finalization failed: {exc}'
                )

        capture_error = self._persist_expert_capture()

        if capture_error is not None:
            reason = f'{reason}; {capture_error}'

        self._terminal = True
        self._terminal_reason = reason
        self._expert.phase = InsertionPhase.ABORT

        self._publish_status(
            STATUS_FAILED,
            reason,
        )

        self.get_logger().error(
            f'Local insertion expert aborted: {reason}'
        )

        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except RuntimeError:
                pass

    def _complete(
        self,
        reason: str,
    ) -> None:
        if self._terminal:
            return

        if self._expert_frame_buffer is not None:
            try:
                self._expert_frame_buffer.mark_terminal(
                    success=True
                )
            except (TypeError, ValueError) as exc:
                self._abort(
                    f'expert capture finalization failed: {exc}'
                )
                return

        capture_error = self._persist_expert_capture()

        if capture_error is not None:
            self._terminal = True
            self._terminal_reason = capture_error
            self._expert.phase = InsertionPhase.ABORT
            self._publish_status(
                STATUS_FAILED,
                capture_error,
            )
            self.get_logger().error(capture_error)
            return

        self._terminal = True
        self._terminal_reason = reason

        self._publish_status(
            STATUS_SUCCESS,
            reason,
        )

        self.get_logger().info(
            f'Local insertion expert completed: {reason}'
        )

    def _goal_timed_out(self) -> bool:
        if (
            not self._goal_in_flight
            or self._goal_started_ns is None
        ):
            return False

        elapsed_s = (
            self.get_clock().now().nanoseconds
            - self._goal_started_ns
        ) / 1.0e9

        return elapsed_s >= self._goal_timeout_s

    def _send_joint_target(
        self,
        target: FloatArray,
    ) -> None:
        if self._terminal:
            return

        if self._goal_in_flight:
            return

        target_array = np.asarray(
            target,
            dtype=np.float64,
        )

        if target_array.shape != (7,):
            self._abort(
                'expert produced a joint target with invalid shape'
            )
            return

        if not np.all(np.isfinite(target_array)):
            self._abort(
                'expert produced non-finite joint targets'
            )
            return

        if not self._trajectory_client.server_is_ready():
            self.get_logger().warn(
                'FollowJointTrajectory action server is not ready'
            )
            return

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = list(
            ROS_PANDA_ARM_JOINT_NAMES
        )

        point = JointTrajectoryPoint()

        point.positions = [
            float(value)
            for value in target_array
        ]

        point.time_from_start = Duration(
            seconds=self._trajectory_duration_s
        ).to_msg()

        goal.trajectory.points = [point]

        self._goal_in_flight = True

        self._goal_started_ns = (
            self.get_clock().now().nanoseconds
        )

        try:
            future = self._trajectory_client.send_goal_async(
                goal
            )
        except Exception:
            # Submission failed before the controller could
            # accept or reject the trajectory goal.
            self._goal_in_flight = False
            self._goal_started_ns = None

            if self._expert_frame_buffer is not None:
                self._expert_frame_buffer.reject_pending_command()

            # Preserve the original submission error for the
            # existing control-loop abort handler.
            raise

        future.add_done_callback(
            self._goal_response_callback
        )

    def _goal_response_callback(
        self,
        future: object,
    ) -> None:
        if self._terminal:
            self._goal_in_flight = False

            if self._expert_frame_buffer is not None:
                try:
                    self._expert_frame_buffer.reject_pending_command()
                except ValueError:
                    # An already-resolved command has no pending frame.
                    pass

            return

        try:
            goal_handle = future.result()
        except Exception as exc:
            self._goal_in_flight = False
            if self._expert_frame_buffer is not None:
                self._expert_frame_buffer.reject_pending_command()
            self._abort(
                f'trajectory goal request failed: {exc}'
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self._goal_in_flight = False
            if self._expert_frame_buffer is not None:
                self._expert_frame_buffer.reject_pending_command()
            self._abort(
                'panda_arm_controller rejected the trajectory goal'
            )
            return

        self._goal_handle = goal_handle

        if self._expert_frame_buffer is not None:
            try:
                self._expert_frame_buffer.accept_pending_command()
            except ValueError as exc:
                self._abort(
                    f'expert capture acceptance failed: {exc}'
                )
                return

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self._goal_result_callback
        )

    def _goal_result_callback(
        self,
        future: object,
    ) -> None:
        self._goal_in_flight = False
        self._goal_started_ns = None
        self._goal_handle = None

        if self._terminal:
            return

        try:
            wrapped_result = future.result()
        except Exception as exc:
            self._abort(
                f'trajectory result failed: {exc}'
            )
            return

        if wrapped_result is None:
            self._abort(
                'trajectory controller returned no result'
            )
            return

        result = wrapped_result.result

        if (
            result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._abort(
                'trajectory controller reported failure '
                f'code {result.error_code}: '
                f'{result.error_string}'
            )

    def _control_tick(self) -> None:
        if self._terminal:
            return

        if not self._armed:
            return

        if self._goal_timed_out():
            self._abort(
                'trajectory goal exceeded '
                f'{self._goal_timeout_s:.3f} s timeout'
            )
            return

        if (
            self._latest_joint_positions is None
            or self._latest_wrist_force_sensor is None
            or self._latest_wrist_torque_sensor is None
        ):
            return

        if (
            self._command_enabled
            and self._latest_controller_command is None
        ):
            return

        try:
            snapshot = self._kinematics.update(
                self._latest_joint_positions
            )

            wrist_rotation_world = (
                self._kinematics.wrist_rotation_world()
            )

            wrist_force_world = (
                wrist_rotation_world
                @ self._latest_wrist_force_sensor
            )

            wrist_torque_world = (
                wrist_rotation_world
                @ self._latest_wrist_torque_sensor
            )

            if not (
                np.all(np.isfinite(wrist_force_world))
                and np.all(np.isfinite(wrist_torque_world))
            ):
                raise RuntimeError(
                    'Transformed wrist wrench is non-finite'
                )

            task_status = evaluate_insertion_task(
                snapshot.peg_tip_position,
                snapshot.receptacle_center_position,
                snapshot.insertion_axis_position,
                config=self._task_config,
            )

            if task_status.failure:
                self._abort(
                    'task geometry entered failure state'
                )
                return

            if task_status.success:
                self._complete(
                    'canonical insertion task reached success'
                )
                return

            measured_target = self._expert.joint_targets(
                current_joint_targets=snapshot.joint_positions,
                peg_tip_position=snapshot.peg_tip_position,
                receptacle_center_position=(
                    snapshot.receptacle_center_position
                ),
                insertion_axis_position=(
                    snapshot.insertion_axis_position
                ),
                peg_tip_jacobian=snapshot.peg_tip_jacobian,
                success_insertion_depth_m=(
                    self._task_config.success_insertion_depth_m
                ),
                wrist_force_world_n=wrist_force_world,
                failed=task_status.failure,
            )

            joint_delta = (
                measured_target
                - snapshot.joint_positions
            )

            if joint_delta.shape != (7,):
                raise RuntimeError(
                    'Expert joint delta has invalid shape'
                )

            if not np.all(np.isfinite(joint_delta)):
                raise RuntimeError(
                    'Expert joint delta is non-finite'
                )

            command_magnitude = float(
                np.linalg.norm(joint_delta)
            )

            force_norm = float(
                np.linalg.norm(
                    wrist_force_world
                )
            )

            torque_norm = float(
                np.linalg.norm(
                    wrist_torque_world
                )
            )

            controller_offset_norm = 0.0

            if self._latest_controller_command is not None:
                controller_offset_norm = float(
                    np.linalg.norm(
                        self._latest_controller_command
                        - snapshot.joint_positions
                    )
                )

            diagnostics = np.asarray(
                [
                    task_status.lateral_error,
                    task_status.axial_offset,
                    task_status.insertion_depth,
                    *wrist_force_world,
                    *wrist_torque_world,
                    force_norm,
                    torque_norm,
                    command_magnitude,
                    controller_offset_norm,
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(diagnostics)):
                raise RuntimeError(
                    'Expert diagnostics became non-finite'
                )

            self.get_logger().debug(
                f'phase={self._expert.phase.value} '
                f'lateral_error={task_status.lateral_error:.6f} '
                f'axial_offset={task_status.axial_offset:.6f} '
                f'insertion_depth={task_status.insertion_depth:.6f} '
                f'force_norm={force_norm:.6f} '
                f'torque_norm={torque_norm:.6f} '
                f'command_norm={command_magnitude:.6f} '
                f'controller_offset_norm={controller_offset_norm:.6f}'
            )

            if not self._command_enabled:
                return

            if self._goal_in_flight:
                return

            if command_magnitude <= np.finfo(np.float64).eps:
                return

            if self._latest_controller_command is None:
                return

            command_target = (
                self._latest_controller_command
                + joint_delta
            )

            if command_target.shape != (7,):
                raise RuntimeError(
                    'Controller-relative joint target has invalid shape'
                )

            if not np.all(np.isfinite(command_target)):
                raise RuntimeError(
                    'Controller-relative joint target is non-finite'
                )

            if self._expert_frame_buffer is not None:
                self._expert_frame_buffer.stage_submitted_command(
                    timestamp_s=monotonic(),
                    joint_position=tuple(
                        float(value)
                        for value in snapshot.joint_positions
                    ),
                    joint_velocity=(
                        tuple(
                            float(value)
                            for value in self._latest_joint_velocities
                        )
                        if self._latest_joint_velocities is not None
                        else None
                    ),
                    task_relative_state=tuple(
                        float(value)
                        for value in (
                            snapshot.receptacle_center_position
                            - snapshot.peg_tip_position
                        )
                    ),
                    wrist_wrench=tuple(
                        float(value)
                        for value in np.concatenate(
                            (wrist_force_world, wrist_torque_world)
                        )
                    ),
                    submitted_joint_target=tuple(
                        float(value)
                        for value in command_target
                    ),
                )

            self._send_joint_target(command_target)

            if (
                self._expert_frame_buffer is not None
                and not self._goal_in_flight
            ):
                self._expert_frame_buffer.reject_pending_command()

        except (RuntimeError, ValueError) as exc:
            self._abort(
                str(exc)
            )


def main(args: list[str] | None = None) -> None:
    """Run the ROS local insertion expert node."""
    rclpy.init(args=args)

    node = LocalInsertionExpertNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Jazzy can surface a subscription-take RuntimeError when Ctrl+C
        # invalidates the ROS context while the executor is still unwinding.
        # Preserve genuine runtime failures while allowing clean shutdown.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
