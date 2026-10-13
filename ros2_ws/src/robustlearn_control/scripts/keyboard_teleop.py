#!/usr/bin/env python3
"""Bounded keyboard teleoperation for the Panda arm."""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty

from control_msgs.action import FollowJointTrajectory
import geometry_msgs.msg
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from robustlearn.teleop import (
    bounded_joint_target,
    command_due,
    deadman_active,
    keyboard_joint_delta,
    TeleopConfig,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

PANDA_JOINT_NAMES: tuple[str, ...] = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
)


class KeyboardTeleop(Node):
    """Keyboard-driven Panda teleoperation through ros2_control."""

    def __init__(self) -> None:
        super().__init__('keyboard_teleop')

        self.declare_parameter('joint_step_rad', 0.01)
        self.declare_parameter('deadman_timeout_s', 0.25)
        self.declare_parameter('min_command_period_s', 0.05)
        self.declare_parameter('trajectory_duration_s', 0.10)
        self.declare_parameter('poll_period_s', 0.02)
        self.declare_parameter('operator_status_period_s', 0.50)
        self.declare_parameter('use_terminal_input', True)
        self.declare_parameter(
            'input_topic',
            '/keyboard_teleop/input',
        )

        self._config = TeleopConfig(
            joint_step_rad=float(
                self.get_parameter('joint_step_rad').value
            ),
            deadman_timeout_s=float(
                self.get_parameter('deadman_timeout_s').value
            ),
            min_command_period_s=float(
                self.get_parameter(
                    'min_command_period_s'
                ).value
            ),
        )

        self._trajectory_duration_s = float(
            self.get_parameter(
                'trajectory_duration_s'
            ).value
        )
        self._poll_period_s = float(
            self.get_parameter('poll_period_s').value
        )
        self._operator_status_period_s = float(
            self.get_parameter(
                'operator_status_period_s'
            ).value
        )
        self._use_terminal_input = bool(
            self.get_parameter(
                'use_terminal_input'
            ).value
        )
        self._input_topic = str(
            self.get_parameter(
                'input_topic'
            ).value
        ).strip()

        if self._trajectory_duration_s <= 0.0:
            raise ValueError(
                'trajectory_duration_s must be positive'
            )

        if self._poll_period_s <= 0.0:
            raise ValueError(
                'poll_period_s must be positive'
            )

        if (
            not math.isfinite(
                self._operator_status_period_s
            )
            or self._operator_status_period_s <= 0.0
        ):
            raise ValueError(
                'operator_status_period_s must be finite and positive'
            )

        if not self._use_terminal_input and not self._input_topic:
            raise ValueError(
                'input_topic must be non-empty when terminal input is disabled'
            )

        self._joint_positions: list[float] | None = None
        self._last_deadman_refresh_s: float | None = None
        self._last_command_s: float | None = None
        self._last_operator_status_s: float | None = None

        self._latest_wrist_force: tuple[float, float, float] | None = None
        self._latest_wrist_torque: tuple[float, float, float] | None = None
        self._latest_expert_status: str | None = None
        self._injected_keys: list[str] = []

        self._goal_handle = None
        self._goal_pending = False
        self._deadman_was_active = False
        self._aborted = False

        self._terminal_settings = None

        self._joint_state_subscription = (
            self.create_subscription(
                JointState,
                '/joint_states',
                self._on_joint_state,
                10,
            )
        )

        self._wrench_subscription = self.create_subscription(
            geometry_msgs.msg.WrenchStamped,
            '/wrist_ft/wrench',
            self._on_wrench,
            10,
        )

        self._expert_status_subscription = self.create_subscription(
            String,
            '/local_insertion_expert/status',
            self._on_expert_status,
            10,
        )

        self._input_subscription = None

        if not self._use_terminal_input:
            self._input_subscription = self.create_subscription(
                String,
                self._input_topic,
                self._on_injected_input,
                10,
            )

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/'
            'follow_joint_trajectory',
        )

        if self._use_terminal_input:
            self._enable_terminal()

        self._timer = self.create_timer(
            self._poll_period_s,
            self._poll_keyboard,
        )

        self.get_logger().info(
            'Keyboard teleop ready: '
            'task=panda_insertion source=teleop. '
            'SPACE refreshes deadman; '
            'q/a w/s e/d r/f t/g y/h u/j jog joints 1-7; '
            'x stops; Esc aborts.'
        )

    def _enable_terminal(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                'keyboard teleop requires an interactive terminal'
            )

        self._terminal_settings = termios.tcgetattr(
            sys.stdin
        )
        tty.setcbreak(
            sys.stdin.fileno()
        )

    def _restore_terminal(self) -> None:
        if self._terminal_settings is None:
            return

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            self._terminal_settings,
        )
        self._terminal_settings = None

    def destroy_node(self) -> bool:
        self._restore_terminal()
        return super().destroy_node()

    @staticmethod
    def _now_s() -> float:
        return time.monotonic()

    def _on_joint_state(
        self,
        message: JointState,
    ) -> None:
        position_by_name = dict(
            zip(
                message.name,
                message.position,
                strict=False,
            )
        )

        if not all(
            name in position_by_name
            for name in PANDA_JOINT_NAMES
        ):
            return

        self._joint_positions = [
            float(position_by_name[name])
            for name in PANDA_JOINT_NAMES
        ]

    def _on_wrench(
        self,
        message: geometry_msgs.msg.WrenchStamped,
    ) -> None:
        force = (
            float(message.wrench.force.x),
            float(message.wrench.force.y),
            float(message.wrench.force.z),
        )
        torque = (
            float(message.wrench.torque.x),
            float(message.wrench.torque.y),
            float(message.wrench.torque.z),
        )

        if not all(
            math.isfinite(value)
            for value in force + torque
        ):
            self.get_logger().warning(
                'Ignoring non-finite wrist wrench feedback.'
            )
            return

        self._latest_wrist_force = force
        self._latest_wrist_torque = torque

    def _on_expert_status(
        self,
        message: String,
    ) -> None:
        status = message.data.strip()

        if status:
            self._latest_expert_status = status

    def _on_injected_input(
        self,
        message: String,
    ) -> None:
        if self._use_terminal_input:
            return

        if len(message.data) != 1:
            self.get_logger().warning(
                'Ignoring injected teleop input that is not one character.'
            )
            return

        self._injected_keys.append(
            message.data
        )

    def _log_operator_status(
        self,
        now_s: float,
    ) -> None:
        active = deadman_active(
            now_s=now_s,
            last_refresh_s=self._last_deadman_refresh_s,
            config=self._config,
        )

        joint_state = (
            'ready'
            if self._joint_positions is not None
            else 'unavailable'
        )

        if (
            self._latest_wrist_force is None
            or self._latest_wrist_torque is None
        ):
            wrench_status = 'wrench=unavailable'
        else:
            force_norm = math.sqrt(
                sum(
                    value * value
                    for value in self._latest_wrist_force
                )
            )
            torque_norm = math.sqrt(
                sum(
                    value * value
                    for value in self._latest_wrist_torque
                )
            )
            wrench_status = (
                f'force_norm={force_norm:.3f}N '
                f'torque_norm={torque_norm:.3f}Nm'
            )

        expert_status = (
            self._latest_expert_status
            if self._latest_expert_status is not None
            else 'unavailable'
        )

        deadman_status = (
            'active'
            if active
            else 'inactive'
        )

        self.get_logger().info(
            'operator_status '
            'task=panda_insertion '
            'source=teleop '
            f'deadman={deadman_status} '
            f'joints={joint_state} '
            f'{wrench_status} '
            f'expert_status={expert_status}'
        )

    def _read_key(self) -> str | None:
        if self._injected_keys:
            return self._injected_keys.pop(0)

        if not self._use_terminal_input:
            return None

        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0.0,
        )

        if not readable:
            return None

        return sys.stdin.read(1)

    def _poll_keyboard(self) -> None:
        if self._aborted:
            return

        now_s = self._now_s()

        if (
            self._last_operator_status_s is None
            or (
                now_s - self._last_operator_status_s
                >= self._operator_status_period_s
            )
        ):
            self._log_operator_status(now_s)
            self._last_operator_status_s = now_s

        active = deadman_active(
            now_s=now_s,
            last_refresh_s=self._last_deadman_refresh_s,
            config=self._config,
        )

        if self._deadman_was_active and not active:
            self.get_logger().info(
                'Deadman expired; stopping teleoperation progression.'
            )
            self._cancel_active_goal()

        self._deadman_was_active = active

        key = self._read_key()

        if key is None:
            return

        if key == ' ':
            self._last_deadman_refresh_s = now_s
            self._deadman_was_active = True
            return

        if key == 'x':
            self._stop()
            return

        if key == '\x1b':
            self._abort()
            return

        try:
            delta = keyboard_joint_delta(
                key,
                config=self._config,
            )
        except ValueError:
            return

        active = deadman_active(
            now_s=now_s,
            last_refresh_s=self._last_deadman_refresh_s,
            config=self._config,
        )

        if not active:
            self.get_logger().warning(
                'Jog ignored because deadman is inactive.'
            )
            return

        if not command_due(
            now_s=now_s,
            last_command_s=self._last_command_s,
            config=self._config,
        ):
            return

        if self._goal_pending:
            return

        if self._joint_positions is None:
            self.get_logger().warning(
                'Jog ignored because joint state is unavailable.'
            )
            return

        try:
            target = bounded_joint_target(
                self._joint_positions,
                delta,
                enabled=True,
                config=self._config,
            )
        except ValueError as exc:
            self.get_logger().error(
                f'Teleop command rejected: {exc}'
            )
            return

        self._send_target(
            target.tolist()
        )
        self._last_command_s = now_s

    def _send_target(
        self,
        positions: list[float],
    ) -> None:
        if not self._trajectory_client.server_is_ready():
            self.get_logger().warning(
                'FollowJointTrajectory server is not ready.'
            )
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(
            PANDA_JOINT_NAMES
        )

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            seconds=self._trajectory_duration_s
        ).to_msg()

        goal.trajectory.points = [point]

        self._goal_pending = True

        future = self._trajectory_client.send_goal_async(
            goal
        )
        future.add_done_callback(
            self._on_goal_response
        )

    def _on_goal_response(
        self,
        future,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._goal_pending = False
            self.get_logger().error(
                f'Failed to send teleop trajectory: {exc}'
            )
            return

        if not goal_handle.accepted:
            self._goal_pending = False
            self.get_logger().warning(
                'Teleop trajectory goal was rejected.'
            )
            return

        self._goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self._on_goal_result
        )

    def _on_goal_result(
        self,
        future,
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Teleop trajectory result failed: {exc}'
            )

        self._goal_handle = None
        self._goal_pending = False

    def _cancel_active_goal(self) -> None:
        goal_handle = self._goal_handle

        if goal_handle is None:
            return

        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:
            self.get_logger().error(
                f'Failed to cancel teleop trajectory: {exc}'
            )

        self._goal_handle = None
        self._goal_pending = False

    def _stop(self) -> None:
        self._last_deadman_refresh_s = None
        self._deadman_was_active = False
        self._cancel_active_goal()

        self.get_logger().info(
            'Teleoperation stopped.'
        )

    def _abort(self) -> None:
        self._aborted = True
        self._stop()

        self.get_logger().info(
            'Teleoperation aborted by operator.'
        )

        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = KeyboardTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
