#!/usr/bin/env python3
"""Bounded keyboard teleoperation for the Panda arm."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import select
import sys
import termios
import time
import tty
from uuid import uuid4

from action_msgs.msg import GoalStatus
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
from robustlearn.teleop_collection_summary import (
    summarize_teleop_attempt,
)
from robustlearn.teleop_demo_collection import (
    apply_teleop_operator_decision,
    build_teleop_episode_result,
    build_teleop_observation_frame,
    create_teleop_recording_components,
    teleop_review_key_action,
    TeleopCollectionConfig,
    TeleopControllerCompletionTracker,
    TeleopReviewMotionGuard,
)
from robustlearn.teleop_summary_history import (
    update_teleop_summary_history,
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
        self.declare_parameter('record_demonstrations', False)
        self.declare_parameter('teleop_output_root', '')
        self.declare_parameter('teleop_episode_index', -1)
        self.declare_parameter('teleop_seed', -1)
        self.declare_parameter('teleop_scenario', '')
        self.declare_parameter('teleop_git_commit', '')
        self.declare_parameter('teleop_config_id', '')
        self.declare_parameter('teleop_task', 'panda_insertion')
        self.declare_parameter('teleop_failure_reason', '')
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
        self._joint_velocities: list[float] | None = None
        self._last_deadman_refresh_s: float | None = None
        self._last_command_s: float | None = None
        self._last_operator_status_s: float | None = None

        self._latest_wrist_force: tuple[float, float, float] | None = None
        self._latest_wrist_torque: tuple[float, float, float] | None = None
        self._latest_expert_status: str | None = None
        self._injected_keys: list[str] = []

        self._goal_handle = None
        self._goal_pending = False
        self._request_sequence = 0
        self._active_request_id: int | None = None
        self._deadman_was_active = False

        capture_requested = self.get_parameter(
            'record_demonstrations'
        ).value

        (
            self._teleop_session,
            self._capture_kinematics,
        ) = create_teleop_recording_components(
            capture_requested
        )

        self._teleop_collection_config = None
        self._teleop_attempt_id = (
            uuid4().hex if capture_requested else None
        )

        if capture_requested:
            failure_reason = str(
                self.get_parameter(
                    'teleop_failure_reason'
                ).value
            ).strip()

            self._teleop_collection_config = TeleopCollectionConfig(
                output_root=Path(
                    str(
                        self.get_parameter(
                            'teleop_output_root'
                        ).value
                    ).strip()
                ),
                episode_index=self.get_parameter(
                    'teleop_episode_index'
                ).value,
                seed=self.get_parameter(
                    'teleop_seed'
                ).value,
                scenario=self.get_parameter(
                    'teleop_scenario'
                ).value,
                git_commit=self.get_parameter(
                    'teleop_git_commit'
                ).value,
                config_id=self.get_parameter(
                    'teleop_config_id'
                ).value,
                task=self.get_parameter(
                    'teleop_task'
                ).value,
                failure_reason=(
                    failure_reason if failure_reason else None
                ),
            )

        self._review_motion_guard = (
            TeleopReviewMotionGuard()
            if self._teleop_session is not None
            else None
        )

        self._controller_completion_tracker = (
            TeleopControllerCompletionTracker()
            if self._teleop_session is not None
            else None
        )
        self._aborted = False
        self._review_mode = False

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
        guard = getattr(
            self, '_review_motion_guard', None
        )

        # Incomplete joint messages invalidate stationary evidence.
        if guard is not None and not all(
            name in message.name
            for name in PANDA_JOINT_NAMES
        ):
            guard.reset()

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
            if guard is not None:
                guard.reset()
            return

        self._joint_positions = [
            float(position_by_name[name])
            for name in PANDA_JOINT_NAMES
        ]

        # Missing or incomplete velocity feedback must not reuse
        # a previous sample.
        self._joint_velocities = None

        if len(message.velocity) != len(message.name):
            if guard is not None:
                guard.reset()
            return

        velocity_by_name = dict(
            zip(
                message.name,
                message.velocity,
                strict=False,
            )
        )

        if not all(
            name in velocity_by_name
            for name in PANDA_JOINT_NAMES
        ):
            if guard is not None:
                guard.reset()
            return

        velocities = [
            float(velocity_by_name[name])
            for name in PANDA_JOINT_NAMES
        ]

        if all(math.isfinite(value) for value in velocities):
            self._joint_velocities = velocities

        if guard is not None:
            # Use the local monotonic receipt time, not a
            # potentially unrelated ROS/simulation clock.
            velocities = (
                tuple(self._joint_velocities)
                if (
                    self._joint_velocities is not None
                    and self._joint_positions is not None
                    and all(
                        math.isfinite(value)
                        for value in self._joint_positions
                    )
                )
                else None
            )

            guard.observe(
                timestamp_s=self._now_s(),
                joint_velocities=velocities,
            )

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

    def _record_collection_outcome(
        self,
        outcome: str,
        *,
        episode_path: Path | None = None,
        error: str | None = None,
    ) -> Path:
        """Validate and durably record one collection outcome."""
        config = getattr(
            self, '_teleop_collection_config', None
        )
        session = getattr(self, '_teleop_session', None)
        attempt_id = getattr(
            self, '_teleop_attempt_id', None
        )

        if (
            config is None
            or session is None
            or not isinstance(attempt_id, str)
            or not attempt_id.strip()
        ):
            raise ValueError(
                'Cannot report collection without session identity.'
            )

        attempt = summarize_teleop_attempt(
            config=config,
            session=session,
            outcome=outcome,
            episode_path=episode_path,
            error=error,
        )

        return update_teleop_summary_history(
            replace(attempt, attempt_id=attempt_id),
            output_root=config.output_root,
        )

    def _retry_pending_summary(self) -> None:
        """Retry reporting without rewriting an accepted episode."""
        pending = getattr(self, '_teleop_pending_summary', None)
        if pending is None:
            self.get_logger().warning('No summary update to retry.')
            return

        outcome, episode_path, config = pending
        if config != getattr(self, '_teleop_collection_config', None):
            self.get_logger().error('Summary retry metadata changed.')
            return

        try:
            summary_path = self._record_collection_outcome(
                outcome, episode_path=episode_path,
            )
        except (OSError, ValueError, TypeError) as exc:
            self.get_logger().error(f'Summary retry failed: {exc}')
            return

        self._teleop_pending_summary = None
        self.get_logger().info(
            f'Collection summary recovered: {summary_path}'
        )

    def _approve_reviewed_recording(
        self,
        decision: str,
    ) -> None:
        """Persist approval, or retry an unchanged failed write."""
        if decision not in ('accept_success', 'accept_failure'):
            raise ValueError('unsupported recording approval decision')

        session = getattr(self, '_teleop_session', None)
        config = getattr(self, '_teleop_collection_config', None)

        if session is None or config is None:
            self.get_logger().error(
                'Recording or collection configuration unavailable.'
            )
            return

        if session.status == 'recording':
            if not self._review_ready_for_decision():
                self.get_logger().warning(
                    'Approval blocked: controller or motion not ready.'
                )
                return

            try:
                result = build_teleop_episode_result(
                    session,
                    episode_index=config.episode_index,
                    seed=config.seed,
                    scenario=config.scenario,
                    success=(decision == 'accept_success'),
                    failure_reason=(
                        None
                        if decision == 'accept_success'
                        else config.failure_reason
                    ),
                )
            except (TypeError, ValueError) as exc:
                self.get_logger().error(
                    f'Invalid teleop approval metadata: {exc}'
                )
                return

        elif session.status == 'accepted':
            pending = getattr(
                self, '_teleop_pending_approval', None
            )

            if (
                pending is None
                or pending[0] != decision
                or pending[2] != config
            ):
                self.get_logger().error(
                    'Retry denied: no matching failed write.'
                )
                return

            if not self._review_ready_for_decision(
                allow_accepted_retry=True
            ):
                self.get_logger().warning(
                    'Retry blocked: controller or motion not ready.'
                )
                return

            result = pending[1]

        else:
            self.get_logger().error(
                'Rejected or aborted recording cannot be approved.'
            )
            return

        try:
            path = apply_teleop_operator_decision(
                session,
                decision,
                result=result,
                output_root=config.output_root,
                git_commit=config.git_commit,
                config_id=config.config_id,
                task=config.task,
            )
        except (OSError, ValueError, TypeError) as exc:
            # A write can fail after session.accept(). Retain the
            # exact original metadata for an explicit operator retry.
            if session.status == 'accepted':
                self._teleop_pending_approval = (
                    decision, result, config
                )
            else:
                self._teleop_pending_approval = None

            try:
                self._record_collection_outcome(
                    'persistence_failed',
                    error=str(exc),
                )
            except (OSError, ValueError, TypeError) as report_exc:
                self.get_logger().error(
                    f'Could not update collection summary: '
                    f'{report_exc}'
                )

            self.get_logger().error(
                f'Teleop approval/persistence failed: {exc}'
            )
            return

        # The dataset save succeeded. Summary failures must not
        # falsely report the dataset write itself as failed.
        self._teleop_pending_approval = None

        outcome = (
            'accepted_success'
            if decision == 'accept_success'
            else 'accepted_failure'
        )

        try:
            summary_path = self._record_collection_outcome(
                outcome,
                episode_path=path,
            )
        except (OSError, ValueError, TypeError) as exc:
            self._teleop_pending_summary = (
                outcome, path, config
            )
            self.get_logger().error(
                f'Dataset episode saved at {path}, but '
                f'collection summary update failed: {exc}'
            )
            return

        self._teleop_pending_summary = None

        self.get_logger().info(
            f'Teleop demonstration accepted: '
            f'outcome={decision} path={path} '
            f'summary={summary_path}'
        )

    def _review_ready_for_decision(
        self,
        *,
        allow_accepted_retry: bool = False,
    ) -> bool:
        """Require controller completion and stationary review feedback."""
        session = getattr(
            self, '_teleop_session', None
        )

        if session is None:
            return False

        eligible = (
            session.status == 'recording'
            or (
                allow_accepted_retry
                and session.status == 'accepted'
                and getattr(
                    self, '_teleop_pending_approval', None
                ) is not None
            )
        )

        if (
            not eligible
            or not getattr(self, '_review_mode', False)
            or getattr(self, '_aborted', False)
        ):
            return False

        tracker = getattr(
            self, '_controller_completion_tracker', None
        )
        guard = getattr(
            self, '_review_motion_guard', None
        )

        # Missing tracking components must never permit approval.
        if tracker is None or guard is None:
            return False

        # This remains false if any old/stale goal is outstanding,
        # or a controller result could not be verified.
        if not tracker.settled:
            return False

        request_pending = (
            self._goal_pending
            or self._active_request_id is not None
        )

        goal_active = self._goal_handle is not None

        # A normally completed trajectory does not require
        # cancellation; every trajectory must be terminal.
        return guard.ready(
            now_s=self._now_s(),
            in_review=True,
            controller_request_pending=request_pending,
            controller_goal_active=goal_active,
            cancellation_confirmed=tracker.settled,
        )

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

        # Escape remains available in every operating mode.
        if key == '\x1b':
            self._abort()
            return

        if self._review_mode:
            if key == 's':
                self._retry_pending_summary()
                return

            action = teleop_review_key_action(
                key,
                in_review=True,
            )

            if action == 'resume':
                session = self._teleop_session

                if (
                    session is None
                    or session.status != 'recording'
                ):
                    self.get_logger().warning(
                        'Finalized recording cannot resume; '
                        'restart teleoperation for a new session.'
                    )
                    return

                if not self._review_ready_for_decision():
                    self.get_logger().warning(
                        'Cannot resume while controller completion '
                        'or stationary feedback is unverified.'
                    )
                    return

                self._review_mode = False

                guard = getattr(
                    self, '_review_motion_guard', None
                )

                if guard is not None:
                    guard.reset()
                self.get_logger().info(
                    'Review exited; refresh deadman before jogging.'
                )
            elif action == 'reject':
                if not self._review_ready_for_decision():
                    self.get_logger().warning(
                        'Recording rejection deferred until controller '
                        'is settled and feedback is stationary.'
                    )
                else:
                    session = self._teleop_session

                    if session is not None:
                        session.reject()
                        try:
                            self._record_collection_outcome('rejected')
                        except (OSError, ValueError, TypeError) as exc:
                            self._teleop_pending_summary = (
                                'rejected',
                                None,
                                getattr(self, '_teleop_collection_config', None),
                            )
                            self.get_logger().error(
                                f'Rejection summary failed: {exc}'
                            )
                        self.get_logger().info(
                            'Recording rejected; no episode exported. '
                            'Restart teleoperation for a new session.'
                        )

            elif action in ('accept_success', 'accept_failure'):
                self._approve_reviewed_recording(action)

            # Never interpret keys as jog or deadman input
            # while recording review is active.
            return

        session = self._teleop_session

        if (
            session is not None
            and session.status == 'recording'
        ):
            action = teleop_review_key_action(
                key,
                in_review=False,
            )

            if action == 'enter_review':
                self._stop()

                guard = getattr(
                    self, '_review_motion_guard', None
                )

                if guard is not None:
                    guard.reset()

                self._review_mode = True
                self.get_logger().info(
                    'Recording review entered; '
                    'robot commands are blocked.'
                )
                return

        if key == ' ':
            self._last_deadman_refresh_s = now_s
            self._deadman_was_active = True
            return

        if key == 'x':
            self._stop()
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

    def _stage_capture_for_target(
        self,
        positions: list[float],
    ) -> bool:
        """Stage real feedback before submitting a controller goal."""
        session = self._teleop_session

        if session is None:
            return True

        if session.status != 'recording':
            self.get_logger().warning(
                'Teleop recording is finalized; command blocked.'
            )
            return False

        if self._capture_kinematics is None:
            self.get_logger().error(
                'Teleop capture kinematics are unavailable.'
            )
            return False

        try:
            frame = build_teleop_observation_frame(
                kinematics=self._capture_kinematics,
                frame_index=len(session.frames),
                timestamp_s=self._now_s(),
                joint_position=(
                    tuple(self._joint_positions)
                    if self._joint_positions is not None
                    else None
                ),
                joint_velocity=(
                    tuple(self._joint_velocities)
                    if self._joint_velocities is not None
                    else None
                ),
                wrist_force_sensor=self._latest_wrist_force,
                wrist_torque_sensor=self._latest_wrist_torque,
                submitted_joint_target=tuple(positions),
            )

            session.stage_frame(frame)

        except (ValueError, TypeError, RuntimeError) as exc:
            self.get_logger().warning(
                f'Teleop capture blocked command: {exc}'
            )
            return False

        return True

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

        if not self._stage_capture_for_target(positions):
            return

        self._request_sequence = (
            getattr(self, '_request_sequence', 0) + 1
        )
        request_id = self._request_sequence
        self._active_request_id = request_id

        tracker = getattr(
            self, '_controller_completion_tracker', None
        )

        if tracker is not None:
            tracker.submitted(request_id)

        self._goal_pending = True

        try:
            future = self._trajectory_client.send_goal_async(
                goal
            )
        except Exception as exc:
            self._goal_pending = False
            self._active_request_id = None
            self._resolve_pending_capture(accepted=False)

            if tracker is not None:
                tracker.invalidate()

            self.get_logger().error(
                f'Failed to submit teleop trajectory: {exc}'
            )
            return

        future.add_done_callback(
            lambda completed, request_id=request_id: (
                self._on_goal_response(
                    completed,
                    request_id=request_id,
                )
            )
        )

    def _resolve_pending_capture(
        self,
        *,
        accepted: bool,
    ) -> None:
        """Commit only controller-accepted, non-aborted frames."""
        session = self._teleop_session

        if session is None or session.status != 'recording':
            return

        if not session.has_pending_goal:
            return

        if accepted and not self._aborted:
            session.accept_pending_frame()
        else:
            session.reject_pending_frame()

    def _on_goal_response(
        self,
        future,
        request_id: int | None = None,
    ) -> None:
        tracker = getattr(
            self, '_controller_completion_tracker', None
        )
        if (
            request_id is not None
            and request_id != getattr(
                self, '_active_request_id', None
            )
        ):
            # This response belongs to an older, canceled request.
            # Never modify the current recording or controller state.
            try:
                stale_handle = future.result()
            except Exception:
                if tracker is not None:
                    tracker.invalidate()
                return

            if tracker is not None and request_id is not None:
                try:
                    tracker.goal_response(
                        request_id,
                        accepted=(
                            stale_handle is not None
                            and bool(stale_handle.accepted)
                        ),
                    )
                except (TypeError, ValueError) as exc:
                    tracker.invalidate()
                    self.get_logger().error(
                        f'Cannot track stale teleop goal: {exc}'
                    )

            if (
                stale_handle is not None
                and stale_handle.accepted
            ):
                try:
                    if (
                        tracker is not None
                        and request_id is not None
                    ):
                        tracker.cancellation_requested(
                            request_id
                        )

                        result_future = (
                            stale_handle.get_result_async()
                        )
                        result_future.add_done_callback(
                            lambda completed, request_id=request_id: (
                                self._on_goal_result(
                                    completed,
                                    request_id=request_id,
                                )
                            )
                        )

                    cancel_future = (
                        stale_handle.cancel_goal_async()
                    )

                    if (
                        tracker is not None
                        and request_id is not None
                    ):
                        cancel_future.add_done_callback(
                            lambda completed, request_id=request_id: (
                                self._on_cancel_response(
                                    completed,
                                    request_id=request_id,
                                )
                            )
                        )
                except Exception as exc:
                    if tracker is not None:
                        tracker.invalidate()
                    self.get_logger().error(
                        f'Failed to cancel stale teleop goal: {exc}'
                    )
            return

        try:
            goal_handle = future.result()
        except Exception as exc:
            if tracker is not None:
                tracker.invalidate()

            self._goal_pending = False
            self._active_request_id = None
            self._resolve_pending_capture(accepted=False)
            self.get_logger().error(
                f'Failed to send teleop trajectory: {exc}'
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            if tracker is not None and request_id is not None:
                try:
                    tracker.goal_response(
                        request_id, accepted=False
                    )
                except (TypeError, ValueError) as exc:
                    tracker.invalidate()
                    self.get_logger().error(
                        f'Cannot track rejected teleop goal: {exc}'
                    )

            self._goal_pending = False
            self._active_request_id = None
            self._resolve_pending_capture(accepted=False)
            self.get_logger().warning(
                'Teleop trajectory goal was rejected.'
            )
            return

        if tracker is not None and request_id is not None:
            try:
                tracker.goal_response(
                    request_id, accepted=True
                )
            except (TypeError, ValueError) as exc:
                tracker.invalidate()
                self.get_logger().error(
                    f'Cannot track accepted teleop goal: {exc}'
                )

        session = self._teleop_session

        capture_finalized = (
            session is not None
            and session.status != 'recording'
        )

        if (
            not self._goal_pending
            or self._aborted
            or capture_finalized
        ):
            # The controller accepted a goal after the operator
            # canceled or finalized this recording.
            # Never turn this late response into a training frame.
            self._resolve_pending_capture(accepted=False)

            try:
                if (
                    tracker is not None
                    and request_id is not None
                ):
                    tracker.cancellation_requested(
                        request_id
                    )

                    result_future = goal_handle.get_result_async()
                    result_future.add_done_callback(
                        lambda completed, request_id=request_id: (
                            self._on_goal_result(
                                completed,
                                request_id=request_id,
                            )
                        )
                    )

                cancel_future = goal_handle.cancel_goal_async()

                if (
                    tracker is not None
                    and request_id is not None
                ):
                    cancel_future.add_done_callback(
                        lambda completed, request_id=request_id: (
                            self._on_cancel_response(
                                completed,
                                request_id=request_id,
                            )
                        )
                    )
            except Exception as exc:
                if tracker is not None:
                    tracker.invalidate()
                self.get_logger().error(
                    f'Failed to cancel late teleop goal: {exc}'
                )

            self._goal_handle = None
            self._goal_pending = False
            self._active_request_id = None
            return

        self._goal_handle = goal_handle
        self._resolve_pending_capture(accepted=True)

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, request_id=request_id: (
                self._on_goal_result(
                    completed,
                    request_id=request_id,
                )
            )
        )

    def _on_goal_result(
        self,
        future,
        request_id: int | None = None,
    ) -> None:
        tracker = getattr(
            self, '_controller_completion_tracker', None
        )

        if tracker is not None and request_id is not None:
            try:
                response = future.result()
            except Exception as exc:
                tracker.invalidate()
                self.get_logger().error(
                    f'Teleop trajectory result failed: {exc}'
                )
            else:
                terminal_statuses = (
                    GoalStatus.STATUS_SUCCEEDED,
                    GoalStatus.STATUS_CANCELED,
                    GoalStatus.STATUS_ABORTED,
                )
                confirmed = (
                    getattr(response, 'status', None)
                    in terminal_statuses
                )

                try:
                    tracker.terminal_result(
                        request_id,
                        confirmed=confirmed,
                    )
                except (TypeError, ValueError) as exc:
                    tracker.invalidate()
                    self.get_logger().error(
                        f'Cannot track teleop terminal result: {exc}'
                    )

        if (
            request_id is not None
            and request_id != getattr(
                self, '_active_request_id', None
            )
        ):
            return

        # Existing callbacks without a tracker keep their behavior.
        if tracker is None or request_id is None:
            try:
                future.result()
            except Exception as exc:
                self.get_logger().error(
                    f'Teleop trajectory result failed: {exc}'
                )

        self._goal_handle = None
        self._goal_pending = False
        self._active_request_id = None

    def _on_cancel_response(
        self,
        future,
        request_id: int,
    ) -> None:
        """Track a cancellation reply without assuming termination."""
        tracker = getattr(
            self, '_controller_completion_tracker', None
        )

        if tracker is None:
            return

        try:
            response = future.result()
        except Exception as exc:
            tracker.invalidate()
            self.get_logger().error(
                f'Teleop cancellation response failed: {exc}'
            )
            return

        return_code = getattr(response, 'return_code', None)
        goals_canceling = getattr(
            response, 'goals_canceling', None
        )

        if (
            not isinstance(return_code, int)
            or not isinstance(goals_canceling, (list, tuple))
        ):
            tracker.invalidate()
            self.get_logger().error(
                'Malformed teleop cancellation response.'
            )
            return

        acknowledged = (
            return_code == 0
            and bool(goals_canceling)
        )

        try:
            tracker.cancellation_response(
                request_id,
                acknowledged=acknowledged,
            )
        except (TypeError, ValueError) as exc:
            tracker.invalidate()
            self.get_logger().error(
                f'Cannot track teleop cancellation: {exc}'
            )

    def _cancel_active_goal(self) -> None:
        """Cancel control and preserve outstanding goal accounting."""
        # A trajectory submission may still await goal acceptance.
        # Discard its unaccepted frame without losing its request ID.
        self._resolve_pending_capture(accepted=False)

        tracker = getattr(
            self, '_controller_completion_tracker', None
        )
        request_id = getattr(
            self, '_active_request_id', None
        )
        goal_handle = self._goal_handle

        if tracker is not None:
            if request_id is None:
                if goal_handle is not None or self._goal_pending:
                    tracker.invalidate()
            else:
                try:
                    tracker.cancellation_requested(request_id)
                except (TypeError, ValueError) as exc:
                    tracker.invalidate()
                    self.get_logger().error(
                        f'Cannot track cancel request: {exc}'
                    )

        # Preserve the existing stop behavior. Clearing these fields
        # does not remove the outstanding request from the tracker.
        self._goal_handle = None
        self._goal_pending = False
        self._active_request_id = None

        if goal_handle is None:
            return

        try:
            cancel_future = goal_handle.cancel_goal_async()

            if tracker is not None and request_id is not None:
                cancel_future.add_done_callback(
                    lambda completed, request_id=request_id: (
                        self._on_cancel_response(
                            completed,
                            request_id=request_id,
                        )
                    )
                )
        except Exception as exc:
            if tracker is not None:
                tracker.invalidate()

            self.get_logger().error(
                f'Failed to cancel teleop trajectory: {exc}'
            )

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

        session = getattr(self, '_teleop_session', None)
        config = getattr(self, '_teleop_collection_config', None)

        if session is not None and session.status == 'recording':
            session.abort()
            if config is not None:
                try:
                    self._record_collection_outcome('aborted')
                except (OSError, ValueError, TypeError) as exc:
                    # Reporting must never prevent operator abort.
                    self.get_logger().error(
                        f'Abort summary failed: {exc}'
                    )

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
