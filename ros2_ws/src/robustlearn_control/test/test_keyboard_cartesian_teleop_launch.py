import math
import threading
import time
import unittest

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import WrenchStamped
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.action import (
    ActionServer,
    CancelResponse,
    GoalResponse,
)
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String

PANDA_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]


INITIAL_POSITIONS = [
    0.0,
    0.02,
    0.0,
    -1.50,
    0.0,
    1.51,
    -0.78,
]


@pytest.mark.launch_test
def generate_test_description():
    keyboard_teleop = launch_ros.actions.Node(
        package='robustlearn_control',
        executable='keyboard_teleop.py',
        name='keyboard_cartesian_teleop',
        output='screen',
        prefix='uv run',
        parameters=[
            {
                'use_terminal_input': False,
                'input_topic': '/keyboard_teleop/cartesian_test_input',
                'joint_step_rad': 0.01,
                'deadman_timeout_s': 0.45,
                'teleop_mode': 'cartesian',
                'cartesian_step_m': 0.001,
                'cartesian_feedback_timeout_s': 0.25,
                'min_command_period_s': 0.01,
                'trajectory_duration_s': 0.10,
                'poll_period_s': 0.01,
                'operator_status_period_s': 10.0,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'warn',
        ],
    )

    return (
        launch.LaunchDescription(
            [
                keyboard_teleop,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'keyboard_teleop': keyboard_teleop,
        },
    )


class TestKeyboardCartesianTeleop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.node = rclpy.create_node(
            'keyboard_cartesian_teleop_launch_test'
        )

        cls.executor = MultiThreadedExecutor(
            num_threads=3
        )
        cls.executor.add_node(
            cls.node
        )

        cls.executor_thread = threading.Thread(
            target=cls.executor.spin,
            daemon=True,
        )
        cls.executor_thread.start()

        cls.joint_state_publisher = cls.node.create_publisher(
            JointState,
            '/joint_states',
            10,
        )

        cls.wrench_publisher = cls.node.create_publisher(
            WrenchStamped,
            '/wrist_ft/wrench',
            10,
        )

        cls.controller_state_publisher = cls.node.create_publisher(
            JointTrajectoryControllerState,
            '/panda_arm_controller/controller_state',
            10,
        )

        cls.input_publisher = cls.node.create_publisher(
            String,
            '/keyboard_teleop/cartesian_test_input',
            10,
        )

        cls.state_lock = threading.Lock()
        cls.received_goals = []
        cls.cancel_count = 0
        cls.completed_goal_count = 0
        cls.cancel_release = threading.Event()
        cls.cancel_release.set()
        cls.cancel_blocked = threading.Event()
        cls.interpolation_mode = 'enabled'
        cls.parameter_service = cls.node.create_service(
            GetParameters,
            '/panda_arm_controller/get_parameters',
            cls.on_interpolation_parameter_request,
        )

        cls.action_server = ActionServer(
            cls.node,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
            execute_callback=cls.execute_goal,
            goal_callback=cls.goal_callback,
            cancel_callback=cls.cancel_callback,
        )

    def setUp(self):
        with self.state_lock:
            type(self).interpolation_mode = 'enabled'

    def tearDown(self):
        """Cancel and settle mock goals before the next test."""
        # A delayed cancellation must never hold another test.
        self.cancel_release.set()

        # Stop any goal left active by the preceding test.
        self.publish_key('x')

        # Allow the ROS input subscription to process the stop.
        time.sleep(0.10)

        deadline = time.monotonic() + 4.0

        while time.monotonic() < deadline:
            with self.state_lock:
                submitted = len(self.received_goals)
                completed = self.completed_goal_count

            if completed >= submitted:
                # Allow the client's asynchronous action-result
                # callback to process the mock terminal result.
                time.sleep(0.10)
                return

            time.sleep(0.02)

        self.fail(
            'Mock controller did not settle before the next test: '
            f'submitted={submitted}, completed={completed}'
        )

    @classmethod
    def tearDownClass(cls):
        cls.parameter_service.destroy()
        cls.action_server.destroy()

        cls.executor.shutdown()

        cls.executor_thread.join(
            timeout=2.0
        )

        cls.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    @classmethod
    def on_interpolation_parameter_request(cls, request, response):
        """Return the configured simulated controller parameter."""
        with cls.state_lock:
            mode = cls.interpolation_mode

        if request.names != ['interpolate_from_desired_state']:
            return response

        if mode == 'missing':
            response.values = []
            return response

        value = ParameterValue()

        if mode == 'malformed':
            value.type = ParameterType.PARAMETER_STRING
            value.string_value = 'true'
        else:
            value.type = ParameterType.PARAMETER_BOOL
            value.bool_value = mode == 'enabled'

        response.values = [value]
        return response

    @staticmethod
    def goal_callback(_goal_request):
        return GoalResponse.ACCEPT

    @classmethod
    def cancel_callback(cls, _goal_handle):
        with cls.state_lock:
            cls.cancel_count += 1

        return CancelResponse.ACCEPT

    @classmethod
    def execute_goal(cls, goal_handle):
        trajectory = goal_handle.request.trajectory

        if not trajectory.points:
            goal_handle.abort()

            result = FollowJointTrajectory.Result()
            result.error_code = (
                FollowJointTrajectory.Result.INVALID_GOAL
            )
            return result

        target = list(
            trajectory.points[-1].positions
        )

        with cls.state_lock:
            cls.received_goals.append(
                target
            )

        deadline = time.monotonic() + 2.0

        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                cls.cancel_blocked.set()
                cls.cancel_release.wait(timeout=4.0)
                goal_handle.canceled()

                with cls.state_lock:
                    cls.completed_goal_count += 1

                result = FollowJointTrajectory.Result()
                result.error_code = (
                    FollowJointTrajectory.Result.SUCCESSFUL
                )
                return result

            time.sleep(0.01)

        goal_handle.succeed()

        with cls.state_lock:
            cls.completed_goal_count += 1

        result = FollowJointTrajectory.Result()
        result.error_code = (
            FollowJointTrajectory.Result.SUCCESSFUL
        )
        return result

    @classmethod
    def publish_joint_state(cls, reference_offset=None):
        message = JointState()
        message.header.stamp = (
            cls.node.get_clock().now().to_msg()
        )
        message.name = list(
            PANDA_JOINT_NAMES
        )
        message.position = list(
            INITIAL_POSITIONS
        )
        message.velocity = [
            0.0
            for _ in PANDA_JOINT_NAMES
        ]

        cls.joint_state_publisher.publish(
            message
        )

        if reference_offset is None:
            reference_offset = (0.0,) * 7

        controller_state = JointTrajectoryControllerState()
        controller_state.joint_names = list(PANDA_JOINT_NAMES)
        controller_state.reference.positions = [
            measured + offset
            for measured, offset in zip(
                INITIAL_POSITIONS,
                reference_offset,
                strict=True,
            )
        ]

        cls.controller_state_publisher.publish(controller_state)
        cls.publish_wrench((0.0, 0.0, 0.0))

    @classmethod
    def publish_wrench(
        cls,
        force_xyz,
        torque_xyz=(0.0, 0.0, 0.0),
    ):
        message = WrenchStamped()
        message.header.stamp = (
            cls.node.get_clock().now().to_msg()
        )
        message.header.frame_id = 'panda_wrist_ft'

        message.wrench.force.x = float(force_xyz[0])
        message.wrench.force.y = float(force_xyz[1])
        message.wrench.force.z = float(force_xyz[2])

        message.wrench.torque.x = float(torque_xyz[0])
        message.wrench.torque.y = float(torque_xyz[1])
        message.wrench.torque.z = float(torque_xyz[2])

        cls.wrench_publisher.publish(message)

    @classmethod
    def publish_key(cls, key):
        message = String()
        message.data = key

        cls.input_publisher.publish(
            message
        )

    @classmethod
    def wait_for_connections(
        cls,
        timeout_sec,
    ):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            if (
                cls.joint_state_publisher.get_subscription_count() > 0
                and cls.input_publisher.get_subscription_count() > 0
            ):
                return True

            time.sleep(0.02)

        return (
            cls.joint_state_publisher.get_subscription_count() > 0
            and cls.input_publisher.get_subscription_count() > 0
        )

    @classmethod
    def wait_for_goal_count(
        cls,
        expected,
        timeout_sec,
    ):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            with cls.state_lock:
                if len(cls.received_goals) >= expected:
                    return True

            time.sleep(0.01)

        with cls.state_lock:
            return len(cls.received_goals) >= expected

    @classmethod
    def wait_for_cancel(
        cls,
        timeout_sec,
    ):
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            with cls.state_lock:
                if cls.cancel_count > 0:
                    return True

            time.sleep(0.01)

        with cls.state_lock:
            return cls.cancel_count > 0

    def test_xz_interpolation_revocation_cancels_active_goal(self):
        """Revoke verified interpolation during an active trajectory."""
        self.assertTrue(
            self.wait_for_connections(timeout_sec=5.0)
        )

        with self.state_lock:
            initial_goals = len(self.received_goals)
            initial_cancels = self.cancel_count

        # Establish fresh feedback and verified true interpolation.
        for _ in range(30):
            self.publish_joint_state()
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')

        # Keep deadman and feedback alive while awaiting acceptance.
        deadline = time.monotonic() + 0.7
        accepted = False

        while time.monotonic() < deadline:
            self.publish_joint_state()
            self.publish_key(' ')

            with self.state_lock:
                accepted = len(self.received_goals) > initial_goals

            if accepted:
                break

            time.sleep(0.02)

        self.assertTrue(
            accepted,
            'Could not establish an active Cartesian trajectory.'
        )

        with self.state_lock:
            self.assertEqual(
                self.cancel_count,
                initial_cancels,
                'Goal canceled before interpolation was revoked.'
            )
            type(self).interpolation_mode = 'disabled'

        # Refresh deadman and joint/wrench/reference feedback.
        # Cancellation must therefore result from revocation.
        deadline = time.monotonic() + 0.35
        canceled = False

        while time.monotonic() < deadline:
            self.publish_joint_state()
            self.publish_key(' ')

            with self.state_lock:
                canceled = self.cancel_count > initial_cancels

            if canceled:
                break

            time.sleep(0.02)

        self.assertTrue(
            canceled,
            'Revoking interpolation did not cancel the active goal.'
        )

    def _assert_unverified_interpolation_blocks_jog(self, mode):
        self.assertTrue(
            self.wait_for_connections(timeout_sec=5.0)
        )

        with self.state_lock:
            baseline = len(self.received_goals)
            type(self).interpolation_mode = mode

        # Maintain normal joint, reference, and wrench feedback.
        # Allow old positive verification to expire.
        for _ in range(32):
            self.publish_joint_state()
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')
        time.sleep(0.18)

        with self.state_lock:
            observed = len(self.received_goals)

        self.assertEqual(
            observed,
            baseline,
            f'Cartesian goal accepted with interpolation={mode}',
        )

    def test_x_disabled_interpolation_blocks_jog(self):
        self._assert_unverified_interpolation_blocks_jog('disabled')

    def test_xx_malformed_interpolation_blocks_jog(self):
        self._assert_unverified_interpolation_blocks_jog('malformed')

    def test_xy_missing_interpolation_blocks_jog(self):
        self._assert_unverified_interpolation_blocks_jog('missing')

    def test_cartesian_goal_deadman_and_stale_feedback(self):
        self.assertTrue(
            self.wait_for_connections(timeout_sec=5.0),
            'Cartesian teleop ROS connections unavailable.',
        )

        # Provide measured, stationary joint feedback.
        for _ in range(20):
            self.publish_joint_state()
            time.sleep(0.02)

        # A jog without deadman enable must do nothing.
        self.publish_key('w')
        time.sleep(0.15)

        with self.state_lock:
            self.assertEqual(len(self.received_goals), 0)

        # Exercise reference-relative control with a nonzero
        # measured-versus-commanded offset in panda_joint2.
        for _ in range(10):
            self.publish_joint_state(
                reference_offset=(
                    0.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0
                )
            )
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')

        self.assertTrue(
            self.wait_for_goal_count(
                expected=1,
                timeout_sec=3.0,
            ),
            'Cartesian jog produced no trajectory goal.',
        )

        with self.state_lock:
            target = list(self.received_goals[0])

        initial = list(INITIAL_POSITIONS)

        self.assertEqual(len(target), 7)

        # Frozen regression fixture: +1 mm X using the
        # measured Panda configuration and a +0.002 rad
        # controller-reference offset on panda_joint2.
        # ROS launch_testing uses system Python, so this
        # assertion must not import uv-only project modules.
        expected = [
            -1.2870052323161666e-18,
            0.025360392221669546,
            -1.2619822855630713e-18,
            -1.4958886143710941,
            2.1291848217318154e-20,
            1.509255725275479,
            -0.78000000000000003,
        ]

        for actual_joint, expected_joint in zip(
            target, expected, strict=True
        ):
            self.assertAlmostEqual(
                actual_joint,
                float(expected_joint),
                delta=1e-5,
            )

        self.assertTrue(
            all(math.isfinite(value) for value in target)
        )

        joint_deltas = [
            commanded - measured
            for commanded, measured in zip(
                target, initial, strict=True
            )
        ]

        # The ROS interface must submit a real, bounded
        # seven-joint target instead of a no-op or unsafe step.
        self.assertGreater(
            max(abs(delta) for delta in joint_deltas),
            1e-7,
        )

        self.assertLessEqual(
            max(abs(delta) for delta in joint_deltas),
            0.02000001,
        )

        # Peg-tip XYZ accuracy is validated separately by
        # test_cartesian_teleop_model.py using real MuJoCo FK.

        # The mock action deliberately stays active until
        # cancelled or its execution timeout is reached.
        self.assertTrue(
            self.wait_for_cancel(timeout_sec=3.0),
            'Deadman expiry failed to cancel the goal.',
        )

        with self.state_lock:
            goal_count = len(self.received_goals)

        # No new deadman refresh: reject additional motion.
        self.publish_key('w')
        time.sleep(0.25)

        with self.state_lock:
            self.assertEqual(
                len(self.received_goals),
                goal_count,
                'Jog progressed without deadman refresh.',
            )

        # Deadman is refreshed, but measured joint feedback
        # is now stale. This must also block motion.
        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')
        time.sleep(0.25)

        with self.state_lock:
            self.assertEqual(
                len(self.received_goals),
                goal_count,
                'Cartesian jog accepted stale joint feedback.',
            )

    def test_z_delayed_cancellation_blocks_overlapping_goals(self):
        """Require terminal completion before accepting another jog."""
        self.assertTrue(self.wait_for_connections(timeout_sec=5.0))

        with self.state_lock:
            initial_count = len(self.received_goals)

        self.cancel_blocked.clear()
        self.cancel_release.clear()

        try:
            for _ in range(15):
                self.publish_joint_state()
                time.sleep(0.02)

            self.publish_key(' ')
            time.sleep(0.03)
            self.publish_key('w')

            self.assertTrue(
                self.wait_for_goal_count(initial_count + 1, 2.0),
                'Initial Cartesian trajectory was not submitted.',
            )

            self.assertTrue(
                self.cancel_blocked.wait(timeout=2.0),
                'Mock controller did not enter delayed cancellation.',
            )

            # Keep joint feedback fresh while deliberately holding
            # the controller in an unresolved cancellation state.
            for _ in range(10):
                self.publish_joint_state()
                time.sleep(0.02)

            self.publish_key(' ')
            time.sleep(0.03)
            self.publish_key('w')
            time.sleep(0.15)

            with self.state_lock:
                self.assertEqual(
                    len(self.received_goals),
                    initial_count + 1,
                    'New goal overlapped unresolved cancellation.',
                )

        finally:
            # Always allow the mock controller to finish, including
            # when a preceding assertion fails.
            self.cancel_release.set()

        # Once the terminal result is processed, jogging must
        # become available again with fresh feedback and deadman.
        recovered = False

        for _ in range(12):
            self.publish_joint_state()
            self.publish_key(' ')
            time.sleep(0.03)
            self.publish_key('w')

            if self.wait_for_goal_count(initial_count + 2, 0.15):
                recovered = True
                break

        self.assertTrue(
            recovered,
            'Jogging did not recover after terminal cancellation.',
        )

        with self.state_lock:
            self.assertEqual(
                len(self.received_goals),
                initial_count + 2,
            )

    def test_zy_stale_wrench_cancels_active_goal(self):
        """Cancel an active Cartesian goal when wrench data expires."""
        self.assertTrue(
            self.wait_for_connections(timeout_sec=5.0)
        )

        with self.state_lock:
            initial_goals = len(self.received_goals)
            initial_cancels = self.cancel_count

        for _ in range(15):
            self.publish_joint_state()
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')

        self.assertTrue(
            self.wait_for_goal_count(
                initial_goals + 1, 2.0
            ),
            'Fresh wrench did not allow the initial jog.',
        )

        # Stop publishing wrench feedback. It expires
        # before the configured 0.45 s deadman timeout.
        deadline = time.monotonic() + 1.5
        observed_cancel = False

        while time.monotonic() < deadline:
            with self.state_lock:
                observed_cancel = (
                    self.cancel_count > initial_cancels
                )

            if observed_cancel:
                break

            time.sleep(0.01)

        self.assertTrue(
            observed_cancel,
            'Stale wrench failed to cancel active motion.',
        )

        time.sleep(0.15)

    def test_zz_force_limit_cancels_active_goal(self):
        """Cancel motion when the wrist force exceeds 5 N."""
        self.assertTrue(
            self.wait_for_connections(timeout_sec=5.0)
        )

        with self.state_lock:
            initial_goals = len(self.received_goals)
            initial_cancels = self.cancel_count

        for _ in range(15):
            self.publish_joint_state()
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')

        self.assertTrue(
            self.wait_for_goal_count(
                initial_goals + 1, 2.0
            ),
            'Initial safe Cartesian goal was not submitted.',
        )

        self.publish_wrench((6.0, 0.0, 0.0))

        deadline = time.monotonic() + 1.5
        observed_cancel = False

        while time.monotonic() < deadline:
            with self.state_lock:
                observed_cancel = (
                    self.cancel_count > initial_cancels
                )

            if observed_cancel:
                break

            time.sleep(0.01)

        self.assertTrue(
            observed_cancel,
            'Excessive wrist force failed to cancel motion.',
        )

        # Re-arming deadman cannot bypass the unsafe
        # measured force.
        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('w')
        time.sleep(0.15)

        with self.state_lock:
            self.assertEqual(
                len(self.received_goals),
                initial_goals + 1,
                'Unsafe force permitted another goal.',
            )
