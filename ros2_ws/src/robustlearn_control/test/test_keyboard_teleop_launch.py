import math
import threading
import time
import unittest

from control_msgs.action import FollowJointTrajectory
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
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
    0.013358,
    -0.000005,
    -1.583575,
    -0.000669,
    1.568475,
    -0.785300,
]


@pytest.mark.launch_test
def generate_test_description():
    keyboard_teleop = launch_ros.actions.Node(
        package='robustlearn_control',
        executable='keyboard_teleop.py',
        name='keyboard_teleop',
        output='screen',
        prefix='uv run',
        parameters=[
            {
                'use_terminal_input': False,
                'input_topic': '/keyboard_teleop/test_input',
                'joint_step_rad': 0.01,
                'deadman_timeout_s': 0.15,
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


class TestKeyboardTeleop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.node = rclpy.create_node(
            'keyboard_teleop_launch_test'
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

        cls.input_publisher = cls.node.create_publisher(
            String,
            '/keyboard_teleop/test_input',
            10,
        )

        cls.state_lock = threading.Lock()
        cls.received_goals = []
        cls.cancel_count = 0

        cls.action_server = ActionServer(
            cls.node,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
            execute_callback=cls.execute_goal,
            goal_callback=cls.goal_callback,
            cancel_callback=cls.cancel_callback,
        )

    @classmethod
    def tearDownClass(cls):
        cls.action_server.destroy()

        cls.executor.shutdown()

        cls.executor_thread.join(
            timeout=2.0
        )

        cls.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

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
                goal_handle.canceled()

                result = FollowJointTrajectory.Result()
                result.error_code = (
                    FollowJointTrajectory.Result.SUCCESSFUL
                )
                return result

            time.sleep(0.01)

        goal_handle.succeed()

        result = FollowJointTrajectory.Result()
        result.error_code = (
            FollowJointTrajectory.Result.SUCCESSFUL
        )
        return result

    @classmethod
    def publish_joint_state(cls):
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

    def test_bounded_goal_and_deadman_stop(
        self,
    ):
        self.assertTrue(
            self.wait_for_connections(
                timeout_sec=5.0,
            ),
            'Teleop ROS connections were not discovered.',
        )

        for _ in range(20):
            self.publish_joint_state()
            time.sleep(0.02)

        self.publish_key(' ')
        time.sleep(0.03)
        self.publish_key('q')

        self.assertTrue(
            self.wait_for_goal_count(
                expected=1,
                timeout_sec=2.0,
            ),
            'Teleop did not send a trajectory goal.',
        )

        with self.state_lock:
            first_goal = list(
                self.received_goals[0]
            )

        self.assertEqual(
            len(first_goal),
            7,
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in first_goal
            )
        )

        for index in range(7):
            delta = (
                first_goal[index]
                - INITIAL_POSITIONS[index]
            )

            self.assertLessEqual(
                abs(delta),
                0.01 + 1e-12,
            )

        self.assertAlmostEqual(
            first_goal[0],
            INITIAL_POSITIONS[0] + 0.01,
            places=9,
        )

        for index in range(1, 7):
            self.assertAlmostEqual(
                first_goal[index],
                INITIAL_POSITIONS[index],
                places=9,
            )

        self.assertTrue(
            self.wait_for_cancel(
                timeout_sec=2.0,
            ),
            'Deadman expiry did not cancel the active trajectory.',
        )

        with self.state_lock:
            goal_count_after_deadman = len(
                self.received_goals
            )

        self.publish_key('q')
        time.sleep(0.30)

        with self.state_lock:
            final_goal_count = len(
                self.received_goals
            )

        self.assertEqual(
            final_goal_count,
            goal_count_after_deadman,
            (
                'Teleop progressed after deadman expiration '
                'without a new enable refresh.'
            ),
        )
