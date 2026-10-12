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
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
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


CANONICAL_PRE_INSERTION_JOINT_POSITIONS = [
    0.0,
    0.013358,
    -0.000005,
    -1.583575,
    -0.000669,
    1.568475,
    -0.785300,
]


EXPECTED_FAILURE_STATUS = (
    'failed|trajectory goal exceeded 0.250 s timeout'
)


@pytest.mark.launch_test
def generate_test_description():
    local_expert = launch_ros.actions.Node(
        package='robustlearn_control',
        executable='local_insertion_expert.py',
        name='local_insertion_expert',
        output='screen',
        prefix='uv run',
        parameters=[
            {
                'command_enabled': True,
                'control_period_s': 0.02,
                'trajectory_duration_s': 0.10,
                'goal_timeout_s': 0.25,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'warn',
            '--log-level',
            'local_insertion_expert:=debug',
        ],
    )

    return (
        launch.LaunchDescription(
            [
                local_expert,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'local_expert': local_expert,
        },
    )


class TestLocalInsertionExpertTimeout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.node = rclpy.create_node(
            'local_insertion_expert_timeout_test'
        )

        cls.executor = MultiThreadedExecutor(
            num_threads=2
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

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        cls.status_lock = threading.Lock()
        cls.status_history = []

        cls.status_subscription = cls.node.create_subscription(
            String,
            '/local_insertion_expert/status',
            cls.status_callback,
            status_qos,
        )

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

        cls.node.destroy_subscription(
            cls.status_subscription
        )

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

    @staticmethod
    def cancel_callback(_goal_handle):
        return CancelResponse.ACCEPT

    @classmethod
    def status_callback(
        cls,
        message: String,
    ):
        with cls.status_lock:
            cls.status_history.append(
                message.data
            )

    @staticmethod
    def execute_goal(goal_handle):
        deadline = time.monotonic() + 2.0

        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()

                result = FollowJointTrajectory.Result()
                result.error_code = (
                    FollowJointTrajectory.Result.SUCCESSFUL
                )

                return result

            time.sleep(0.02)

        goal_handle.succeed()

        result = FollowJointTrajectory.Result()
        result.error_code = (
            FollowJointTrajectory.Result.SUCCESSFUL
        )

        return result

    @classmethod
    def wait_for_feedback_subscribers(
        cls,
        timeout_sec: float,
    ) -> bool:
        deadline = (
            time.monotonic()
            + timeout_sec
        )

        while time.monotonic() < deadline:
            joint_state_ready = (
                cls.joint_state_publisher.get_subscription_count()
                > 0
            )

            wrench_ready = (
                cls.wrench_publisher.get_subscription_count()
                > 0
            )

            controller_state_ready = (
                cls.controller_state_publisher.get_subscription_count()
                > 0
            )

            if (
                joint_state_ready
                and wrench_ready
                and controller_state_ready
            ):
                return True

            time.sleep(0.02)

        return (
            cls.joint_state_publisher.get_subscription_count() > 0
            and cls.wrench_publisher.get_subscription_count() > 0
            and (
                cls.controller_state_publisher.get_subscription_count()
                > 0
            )
        )

    @classmethod
    def publish_valid_feedback(cls):
        now = cls.node.get_clock().now().to_msg()

        joint_state = JointState()
        joint_state.header.stamp = now
        joint_state.name = list(
            PANDA_JOINT_NAMES
        )

        joint_state.position = list(
            CANONICAL_PRE_INSERTION_JOINT_POSITIONS
        )

        joint_state.velocity = [
            0.0
            for _ in PANDA_JOINT_NAMES
        ]

        cls.joint_state_publisher.publish(
            joint_state
        )

        wrench = WrenchStamped()
        wrench.header.stamp = now
        wrench.header.frame_id = 'panda_wrist_ft'

        cls.wrench_publisher.publish(
            wrench
        )

        controller_state = JointTrajectoryControllerState()
        controller_state.header.stamp = now
        controller_state.joint_names = list(
            PANDA_JOINT_NAMES
        )

        controller_state.output.positions = list(
            CANONICAL_PRE_INSERTION_JOINT_POSITIONS
        )

        cls.controller_state_publisher.publish(
            controller_state
        )

    @classmethod
    def wait_for_status(
        cls,
        expected_status: str,
        timeout_sec: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            with cls.status_lock:
                if expected_status in cls.status_history:
                    return True

            time.sleep(0.01)

        with cls.status_lock:
            return expected_status in cls.status_history

    def test_trajectory_goal_timeout_aborts(
        self,
        proc_output,
        local_expert,
    ):
        self.assertTrue(
            self.wait_for_feedback_subscribers(
                timeout_sec=10.0,
            ),
            (
                'Local insertion expert feedback subscriptions were not '
                'discovered before the test deadline; '
                'joint_state_subscribers='
                f'{self.joint_state_publisher.get_subscription_count()}, '
                'wrench_subscribers='
                f'{self.wrench_publisher.get_subscription_count()}, '
                'controller_state_subscribers='
                f'{self.controller_state_publisher.get_subscription_count()}'
            ),
        )

        publish_deadline = (
            time.monotonic()
            + 2.0
        )

        while time.monotonic() < publish_deadline:
            self.publish_valid_feedback()
            time.sleep(0.02)

        proc_output.assertWaitFor(
            'Local insertion expert aborted: '
            'trajectory goal exceeded 0.250 s timeout',
            process=local_expert,
            timeout=5.0,
        )

        self.assertTrue(
            self.wait_for_status(
                EXPECTED_FAILURE_STATUS,
                timeout_sec=2.0,
            ),
            (
                'Local insertion expert did not publish terminal failure '
                f'status; history={self.status_history}'
            ),
        )

        with self.status_lock:
            statuses = list(
                self.status_history
            )

        self.assertIn(
            'running',
            statuses,
            (
                'Expected running status before timeout failure; '
                f'history={statuses}'
            ),
        )

        self.assertIn(
            EXPECTED_FAILURE_STATUS,
            statuses,
            (
                'Expected terminal timeout failure status; '
                f'history={statuses}'
            ),
        )
