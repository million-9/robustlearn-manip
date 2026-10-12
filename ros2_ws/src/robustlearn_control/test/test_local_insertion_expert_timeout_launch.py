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
from sensor_msgs.msg import JointState

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

    @staticmethod
    def cancel_callback(_goal_handle):
        return CancelResponse.ACCEPT

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

    def test_trajectory_goal_timeout_aborts(
        self,
        proc_output,
        local_expert,
    ):
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
