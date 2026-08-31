import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from std_msgs.msg import String


@pytest.mark.launch_test
def generate_test_description():
    status_publisher = launch_ros.actions.Node(
        package='robustlearn_control',
        executable='status_publisher',
        name='status_publisher',
        output='screen',
    )

    return launch.LaunchDescription(
        [
            status_publisher,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestStatusPublisher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('status_publisher_test')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_status_message_is_published(self):
        received_messages = []

        def callback(msg):
            received_messages.append(msg.data)

        subscription = self.node.create_subscription(
            String,
            '/system_status',
            callback,
            10,
        )

        timeout = time.monotonic() + 5.0

        while time.monotonic() < timeout and not received_messages:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

        self.assertTrue(
            received_messages,
            'No message received on /system_status',
        )

        self.node.destroy_subscription(subscription)
