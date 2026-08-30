from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [
            FindPackageShare('robustlearn_control'),
            'config',
            'status_publisher.yaml',
        ]
    )

    return LaunchDescription(
        [
            Node(
                package='robustlearn_control',
                executable='status_publisher',
                name='status_publisher',
                parameters=[params_file],
                output='screen',
            ),
            Node(
                package='robustlearn_control',
                executable='status_subscriber',
                name='status_subscriber',
                output='screen',
            ),
        ]
    )
