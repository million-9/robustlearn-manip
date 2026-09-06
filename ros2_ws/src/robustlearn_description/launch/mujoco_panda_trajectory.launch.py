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

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(
        get_package_share_directory('robustlearn_description')
    )

    panda_xacro = package_share / 'urdf' / 'panda.urdf.xacro'
    controllers_yaml = (
        package_share / 'config' / 'mujoco_trajectory_controllers.yaml'
    )

    model_path_argument = DeclareLaunchArgument(
        'model_path',
        description=(
            'Absolute path to the project-owned MuJoCo MJCF '
            'entry-point model.'
        ),
    )

    model_path = LaunchConfiguration('model_path')

    robot_description = {
        'robot_description': Command(
            [
                'xacro ',
                str(panda_xacro),
                ' hardware_plugin:=robustlearn_mujoco_hardware/'
                'MuJoCoSystem',
                ' model_path:=',
                model_path,
            ]
        )
    }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[str(controllers_yaml)],
        remappings=[
            ('~/robot_description', '/robot_description'),
        ],
        output='screen',
    )

    controllers_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'panda_arm_controller',
            '--controller-manager',
            '/controller_manager',
            '--activate-as-group',
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            model_path_argument,
            robot_state_publisher,
            controller_manager,
            controllers_spawner,
        ]
    )
