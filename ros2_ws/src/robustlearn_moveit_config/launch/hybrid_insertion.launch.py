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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(
        get_package_share_directory(
            'robustlearn_moveit_config'
        )
    )

    model_path_argument = DeclareLaunchArgument(
        'model_path',
        description=(
            'Absolute path to the project-owned MuJoCo MJCF '
            'entry-point model.'
        ),
    )

    use_rviz_argument = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Start RViz for the hybrid MoveIt workflow.',
    )

    moveit_mujoco = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                package_share
                / 'launch'
                / 'moveit_mujoco.launch.py'
            )
        ),
        launch_arguments={
            'model_path': LaunchConfiguration('model_path'),
            'initial_keyframe': 'week9_approach',
            'use_rviz': LaunchConfiguration('use_rviz'),
        }.items(),
    )

    local_expert = Node(
        package='robustlearn_control',
        executable='local_insertion_expert.py',
        name='local_insertion_expert',
        output='screen',
        prefix='uv run',
        parameters=[
            {
                'command_enabled': True,
                'start_armed': False,
                'control_period_s': 0.02,
                'trajectory_duration_s': 0.10,
                'goal_timeout_s': 1.0,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'info',
            '--log-level',
            'local_insertion_expert:=debug',
        ],
    )

    coordinator = Node(
        package='robustlearn_moveit_config',
        executable='hybrid_insertion_coordinator.py',
        name='hybrid_insertion_coordinator',
        output='screen',
        parameters=[
            {
                'handoff_tolerance_rad': 0.002,
                'moveit_ready_timeout_s': 10.0,
                'planning_timeout_s': 15.0,
                'execution_timeout_s': 20.0,
                'handoff_settle_s': 0.5,
            }
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            'info',
            '--log-level',
            'hybrid_insertion_coordinator:=debug',
        ],
    )

    return LaunchDescription(
        [
            model_path_argument,
            use_rviz_argument,
            moveit_mujoco,
            local_expert,
            coordinator,
        ]
    )
