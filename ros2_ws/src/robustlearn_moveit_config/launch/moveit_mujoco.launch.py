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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
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
        description='Start RViz for interactive MoveIt visualization.',
    )

    description_share = Path(
        get_package_share_directory('robustlearn_description')
    )

    panda_config_share = Path(
        get_package_share_directory(
            'moveit_resources_panda_moveit_config'
        )
    )

    moveit_config = (
        MoveItConfigsBuilder(
            'moveit_resources_panda',
            package_name='moveit_resources_panda_moveit_config',
        )
        .robot_description(
            file_path='config/panda.urdf.xacro'
        )
        .robot_description_semantic(
            file_path='config/panda.srdf'
        )
        .trajectory_execution(
            file_path='config/moveit_controllers.yaml'
        )
        .planning_pipelines(
            pipelines=['ompl']
        )
        .to_moveit_configs()
    )

    mujoco_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                description_share
                / 'launch'
                / 'mujoco_panda_trajectory.launch.py'
            )
        ),
        launch_arguments={
            'model_path': LaunchConfiguration('model_path'),
        }.items(),
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_panda_base',
        arguments=[
            '--x',
            '0.0',
            '--y',
            '0.0',
            '--z',
            '0.0',
            '--roll',
            '0.0',
            '--pitch',
            '0.0',
            '--yaw',
            '0.0',
            '--frame-id',
            'world',
            '--child-frame-id',
            'panda_link0',
        ],
        output='screen',
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        parameters=[
            moveit_config.to_dict(),
            {
                'publish_robot_description_semantic': True,
                'publish_planning_scene': True,
                'publish_geometry_updates': True,
                'publish_state_updates': True,
                'publish_transforms_updates': True,
            },
        ],
        output='screen',
    )

    rviz_config = (
        panda_config_share / 'launch' / 'moveit.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(
            LaunchConfiguration('use_rviz')
        ),
        arguments=[
            '-d',
            str(rviz_config),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            model_path_argument,
            use_rviz_argument,
            mujoco_bringup,
            static_tf,
            move_group,
            rviz,
        ]
    )
