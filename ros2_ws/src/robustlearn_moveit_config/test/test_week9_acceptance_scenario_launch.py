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
# ruff: noqa: I001

import json
import math
import os
import pathlib
import time
import unittest

import ament_index_python.packages
from geometry_msgs.msg import WrenchStamped
import launch
import launch.actions
import launch.launch_description_sources
import launch_testing
import launch_testing.actions
import mujoco
import numpy as np
import pytest
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from robustlearn.sim.control import panda_site_spatial_jacobian
from robustlearn.sim.insertion import load_insertion_model
from robustlearn.week9_acceptance import evaluate_week9_insertion_metrics

from sensor_msgs.msg import JointState
from std_msgs.msg import String


os.environ.setdefault(
    'ROS_DOMAIN_ID',
    '188',
)


PANDA_JOINTS = tuple(f'panda_joint{index}' for index in range(1, 8))


MUJOCO_PANDA_JOINTS = tuple(f'joint{index}' for index in range(1, 8))


PRE_INSERTION_MEASURED_JOINT_POSITIONS = (
    -0.000154117003526,
    0.028393605524688,
    0.000674241873063,
    -1.513153749958756,
    0.000042615824978,
    1.512449322659168,
    -0.784579660416053,
)


EXPECTED_EXPERT_SUCCESS = 'success|canonical insertion task reached success'


DEFAULT_SCENARIO = 'week9_case_01'


def _scenario_name() -> str:
    scenario = os.environ.get(
        'ROBUSTLEARN_WEEK9_SCENARIO',
        DEFAULT_SCENARIO,
    ).strip()

    if not scenario:
        raise RuntimeError('ROBUSTLEARN_WEEK9_SCENARIO must not be empty')

    return scenario


@pytest.mark.launch_test
def generate_test_description():
    package_share = pathlib.Path(
        ament_index_python.packages.get_package_share_directory('robustlearn_moveit_config')
    )

    launch_file = package_share / 'launch' / 'hybrid_insertion.launch.py'

    repository_root = pathlib.Path(__file__).resolve().parents[4]

    model_path = (
        repository_root / 'robot_description' / 'mjcf' / 'insertion' / 'panda_insertion.xml'
    )

    if not model_path.is_file():
        raise RuntimeError(f'Week 9 acceptance model not found: {model_path}')

    scenario = _scenario_name()

    hybrid_insertion = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(str(launch_file)),
        launch_arguments={
            'model_path': str(model_path),
            'initial_keyframe': scenario,
            'use_rviz': 'false',
        }.items(),
    )

    return (
        launch.LaunchDescription(
            [
                hybrid_insertion,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            'scenario': scenario,
        },
    )


class ShadowWeek9Kinematics:
    """Reconstruct Week 9 task geometry from observed Panda joints."""

    def __init__(self) -> None:
        self.model = load_insertion_model()
        self.data = mujoco.MjData(self.model)

        self._joint_qpos_addresses = tuple(
            self._joint_qpos_address(name) for name in MUJOCO_PANDA_JOINTS
        )

        self._peg_tip_site_id = self._site_id('peg_tip')

        self._receptacle_center_site_id = self._site_id('receptacle_center')

        self._insertion_axis_site_id = self._site_id('insertion_axis')

        self._wrist_ft_site_id = self._site_id('panda_wrist_ft')

    def _joint_qpos_address(
        self,
        name: str,
    ) -> int:
        joint_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
        )

        if joint_id < 0:
            raise RuntimeError(f'Missing MuJoCo joint {name!r}')

        return int(self.model.jnt_qposadr[joint_id])

    def _site_id(
        self,
        name: str,
    ) -> int:
        site_id = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
        )

        if site_id < 0:
            raise RuntimeError(f'Missing MuJoCo site {name!r}')

        return site_id

    def evaluate(
        self,
        joint_positions: tuple[float, ...],
        wrist_force_sensor: tuple[float, float, float],
    ):
        positions = np.asarray(
            joint_positions,
            dtype=np.float64,
        )

        force_sensor = np.asarray(
            wrist_force_sensor,
            dtype=np.float64,
        )

        if positions.shape != (7,):
            raise ValueError('joint_positions must have shape (7,)')

        if force_sensor.shape != (3,):
            raise ValueError('wrist_force_sensor must have shape (3,)')

        if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(force_sensor))):
            raise ValueError('Week 9 observed state must be finite')

        for qpos_address, position in zip(
            self._joint_qpos_addresses,
            positions,
            strict=True,
        ):
            self.data.qpos[qpos_address] = position

        self.data.qvel.fill(0.0)

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        peg_tip_position = np.asarray(
            self.data.site_xpos[self._peg_tip_site_id],
            dtype=np.float64,
        ).copy()

        receptacle_center_position = np.asarray(
            self.data.site_xpos[self._receptacle_center_site_id],
            dtype=np.float64,
        ).copy()

        insertion_axis_position = np.asarray(
            self.data.site_xpos[self._insertion_axis_site_id],
            dtype=np.float64,
        ).copy()

        wrist_rotation_world = (
            np.asarray(
                self.data.site_xmat[self._wrist_ft_site_id],
                dtype=np.float64,
            )
            .reshape(
                3,
                3,
            )
            .copy()
        )

        wrist_force_world = wrist_rotation_world @ force_sensor

        # Exercise the same project-owned Jacobian path used by the
        # classical ROS expert and reject non-finite reconstruction.
        peg_tip_jacobian = panda_site_spatial_jacobian(
            self.model,
            self.data,
            'peg_tip',
        )

        arrays = (
            peg_tip_position,
            receptacle_center_position,
            insertion_axis_position,
            wrist_rotation_world,
            wrist_force_world,
            peg_tip_jacobian,
        )

        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError('Week 9 shadow reconstruction became non-finite')

        return evaluate_week9_insertion_metrics(
            peg_tip_position,
            receptacle_center_position,
            insertion_axis_position,
            wrist_force_world,
        )


class TestWeek9AcceptanceScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('week9_acceptance_scenario_test')

        self.kinematics = ShadowWeek9Kinematics()

        self.latest_joint_positions = None
        self.latest_wrist_force_sensor = None

        self.first_joint_positions = None
        self.handoff_joint_positions = None

        self.status_history = []

        self.expert_running = False

        self.latest_success_metrics = None
        self.terminal_metrics = None

        self.peak_axial_force_n = 0.0
        self.peak_lateral_force_n = 0.0

        self.joint_subscription = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        self.wrench_subscription = self.node.create_subscription(
            WrenchStamped,
            '/wrist_ft/wrench',
            self._wrench_callback,
            qos_profile_sensor_data,
        )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.status_subscription = self.node.create_subscription(
            String,
            '/local_insertion_expert/status',
            self._status_callback,
            status_qos,
        )

    def tearDown(self):
        self.node.destroy_subscription(self.joint_subscription)

        self.node.destroy_subscription(self.wrench_subscription)

        self.node.destroy_subscription(self.status_subscription)

        self.node.destroy_node()

    def _arm_positions(
        self,
        message: JointState,
    ):
        if len(message.name) != len(message.position):
            return None

        lookup = dict(
            zip(
                message.name,
                message.position,
                strict=True,
            )
        )

        if not all(joint_name in lookup for joint_name in PANDA_JOINTS):
            return None

        positions = tuple(float(lookup[joint_name]) for joint_name in PANDA_JOINTS)

        if not all(math.isfinite(value) for value in positions):
            return None

        return positions

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        positions = self._arm_positions(message)

        if positions is None:
            return

        self.latest_joint_positions = positions

        if self.first_joint_positions is None:
            self.first_joint_positions = positions

        self._update_force_metrics()

    def _wrench_callback(
        self,
        message: WrenchStamped,
    ) -> None:
        values = (
            float(message.wrench.force.x),
            float(message.wrench.force.y),
            float(message.wrench.force.z),
        )

        if not all(math.isfinite(value) for value in values):
            return

        self.latest_wrist_force_sensor = values

        self._update_force_metrics()

    def _status_callback(
        self,
        message: String,
    ) -> None:
        status = message.data.strip()

        if not status:
            return

        self.status_history.append(status)

        if status == 'disarmed':
            self.expert_running = False
            return

        if status == 'running':
            if self.handoff_joint_positions is None and self.latest_joint_positions is not None:
                self.handoff_joint_positions = tuple(self.latest_joint_positions)

            self.expert_running = True
            return

        if status.startswith('success|'):
            self.terminal_metrics = (
                self.latest_success_metrics
                if self.latest_success_metrics is not None
                else self._current_metrics()
            )

            self.expert_running = False
            return

        if status.startswith('failed|'):
            self.terminal_metrics = self._current_metrics()
            self.expert_running = False

    def _current_metrics(self):
        if self.latest_joint_positions is None or self.latest_wrist_force_sensor is None:
            return None

        return self.kinematics.evaluate(
            self.latest_joint_positions,
            self.latest_wrist_force_sensor,
        )

    def _update_force_metrics(
        self,
    ) -> None:
        metrics = self._current_metrics()

        if metrics is None:
            return

        if not self.expert_running:
            return

        self.peak_axial_force_n = max(
            self.peak_axial_force_n,
            metrics.axial_force_n,
        )

        self.peak_lateral_force_n = max(
            self.peak_lateral_force_n,
            metrics.lateral_force_n,
        )

        if metrics.task_success:
            self.latest_success_metrics = metrics

    def _spin_until(
        self,
        condition,
        timeout_sec: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

            if condition():
                return True

        return bool(condition())

    def _terminal_status_seen(
        self,
    ) -> bool:
        return any(
            status.startswith('success|') or status.startswith('failed|')
            for status in self.status_history
        )

    def test_week9_clean_scenario(
        self,
        proc_output,
        scenario,
    ):
        self.assertTrue(
            self._spin_until(
                lambda: (
                    self.first_joint_positions is not None
                    and self.latest_wrist_force_sensor is not None
                ),
                timeout_sec=20.0,
            ),
            ('Week 9 scenario did not provide finite joint and wrist-force feedback.'),
        )

        initial_distance_from_preinsertion = max(
            abs(actual - expected)
            for actual, expected in zip(
                self.first_joint_positions,
                PRE_INSERTION_MEASURED_JOINT_POSITIONS,
                strict=True,
            )
        )

        self.assertGreater(
            initial_distance_from_preinsertion,
            0.05,
            (
                f'{scenario} did not begin from a meaningful global '
                'approach configuration; '
                f'maximum initial pre-insertion distance='
                f'{initial_distance_from_preinsertion:.6f} rad.'
            ),
        )

        self.assertTrue(
            self._spin_until(
                self._terminal_status_seen,
                timeout_sec=45.0,
            ),
            (f'{scenario} did not reach a terminal expert status; history={self.status_history}'),
        )

        self.assertIn(
            'disarmed',
            self.status_history,
            (
                f'{scenario} did not begin with the local expert disarmed; '
                f'history={self.status_history}'
            ),
        )

        self.assertIn(
            'running',
            self.status_history,
            (f'{scenario} did not enter local expert execution; history={self.status_history}'),
        )

        self.assertIn(
            EXPECTED_EXPERT_SUCCESS,
            self.status_history,
            (f'{scenario} did not complete successfully; history={self.status_history}'),
        )

        self.assertIsNotNone(
            self.handoff_joint_positions,
            (f'{scenario} did not capture a measured handoff state.'),
        )

        handoff_error_rad = max(
            abs(actual - expected)
            for actual, expected in zip(
                self.handoff_joint_positions,
                PRE_INSERTION_MEASURED_JOINT_POSITIONS,
                strict=True,
            )
        )

        self.assertLessEqual(
            handoff_error_rad,
            0.002,
            (
                f'{scenario} handoff error exceeded the Week 9 '
                f'contract: {handoff_error_rad:.6f} rad.'
            ),
        )

        final_metrics = self.terminal_metrics

        self.assertIsNotNone(
            final_metrics,
            (f'{scenario} has no finite terminal metric sample.'),
        )

        self.assertTrue(
            final_metrics.task_success,
            (
                f'{scenario} terminal ROS success did not match '
                'project-owned geometric task success.'
            ),
        )

        self.assertFalse(
            final_metrics.task_failure,
            (f'{scenario} ended in geometric task failure.'),
        )

        scalar_metrics = (
            handoff_error_rad,
            final_metrics.lateral_error_m,
            final_metrics.axial_offset_m,
            final_metrics.insertion_depth_m,
            final_metrics.axial_force_n,
            final_metrics.lateral_force_n,
            final_metrics.force_norm_n,
            self.peak_axial_force_n,
            self.peak_lateral_force_n,
        )

        self.assertTrue(
            all(math.isfinite(value) for value in scalar_metrics),
            (f'{scenario} produced non-finite acceptance metrics.'),
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=planning',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=executing',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion phase=verifying_handoff',
            timeout=5.0,
        )

        proc_output.assertWaitFor(
            'Hybrid insertion completed: '
            'local insertion expert reported success: '
            'canonical insertion task reached success',
            timeout=5.0,
        )

        result = {
            'scenario': scenario,
            'success': True,
            'moveit_planning_success': True,
            'moveit_execution_success': True,
            'handoff_error_rad': handoff_error_rad,
            'local_expert_success': True,
            'final_lateral_error_m': (final_metrics.lateral_error_m),
            'final_axial_offset_m': (final_metrics.axial_offset_m),
            'final_insertion_depth_m': (final_metrics.insertion_depth_m),
            'final_axial_force_n': (final_metrics.axial_force_n),
            'final_lateral_force_n': (final_metrics.lateral_force_n),
            'peak_axial_force_n': (self.peak_axial_force_n),
            'peak_lateral_force_n': (self.peak_lateral_force_n),
            'failure_category': None,
        }

        print(
            'WEEK9_RESULT '
            + json.dumps(
                result,
                sort_keys=True,
            ),
            flush=True,
        )
