# MuJoCo ros2_control Hardware

This document defines the build-time and runtime contracts for the
`robustlearn_mujoco_hardware` ROS 2 package.

## Plugin

The ros2_control hardware plugin is registered as:

    robustlearn_mujoco_hardware/MuJoCoSystem

Its base class is:

    hardware_interface::SystemInterface

Issue #59 provides the package scaffold, shared library, pluginlib
registration, and plugin discovery/loading test.

Week 7 progressively implements the runtime hardware path:

- Issue #60 initializes the MuJoCo model, lifecycle state, and named Panda
  joint/actuator mappings.
- Issue #61 implements the MuJoCo-to-ros2_control Panda joint-state read path.
- Issue #62 implements the ros2_control-to-MuJoCo Panda position-command path
  and deterministic physics stepping.

## MuJoCo C/C++ build dependency

MuJoCo is an external non-ROS dependency.

The package resolves the MuJoCo C headers and shared library in this order:

1. CMake variable `ROBUSTLEARN_MUJOCO_ROOT`
2. Environment variable `ROBUSTLEARN_MUJOCO_ROOT`
3. The installation directory of an importable Python `mujoco` package

The resolved root must contain:

    include/mujoco/mujoco.h

and a MuJoCo shared library such as:

    libmujoco.so
    libmujoco.so.<version>

The repository does not vendor another MuJoCo SDK or duplicate MuJoCo
Menagerie assets.

The built plugin links to the resolved MuJoCo shared library. Because the
project may obtain MuJoCo from its Python environment rather than a system
library directory, the installed plugin retains the resolved MuJoCo root as
its runtime library search path.

The MuJoCo root is therefore part of the source-build/install contract. If
that installation is moved or replaced after the ROS workspace is built, the
hardware package must be rebuilt against the new MuJoCo location.

## Runtime MJCF model-path contract

The ros2_control hardware configuration uses the hardware parameter:

    model_path

Example:

    <ros2_control name="PandaMuJoCoSystem" type="system">
      <hardware>
        <plugin>robustlearn_mujoco_hardware/MuJoCoSystem</plugin>
        <param name="model_path">/absolute/path/to/model.xml</param>
      </hardware>
    </ros2_control>

The runtime contract is:

- the parameter name is exactly `model_path`;
- its value identifies the MJCF entry-point file used by the hardware plugin;
- the path must resolve to an existing readable MJCF file;
- missing or invalid model paths must cause hardware initialization to fail
  clearly;
- project-owned MJCF assets remain under `robot_description/mjcf`;
- vendored MuJoCo Menagerie files must not be duplicated or modified.

Issue #59 defines this contract. Actual model loading and validation are part
of Issue #60.

## Panda command and physics-stepping contract

The hardware plugin exposes one position command interface for each Panda arm
joint from `panda_joint1` through `panda_joint7`.

During initialization, the plugin uses the MJCF `home` keyframe when one is
available and advances MuJoCo derived state with `mj_forward()`. The internal
position command buffers are initialized from the resulting Panda joint
positions so activation begins from a hold-current-position command instead of
introducing an initial command jump.

Each `write()` cycle follows this contract:

1. Read all seven ROS position commands.
2. Reject the cycle if any command is non-finite.
3. Validate each command against the compiled MuJoCo actuator control range
   when that actuator is control-limited.
4. If any command is invalid, return an error before changing any Panda
   actuator control value or advancing simulation time.
5. If all commands are valid, copy them to the cached named Panda actuator
   mappings for `actuator1` through `actuator7`.
6. Leave the gripper/task actuator `actuator8` unchanged.
7. Advance MuJoCo by exactly one call to `mj_step()`.

A successful hardware `write()` therefore advances exactly one compiled MuJoCo
physics timestep. The ROS control-loop `period` argument does not scale the
number of MuJoCo physics steps. This makes simulation evolution deterministic
for a fixed initial state and fixed sequence of hardware commands.

For the current Panda insertion MJCF, the compiled MuJoCo timestep is 0.002 s.
A dedicated real-MuJoCo controller-manager bringup can select an appropriate
control-loop rate separately; changing that bringup rate does not change the
one-step-per-successful-write hardware contract.

After a successful `write()`, the next `read()` copies the resulting finite
Panda `qpos` and `qvel` values into the corresponding ros2_control position and
velocity state interfaces.

## Week 7 MuJoCo ros2_control bringup

The dedicated Week 7 bringup uses the project-owned MuJoCo hardware plugin
without replacing the existing mock-hardware workflow.

The real-MuJoCo launch starts:

    robot_state_publisher
            |
            v
    controller_manager
            |
            +-- robustlearn_mujoco_hardware/MuJoCoSystem
            |
            +-- joint_state_broadcaster

The trajectory controller is intentionally not started by this launch. Joint
trajectory execution is deferred to Week 8.

The dedicated controller-manager configuration runs at 500 Hz, matching the
current Panda insertion model's 0.002 s MuJoCo timestep and the hardware
contract of exactly one mj_step() per successful write().

### Build

From the repository root:

    uv sync --locked --no-dev

    export ROBUSTLEARN_MUJOCO_ROOT="$(
      uv run --no-sync python -c \
        'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)'
    )"

    source /opt/ros/jazzy/setup.bash

    cd ros2_ws
    colcon build --symlink-install
    source install/setup.bash
    cd ..

### Launch

Resolve the project-owned MJCF entry point:

    MODEL_PATH="$(
      realpath robot_description/mjcf/insertion/panda_insertion.xml
    )"

Start the MuJoCo-backed Panda:

    ros2 launch \
      robustlearn_description \
      mujoco_panda.launch.py \
      model_path:="$MODEL_PATH"

The model_path launch argument is required. The MuJoCo acceptance launch
explicitly selects:

    robustlearn_mujoco_hardware/MuJoCoSystem

The existing mock_panda.launch.py remains available separately and continues
to use mock_components/GenericSystem.

### Inspect the hardware component

In another terminal with ROS 2 and the workspace sourced:

    ros2 service call \
      /controller_manager/list_hardware_components \
      controller_manager_msgs/srv/ListHardwareComponents \
      "{}"

The PandaSystem component must report:

    plugin_name='robustlearn_mujoco_hardware/MuJoCoSystem'
    state ... id=3 ... label='active'
    rw_rate=500

This check distinguishes the real MuJoCo hardware path from the mock
GenericSystem path.

### Inspect the controller

Verify that the joint-state broadcaster is active:

    ros2 control list_controllers \
      --controller-manager /controller_manager

Expected controller state:

    joint_state_broadcaster ... active

### Inspect Panda joint state

Read one published joint-state message:

    ros2 topic echo \
      /joint_states \
      sensor_msgs/msg/JointState \
      --once

The message must contain:

    panda_joint1
    panda_joint2
    panda_joint3
    panda_joint4
    panda_joint5
    panda_joint6
    panda_joint7

All seven published position values and all seven published velocity values
must be finite.

The hardware does not export an effort state interface, so the
joint_state_broadcaster may publish unavailable effort entries as NaN.
Week 7 acceptance therefore validates position and velocity state only.

### Automated acceptance

The launch-level acceptance test is:

    ros2_ws/src/robustlearn_description/test/test_mujoco_panda_launch.py

It verifies that:

- exactly one PandaSystem hardware component exists;
- its plugin is robustlearn_mujoco_hardware/MuJoCoSystem;
- the hardware lifecycle state is active;
- the hardware read/write rate is 500 Hz;
- joint_state_broadcaster becomes active;
- /joint_states contains all seven Panda arm joints;
- all seven positions are finite;
- all seven velocities are finite.

The test uses ROS service clients and a typed sensor_msgs/msg/JointState
subscription rather than parsing command-line output.

## Week 8 direct joint-trajectory execution

Issue #69 adds a dedicated trajectory-command bringup on top of the Week 7
MuJoCo hardware path.

The Week 7 joint-state-only launch remains unchanged:

    mujoco_panda.launch.py

The Week 8 trajectory launch is:

    mujoco_panda_trajectory.launch.py

and uses:

    config/mujoco_trajectory_controllers.yaml

The Week 8 launch activates both:

    joint_state_broadcaster
    panda_arm_controller

The arm controller is:

    joint_trajectory_controller/JointTrajectoryController

and commands exactly:

    panda_joint1
    panda_joint2
    panda_joint3
    panda_joint4
    panda_joint5
    panda_joint6
    panda_joint7

using position command interfaces and position/velocity state interfaces.

The gripper/task actuator remains outside this seven-arm-joint trajectory
command path.

### Launch the trajectory-command path

From the repository root, after building and sourcing the ROS workspace:

    MODEL_PATH="$(
      realpath robot_description/mjcf/insertion/panda_insertion.xml
    )"

    ros2 launch \
      robustlearn_description \
      mujoco_panda_trajectory.launch.py \
      model_path:="$MODEL_PATH"

Verify the controllers:

    ros2 control list_controllers

Expected active controllers include:

    joint_state_broadcaster ... active
    panda_arm_controller ... active

Verify the hardware interfaces:

    ros2 control list_hardware_interfaces

The position command interfaces from `panda_joint1/position` through
`panda_joint7/position` must all report both available and claimed.

### Send a bounded FollowJointTrajectory goal

The canonical insertion model starts close to:

    [0.0, 0.012, 0.0, -1.582, 0.0, 1.569, -0.785]

A small bounded direct trajectory can therefore move `panda_joint1` to
0.03 rad while keeping the other arm joints near their initial positions:

    ros2 action send_goal \
      /panda_arm_controller/follow_joint_trajectory \
      control_msgs/action/FollowJointTrajectory \
      "{
        trajectory: {
          joint_names: [
            panda_joint1,
            panda_joint2,
            panda_joint3,
            panda_joint4,
            panda_joint5,
            panda_joint6,
            panda_joint7
          ],
          points: [
            {
              positions: [
                0.03,
                0.012,
                0.0,
                -1.582,
                0.0,
                1.569,
                -0.785
              ],
              time_from_start: {sec: 2, nanosec: 0}
            }
          ]
        }
      }"

The goal must be accepted and finish with:

    error_code: 0
    Goal finished with status: SUCCEEDED

### Trajectory tolerances

The Week 8 controller configuration defines:

    path tolerance:            0.05 rad
    final joint tolerance:     0.01 rad
    stopped velocity tolerance: 0.01 rad/s
    goal-time allowance:       0.5 s

The final position error for every Panda arm joint must therefore be no greater
than 0.01 rad.

### Automated acceptance

The launch-level trajectory acceptance test is:

    ros2_ws/src/robustlearn_description/test/test_mujoco_trajectory_launch.py

It verifies that:

- `joint_state_broadcaster` becomes active;
- `panda_arm_controller` becomes active;
- all seven Panda position command interfaces are available and claimed;
- Panda position and velocity feedback is finite;
- a bounded `FollowJointTrajectory` goal is accepted;
- the action reaches `GoalStatus.STATUS_SUCCEEDED`;
- MuJoCo-backed Panda joint state changes in response to the command;
- the final error for every Panda arm joint is at most 0.01 rad.

The Week 7 joint-state-only acceptance test remains:

    ros2_ws/src/robustlearn_description/test/test_mujoco_panda_launch.py