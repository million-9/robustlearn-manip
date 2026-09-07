# MoveIt 2 Trajectory Execution Through MuJoCo

## Purpose

This document describes the Week 8 MoveIt-to-MuJoCo integration path.

The workflow connects MoveIt 2 planning and trajectory execution to the
project-owned MuJoCo ros2_control hardware implementation:

    MoveIt 2 / OMPL
            |
            v
    /execute_trajectory
            |
            v
    MoveItSimpleControllerManager
            |
            v
    panda_arm_controller
            |
            v
    FollowJointTrajectory
            |
            v
    ros2_control controller_manager
            |
            v
    robustlearn_mujoco_hardware/MuJoCoSystem
            |
            v
    MuJoCo Panda simulation

The existing mock MoveIt workflow remains separate and unchanged.

## Architecture

The integrated launch is:

    ros2_ws/src/robustlearn_moveit_config/launch/moveit_mujoco.launch.py

The launch intentionally gives the ROS control and MoveIt layers different
robot-description responsibilities.

The project-owned Panda description is used by:

- robot_state_publisher
- controller_manager
- robustlearn_mujoco_hardware/MuJoCoSystem
- joint_state_broadcaster
- panda_arm_controller

MoveIt uses the installed Panda MoveIt resource model internally for:

- Panda SRDF
- planning groups
- collision geometry
- kinematics
- OMPL configuration
- trajectory-execution controller configuration

This separation avoids launching a second mock Panda hardware component while
still allowing MoveIt to use the complete Panda planning model.

The cross-model FK validation performed earlier in Week 8 establishes the
kinematic consistency required for this separation.

## Hardware Path

The real-MuJoCo workflow contains exactly one ros2_control hardware component:

    hardware component: PandaSystem
    plugin: robustlearn_mujoco_hardware/MuJoCoSystem

The real-MuJoCo acceptance path must not use:

    mock_components/GenericSystem

The active controllers are:

    joint_state_broadcaster
    panda_arm_controller

The arm controller owns the seven Panda position command interfaces:

    panda_joint1
    panda_joint2
    panda_joint3
    panda_joint4
    panda_joint5
    panda_joint6
    panda_joint7

## Complete MoveIt Joint State

The project MuJoCo hardware exposes the seven physical Panda arm joints.

The MoveIt Panda model also contains:

    panda_finger_joint1
    panda_finger_joint2

The current simulation does not model an independently controlled Panda
gripper in this ros2_control path.

To provide MoveIt with a complete robot state without introducing fake gripper
hardware, joint_state_broadcaster publishes the two finger joints as fixed
extra joints with zero state.

The resulting /joint_states message therefore contains:

    panda_joint1
    panda_joint2
    panda_joint3
    panda_joint4
    panda_joint5
    panda_joint6
    panda_joint7
    panda_finger_joint1
    panda_finger_joint2

The two finger entries are state-completion values only. They are not command
interfaces and do not represent a second hardware component.

## Environment Setup

From the repository root:

    source /opt/ros/jazzy/setup.bash
    source .venv/bin/activate

MuJoCo can be made explicit for CMake discovery with:

    export ROBUSTLEARN_MUJOCO_ROOT="$(
      python3 -c \
        'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)'
    )"

Build the workspace:

    cd ros2_ws
    colcon build --symlink-install
    source install/setup.bash

## Headless Launch

The CI-oriented and automated-validation path does not start RViz.

From ros2_ws:

    MODEL_PATH="$HOME/projects/robustlearn-manip/robot_description/mjcf/insertion/panda_insertion.xml"

    ros2 launch robustlearn_moveit_config moveit_mujoco.launch.py \
      model_path:="$MODEL_PATH" \
      use_rviz:=false

Expected runtime components include:

    robot_state_publisher
    controller_manager
    joint_state_broadcaster
    panda_arm_controller
    move_group
    world_to_panda_base static transform

Expected MoveIt interfaces include:

    /plan_kinematic_path
    /execute_trajectory
    /move_action
    /panda_arm_controller/follow_joint_trajectory

## Interactive RViz Launch

For interactive MoveIt visualization:

    MODEL_PATH="$HOME/projects/robustlearn-manip/robot_description/mjcf/insertion/panda_insertion.xml"

    ros2 launch robustlearn_moveit_config moveit_mujoco.launch.py \
      model_path:="$MODEL_PATH" \
      use_rviz:=true

RViz visualizes the Panda using ROS joint-state feedback.

MuJoCo itself continues to run as the physics backend inside
robustlearn_mujoco_hardware/MuJoCoSystem. The native MuJoCo viewer is not
started by this workflow.

## Planning and Execution

The automated acceptance problem uses the Panda arm planning group:

    panda_arm

A bounded joint-space target is requested from OMPL.

The resulting MoveIt RobotTrajectory is then submitted through:

    /execute_trajectory

MoveIt selects:

    panda_arm_controller

and forwards the JointTrajectory through:

    /panda_arm_controller/follow_joint_trajectory

The controller converts the trajectory into position commands consumed by the
project-owned MuJoCo SystemInterface.

## Automated Integration Test

The permanent launch test is:

    ros2_ws/src/robustlearn_moveit_config/test/test_moveit_mujoco_execution_launch.py

Run the test directly with:

    cd ros2_ws
    source install/setup.bash

    colcon test \
      --packages-select robustlearn_moveit_config \
      --ctest-args -R test_test_moveit_mujoco_execution_launch.py \
      --event-handlers console_direct+

    colcon test-result \
      --verbose \
      --test-result-base build/robustlearn_moveit_config

The test verifies:

- exactly one real Panda hardware component is present
- PandaSystem uses robustlearn_mujoco_hardware/MuJoCoSystem
- GenericSystem is not used in the real-MuJoCo path
- joint_state_broadcaster is active
- panda_arm_controller is active
- complete Panda arm and finger state reaches MoveIt
- Panda position and velocity feedback is finite
- MoveIt planning succeeds
- the planned trajectory is non-empty
- MoveIt accepts the execution request
- trajectory execution reports success
- MuJoCo joint feedback changes during execution
- the final MuJoCo state agrees with the planned final state

## Execution Tolerances

The integration test requires meaningful observed arm motion greater than:

    0.02 rad

The maximum allowed final Panda joint error is:

    0.02 rad

The tolerance is evaluated against the final positions in the actual
MoveIt-generated trajectory rather than only against the originally requested
joint constraints.

## Manual Validation Result

The initial manual end-to-end validation demonstrated:

    MoveIt planning: success
    ExecuteTrajectory goal: accepted
    controller terminal state: succeeded
    MoveIt error code: SUCCESS
    meaningful MuJoCo joint motion: observed
    final joint error: within 0.02 rad

The controller runtime reported:

    Received new action goal
    Accepted new action goal
    Goal reached, success!

MoveIt reported successful completion of the same trajectory.

## Preserved Workflows

The following existing workflows remain intentionally available:

- mock MoveIt demo
- Week 7 real-MuJoCo joint-state bringup
- direct FollowJointTrajectory execution through MuJoCo
- project-owned MuJoCo hardware launch
- headless MoveIt-to-MuJoCo execution
- optional RViz MoveIt visualization

The MoveIt-to-MuJoCo launch does not replace the mock workflow.

## Known Non-Blocking Runtime Messages

On development machines without realtime scheduling privileges,
controller_manager may report that FIFO realtime scheduling could not be
enabled.

MoveIt can also report that no 3D Octomap sensor plugin is configured. The
current Week 8 joint-space planning acceptance does not require live 3D sensor
updates.

During launch-test teardown, MoveIt and ros2_control may emit shutdown-related
messages while processes receive SIGINT/SIGTERM. These messages are considered
teardown diagnostics when the launch test itself completes successfully.

## Week 8 Boundary

This integration proves the central Week 8 functional result:

    MoveIt trajectory executes in MuJoCo.

The complete Week 8 milestone is finalized separately by the Week 8 acceptance
workflow, which also combines the joint-mapping contract and cross-model FK
validation.
