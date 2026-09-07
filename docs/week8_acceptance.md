# Week 8 Acceptance Workflow

Week 8 completes the MoveIt-to-MuJoCo trajectory-execution milestone for
RobustLearn-Manip.

The milestone gate is:

    MoveIt trajectory executes successfully in MuJoCo through
    robustlearn_mujoco_hardware/MuJoCoSystem.

## Scope

Week 8 integrates and validates the following layers:

- real MuJoCo-backed ros2_control hardware;
- seven-joint Panda trajectory control;
- ROS/URDF to MuJoCo/MJCF joint correspondence;
- cross-model forward-kinematics consistency;
- MoveIt 2 planning;
- MoveIt trajectory execution through ros2_control;
- MuJoCo state feedback during execution;
- regression preservation of the existing mock workflows.

The Week 8 acceptance workflow deliberately reuses the dedicated validation
layers implemented during Issues #69 through #72 instead of duplicating their
test logic.

## Real hardware path

The real execution architecture is:

    MoveIt 2
        |
        | planned JointTrajectory
        v
    MoveItSimpleControllerManager
        |
        v
    panda_arm_controller
        |
        | seven position command interfaces
        v
    controller_manager
        |
        v
    PandaSystem
        |
        v
    robustlearn_mujoco_hardware/MuJoCoSystem
        |
        v
    MuJoCo Panda model

Only one Panda hardware component is active in this workflow.

The real hardware component is:

    component: PandaSystem
    plugin: robustlearn_mujoco_hardware/MuJoCoSystem

The mock hardware plugin:

    mock_components/GenericSystem

is not used by the real-MuJoCo acceptance path.

## Seven-joint trajectory controller contract

The active trajectory controller is:

    panda_arm_controller

During execution it claims exactly these seven command interfaces:

    panda_joint1/position
    panda_joint2/position
    panda_joint3/position
    panda_joint4/position
    panda_joint5/position
    panda_joint6/position
    panda_joint7/position

The gripper remains outside the seven-arm-joint command path.

The joint-state broadcaster additionally publishes fixed finger-state entries
required by the richer MoveIt Panda model.

## URDF-to-MJCF mapping validation

The project-owned mapping contract is:

    robot_description/panda_joint_mapping.json

It defines the correspondence between:

    ROS Panda joints
    MuJoCo Panda joints
    MuJoCo actuators

for all seven arm joints.

The mapping is validated by:

    tests/unit/test_panda_joint_mapping.py

The mapping contract also defines the cross-model frame correspondence used by
the forward-kinematics validation.

Detailed documentation is available in:

    docs/panda_joint_mapping.md

## Cross-model forward kinematics

Cross-model FK validation compares the MoveIt/URDF representation against the
native MuJoCo representation using identical mapped joint configurations.

The primary frame pair is:

    panda_link7 <-> link7

The tool frame pair is:

    panda_hand <-> hand

The dedicated test is:

    ros2_ws/src/robustlearn_mujoco_hardware/test/test_cross_model_fk.cpp

The validation tolerance is:

    translation: 1e-6 m
    rotation:    1e-6 rad

Detailed documentation is available in:

    docs/cross_model_fk_validation.md

## MoveIt-to-MuJoCo execution

The integrated launch workflow is:

    ros2_ws/src/robustlearn_moveit_config/launch/moveit_mujoco.launch.py

It combines:

- the project-owned minimal Panda URDF for robot_state_publisher,
  controller_manager, and MuJoCoSystem;
- the upstream MoveIt Panda model for planning, SRDF, kinematics, collision
  checking, and OMPL;
- the real panda_arm_controller trajectory path;
- MuJoCo-backed joint-state feedback.

This avoids launching a second mock controller manager.

The existing mock MoveIt workflow remains separate and available.

## End-to-end launch acceptance

The main MoveIt-to-MuJoCo launch acceptance test is:

    ros2_ws/src/robustlearn_moveit_config/test/test_moveit_mujoco_execution_launch.py

The test verifies:

- PandaSystem becomes active;
- PandaSystem uses robustlearn_mujoco_hardware/MuJoCoSystem;
- exactly one hardware component is active;
- GenericSystem is absent from the real path;
- joint_state_broadcaster becomes active;
- panda_arm_controller becomes active;
- panda_arm_controller claims exactly seven Panda position interfaces;
- complete finite Panda state reaches MoveIt;
- MoveIt planning succeeds;
- the planned trajectory contains all seven Panda arm joints;
- MoveIt execution is accepted;
- execution reaches the SUCCEEDED action state;
- MoveIt reports SUCCESS;
- MuJoCo joint feedback changes during execution;
- final Panda arm state remains finite;
- final joint error is within the execution tolerance.

The documented final-position tolerance is:

    0.02 rad

The test also requires observable motion greater than:

    0.02 rad

## Week 8 acceptance runner

The complete milestone workflow is:

    tools/run_week8_acceptance.sh

Run it from the repository root:

    ./tools/run_week8_acceptance.sh

The runner performs:

- Panda mapping-contract validation;
- Ruff validation;
- mypy validation of the production robustlearn package;
- the complete Python pytest regression;
- ROS workspace build;
- cross-model FK validation;
- MoveIt-to-MuJoCo execution validation;
- complete ROS workspace regression.

The final success banner is:

    WEEK 8 ACCEPTANCE: PASS
    MoveIt trajectory executes in MuJoCo.

## Validated Week 8 result

The complete Week 8 acceptance workflow was executed successfully.

Final ROS regression:

    5 packages finished
    118 tests
    0 errors
    0 failures
    12 skipped

The MoveIt-to-MuJoCo acceptance test passed independently before the complete
workspace regression.

The final milestone gate therefore passes:

    MoveIt trajectory executes successfully in MuJoCo through
    robustlearn_mujoco_hardware/MuJoCoSystem.

## Preserved workflows

Week 8 preserves the previously validated paths:

- Week 7 real-MuJoCo ros2_control bringup;
- direct FollowJointTrajectory execution through MuJoCo;
- mock Panda ros2_control bringup;
- mock MoveIt demo;
- Python manipulation-task and evaluation regressions.

Project-specific task changes remain outside the vendored MuJoCo Menagerie
Panda files.

## Week 8 boundary

Week 8 establishes the planning-to-execution infrastructure required for later
deployment-oriented manipulation work.

Completed by this boundary:

    MoveIt planning
        +
    validated ROS/MuJoCo model correspondence
        +
    real ros2_control trajectory execution
        +
    MuJoCo feedback
        =
    validated MoveIt-to-MuJoCo execution path

The following remain outside the Week 8 milestone:

- classical insertion expert benchmarking;
- demonstration collection;
- Behaviour Cloning;
- DAgger;
- SAC;
- learned-policy deployment;
- large-scale robustness evaluation;
- ONNX deployment;
- hardware deployment.
