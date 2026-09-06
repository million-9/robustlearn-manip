# Cross-Model Panda Forward-Kinematics Validation

## Purpose

RobustLearn-Manip represents the Franka Panda arm in two model stacks:

- ROS 2 / MoveIt 2 for planning and robot-state kinematics.
- MuJoCo for physics simulation and ros2_control hardware execution.

Issue #71 adds a deterministic cross-model forward-kinematics test that verifies both model stacks produce the same Panda end-effector pose for the same seven-joint arm configuration.

The validation uses the machine-readable joint mapping contract:

    robot_description/panda_joint_mapping.json

The test does not modify either robot model to force agreement.

## Mapping contract

The test reads the existing mapping contract rather than duplicating the ROS-to-MuJoCo joint relationship.

For each of the seven Panda arm joints, the contract defines:

- ROS joint name
- MuJoCo joint name
- position sign
- position offset
- canonical home position
- cross-model base and tip frames

The position relationship is:

    ros_position =
        position_sign * mujoco_position
        + position_offset_rad

The test converts a ROS joint position into the corresponding MuJoCo position using:

    mujoco_position =
        (ros_position - position_offset_rad)
        / position_sign

For the current Panda mapping:

    position_sign = +1
    position_offset_rad = 0

for all seven arm joints.

## Seven-joint arm frame correspondence

The mapping contract defines the primary FK correspondence as:

    MoveIt: panda_link0 -> panda_link7
    MuJoCo: link0       -> link7

This represents the neutral seven-DOF Panda arm boundary.

The test computes poses relative to the corresponding base frame:

    T_base_tip = inverse(T_world_base) * T_world_tip

Using relative transforms prevents arbitrary world placement from affecting the comparison.

## Tool-frame correspondence

The test additionally validates the installed MoveIt Panda hand against the MuJoCo hand body:

    MoveIt: panda_link0 -> panda_hand
    MuJoCo: link0       -> hand

The MoveIt Panda description represents the fixed geometry after panda_link7 as:

    panda_link7
        -> translation 0.107 m
    panda_link8
        -> yaw -pi/4
    panda_hand

MuJoCo represents the equivalent fixed transform directly as:

    link7 -> hand

Therefore the validated frame pairs are:

    panda_link7 <-> link7
    panda_hand  <-> hand

The panda_link7/link7 pair validates the seven-joint arm model.

The panda_hand/hand pair additionally validates the fixed tool geometry used by the installed MoveIt Panda description.

## MoveIt FK computation

The test constructs the MoveIt Panda model directly from the installed MoveIt resource URDF and SRDF.

It uses:

    moveit::core::RobotModel
    moveit::core::JointModelGroup
    moveit::core::RobotState

For each configuration, the seven Panda arm positions are assigned to the panda_arm joint group and MoveIt updates the robot-state link transforms.

The test does not require:

- move_group
- RViz
- controller_manager
- a running ROS graph

This keeps the test deterministic and suitable for CI.

## MuJoCo FK computation

The test compiles the project insertion model:

    robot_description/mjcf/insertion/panda_insertion.xml

MuJoCo joint names are obtained from the mapping contract.

The ROS joint values are converted according to the mapping sign and offset, written into the corresponding MuJoCo qpos entries, and then:

    mj_forward()

is used to compute the MuJoCo body poses.

No vendored MuJoCo Menagerie files are changed by this validation.

## Representative configurations

Four deterministic joint configurations are tested.

### Home

The home configuration is read directly from the mapping contract:

    [0.0,
     0.0,
     0.0,
     -1.57079,
     0.0,
     1.57079,
     -0.7853]

### Offset A

    [0.3,
     -0.5,
     0.4,
     -1.2,
     0.5,
     1.8,
     -0.4]

### Offset B

    [-0.6,
     0.7,
     -0.8,
     -2.2,
     0.9,
     0.8,
     1.1]

### Offset C

    [1.0,
     -1.0,
     0.7,
     -0.8,
     -1.2,
     2.5,
     -1.0]

The non-home configurations exercise multiple joints simultaneously and avoid validating only a zero or symmetric pose.

## Translation comparison

Translation error is measured as the Euclidean distance between the MoveIt and MuJoCo positions:

    || p_moveit - p_mujoco ||

The required tolerance is:

    translation error <= 1e-6 m

## Orientation comparison

Orientation error is computed from the relative rotation:

    R_error = transpose(R_moveit) * R_mujoco

The angular error is then obtained from the relative rotation angle:

    orientation_error = angle(R_error)

This avoids quaternion sign ambiguity.

The required tolerance is:

    orientation error <= 1e-6 rad

## Failure diagnostics

GoogleTest scoped traces record:

- configuration name
- seven joint positions
- ROS/MuJoCo frame pair

If a comparison fails, the test additionally reports:

- translation error
- orientation error
- MoveIt translation
- MuJoCo translation
- MoveIt rotation matrix
- MuJoCo rotation matrix

This makes cross-model drift diagnosable directly from test or CI output.

## Implementation

The test is located at:

    ros2_ws/src/robustlearn_mujoco_hardware/test/test_cross_model_fk.cpp

It is registered with the package's existing ament_cmake_gtest infrastructure.

The test directly consumes:

    robot_description/panda_joint_mapping.json

so the FK validation and the joint mapping contract cannot silently diverge.

## Focused validation

Build the package with:

    cd ros2_ws

    colcon build \
      --symlink-install \
      --packages-select robustlearn_mujoco_hardware \
      --allow-overriding robustlearn_mujoco_hardware

Run only the cross-model FK test with:

    ctest \
      --test-dir build/robustlearn_mujoco_hardware \
      -R '^test_cross_model_fk$' \
      --output-on-failure

The focused test currently passes for all four configurations and both validated frame pairs.

## Scope boundary

This validation proves cross-model kinematic consistency.

It does not test:

- MoveIt trajectory execution through ros2_control
- trajectory-controller timing
- physics or dynamic-model agreement
- collision or contact behavior
- motion-planning success

MoveIt trajectory execution through MuJoCo is handled separately by Issue #72.
