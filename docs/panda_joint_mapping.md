# Panda URDF-to-MuJoCo Joint Mapping

This document defines the authoritative Panda arm joint correspondence used by
RobustLearn-Manip between the ROS/URDF representation and the MuJoCo
representation.

The machine-readable contract is:

    robot_description/panda_joint_mapping.json

The validator is:

    robustlearn/panda_joint_mapping.py

The automated tests are:

    tests/unit/test_panda_joint_mapping.py

## Authoritative joint ordering

The ROS/URDF Panda arm ordering is:

    panda_joint1
    panda_joint2
    panda_joint3
    panda_joint4
    panda_joint5
    panda_joint6
    panda_joint7

The corresponding MuJoCo ordering is:

    joint1
    joint2
    joint3
    joint4
    joint5
    joint6
    joint7

The corresponding MuJoCo actuators are:

    actuator1
    actuator2
    actuator3
    actuator4
    actuator5
    actuator6
    actuator7

The mapping is therefore:

| ROS joint | MuJoCo joint | MuJoCo actuator |
| --- | --- | --- |
| `panda_joint1` | `joint1` | `actuator1` |
| `panda_joint2` | `joint2` | `actuator2` |
| `panda_joint3` | `joint3` | `actuator3` |
| `panda_joint4` | `joint4` | `actuator4` |
| `panda_joint5` | `joint5` | `actuator5` |
| `panda_joint6` | `joint6` | `actuator6` |
| `panda_joint7` | `joint7` | `actuator7` |

Every ROS joint maps to exactly one MuJoCo joint and exactly one MuJoCo
actuator.

Duplicate ROS joints, MuJoCo joints, or MuJoCo actuators are invalid.

## Position convention

For every Panda arm joint, the position transformation is:

    ros_position =
        position_sign * mujoco_position
        + position_offset_rad

For the current Panda model:

    position_sign = +1
    position_offset_rad = 0

for all seven joints.

Therefore ROS and MuJoCo use the same joint-position sign and zero-reference
convention.

Both representations define the arm joint axis locally as:

    [0, 0, 1]

The mapping validator checks the ROS joint axis against the compiled MuJoCo
joint axis together with the declared position sign.

## Joint limits

The URDF and compiled MuJoCo joint limits are:

| Joint | Lower [rad] | Upper [rad] |
| --- | ---: | ---: |
| `panda_joint1` / `joint1` | -2.8973 | 2.8973 |
| `panda_joint2` / `joint2` | -1.7628 | 1.7628 |
| `panda_joint3` / `joint3` | -2.8973 | 2.8973 |
| `panda_joint4` / `joint4` | -3.0718 | -0.0698 |
| `panda_joint5` / `joint5` | -2.8973 | 2.8973 |
| `panda_joint6` / `joint6` | -0.0175 | 3.7525 |
| `panda_joint7` / `joint7` | -2.8973 | 2.8973 |

There are currently no deliberate Panda arm joint-limit differences between
the ROS/URDF and compiled MuJoCo representations.

The mapping contract therefore uses an equality policy with an absolute
comparison tolerance of:

    1e-9 rad

The MuJoCo validator reads the compiled joint ranges rather than relying only
on literal XML attributes. This is important because some Panda MJCF joints
inherit their limits from MuJoCo defaults.

## Home/reference configuration

The MuJoCo Panda model defines the keyframe:

    home

The expected mapped ROS-side Panda joint configuration is:

    panda_joint1 =  0.0
    panda_joint2 =  0.0
    panda_joint3 =  0.0
    panda_joint4 = -1.57079
    panda_joint5 =  0.0
    panda_joint6 =  1.57079
    panda_joint7 = -0.7853

The validator verifies that:

- the `home` keyframe exists;
- every mapped MuJoCo joint has a finite home position;
- the declared sign and offset transform the MuJoCo position into the expected
  ROS position;
- every home position lies within the corresponding URDF joint limits.

This is the reference configuration used to establish that ROS and MuJoCo
describe the same arm configuration before later cross-model forward
kinematics validation.

## MuJoCo ordering and actuator correspondence

The compiled MuJoCo Panda arm qpos addresses are expected to follow the
contract ordering:

    joint1 -> qpos 0
    joint2 -> qpos 1
    joint3 -> qpos 2
    joint4 -> qpos 3
    joint5 -> qpos 4
    joint6 -> qpos 5
    joint7 -> qpos 6

The validator requires mapped qpos addresses to be unique and ordered
consistently with the contract.

Each mapped actuator must:

- exist;
- use a joint transmission;
- target exactly the corresponding mapped MuJoCo joint.

This makes the joint-state and joint-command correspondence explicit instead
of relying on implicit array positions.

## Cross-model FK boundary

Issue #70 validates model correspondence only at the joint-contract level.

The contract reserves the following endpoints for the later cross-model
forward-kinematics validation:

    ROS base frame:      panda_link0
    MuJoCo base body:    link0

    ROS tip frame:       panda_link7
    MuJoCo tip body:     link7

All seven mapped arm joints are included in that later comparison.

Full cross-model forward-kinematics comparison is intentionally deferred to
the next Week 8 issue.

## Validation

Run the focused mapping tests with:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
      tests/unit/test_panda_joint_mapping.py \
      -v

The validator checks:

- schema version;
- exactly seven mapped arm joints;
- deterministic ROS joint ordering;
- deterministic MuJoCo joint ordering;
- unique ROS joint names;
- unique MuJoCo joint names;
- unique MuJoCo actuator names;
- required URDF joints;
- required MuJoCo joints;
- required MuJoCo actuators;
- revolute/hinge joint type correspondence;
- local axis and position-sign convention;
- URDF versus compiled MuJoCo joint limits;
- MuJoCo actuator-to-joint targets;
- unique ordered MuJoCo qpos addresses;
- `home` keyframe existence;
- mapped home/reference positions;
- existence of the declared ROS and MuJoCo FK endpoints.

Invalid mappings fail deterministically with `MappingValidationError`.

## Vendored model policy

The MuJoCo Menagerie Panda model remains vendored under:

    robot_description/mjcf/franka_emika_panda/

Those files are treated as third-party source assets and are not modified for
this mapping contract.

All RobustLearn-specific mapping metadata and validation logic live outside the
vendored model.