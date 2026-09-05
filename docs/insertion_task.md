# Panda Insertion Task

This document describes the initial MuJoCo insertion-task geometry introduced
for Week 4 of RobustLearn-Manip.

The objective of this stage is not to implement a complete insertion
controller. It establishes a minimal, testable physical scene that later
control, imitation-learning, and reinforcement-learning work can use.

## Model Composition

The task is composed from three independent MJCF components:

```text
robot_description/mjcf/
├── franka_emika_panda/
│   └── panda.xml
└── insertion/
    ├── workcell.xml
    └── fixed_peg.xml
```

The Franka Panda model is vendored from MuJoCo Menagerie and is not modified
for task-specific geometry.

The complete task model is assembled programmatically using MuJoCo `MjSpec`.

The composition is:

```text
Panda model
    |
    +-- hand
         |
         +-- fixed insertion tool
              |
              +-- peg

World
    |
    +-- workstation
    +-- insertion fixture
         |
         +-- receptacle
```

The implementation is provided by:

```text
robustlearn/sim/insertion.py
```

## Initial Tool Strategy

The initial task uses a **fixed insertion tool** attached to the Panda hand.

The peg is therefore not a free body and is not dynamically grasped by the
Panda fingers.

This strategy is intentional.

It avoids introducing grasp stability, object slipping, gripper-force tuning,
and additional reset state before the basic insertion scene has been
validated.

A simplified adapter is included visually so that the peg appears as a
deliberate tool attachment rather than as an unsupported object between the
gripper fingers.

A dynamically grasped connector can be introduced in a later milestone if it
becomes useful for the learning task.

## Peg Geometry

The contact-critical peg uses primitive box geometry.

Nominal dimensions:

```text
width:   16 mm
depth:   16 mm
length:  60 mm
```

MuJoCo box dimensions are specified as half-sizes, therefore the collision
geometry uses:

```text
size="0.008 0.008 0.03"
```

The peg visual geometry and collision geometry are separate.

The collision representation intentionally remains simple to make contact
behavior easier to inspect and debug.

## Receptacle Geometry

The receptacle is represented as a square opening constructed from four
primitive box collision geoms.

The opening is approximately:

```text
19 mm x 19 mm
```

With the 16 mm square peg, this produces approximately:

```text
1.5 mm clearance per side
```

The opening is a real empty collision volume.

It is not represented by placing a decorative geometry on top of a solid
fixture block.

The four collision walls are:

```text
fixture_left_collision
fixture_right_collision
fixture_front_collision
fixture_rear_collision
```

The use of simple box geometry is deliberate. Detailed meshes are unnecessary
for the initial contact model and would make collision debugging more
difficult.

## Workstation

The task contains a static workstation and fixed insertion fixture.

Important bodies are:

```text
workstation
insertion_fixture
receptacle
peg_tool
```

The workstation and fixture are fixed to the MuJoCo world.

They therefore do not require joints and do not move under gravity.

## Stable Task Sites

The task exposes the following sites as stable programmatic references:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

### `peg_tip`

Attached to the distal end of the fixed insertion peg.

This will later be useful for:

- Cartesian observations
- insertion depth calculations
- alignment error
- reward computation
- success metrics

### `receptacle_center`

Defines the nominal center of the insertion opening.

### `insertion_axis`

Defines the nominal insertion direction.

The peg axis and insertion axis are parallel and point in opposite local
directions in the initial model:

```text
peg local +Z       -> downward
insertion site +Z  -> upward
dot product        -> approximately -1
```

### `pre_insertion`

Defines the nominal starting reference above the receptacle.

At the Panda `home` keyframe, the peg tip is approximately coincident with
this site.

## Nominal Initial Configuration

The Panda Menagerie `home` keyframe is used as the initial task pose.

The workcell is positioned so that the peg is centered above the receptacle
when the Panda is in this configuration.

Measured nominal geometry is approximately:

```text
horizontal alignment error: 0.001 mm
peg-to-pre-insertion error:  0.002 mm
peg height above receptacle: 19.5 mm
initial contacts:            0
```

These values are diagnostic measurements rather than exact task invariants.

Automated tests use tolerances rather than requiring exact floating-point
coordinates.

## Gravity and Stability

The nominal scene has also been stepped for approximately five simulated
seconds.

Observed behavior:

```text
workcell/receptacle drift: 0 mm
maximum task contacts:     0
simulation state:          finite
peg-tip settling:          approximately 6.8 mm
```

The small peg-tip motion results from the Panda arm settling under gravity
against the Menagerie position actuators.

Controller tuning is intentionally outside the scope of the geometry
milestone.

The automated regression test therefore checks for gross instability rather
than requiring a perfectly motionless end-effector.

## Viewer

The insertion scene can be inspected locally with:

```bash
uv run python tools/view_insertion.py
```

The viewer:

- loads the complete Panda insertion model
- resets the Panda to the `home` keyframe
- steps MuJoCo physics
- provides an interactive camera for inspecting the task geometry

The viewer does **not** execute an insertion trajectory.

Issue #28 establishes the physical task scene only. Motion generation,
controllers, policies, and environment actions are handled by later work.

## Automated Validation

Task geometry is tested in:

```text
tests/unit/test_insertion_model.py
```

The tests verify:

- the complete Panda + workcell model compiles
- required task bodies exist
- required task sites exist
- contact-critical geometry uses primitive boxes
- the home pose is centered over the receptacle
- the peg axis aligns with the insertion axis
- the nominal initial pose has no unintended contacts
- the scene remains finite and stable while physics is stepped

The full Python validation commands are:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```