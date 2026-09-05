# Panda sensing

RobustLearn-Manip defines a project-owned MuJoCo sensing layer for the
Franka Emika Panda insertion task.

The sensing layer provides explicit Panda arm joint state and wrist
force/torque measurements without modifying the vendored MuJoCo
Menagerie Panda model.

## Sensor outputs

The compiled insertion model contains 16 project-owned sensors with
20 scalar output values:

| Quantity | Count | Shape | Units |
| --- | ---: | ---: | --- |
| Panda arm joint position | 7 | `(7,)` | rad |
| Panda arm joint velocity | 7 | `(7,)` | rad/s |
| Wrist force | 1 | `(3,)` | N |
| Wrist torque | 1 | `(3,)` | N m |

The seven Panda arm joints are:

- `joint1`
- `joint2`
- `joint3`
- `joint4`
- `joint5`
- `joint6`
- `joint7`

## Stable sensor names

Joint-position sensors:

- `panda_joint1_position`
- `panda_joint2_position`
- `panda_joint3_position`
- `panda_joint4_position`
- `panda_joint5_position`
- `panda_joint6_position`
- `panda_joint7_position`

Joint-velocity sensors:

- `panda_joint1_velocity`
- `panda_joint2_velocity`
- `panda_joint3_velocity`
- `panda_joint4_velocity`
- `panda_joint5_velocity`
- `panda_joint6_velocity`
- `panda_joint7_velocity`

Wrist wrench sensors:

- `panda_wrist_force`
- `panda_wrist_torque`

The corresponding wrist sensing site is:

- `panda_wrist_ft`

These names are defined centrally in
`robustlearn/sim/sensing.py`.

Application code should not depend directly on raw
`MjModel.sensor_adr` or `MjData.sensordata` indexing.

## Wrist sensing frame

The `panda_wrist_ft` site is project-owned and belongs to the
`peg_tool` body.

The compiled body relationship is:

```text
hand
└── peg_tool
    └── panda_wrist_ft
```

The force and torque sensors therefore measure the wrench transmitted
across the fixed `hand` to `peg_tool` relationship.

MuJoCo's force-sensor convention defines the sensing site on the child
body. The measured force points from the child body (`peg_tool`) toward
its parent body (`hand`). Torque uses the corresponding child-parent
interaction convention.

The sensing site has no additional local rotation relative to
`peg_tool`. Wrist-force and wrist-torque components are therefore
expressed in the `panda_wrist_ft` site frame, whose axes coincide with
the local axes of `peg_tool`.

Force is reported in newtons (N).

Torque is reported in newton-metres (N m).

The reset wrench is not required to be zero. Gravity and other model
dynamics can produce a finite baseline wrench. Consumers should not
interpret any nonzero value by itself as contact with the insertion
fixture.

## Python API

`MuJoCoSimulation.sensor_snapshot()` returns a
`PandaSensorSnapshot` containing independent NumPy array copies.

Example:

```python
from robustlearn.sim import MuJoCoSimulation

sim = MuJoCoSimulation()
sim.reset(seed=2026)

sensors = sim.sensor_snapshot()

print(sensors.joint_positions)
print(sensors.joint_velocities)
print(sensors.wrist_force)
print(sensors.wrist_torque)
```

The snapshot fields are:

```text
joint_positions   shape (7,)   rad
joint_velocities  shape (7,)   rad/s
wrist_force       shape (3,)   N
wrist_torque      shape (3,)   N m
```

`PandaSensorReader` resolves MuJoCo sensor addresses when the
simulation is constructed and keeps raw `sensordata` indexing behind
the simulation and sensing abstraction.

`PandaSensorSnapshot` arrays are independent copies. Later changes to
`MjData.sensordata` therefore do not mutate an already returned
snapshot.

## Joint-state agreement

The project uses explicit MuJoCo `jointpos` and `jointvel` sensors for
the seven scalar Panda arm joints.

The current unit tests require exact equality between these sensor
outputs and the corresponding simulator joint state.

This exact comparison is intentional for the current noise-free and
delay-free simulation model. Sensor noise and delay randomization are
outside the scope of this sensing layer and would require revisiting
that contract.

## Deterministic reset behavior

Sensor state follows the deterministic simulation behavior established
by the simulation layer.

For the same seed and initial conditions, independently constructed
simulations must produce identical Panda sensor snapshots after reset.

The tests also verify that all sensor values are finite at reset.

## Controlled wrist-load sanity test

The unit suite applies a known Cartesian external force and torque to
the fixed `peg_tool` body using MuJoCo's external-wrench input.

The controlled test load is:

```text
force  = [12.0, -7.0, 5.0] N
torque = [1.2, -0.7, 0.4] N m
```

After forwarding the model dynamics, the wrist-force and wrist-torque
measurements must change by clearly measurable amounts.

The test requires:

```text
force-change norm  > 0.1 N
torque-change norm > 0.01 N m
```

These thresholds are intentionally much smaller than the response
observed during development. They verify that the sensor responds to a
controlled load without hard-coding one exact rigid-body dynamic
solution.

After the external wrench is cleared and the model is forwarded again,
the wrist measurements must return to the original baseline.

During the Issue #37 development probe, the controlled load produced
approximately:

```text
force-change norm  = 14.3944 N
torque-change norm = 1.5209 N m
```

After clearing the applied wrench, both force and torque returned
exactly to their original baseline values in the development probe.

The permanent test checks measurable response and baseline restoration
rather than exact component values.

This is important because wrist sensor components are expressed in the
local sensing-site frame, while an externally applied Cartesian wrench
and the resulting measurements also depend on the current robot
configuration and rigid-body dynamics.

## Reset baseline

At the deterministic home reset used during Issue #37 development, the
wrist sensors produced a small finite baseline wrench rather than
exactly zero.

An example reset measurement was:

```text
wrist force [N]
[-0.012622146, -0.102360981, -0.004494517]

wrist torque [N m]
[0.004371763, -0.000503851, -0.000001138]
```

This baseline is not considered an error.

The sensing contract requires finite and deterministic readings, not a
zero wrench at reset.

## Sensor model ownership

The original MuJoCo Menagerie Panda model remains vendored and
unmodified.

Project-specific sensing is introduced through two project-owned
mechanisms:

1. The `panda_wrist_ft` site is defined in
   `robot_description/mjcf/insertion/fixed_peg.xml`.
2. Panda joint-state and wrist-wrench sensors are added during model
   composition from `robustlearn/sim/sensing.py`.

This keeps task-specific instrumentation outside the vendored Panda
model.

## Model composition

The insertion model is assembled from:

- the vendored Panda MJCF;
- the project-owned fixed insertion tool;
- the project-owned insertion workcell.

The fixed tool is attached to the Panda `hand` body.

The workcell is attached to the Panda world body.

After attachment, the project-owned Panda sensors are added to the
composed `MjSpec` and the final MuJoCo model is compiled.

Adding the sensors after tool attachment is important because the
`panda_wrist_ft` site belongs to the project-owned tool model and must
exist in the composed specification before the force and torque
sensors reference it.

## Sensor data layout

The sensing model contains:

```text
7 joint-position sensors
7 joint-velocity sensors
1 three-axis wrist-force sensor
1 three-axis wrist-torque sensor
```

Therefore:

```text
model.nsensor     = 16
model.nsensordata = 20
```

The application layer does not depend on those raw offsets.

`PandaSensorReader` resolves each required sensor by stable name and
stores its address and dimension internally.

## Reproduction

From the repository root, run the focused sensing tests with:

```bash
uv run pytest tests/unit/test_sensing.py -v
```

Run Ruff with:

```bash
uv run ruff check .
```

Run mypy with:

```bash
uv run mypy robustlearn
```

Run the complete Python test suite with:

```bash
uv run pytest
```

Run the complete local Python validation with:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

## ROS 2 regression validation

The sensing implementation is part of the Python/MuJoCo layer, but
Issue #37 also requires verifying that the existing ROS 2 workspace
continues to build and test successfully.

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash

cd ros2_ws

rosdep install \
  --from-paths src \
  --ignore-src \
  -r \
  -y \
  --rosdistro jazzy

colcon build --symlink-install

source install/setup.bash

colcon test

colcon test-result --verbose
```

The sensing implementation must not introduce regressions into the
existing ROS 2 workspace.

## Scope

This sensing layer intentionally does not define:

- sensor noise or delay randomization;
- the final learned-policy observation schema;
- ROS sensor-message publication;
- force-control policies;
- camera sensing or rendering;
- scripted insertion behavior.

Those capabilities belong to later Week 5 issues.

## Issue #37 acceptance mapping

The implementation satisfies the Issue #37 sensing requirements as
follows:

- seven Panda arm joint-position values are exposed through explicit
  MuJoCo sensors;
- seven Panda arm joint-velocity values are exposed through explicit
  MuJoCo sensors;
- wrist force exposes three values;
- wrist torque exposes three values;
- stable project-owned names are defined centrally;
- raw MuJoCo sensor indexing is hidden behind `PandaSensorReader`;
- `PandaSensorSnapshot` exposes typed sensor arrays;
- all readings are checked for finite values at reset;
- same-seed resets are checked for identical sensor state;
- joint sensors are checked against the corresponding simulator state;
- a controlled external tool load is checked for a measurable wrist
  wrench response;
- clearing the controlled load is checked for restoration of the
  baseline wrench;
- the vendored Panda model remains untouched;
- complete Python and ROS 2 regression validation is required before
  Issue #37 is committed and merged.
