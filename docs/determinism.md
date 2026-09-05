# MuJoCo Determinism Contract

This document defines the Week 4 determinism contract for the
RobustLearn-Manip MuJoCo simulation layer.

The goal is to make simulator reset behavior explicit, reproducible, and
independent of state left behind by a previous episode.

The implementation is provided by:

```text
robustlearn/sim/simulation.py
```

The primary public classes are:

```text
MuJoCoSimulation
SimulationSnapshot
```

## Purpose

Robot-learning experiments depend on repeatable simulator initialization.

A reset must not accidentally inherit:

- joint positions from a previous rollout
- joint velocities from a previous rollout
- control commands
- applied forces
- solver warm-start state
- actuator state
- mocap state
- simulation time
- random-number-generator history when an explicit seed is supplied

For Week 4, the required deterministic relationship is:

```text
same model
+ same explicit seed
+ same reset configuration
=
same controlled initial state
```

The initial Week 4 task configuration is mostly fixed.

Large randomized start distributions are intentionally deferred to the later
domain-randomization milestone.

The seed mechanism is implemented now so future randomization can use the same
reset API without redesigning simulator ownership.

## Simulation Ownership

`MuJoCoSimulation` centralizes ownership of the simulation resources required
by later environments and learning code.

It owns:

```text
MjModel
MjData
physics timestep
simulation stepping
reset behavior
NumPy random generator
last explicit seed
```

This avoids spreading reset logic and random-number generation across
controllers, environments, training scripts, and tests.

## Canonical Initial Configuration

The canonical Week 4 initial state uses the Panda `home` keyframe inherited
from the vendored MuJoCo Menagerie Panda model.

The insertion workcell introduced in Issue #28 is positioned relative to that
configuration so that the fixed insertion peg begins at the nominal
pre-insertion reference.

The canonical reset therefore starts from:

```text
Panda:          home keyframe
peg:            fixed to Panda hand
workstation:    fixed to world
fixture:        fixed to world
receptacle:     fixed to world
external force: zero
simulation time: zero
```

Task geometry is not randomized in Week 4.

## Reset API

The reset entry point is:

```python
snapshot = simulation.reset(seed=42)
```

An explicit seed constructs a project-owned NumPy random generator using:

```python
np.random.default_rng(seed)
```

The generator belongs to the simulation instance.

No global NumPy RNG is used by the reset layer.

## Explicit Seed Semantics

Calling:

```python
simulation.reset(seed=42)
```

reconstructs both:

1. the canonical controlled MuJoCo state
2. the project-owned RNG stream associated with seed `42`

Repeated calls with the same explicit seed therefore reconstruct the same RNG
sequence.

For example:

```python
simulation.reset(seed=42)
first = simulation.rng.random(10)

simulation.reset(seed=42)
second = simulation.rng.random(10)
```

must produce identical `first` and `second` arrays.

Two independent simulator instances reset with the same explicit seed must
also produce matching RNG streams.

## `seed=None`

Calling:

```python
simulation.reset()
```

still reconstructs the canonical MuJoCo state.

However, it does **not** reseed the project-owned random generator.

This behavior is intentional.

An explicit seed is required when reproducibility of the RNG stream itself is
part of the experiment contract.

Therefore, deterministic experiment entry points should provide an explicit
seed.

## Reset Procedure

The current reset sequence is conceptually:

```text
reset(seed)
    |
    +-- optionally reconstruct project RNG from explicit seed
    |
    +-- restore Panda home keyframe
    |
    +-- clear externally applied generalized forces
    |
    +-- clear externally applied Cartesian body forces
    |
    +-- clear solver warm-start acceleration state
    |
    +-- run MuJoCo forward computation
    |
    +-- return independent SimulationSnapshot
```

The reset uses MuJoCo's keyframe reset functionality rather than manually
assigning only the arm joint positions.

This restores the keyframe-controlled MuJoCo state consistently.

## Controlled State

The Week 4 determinism contract explicitly includes the following state.

### Simulation Time

```text
time
```

Reset reconstructs simulation time at the canonical start.

A rollout must therefore not cause the following episode to begin at the
previous episode's simulation time.

### Generalized Positions

```text
qpos
```

This currently contains:

- seven Panda arm joint positions
- two Panda finger joint positions

The insertion peg does not add a free joint because it is fixed to the Panda
hand.

The workcell is fixed to the world.

### Generalized Velocities

```text
qvel
```

Reset restores the canonical velocity state rather than inheriting velocity
from the previous rollout.

### Actuator Activation State

```text
act
```

The current Panda model has no actuator activation variables, so this array is
currently empty.

It remains part of the snapshot contract so the simulation layer continues to
handle actuator activation state if a future model uses it.

### Control Inputs

```text
ctrl
```

The current model contains eight actuators:

```text
7 Panda arm actuators
1 Panda gripper actuator
```

Control state is restored from the canonical keyframe.

Previous episode control commands must not leak through reset.

### Mocap State

```text
mocap_pos
mocap_quat
```

The current insertion model contains no mocap bodies, so these arrays are
currently empty.

They are included in the snapshot contract for forward compatibility.

### Externally Applied Generalized Forces

```text
qfrc_applied
```

These values are explicitly cleared during reset.

A disturbance or experimental force applied during one episode must not leak
into the next episode.

### Externally Applied Cartesian Body Forces

```text
xfrc_applied
```

These values are also explicitly cleared during reset.

### Solver Warm-Start State

The reset explicitly clears:

```text
qacc_warmstart
```

This is solver auxiliary state that can depend on previous simulation history.

It is intentionally cleared to avoid previous-episode solver state affecting
the newly reconstructed episode.

`qacc_warmstart` is not currently exposed as part of
`SimulationSnapshot`, because the public snapshot focuses on the
experiment-controlled state used for deterministic comparison.

Its reset behavior is nevertheless covered by automated tests.

## Task Configuration

The Week 4 snapshot also records the world positions of the stable task sites:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

These positions are exposed through:

```text
task_site_xpos
```

This allows deterministic tests to verify not only the underlying Panda state,
but also the resulting task-space configuration.

For the current fixed workcell this provides a compact check that the
Panda/tool/workcell composition reconstructs the same nominal task state.

## Simulation Snapshot

`SimulationSnapshot` is an immutable dataclass containing independent copies of
the determinism-relevant arrays.

The snapshot currently contains:

```text
time
qpos
qvel
act
ctrl
mocap_pos
mocap_quat
qfrc_applied
xfrc_applied
task_site_xpos
```

Array values are copied when the snapshot is created.

Changing the live `MjData` after taking a snapshot must therefore not mutate
the previously captured snapshot.

This is important for regression testing and later trajectory comparisons.

## Deterministic Reset Requirements

The automated tests verify the following properties.

### Independent Simulator Equivalence

Two separately created simulations:

```python
sim_a = MuJoCoSimulation()
sim_b = MuJoCoSimulation()
```

reset with:

```python
sim_a.reset(seed=42)
sim_b.reset(seed=42)
```

must produce matching controlled snapshots.

### Repeated Same-Seed Reset

Repeated calls such as:

```python
simulation.reset(seed=123)
simulation.reset(seed=123)
simulation.reset(seed=123)
```

must reproduce the same controlled initial state.

### Previous-Episode Independence

Tests deliberately modify state including:

```text
simulation time
qpos
qvel
ctrl
qfrc_applied
xfrc_applied
qacc_warmstart
```

and actuator or mocap state where available.

A subsequent explicit reset must reconstruct the canonical state rather than
preserving any of those modifications.

### Rollout Independence

The simulation is stepped for hundreds of physics steps.

A subsequent same-seed reset must match the baseline snapshot that existed
before the rollout.

### RNG Reproducibility

The same explicit seed must reconstruct the same project-owned NumPy RNG
stream.

Two independent simulator instances must also produce identical draws when
initialized with the same explicit seed.

Different explicit seeds must produce different RNG streams.

### Simulation After Reset

A reset must leave the simulator in a valid state that can immediately be
stepped normally.

The simulation time must advance according to:

```text
number of physics steps x MuJoCo timestep
```

## What Same Seed Does Not Yet Mean

Week 4 does not yet implement large randomized reset distributions.

Therefore:

```python
simulation.reset(seed=1)
simulation.reset(seed=2)
```

currently produce the same nominal Panda and task geometry.

Their project-owned RNG streams are different, but the initial physical state
is intentionally mostly fixed.

This is not a limitation of the seed API.

It establishes the deterministic mechanism that later randomization will use.

## Future Domain Randomization

Later domain-randomization code must use:

```text
simulation.rng
```

rather than:

```text
np.random.*
Python global random state
ad-hoc independently seeded generators
```

This preserves a single explicit source of randomness for reset-time
randomization.

A future reset sequence can therefore extend the current mechanism:

```text
reset(seed)
    |
    +-- reconstruct RNG
    |
    +-- restore canonical state
    |
    +-- sample randomized task parameters from simulation.rng
    |
    +-- apply sampled parameters
    |
    +-- run forward computation
    |
    +-- return snapshot
```

The deterministic invariant remains:

```text
same model
+ same seed
+ same reset options
=
same randomized initial state
```

## Explicitly Outside the Week 4 Contract

The current deterministic-reset layer does not implement:

- full domain randomization
- large randomized joint configurations
- randomized peg/receptacle geometry
- sensor noise
- camera noise
- stochastic policies
- dataset logging
- evaluation sweeps
- ROS integration
- cross-platform bitwise equivalence guarantees

The current acceptance target is deterministic reconstruction within the
supported project software environment.

## Validation

The deterministic simulation tests are located in:

```text
tests/unit/test_simulation.py
```

Current Python validation is run with:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

The tests cover:

- independent same-seed simulator equivalence
- repeated same-seed resets
- corrupted-state recovery
- rollout-then-reset recovery
- project-owned RNG reproducibility
- distinct RNG streams for different seeds
- stepping after reset
- invalid step-count handling
- snapshot independence from live simulator state
- solver warm-start clearing