# Gymnasium Panda Insertion Environment

This document describes the Week 4 Gymnasium environment scaffold for the
RobustLearn-Manip Panda insertion task.

The environment provides a standards-compliant Gymnasium interface around the
deterministic MuJoCo simulation layer.

Implementation:

```text
robustlearn/envs/panda_insertion.py
```

Public environment class:

```text
PandaInsertionEnv
```

## Purpose

The Week 4 environment establishes the boundary between the MuJoCo simulator
and later control or learning algorithms.

The architecture is:

```text
MuJoCo Panda + insertion workcell
            |
            v
    MuJoCoSimulation
            |
            v
    PandaInsertionEnv
            |
            v
later controllers / RL / IL
```

The environment is intentionally minimal.

It is not yet the final policy interface.

## Gymnasium API

The environment follows the standard Gymnasium API:

```python
observation, info = env.reset(seed=seed, options=options)
```

and:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

## Headless Execution

The environment does not require a viewer.

It can be instantiated and stepped in a headless process:

```python
from robustlearn.envs import PandaInsertionEnv

env = PandaInsertionEnv()

observation, info = env.reset(seed=42)

action = env.action_space.sample()

result = env.step(action)

env.close()
```

Interactive MuJoCo visualization remains separate from the environment itself.

## Action Space

The Week 4 action space exposes the eight MuJoCo actuator control values
directly.

The current actuator layout is:

```text
7 Panda arm actuators
1 Panda gripper actuator
```

The action space is a Gymnasium `Box` with shape:

```text
(8,)
```

and dtype:

```text
float64
```

The lower and upper bounds are taken directly from the compiled MuJoCo model's
actuator control ranges.

This is intentionally a low-level interface.

The Week 4 environment does not yet define the final learned-policy action
space.

In particular, this is not yet a normalized Cartesian or task-space control
interface.

## Observation Space

The Week 4 observation contains:

```text
qpos
qvel
task-site world positions
```

The current dimensions are:

```text
qpos:                 9
qvel:                 9
task-site positions: 12
                      --
total:                30
```

The task-site positions correspond to the four stable task references:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

Each site contributes three Cartesian world coordinates.

The final observation shape is therefore:

```text
(30,)
```

with dtype:

```text
float64
```

## Observation Ordering

The current observation vector is constructed as:

```text
[
    qpos,
    qvel,
    flattened task_site_xpos,
]
```

The task-site order follows the simulation-layer contract:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

This ordering is explicit for Week 4 testing.

It should not yet be treated as the permanent learned-policy state schema.

## Deterministic Reset

The environment delegates reset behavior to:

```text
MuJoCoSimulation.reset(seed=...)
```

The environment does not duplicate MuJoCo reset logic.

Calling:

```python
env.reset(seed=42)
```

therefore uses the deterministic simulator reset mechanism implemented in
Issue #29.

Two independently created environments reset with the same seed reproduce the
same controlled initial observation.

The automated tests also verify that applying the same action after the same
seeded reset produces the same next observation and transition metadata.

## Gymnasium Seed Semantics

The environment calls:

```python
super().reset(seed=seed)
```

to follow Gymnasium's standard seeding workflow.

The same explicit seed is also passed to the RobustLearn simulation layer.

This keeps Gymnasium-level and simulation-level reset semantics aligned.

## Physics Stepping

Each environment step advances a fixed number of MuJoCo physics steps.

The default configuration is:

```text
frame_skip = 5
```

With the current MuJoCo timestep:

```text
0.002 s
```

one Gymnasium environment step therefore advances approximately:

```text
5 x 0.002 s = 0.01 s
```

The exact environment time advancement is determined by:

```text
frame_skip x simulation.timestep
```

## Reward

The current Week 4 reward is:

```text
0.0
```

for every step.

This is intentional.

The purpose of Issue #30 is to establish the environment API, not to commit to
the final insertion reward function.

Reward shaping and final insertion objectives belong to later milestones.

## Termination

The current environment does not yet implement final insertion success or
failure conditions.

Therefore:

```text
terminated = False
```

during the Week 4 scaffold.

Task-specific termination semantics are intentionally deferred.

## Truncation

The environment has an explicit episode step limit.

Default:

```text
max_episode_steps = 200
```

The environment tracks:

```text
elapsed_steps
```

and returns:

```text
truncated = True
```

when:

```text
elapsed_steps >= max_episode_steps
```

This provides a valid episode boundary even before final task success/failure
logic exists.

## Info Dictionary

The current `info` dictionary contains:

```text
simulation_time
elapsed_steps
```

Example:

```python
{
    "simulation_time": 0.01,
    "elapsed_steps": 1,
}
```

These values are intended for debugging and validation.

## Action Validation

Actions must belong to the declared Gymnasium action space.

An out-of-range action raises:

```text
ValueError
```

rather than silently clipping the command.

This makes invalid control commands visible during development and testing.

## Configuration Validation

The environment validates:

```text
frame_skip >= 1
max_episode_steps >= 1
```

Invalid configuration raises `ValueError`.

The current Week 4 implementation also requires all MuJoCo actuator control
ranges to be finite.

## Gymnasium Environment Checker

The environment is tested using:

```python
from gymnasium.utils.env_checker import check_env
```

with rendering checks skipped because the environment is intentionally
headless.

The checker currently reports warnings that:

- the action range is not normalized
- the observation bounds are infinite

These warnings are expected for the Week 4 scaffold.

The action space directly reflects MuJoCo actuator limits rather than the final
policy interface.

The observation bounds are deliberately broad because a formal policy-state
schema and bounded observation normalization are deferred to later milestones.

## Automated Tests

The environment tests are located in:

```text
tests/unit/test_panda_insertion_env.py
```

They verify:

```text
Gymnasium environment checker compatibility
explicit action space
explicit observation space
reset return structure
step five-element return structure
observation shape
observation dtype
finite observations
valid observation-space membership
valid action-space membership
physics-time advancement
same-seed reset determinism
same-seed transition determinism
reset after rollout
episode truncation
invalid-action rejection
invalid frame-skip rejection
invalid episode-length rejection
```

## Current Week 4 Interface Contract

The environment currently guarantees:

```text
action shape:       (8,)
action dtype:       float64

observation shape:  (30,)
observation dtype:  float64

reward:             float
terminated:         bool
truncated:          bool
info:               dict
```

Observations returned by `reset()` and `step()` contain finite numeric values
and belong to the declared observation space.

## Explicitly Out of Scope

The Week 4 Gymnasium scaffold does not yet implement:

- final insertion reward design
- final success conditions
- final failure conditions
- Cartesian policy actions
- normalized policy action space
- wrist force/torque observations
- RGB observations
- depth observations
- domain randomization
- observation normalization
- Stable-Baselines3 training
- behavior cloning
- ROS 2 execution

Those capabilities will build on this environment boundary in later project
milestones.