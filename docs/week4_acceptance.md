# Week 4 Deterministic MuJoCo Acceptance Gate

## Goal

Week 4 establishes the first complete deterministic manipulation-simulation
workflow for RobustLearn-Manip.

The milestone integrates:

```text
committed Panda MJCF
        |
        v
committed insertion workcell
        |
        v
MuJoCo model compilation
        |
        v
deterministic simulation wrapper
        |
        v
Gymnasium environment
        |
        v
seeded reset and rollout
```

The Week 4 acceptance criterion is:

> Seeded environment reproduces state exactly.

## Implemented Components

The Week 4 simulation stack consists of:

```text
robot_description/mjcf/franka_emika_panda/
robot_description/mjcf/insertion/

robustlearn/sim/
robustlearn/envs/
```

The main runtime abstractions are:

```text
MuJoCoSimulation
PandaInsertionEnv
```

## Dependencies

Python dependencies are managed through `uv`.

Install the exact locked environment with:

```bash
uv sync --locked
```

MuJoCo and Gymnasium are declared project dependencies in:

```text
pyproject.toml
```

Exact resolved dependency versions are recorded in:

```text
uv.lock
```

No separate manual `pip install mujoco` or `pip install gymnasium` step is
required.

Verify both libraries:

```bash
uv run python - <<'PY'
import gymnasium
import mujoco

print("MuJoCo:", mujoco.__version__)
print("Gymnasium:", gymnasium.__version__)
PY
```

## Panda MJCF Provenance

The Panda physics model is vendored under:

```text
robot_description/mjcf/franka_emika_panda/
```

It originates from the MuJoCo Menagerie Franka Emika Panda model.

The exact upstream revision is recorded in:

```text
robot_description/mjcf/franka_emika_panda/UPSTREAM.md
```

The upstream license is preserved in:

```text
robot_description/mjcf/franka_emika_panda/LICENSE
```

The vendored robot model is kept separate from project-owned insertion task
geometry.

## Insertion Workcell

Project-owned task geometry is stored under:

```text
robot_description/mjcf/insertion/
```

The workcell contains the fixed insertion tool, workstation, fixture, and
receptacle.

Stable task references include:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

## Deterministic Simulation Layer

`MuJoCoSimulation` owns:

```text
MjModel
MjData
reset behavior
physics stepping
project-owned RNG
controlled-state snapshots
```

The deterministic reset API is:

```python
snapshot = simulation.reset(seed=2026)
```

A seeded reset clears previous episode state and reconstructs the canonical
Week 4 starting configuration.

## Gymnasium Environment

`PandaInsertionEnv` exposes the simulation through the standard Gymnasium API:

```python
observation, info = env.reset(seed=2026)

observation, reward, terminated, truncated, info = env.step(action)
```

The Week 4 environment runs headlessly.

Current interface:

```text
action shape:       (8,)
observation shape:  (30,)
```

The current action interface directly exposes the MuJoCo actuator-control
ranges.

The current observation contains:

```text
qpos
qvel
four task-site world positions
```

This is a Week 4 integration interface rather than the final learned-policy
state/action definition.

## Automated Acceptance Test

The complete Week 4 acceptance workflow is implemented in:

```text
tests/integration/test_week4_acceptance.py
```

Run it with:

```bash
uv run pytest \
  tests/integration/test_week4_acceptance.py \
  -v
```

The test verifies that:

- committed Panda assets exist;
- committed insertion workcell assets exist;
- the complete model compiles;
- the environment instantiates headlessly;
- required robot and task elements exist;
- two independent environments reset with the same seed;
- controlled reset snapshots match exactly;
- reset observations match exactly;
- observations remain finite;
- a deterministic fixed action sequence can be executed;
- identical action sequences produce identical transitions;
- simulator state remains finite during the rollout;
- final controlled states match;
- resetting after the rollout reconstructs the initial state.

## Acceptance Workflow

The automated workflow is:

```text
Environment A
    |
    v
reset(seed=2026)
    |
    v
snapshot A


Environment B
    |
    v
reset(seed=2026)
    |
    v
snapshot B


snapshot A == snapshot B
```

The environments then receive the same fixed action sequence:

```text
A + fixed actions
        |
        v
final state A


B + fixed actions
        |
        v
final state B


final state A == final state B
```

All equality checks use exact NumPy array equality for the controlled state.

## Full Python Validation

Run from the repository root:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

The Week 4 acceptance test is included automatically in the normal pytest
suite.

## ROS 2 Regression Validation

Week 4 simulation development must not break the existing ROS 2 stack.

Run:

```bash
source /opt/ros/jazzy/setup.bash

cd ros2_ws

rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro jazzy \
  -r -y

colcon build --symlink-install

source install/setup.bash

colcon test
colcon test-result --verbose
```

The required result is:

```text
0 errors
0 failures
```

## Week 4 Result

The Week 4 milestone demonstrates a reproducible, headless Panda insertion
simulation environment with deterministic seeded reset behavior and
deterministic fixed rollouts.

The key acceptance result is:

```text
same model
+ same explicit seed
+ same reset configuration
+ same action sequence
=
same controlled simulator state
```

This establishes the deterministic simulation foundation required for later
sensing, randomization, evaluation, classical control, and robot-learning
work.

## Deferred

The following are intentionally outside Week 4:

```text
wrist force/torque sensing
camera definitions
final success/failure conditions
task-debug visualization
scripted full insertion completion
domain randomization
final policy state/action API
learning algorithms
ROS 2 MuJoCo hardware execution
```