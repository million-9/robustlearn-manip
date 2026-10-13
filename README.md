## Reproducibility Principle

Detailed fresh-clone setup and validation instructions are available in
[`docs/reproducibility.md`](docs/reproducibility.md).

A project result is not considered complete merely because it works in the original development environment.

# RobustLearn-Manip

Deployment-oriented robot-learning manipulation project for contact-rich assembly.

## Project Goal

RobustLearn-Manip is an end-to-end robotics and robot-learning project built around a Franka Panda-class manipulator performing precision peg/connector insertion under uncertainty.

The project is simulation-first and is being developed incrementally toward a deployment-oriented architecture combining:

- ROS 2 Jazzy
- modern C++
- MuJoCo
- ros2_control
- MoveIt 2
- PyTorch
- Behaviour Cloning
- DAgger
- SAC
- domain randomisation
- ONNX Runtime
- reproducible evaluation and CI

The intended system separates global collision-free motion planning from local contact-rich control. MoveIt 2 will eventually handle the global approach to a pre-insertion pose, while classical and learned controllers will handle the final insertion phase.

## Current Status

**Week 9: validated the complete hybrid MoveIt 2 + classical local insertion
expert workflow through the project-owned MuJoCo ros2_control hardware path.**

RobustLearn-Manip now implements the intended deployment-oriented manipulation
architecture:

    MoveIt 2 global approach
        |
        v
    collision-free staging configuration
        |
        v
    measured handoff verification
        |
        v
    phase-based classical local expert
        |
        v
    contact-aware insertion
        |
        v
    task success

MoveIt handles global collision-aware motion while the local expert owns the
fine-alignment and contact-rich insertion phase.

Both global and local control use the same project-owned runtime stack:

    MoveIt 2 / local expert
            |
            v
    panda_arm_controller
            |
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
    MuJoCo Panda

Completed through Week 9:

- Ubuntu 24.04 LTS development environment configured
- ROS 2 Jazzy installed and verified
- Git/GitHub issue, branch, pull-request, and project-board workflow established
- Python 3.12 environment managed with `uv`
- CMake and `colcon` build workflow verified
- `rosdep` configured
- pytest, Ruff, and mypy configured
- Python CI running in GitHub Actions
- ROS 2 CI running in GitHub Actions
- clean-checkout reproducibility workflow established
- project-specific ROS 2 communication and launch workflows implemented
- Franka Panda URDF/Xacro and MoveIt robot-description foundation established
- Franka Panda MuJoCo model vendored with provenance and licensing preserved
- insertion workcell and stable task geometry implemented
- deterministic MuJoCo simulation and reset layer implemented
- seeded Gymnasium Panda insertion environment implemented
- Panda joint-state, force/torque, RGB, and depth sensing implemented
- deterministic Jacobian-based scripted insertion controller implemented
- explicit task configuration and seeded domain randomization implemented
- stable learned-policy observation and action APIs implemented
- reproducible seeded multi-episode evaluation implemented
- 100-episode Week 6 reproducibility milestone completed
- project-owned `robustlearn_mujoco_hardware` package implemented
- custom `robustlearn_mujoco_hardware/MuJoCoSystem` plugin implemented
- real MuJoCo Panda hardware lifecycle integrated with ros2_control
- MuJoCo position and velocity feedback exposed through ros2_control
- transactional seven-joint position-command path implemented
- real `joint_state_broadcaster` bringup implemented
- real `panda_arm_controller` trajectory controller implemented
- direct `FollowJointTrajectory` execution through MuJoCo validated
- all seven Panda position command interfaces verified claimed during execution
- ROS Panda joints mapped explicitly to MuJoCo joints and actuators
- project-owned URDF-to-MJCF mapping contract implemented and validated
- cross-model forward kinematics validated between MoveIt/URDF and MuJoCo/MJCF
- MoveIt 2 OMPL planning integrated with the real MuJoCo hardware path
- MoveIt receives complete finite Panda state from MuJoCo
- MoveIt-planned trajectories execute through `panda_arm_controller`
- MuJoCo joint feedback verified to change during MoveIt execution
- exactly one real `PandaSystem` hardware component verified active
- `mock_components/GenericSystem` verified absent from the real-MuJoCo path
- existing mock Panda ros2_control bringup preserved
- existing mock MoveIt demo preserved
- vendored MuJoCo Menagerie Panda files remain untouched
- automated Week 8 acceptance runner implemented
- phase-based deterministic classical insertion expert implemented
- explicit expert configuration and phase/state model implemented
- bounded finite local joint commands preserved
- MuJoCo wrist force/torque exposed through ros2_control
- six-axis wrist wrench published through `/wrist_ft/wrench`
- ROS-side classical local insertion expert runner implemented
- local expert consumes ROS joint-state feedback
- local expert consumes ROS wrist force/torque feedback
- local expert commands through `panda_arm_controller`
- direct MuJoCo actuator bypass avoided in the ROS expert path
- force-aware local insertion behavior implemented
- deterministic local-expert timeout and abort behavior implemented
- hybrid MoveIt-to-local-expert coordinator implemented
- project-owned planning-scene workcell representation implemented
- fixed peg represented as an attached planning object
- deterministic MoveIt staging target implemented
- measured MuJoCo staging equilibrium characterized
- explicit measured-state handoff contract implemented
- strict `0.002 rad` maximum handoff error retained
- MoveIt global approach and local contact-rich control separated explicitly
- deterministic planning, execution, handoff, and expert failure paths tested
- 20-case deterministic Week 9 clean acceptance matrix implemented
- per-case planning, execution, handoff, task, and force metrics recorded
- non-finite Week 9 states and metrics rejected
- project-owned Week 9 acceptance runner implemented
- Week 9 minimum clean-success gate of 90% exceeded
- Week 9 target clean-success gate of 95% exceeded
- final Week 9 result reached 20/20 successful scenarios
- final Week 9 clean success rate reached 100%
- complete ROS regression suite passes with zero errors and zero failures

The Week 9 issues were:

    #79  Phase-based classical insertion expert
         COMPLETED

    #80  MuJoCo wrist force/torque through ros2_control
         COMPLETED

    #81  Classical local expert execution through ros2_control
         COMPLETED

    #82  MoveIt global approach -> classical expert handoff
         COMPLETED

    #83  Week 9 deterministic hybrid acceptance workflow
         COMPLETED

The Week 9 milestone gate is:

    MoveIt global approach
        PASS

    measured staging handoff
        PASS

    local classical insertion expert
        PASS

    final insertion task
        PASS

    >= 90% clean-scenario success
        PASS

    >= 95% clean-scenario target
        PASS

    final deterministic acceptance
        20 / 20 successful
        100.0%

The real runtime contract now includes:

    hardware component: PandaSystem
    hardware plugin: robustlearn_mujoco_hardware/MuJoCoSystem
    hardware lifecycle: active
    joint_state_broadcaster: active
    wrist_ft_broadcaster: active
    panda_arm_controller: active
    commanded arm joints: 7
    claimed command interfaces: 7 position interfaces
    wrist wrench topic: /wrist_ft/wrench
    MoveIt planning: success
    MoveIt execution: success
    measured handoff tolerance: 0.002 rad
    local expert execution: success
    GenericSystem in real path: no

The hybrid execution workflow is:

    ros2_ws/src/robustlearn_moveit_config/launch/hybrid_insertion.launch.py

The Week 9 scenario acceptance test is:

    ros2_ws/src/robustlearn_moveit_config/test/test_week9_acceptance_scenario_launch.py

The complete Week 9 acceptance runner is:

    tools/run_week9_acceptance.py

The validated final Week 9 acceptance result is:

    WEEK 9 ACCEPTANCE: PASS

    Scenarios:       20
    Successes:       20
    Failures:         0
    Success rate:   100.0%

    Minimum gate:    PASS
    Target gate:     PASS

The validated final ROS workspace test result is:

    165 tests
    0 errors
    0 failures
    13 skipped

Detailed Week 9 documentation is available in:

    docs/week9_acceptance.md

The Week 8 execution documentation remains available in:

    docs/panda_joint_mapping.md
    docs/cross_model_fk_validation.md
    docs/moveit_mujoco_execution.md
    docs/week8_acceptance.md

The deterministic simulation, task, policy-interface, reproducibility,
MuJoCo-hardware, and MoveIt execution foundations from earlier milestones remain
part of the completed project base.

Work beyond the Week 9 boundary includes:

- formal 200-episode classical expert benchmarking
- episode-management infrastructure
- MCAP/rosbag logging
- dataset writing
- demonstration collection
- keyboard/gamepad teleoperation
- Behaviour Cloning
- DAgger
- SAC
- learned-policy deployment
- large-scale robustness evaluation
- ONNX deployment
- physical-hardware deployment

## Development Environment

- **OS:** Ubuntu 24.04 LTS
- **ROS:** ROS 2 Jazzy
- **Python:** 3.12
- **Python environment:** `uv`
- **C++ build system:** CMake / ament
- **ROS workspace tool:** `colcon`
- **Version control:** Git / GitHub
- **CI:** GitHub Actions

## Repository Structure

Current relevant structure:

```text
robustlearn-manip/
├── .github/
│   └── workflows/
│       ├── python-ci.yml
│       └── ros2-ci.yml
│
├── docs/
│   ├── determinism.md
│   ├── gymnasium_environment.md
│   ├── insertion_task.md
│   ├── reproducibility.md
│   └── week4_acceptance.md
│
├── robot_description/
│   └── mjcf/
│       ├── franka_emika_panda/
│       │   ├── assets/
│       │   ├── panda.xml
│       │   ├── LICENSE
│       │   └── UPSTREAM.md
│       └── insertion/
│           ├── fixed_peg.xml
│           └── workcell.xml
│
├── robustlearn/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── panda_insertion.py
│   └── sim/
│       ├── __init__.py
│       ├── insertion.py
│       ├── panda.py
│       └── simulation.py
│
├── ros2_ws/
│   └── src/
│       ├── robustlearn_interfaces/
│       ├── robustlearn_control/
│       └── robustlearn_moveit_config/
│
├── tests/
│   ├── integration/
│   │   └── test_week4_acceptance.py
│   └── unit/
│       ├── test_import.py
│       ├── test_insertion_model.py
│       ├── test_panda_insertion_env.py
│       ├── test_panda_model.py
│       └── test_simulation.py
│
├── tools/
│   └── view_insertion.py
│
├── pyproject.toml
├── uv.lock
└── README.md
```

The repository separates:

```text
ROS 2 / MoveIt representation
        from
MuJoCo physics / task representation
```

The MJCF model is authoritative for MuJoCo physics and manipulation-task
simulation, while URDF/Xacro remains responsible for the ROS 2, MoveIt 2, and
`ros2_control` software representation.

## Python Environment Setup

Create the virtual environment using Ubuntu's system Python:

```bash
uv venv \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Prevent `colcon` from scanning the virtual environment:

```bash
touch .venv/COLCON_IGNORE
```

Install the locked Python dependencies:

```bash
uv sync --locked
```

## Python Validation

Run the Python quality checks from the repository root:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

These checks are also run automatically by GitHub Actions.

## Week 4 MuJoCo Manipulation Foundation

Week 4 introduces the first complete manipulation-simulation stack in
RobustLearn-Manip.

The implemented pipeline is:

```text
vendored Franka Panda MJCF
          |
          v
project-owned insertion workcell
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
seeded deterministic reset and rollout
```

### Franka Panda MuJoCo Model

The Panda physics model is stored under:

```text
robot_description/mjcf/franka_emika_panda/
```

The model is vendored from MuJoCo Menagerie so that simulation does not depend
on an external checkout or machine-specific paths.

The exact upstream revision is documented in:

```text
robot_description/mjcf/franka_emika_panda/UPSTREAM.md
```

and the upstream license is preserved in:

```text
robot_description/mjcf/franka_emika_panda/LICENSE
```

The vendored Panda model is not modified with project-specific task geometry.

### Insertion Workcell

Insertion-specific geometry is maintained separately under:

```text
robot_description/mjcf/insertion/
```

The current workcell contains:

```text
fixed Panda-mounted insertion tool
workstation
insertion fixture
square receptacle
```

Stable task references include:

```text
peg_tip
receptacle_center
insertion_axis
pre_insertion
```

### Deterministic Simulation

The deterministic MuJoCo wrapper is:

```text
robustlearn.sim.MuJoCoSimulation
```

It owns:

```text
MjModel
MjData
physics stepping
reset behavior
project RNG
controlled-state snapshots
```

Example:

```python
from robustlearn.sim import MuJoCoSimulation

simulation = MuJoCoSimulation()

snapshot = simulation.reset(seed=2026)

simulation.step(5)
```

The reset workflow explicitly clears episode-dependent simulator inputs and
solver warm-start state before reconstructing the canonical start state.

### Gymnasium Environment

The MuJoCo simulator is exposed through:

```text
robustlearn.envs.PandaInsertionEnv
```

Example:

```python
from robustlearn.envs import PandaInsertionEnv

env = PandaInsertionEnv()

observation, info = env.reset(seed=2026)

action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

The current Week 4 environment has:

```text
action shape:       (8,)
observation shape:  (30,)
```

The action vector currently corresponds to the MuJoCo actuator controls.

The observation contains:

```text
qpos
qvel
four task-site world positions
```

This is intentionally a Week 4 scaffold and is not yet the final learned-policy
state/action interface.

### Week 4 Determinism Acceptance Gate

The complete acceptance test is:

```text
tests/integration/test_week4_acceptance.py
```

Run it with:

```bash
uv run pytest \
  tests/integration/test_week4_acceptance.py \
  -v
```

The test creates two independent environments:

```text
Environment A                 Environment B
      |                             |
      v                             v
reset(seed=2026)              reset(seed=2026)
      |                             |
      v                             v
snapshot A                    snapshot B
      |                             |
      +------------ == -------------+
```

The same deterministic action sequence is then applied to both environments.

The acceptance requirement is:

```text
same model
+ same seed
+ same reset configuration
+ same action sequence
=
same controlled simulator state
```

All controlled reset-state comparisons use exact NumPy array equality.

The environment also verifies that simulation state remains finite throughout
the acceptance rollout.

More details are available in:

```text
docs/determinism.md
docs/insertion_task.md
docs/gymnasium_environment.md
docs/week4_acceptance.md
```

## ROS 2 Workspace Setup

Source ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

Move into the ROS workspace:

```bash
cd ros2_ws
```

Install package dependencies:

```bash
rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro jazzy \
  -r -y
```

Build the workspace:

```bash
colcon build --symlink-install
```

Source the workspace overlay:

```bash
source install/setup.bash
```

Run ROS tests:

```bash
colcon test
colcon test-result --verbose
```

The expected result is zero errors and zero failures.

## ROS 2 Status Demo

The `robustlearn_control` package currently provides the first project-specific ROS 2 C++ communication workflow.

The system contains:

```text
/status_publisher
       |
       | std_msgs/msg/String
       v
/system_status
       ^
       |
       |
/status_subscriber
```

The publisher sends status messages over `/system_status`, while the subscriber receives and logs them.

### Run the Publisher

After building and sourcing the workspace:

```bash
ros2 run robustlearn_control status_publisher
```

By default, the publisher sends a message approximately once per second.

### Run the Subscriber

In another sourced terminal:

```bash
ros2 run robustlearn_control status_subscriber
```

The subscriber should receive the messages published on `/system_status`.

### Inspect the ROS Graph

List running nodes:

```bash
ros2 node list
```

Inspect the publisher:

```bash
ros2 node info /status_publisher
```

Inspect the topic:

```bash
ros2 topic info /system_status
```

Echo messages directly:

```bash
ros2 topic echo /system_status
```

With both nodes running, `/system_status` should report one publisher and one subscriber.

## ROS 2 Parameters

The publisher is configurable through ROS 2 parameters.

Current parameters:

- `message_prefix` — text prepended to each generated status message
- `publish_period_ms` — publication interval in milliseconds

The default configuration preserves the original publisher behaviour.

### Command-Line Parameter Overrides

For example:

```bash
ros2 run robustlearn_control status_publisher \
  --ros-args \
  -p message_prefix:="Robot OK " \
  -p publish_period_ms:=500
```

This runs the same compiled executable with a different message prefix and a 500 ms publication period.

No recompilation is required.

Inspect the active values:

```bash
ros2 param get /status_publisher message_prefix
ros2 param get /status_publisher publish_period_ms
```

The publisher rejects invalid non-positive publication periods rather than creating an invalid timer configuration.

## YAML Parameter Configuration

The publisher configuration is also stored in:

```text
ros2_ws/src/robustlearn_control/config/status_publisher.yaml
```

Current configuration:

```yaml
status_publisher:
  ros__parameters:
    message_prefix: "RobustLearn configured status "
    publish_period_ms: 750
```

The configuration can be loaded directly:

```bash
ros2 run robustlearn_control status_publisher \
  --ros-args \
  --params-file src/robustlearn_control/config/status_publisher.yaml
```

This separates runtime configuration from compiled C++ behaviour.

## ROS 2 Launch

The publisher and subscriber can be started together using the project launch file:

```bash
ros2 launch robustlearn_control status_system.launch.py
```

The launch file:

- starts `status_publisher`
- starts `status_subscriber`
- loads the publisher YAML parameter configuration
- sends node output to the terminal

This provides the first small multi-node ROS 2 system in the project.

Conceptually:

```text
status_system.launch.py
        |
        +---- status_publisher.yaml
        |             |
        |             v
        +---- /status_publisher
        |             |
        |             v
        |       /system_status
        |             |
        |             v
        +---- /status_subscriber
```
## ROS 2 Communication Patterns

RobustLearn-Manip currently demonstrates three ROS 2 communication patterns: topics, services, and actions.

### Topics

Topics are used for continuous or event-driven streams of data.

Current example:

```text
/status_publisher
       |
       | std_msgs/msg/String
       v
/system_status
       |
       v
/status_subscriber
```

Topics are appropriate for data such as:

- joint states
- force/torque measurements
- camera frames
- robot status
- sensor streams

A topic publisher sends data without waiting for a response from each subscriber.

### Services

Services provide request-response communication.

The project defines:

```text
robustlearn_interfaces/srv/SetSystemMode
```

Interface:

```text
string mode
---
bool success
string message
```

The communication pattern is:

```text
system_mode_client
        |
        | request
        v
/set_system_mode
        |
        v
system_mode_server
        |
        | response
        v
system_mode_client
```

Example request:

```text
mode: "READY"
```

Example response:

```text
success: true
message: "System mode set to READY"
```

Services are appropriate for relatively short operations where the caller expects a direct response, such as configuration changes, resets, or state queries.

### Actions

Actions support longer-running operations with goal acceptance, intermediate feedback, final results, and cancellation.

The project defines:

```text
robustlearn_interfaces/action/ExecuteSystemCheck
```

Interface:

```text
int32 total_steps
---
bool success
string message
---
int32 completed_steps
float32 progress
```

The communication pattern is:

```text
system_check_action_client
            |
            | goal
            v
/execute_system_check
            |
            v
system_check_action_server
            |
            +---- feedback
            +---- feedback
            +---- feedback
            |
            +---- result
```

The server rejects goals with non-positive step counts.

Accepted goals publish progress feedback while executing and return a final success result when complete.

Clients can also request cancellation while an action is executing.

Actions are appropriate for operations such as:

- trajectory execution
- motion to a target pose
- insertion procedures
- calibration routines
- other operations that take time and may need cancellation

### Topic vs Service vs Action

| Pattern | Communication | Feedback | Result | Cancellation | Typical use |
|---|---|---|---|---|---|
| Topic | streaming | no direct request feedback | no | no | sensors, status, joint states |
| Service | request-response | no intermediate feedback | yes | generally no | reset, configure, query |
| Action | goal-based asynchronous operation | yes | yes | yes | motion, trajectory execution, long-running tasks |
## Week 10 MCAP Episode Logging

Hybrid manipulation episodes can be wrapped with project-owned ROS 2
`rosbag2` logging using MCAP storage.

The logging path is implemented in:

    robustlearn/mcap_logging.py

Logging is explicitly enabled or disabled through `McapLoggingConfig`.
When disabled, the underlying episode backend executes unchanged.

A recorded episode uses the deterministic layout:

    <output_root>/
      <run_id>/
        episode_<index>__seed_<seed>__<scenario>/
          mcap/
            metadata.yaml
            mcap_0.mcap

Each recording stores metadata for:

- episode index
- seed
- scenario
- git commit
- recording timestamp
- ROS domain ID

The default MCAP topic set is:

    /joint_states
    /wrist_ft/wrench
    /local_insertion_expert/status
    /panda_arm_controller/controller_state
    /execute_trajectory/_action/feedback
    /execute_trajectory/_action/status
    /panda_arm_controller/follow_joint_trajectory/_action/feedback
    /panda_arm_controller/follow_joint_trajectory/_action/status

The action feedback and status streams are hidden ROS 2 topics, so the recorder
uses `--include-hidden-topics`.

The recorder is started before episode execution and is shut down in a
`finally` path so episode failures do not leave the recorder running.

### Inspect a Recording

Set the bag directory:

    BAG_DIR="build/week10_mcap_smoke/smoke_run/episode_000000__seed_0__week9_case_01/mcap"

Inspect its metadata and topic counts:

    ros2 bag info "$BAG_DIR"

List only recorded topic names:

    ros2 bag info -t "$BAG_DIR"

The project validator can also verify that the bag is readable, uses MCAP,
contains messages, and contains every required topic:

    uv run python -c '
    from pathlib import Path
    from robustlearn.mcap_logging import validate_mcap_bag
    result = validate_mcap_bag(
        Path(
            "build/week10_mcap_smoke/"
            "smoke_run/"
            "episode_000000__seed_0__week9_case_01/"
            "mcap"
        )
    )
    print(result)
    '

### Replay a Recording

Source ROS 2 and the workspace before replay:

    source /opt/ros/jazzy/setup.bash
    source ros2_ws/install/setup.bash

Replay the bag:

    ros2 bag play "$BAG_DIR"

If the episode was recorded on an isolated ROS domain, select the desired
domain before replay:

    export ROS_DOMAIN_ID=190
    ros2 bag play "$BAG_DIR"

MCAP recordings are intended for runtime inspection, debugging, and replay.
They are separate from the training-dataset representation used by later
learning workflows.

## Week 10 Structured Expert Rollout Dataset Writer

Expert rollout data has a project-owned, versioned structured schema in:

    robustlearn/rollout_dataset.py

This layer is intentionally separate from MCAP logging.

MCAP records the ROS 2 runtime for debugging and replay. The rollout dataset
writer stores structured state/action examples intended for later imitation
learning and dataset collection workflows.

### Episode Schema

Each rollout episode records:

- schema version
- episode index
- seed
- scenario
- source
- task
- success/failure
- explicit failure reason for failed episodes
- start and end timestamps
- git commit
- configuration identifier
- an explicitly bounded sequence of frames

One episode is written to one deterministic JSON file:

    episode_<index>__seed_<seed>__<scenario>.json

Failed or incomplete episodes may contain zero frames. They are still written
with their failure reason rather than being silently dropped.

### Frame Schema

Each frame records:

- frame index
- timestamp
- 7 Panda joint positions
- optional 7 Panda joint velocities
- task-relative state:
  - lateral error
  - axial offset
  - insertion depth
- 6-axis wrist wrench:
  - force x/y/z
  - torque x/y/z
- 7-dimensional expert joint-delta action
- task success flag
- task failure flag
- termination flag

All supplied numeric values must be finite, and fixed-size state/action fields
are shape validated.

Joint velocity is optional because it is available from the simulator sensing
path but is not yet guaranteed by every live expert rollout path.

### Writer and Reader

A validated episode can be written with:

    from pathlib import Path
    from robustlearn.rollout_dataset import write_episode

    path = write_episode(
        Path("build/expert_rollouts"),
        episode,
    )

The same file can be read and validated with:

    from robustlearn.rollout_dataset import read_episode

    episode = read_episode(path)

Serialization uses deterministic JSON formatting so identical supplied episode
records produce identical file contents.

### Week 11 Demonstration Collection

Week 11 collection will build on this writer rather than defining another
dataset representation.

The collection layer will be responsible for constructing `RolloutFrame`
records from the live manipulation signals, including joint state, wrist
wrench, task-relative geometry, and classical expert action, then passing the
completed episode to this writer.

Dataset v0.1 freezing, train/validation/test splitting, statistics generation,
and the large demonstration collection campaign remain separate later
milestones.

## Continuous Integration

Two GitHub Actions workflows currently validate the repository.

### Python CI

The Python workflow performs:

```text
dependency installation
        |
        v
      Ruff
        |
        v
      mypy
        |
        v
     pytest
```

### ROS 2 CI

The ROS 2 workflow runs on Ubuntu 24.04 with ROS 2 Jazzy and performs:

```text
repository checkout
        |
        v
ROS 2 Jazzy setup
        |
        v
rosdep dependency installation
        |
        v
colcon build
        |
        v
colcon test
        |
        v
colcon test-result
```

Generated workspace artifacts such as `build/`, `install/`, and `log/` are local build products and are not committed.

## Development Workflow

Project work is organized through GitHub Issues and short-lived feature branches.

The intended workflow is:

```text
GitHub Issue
     |
     v
feature branch from main
     |
     v
implementation
     |
     v
local build and tests
     |
     v
commit and push
     |
     v
Pull Request
     |
     v
GitHub Actions CI
     |
     v
merge into main
     |
     v
close issue
     |
     v
delete feature branch
```

Each issue should represent a focused, reviewable unit of work rather than mixing unrelated changes into one branch.

## Planned Project Architecture

The project will progressively add:

1. Franka Panda robot descriptions for ROS 2 and MuJoCo
2. deterministic MuJoCo manipulation environment
3. joint, wrench, and camera sensing
4. randomisation and evaluation infrastructure
5. `ros2_control` integration
6. C++ MuJoCo hardware interface
7. MoveIt 2 global motion planning
8. classical contact-rich insertion controller
9. episode management and demonstration logging
10. Behaviour Cloning
11. DAgger
12. SAC reinforcement learning
13. domain randomisation and robustness evaluation
14. ONNX policy export and ROS inference
15. runtime latency and safety evaluation
16. Docker-based reproducibility
17. optional secondary Isaac validation

The target architecture is:

```text
MoveIt 2 / Learned Policy
           |
           v
     ros2_control
           |
           v
 MuJoCo Hardware Interface
           |
           v
        MuJoCo
           |
           +---- joint state
           +---- force / torque
           +---- RGB-D
           |
           v
        ROS 2
```

Training and evaluation infrastructure will later operate alongside this runtime system using PyTorch, experiment tracking, dataset tooling, and reproducible benchmark configurations.

## Project Scope

The flagship task is precision peg/connector insertion under uncertainty.

The intended control decomposition is:

```text
random robot configuration
          |
          v
MoveIt 2 global planning
          |
          v
safe pre-insertion pose
          |
          v
local contact-rich controller
          |
          v
successful insertion
```

The learned policy will not initially be responsible for arbitrary global robot motion. Global collision-free motion and local uncertain contact control are intentionally separated.

The initial learned policy will be state-based. Vision-conditioned policies, diffusion policies, VLAs, and sensor-fusion extensions are outside the core flagship scope and will only be considered after the core system is complete.

## Reproducibility Principle

A project result is not considered complete merely because it works in the original development environment.

Important project milestones should eventually be reproducible through:

- pinned dependencies
- deterministic configuration where applicable
- explicit random seeds
- clean-checkout builds
- automated tests
- GitHub Actions
- documented commands
- Docker
- versioned experiment configurations
- recorded benchmark methodology

Claims about task success, robustness, latency, or learning performance will only be added after they have been experimentally measured.

## Project Status

RobustLearn-Manip has completed the **Week 4 deterministic MuJoCo manipulation
foundation**.

The repository now demonstrates:

```text
ROS 2 communication foundations
        |
        v
Franka Panda robot representations
        |
        v
MuJoCo Panda physics model
        |
        v
insertion workcell geometry
        |
        v
deterministic simulation wrapper
        |
        v
Gymnasium manipulation environment
        |
        v
deterministic Week 4 acceptance workflow
```

The Week 4 milestone is considered successful when:

```text
Seeded environment reproduces state exactly.
```

That behavior is covered by automated regression testing.

Current implemented simulation capabilities include:

- committed Panda MJCF assets
- traceable MuJoCo Menagerie provenance
- committed insertion workcell assets
- headless MuJoCo model compilation and stepping
- deterministic seeded simulator reset
- controlled-state snapshots
- project-owned NumPy RNG
- Gymnasium reset and step APIs
- explicit action and observation spaces
- explicit episode truncation
- deterministic fixed rollouts
- finite-state smoke validation
- automated Week 4 integration acceptance testing

The project has **not yet** implemented:

- wrist force/torque sensing
- RGB or depth cameras
- final task success/failure logic
- full scripted insertion completion
- domain randomization
- formal task configuration schemas
- final learned-policy state/action interfaces
- `ros2_control` execution backed by MuJoCo
- MoveIt trajectory execution in MuJoCo
- behavior cloning
- DAgger
- SAC reinforcement learning
- ONNX policy deployment
- final task-success, robustness, or latency benchmarks

Those capabilities belong to later milestones and will be added only after
their own acceptance criteria are satisfied.

The project continues to follow the principle that capabilities are not treated
as complete until they are reproducible, tested, and documented.