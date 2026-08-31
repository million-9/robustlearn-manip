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

**Week 2: ROS 2 C++ communication and runtime configuration.**

Completed so far:

- Ubuntu 24.04 LTS development environment configured
- ROS 2 Jazzy installed and verified
- Git/GitHub issue, branch, pull-request, and project-board workflow established
- Python 3.12 environment managed with `uv`
- CMake and `colcon` build workflow verified
- `rosdep` configured
- Python package initialized
- pytest, Ruff, and mypy configured
- Python CI running in GitHub Actions
- ROS 2 CI running in GitHub Actions
- clean-checkout reproducibility verified
- initial `robustlearn_interfaces` ROS 2 package created
- `robustlearn_control` C++ ROS 2 package created
- C++ ROS 2 publisher and subscriber implemented
- publisher/subscriber communication verified through ROS 2 CLI tools
- ROS 2 runtime parameters added to the publisher
- command-line parameter overrides verified
- YAML parameter configuration added
- ROS 2 launch file added to start the publisher and subscriber together
- parameter validation added for invalid publication periods
- ROS 2 build, lint, and test checks passing

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
├── robustlearn/
├── ros2_ws/
│   └── src/
│       ├── robustlearn_interfaces/
│       └── robustlearn_control/
│           ├── config/
│           │   └── status_publisher.yaml
│           ├── include/
│           │   └── robustlearn_control/
│           ├── launch/
│           │   └── status_system.launch.py
│           ├── src/
│           │   ├── status_publisher.cpp
│           │   └── status_subscriber.cpp
│           ├── CMakeLists.txt
│           └── package.xml
├── tests/
├── pyproject.toml
└── README.md
```

The repository will expand as simulation, robot control, planning, learning, deployment, and evaluation components are introduced.

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

RobustLearn-Manip is currently in the **ROS 2 foundations phase**.

The repository now demonstrates working project-specific C++ ROS 2 communication, runtime parameter configuration, YAML configuration, launch-based multi-node execution, automated build/test validation, and a reproducible Git/GitHub development workflow.

MuJoCo manipulation, `ros2_control`, MoveIt 2, learning algorithms, domain randomisation, ONNX deployment, and final performance benchmarks have **not yet been implemented** and should not be interpreted as completed capabilities.

The project is being developed incrementally, with each capability required to pass its acceptance gate before later stages are treated as complete.