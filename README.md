# RobustLearn-Manip

Deployment-oriented robot-learning manipulation project for contact-rich assembly.

## Project Goal

Build an end-to-end manipulation system around a Franka Panda-class robot for precision peg/connector insertion under uncertainty.

The planned system will progressively integrate:

- ROS 2 Jazzy
- MuJoCo
- ros2_control
- MoveIt 2
- PyTorch
- Behaviour Cloning
- DAgger
- SAC
- domain randomisation
- ONNX Runtime

## Current Status

Week 1: development environment and repository bootstrap.

Completed so far:

- Ubuntu 24.04 LTS installed
- ROS 2 Jazzy installed and verified
- C++ ROS talker and Python ROS listener communication verified
- rosdep configured
- Git repository initialized
- Python 3.12 environment managed with uv
- ROS Python packages accessible inside the virtual environment
- CMake and colcon verified
- Initial ROS 2 package created
- Python package initialized
- pytest, Ruff, and mypy configured

## Development Environment

- OS: Ubuntu 24.04 LTS
- ROS: ROS 2 Jazzy
- Python: 3.12
- Python environment: uv
- Build system: CMake / ament
- ROS workspace tool: colcon

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

Prevent colcon from scanning the virtual environment:

```bash
touch .venv/COLCON_IGNORE
```

Install the locked Python dependencies:

```bash
uv sync
```

## Python Validation

Run:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

## ROS 2 Workspace

Source ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

Move into the workspace:

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

Run ROS tests:

```bash
colcon test
colcon test-result --verbose
```

## Planned Project Architecture

The project will later include:

1. Franka Panda manipulation task in MuJoCo
2. ROS 2 and ros2_control integration
3. MoveIt 2 global motion planning
4. Classical contact-rich insertion controller
5. Demonstration dataset generation
6. Behaviour Cloning
7. DAgger
8. SAC reinforcement learning
9. Domain randomisation and robustness evaluation
10. ONNX policy deployment and runtime profiling

These components will be implemented incrementally during later project milestones.

## Project Status

This repository is currently in the foundation phase. Results and performance claims will only be added once they have been experimentally validated.
