# Reproducibility Guide

This document describes how to reproduce the RobustLearn-Manip development
environment and validation workflow from a fresh clone.

The purpose of this guide is to ensure that the project does not depend on
hidden state from the original development checkout, such as an existing
Python virtual environment or previously generated ROS 2 build artifacts.

A successful clean-checkout reproduction starts without:

- `.venv/`
- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`

These directories are generated locally and must not be committed.

## Supported Development Environment

The current project environment assumes:

- Ubuntu 24.04 LTS
- ROS 2 Jazzy
- Python 3.12
- `uv` for Python dependency management
- `rosdep` for ROS dependency installation
- `colcon` for ROS workspace builds
- CMake / ament for ROS 2 C++ packages
- Git for version control
- GitHub Actions for continuous integration

The GitHub Actions workflows also run on Ubuntu 24.04 and use ROS 2 Jazzy.

## 1. Fresh Clone

Clone the repository into a new directory:

```bash
git clone https://github.com/million-9/robustlearn-manip.git
cd robustlearn-manip
```

Confirm that the checkout is clean:

```bash
git status
```

Expected result:

```text
On branch main
nothing to commit, working tree clean
```

A fresh checkout should not contain an existing Python virtual environment or
ROS 2 build products.

For example:

```bash
ls -d .venv ros2_ws/build ros2_ws/install ros2_ws/log 2>/dev/null
```

On a completely fresh checkout, these paths should not exist.

## 2. Python Environment

The project uses Python 3.12 and `uv`.

Create the local virtual environment using Ubuntu's system Python:

```bash
uv venv \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Prevent `colcon` from discovering the Python virtual environment as part of
the ROS workspace:

```bash
touch .venv/COLCON_IGNORE
```

Install the locked Python dependencies:

```bash
uv sync --locked
```

The `--locked` option requires dependency installation to remain consistent
with the committed lock file instead of silently changing dependency
resolution.

## 3. Python Validation

Run the Python quality checks from the repository root:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

All checks should pass.

These commands correspond to the validation performed by the Python GitHub
Actions workflow:

```text
uv sync --locked
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

## 4. ROS Environment and pytest Plugin Interaction

ROS 2 modifies the shell environment when its setup script is sourced.

For example:

```bash
source /opt/ros/jazzy/setup.bash
```

ROS can expose system Python packages through environment variables such as
`PYTHONPATH`.

pytest supports automatic discovery of third-party plugins. In an environment
where ROS and additional system Python packages are visible, pytest may
therefore discover plugins that are unrelated to RobustLearn-Manip.

The normal project validation command is:

```bash
uv run pytest
```

If a ROS-configured local shell causes pytest failures from unrelated
automatically discovered plugins, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

This disables automatic third-party pytest plugin loading while still running
the project's own tests.

This command should not be used to hide failures in RobustLearn-Manip tests.
It is intended only for failures caused by external pytest plugin discovery.

For the cleanest Python-only validation environment, run the Python checks in
a terminal where ROS 2 has not been sourced.

## 5. ROS 2 Environment

Open a terminal and source ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

Confirm the ROS distribution:

```bash
echo "$ROS_DISTRO"
```

Expected result:

```text
jazzy
```

Move into the ROS workspace:

```bash
cd ros2_ws
```

## 6. Install ROS Dependencies

Update the rosdep dependency database:

```bash
rosdep update
```

Install dependencies required by packages in the workspace:

```bash
rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro jazzy \
  -r -y
```

The `--ignore-src` option prevents rosdep from attempting to install packages
that already exist as source packages inside the workspace.

## 7. Build the ROS Workspace

From `ros2_ws/`, ensure ROS 2 Jazzy is sourced:

```bash
source /opt/ros/jazzy/setup.bash
```

Build:

```bash
colcon build --symlink-install
```

A successful build creates generated workspace directories:

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

These directories are local build products and are intentionally excluded
from version control.

## 8. Source the Workspace Overlay

After the build completes:

```bash
source install/setup.bash
```

There are now two ROS environments involved:

```text
/opt/ros/jazzy
      |
      | ROS 2 underlay
      v
ros2_ws/install
      |
      | project overlay
      v
RobustLearn ROS packages
```

The system ROS 2 Jazzy installation is the **underlay**.

The project's generated `install/` directory is the **overlay**.

Sourcing:

```bash
source /opt/ros/jazzy/setup.bash
```

makes the standard ROS 2 Jazzy environment available.

After building the project, sourcing:

```bash
source install/setup.bash
```

adds the RobustLearn-Manip packages to the ROS environment.

This distinction is important for ROS commands, launch files, custom
interfaces, and integration tests that locate packages through the ROS package
index.

## 9. Verify Project ROS Packages

After sourcing the workspace overlay:

```bash
ros2 pkg prefix robustlearn_interfaces
ros2 pkg prefix robustlearn_control
```

Both commands should return installation paths associated with the current
workspace.

The returned paths should contain:

```text
ros2_ws/install/
```

Verify the custom service definition:

```bash
ros2 interface show robustlearn_interfaces/srv/SetSystemMode
```

Expected interface:

```text
string mode
---
bool success
string message
```

Verify the custom action definition:

```bash
ros2 interface show robustlearn_interfaces/action/ExecuteSystemCheck
```

Expected structure:

```text
int32 total_steps
---
bool success
string message
---
int32 completed_steps
float32 progress
```

Successful interface discovery confirms that the generated
`robustlearn_interfaces` package is available through the workspace overlay.

## 10. Run ROS Tests

From `ros2_ws/`, with the ROS underlay and project overlay sourced:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

colcon test
colcon test-result --verbose
```

The expected result is:

```text
0 errors
0 failures
```

The exact total number of tests may increase as the project grows.
Reproducibility should therefore be judged by the absence of errors and
failures rather than by a permanently fixed test count.

The ROS test suite includes ROS package checks together with project-written
unit and integration tests.

## 11. Why the Workspace Must Be Sourced Before Integration Tests

Building a ROS package and making that package discoverable by ROS are
separate steps.

For example:

```bash
colcon build --symlink-install
```

creates the workspace installation.

However, a shell that has only sourced:

```bash
source /opt/ros/jazzy/setup.bash
```

knows about the ROS 2 Jazzy installation but does not automatically know about
the project's newly built packages.

A launch-based integration test may therefore fail with an error similar to:

```text
package 'robustlearn_control' not found, searching: ['/opt/ros/jazzy']
```

Source the generated workspace overlay:

```bash
source install/setup.bash
```

After doing this, ROS package discovery includes packages built inside
RobustLearn-Manip.

The ROS 2 GitHub Actions workflow performs the same overlay-sourcing step
before running the test suite.

## 12. Local Validation Equivalent to CI

The repository contains separate Python and ROS 2 GitHub Actions workflows.

The following commands reproduce the essential CI validation locally.

### Python CI Equivalent

From the repository root:

```bash
uv sync --locked

uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

Expected result:

```text
Ruff    PASS
mypy    PASS
pytest  PASS
```

If pytest is affected by unrelated automatically discovered plugins in a
ROS-configured shell:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

### ROS 2 CI Equivalent

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash

rosdep update

rosdep install \
  --from-paths ros2_ws/src \
  --ignore-src \
  --rosdistro jazzy \
  -r -y
```

Move into the workspace:

```bash
cd ros2_ws
```

Build:

```bash
colcon build --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

Run the tests:

```bash
colcon test
colcon test-result --verbose
```

A successful validation should therefore look conceptually like:

```text
Python
  |
  +-- uv sync       PASS
  +-- Ruff          PASS
  +-- mypy          PASS
  +-- pytest        PASS

ROS 2
  |
  +-- rosdep        PASS
  +-- colcon build  PASS
  +-- colcon test   PASS
  +-- test-result   0 errors, 0 failures
```

GitHub Actions independently repeats these validation stages on a clean
runner.

## 13. Generated Files Must Remain Untracked

The following files and directories are generated locally and must not be
committed:

```text
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/

ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

These paths are excluded through the repository `.gitignore`.

After building and testing, verify the repository state:

```bash
git status
```

Generated Python and ROS build products should not appear as untracked or
modified repository files.

Only intentional source-code or documentation changes should appear in Git.

## 14. Full Clean-Checkout Verification

For the strongest reproducibility test, create an entirely separate clone
instead of deleting generated files from the normal development repository.

For example:

```bash
cd ~/projects

git clone https://github.com/million-9/robustlearn-manip.git \
  robustlearn-manip-clean

cd robustlearn-manip-clean
```

This separate checkout must not reuse:

```text
.venv/
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

from the normal development repository.

### Reproduce the Python Environment

Create the environment:

```bash
uv venv \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Prevent colcon from discovering it:

```bash
touch .venv/COLCON_IGNORE
```

Install dependencies:

```bash
uv sync --locked
```

Run Python validation:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

If the local ROS/system Python environment causes unrelated pytest plugin
discovery problems:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

### Reproduce the ROS Workspace

Source ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

Update rosdep:

```bash
rosdep update
```

Install dependencies:

```bash
rosdep install \
  --from-paths ros2_ws/src \
  --ignore-src \
  --rosdistro jazzy \
  -r -y
```

Move into the workspace:

```bash
cd ros2_ws
```

Build:

```bash
colcon build --symlink-install
```

Source the generated overlay:

```bash
source install/setup.bash
```

Run tests:

```bash
colcon test
colcon test-result --verbose
```

Verify package discovery:

```bash
ros2 pkg prefix robustlearn_interfaces
ros2 pkg prefix robustlearn_control
```

Verify the custom service:

```bash
ros2 interface show robustlearn_interfaces/srv/SetSystemMode
```

Verify the custom action:

```bash
ros2 interface show robustlearn_interfaces/action/ExecuteSystemCheck
```

If all Python checks, ROS builds, ROS tests, and package-discovery commands
succeed from this independent clone, the current project environment is
considered reproducible.

## 15. Clean-Checkout Reproducibility Checklist

A clean-checkout verification is complete when:

- the repository was cloned into a separate directory
- no original `.venv/` was reused
- no original ROS `build/` directory was reused
- no original ROS `install/` directory was reused
- no original ROS `log/` directory was reused
- dependencies were installed from committed project metadata
- `uv sync --locked` succeeded
- Ruff passed
- mypy passed
- pytest passed
- ROS 2 Jazzy was sourced
- rosdep dependency installation succeeded
- the ROS workspace built successfully
- the generated workspace overlay was sourced
- `robustlearn_interfaces` was discoverable
- `robustlearn_control` was discoverable
- custom ROS interfaces were discoverable
- `colcon test` completed successfully
- `colcon test-result --verbose` reported zero errors and zero failures
- generated development and build artifacts remained untracked
- the essential local validation stages correspond to GitHub Actions CI

## Reproducibility Principle

A project result is not considered reproducible merely because it works in
the original development checkout.

The project should be reconstructable from:

```text
Git repository
      +
locked dependencies
      +
documented system assumptions
      +
documented setup commands
      +
documented build commands
      +
automated tests
      +
continuous integration
```

without relying on hidden local state.

The clean-checkout process will be extended as MuJoCo, robot control,
planning, learning infrastructure, datasets, trained models, experiment
configuration, evaluation tooling, and deployment components are added to the
project.

## Panda Joint Trajectory Validation

The mock Panda trajectory-control path can be validated without MuJoCo or
physical hardware.

The expected data path is:

```text
FollowJointTrajectory goal
        |
        v
panda_arm_controller
        |
        v
ros2_control position command interfaces
        |
        v
mock_components/GenericSystem
        |
        v
position and velocity state interfaces
        |
        v
joint_state_broadcaster
        |
        v
/joint_states
```

### Start the mock Panda

From the ROS workspace:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch robustlearn_description mock_panda.launch.py
```

Leave this terminal running.

### Verify the trajectory action

In a second terminal:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 action type /panda_arm_controller/follow_joint_trajectory
ros2 action info /panda_arm_controller/follow_joint_trajectory
```

The expected action type is:

```text
control_msgs/action/FollowJointTrajectory
```

The controller should expose one action server.

### Monitor the Panda joint state

In another terminal:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo /joint_states --field position
```

### Send a valid seven-joint trajectory

Send a trajectory goal using all seven Panda arm joints:

```bash
ros2 action send_goal \
  /panda_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{
    trajectory: {
      joint_names: [
        panda_joint1,
        panda_joint2,
        panda_joint3,
        panda_joint4,
        panda_joint5,
        panda_joint6,
        panda_joint7
      ],
      points: [
        {
          positions: [0.2, -0.3, 0.1, -1.0, 0.2, 1.0, 0.3],
          time_from_start: {sec: 5, nanosec: 0}
        }
      ]
    }
  }" \
  --feedback
```

The goal should be accepted and finish successfully.

Expected final action status:

```text
error_code: 0
error_string: Goal successfully reached!

Goal finished with status: SUCCEEDED
```

After execution, inspect one joint-state message:

```bash
ros2 topic echo /joint_states --once
```

The final joint positions should correspond to the commanded target:

```text
[0.2, -0.3, 0.1, -1.0, 0.2, 1.0, 0.3]
```

### Validate trajectory rejection

A malformed trajectory should be rejected instead of being passed to the
hardware.

The following goal intentionally contains seven joint names but only six
position values:

```bash
ros2 action send_goal \
  /panda_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{
    trajectory: {
      joint_names: [
        panda_joint1,
        panda_joint2,
        panda_joint3,
        panda_joint4,
        panda_joint5,
        panda_joint6,
        panda_joint7
      ],
      points: [
        {
          positions: [0.1, -0.2, 0.1, -0.8, 0.2, 0.8],
          time_from_start: {sec: 3, nanosec: 0}
        }
      ]
    }
  }" \
  --feedback
```

Expected result:

```text
Goal was rejected.
```

Verify that the rejected goal did not modify the robot state:

```bash
ros2 topic echo /joint_states --once
```

The Panda should remain at its previously commanded configuration.

This validates both successful trajectory execution and understandable
rejection of structurally invalid trajectory commands.