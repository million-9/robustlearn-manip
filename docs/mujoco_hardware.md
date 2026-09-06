# MuJoCo ros2_control Hardware

This document defines the build-time and runtime contracts for the
`robustlearn_mujoco_hardware` ROS 2 package.

## Plugin

The ros2_control hardware plugin is registered as:

    robustlearn_mujoco_hardware/MuJoCoSystem

Its base class is:

    hardware_interface::SystemInterface

Issue #59 provides the package scaffold, shared library, pluginlib
registration, and plugin discovery/loading test.

Week 7 progressively implements the runtime hardware path:

- Issue #60 initializes the MuJoCo model, lifecycle state, and named Panda
  joint/actuator mappings.
- Issue #61 implements the MuJoCo-to-ros2_control Panda joint-state read path.
- Issue #62 implements the ros2_control-to-MuJoCo Panda position-command path
  and deterministic physics stepping.

## MuJoCo C/C++ build dependency

MuJoCo is an external non-ROS dependency.

The package resolves the MuJoCo C headers and shared library in this order:

1. CMake variable `ROBUSTLEARN_MUJOCO_ROOT`
2. Environment variable `ROBUSTLEARN_MUJOCO_ROOT`
3. The installation directory of an importable Python `mujoco` package

The resolved root must contain:

    include/mujoco/mujoco.h

and a MuJoCo shared library such as:

    libmujoco.so
    libmujoco.so.<version>

The repository does not vendor another MuJoCo SDK or duplicate MuJoCo
Menagerie assets.

The built plugin links to the resolved MuJoCo shared library. Because the
project may obtain MuJoCo from its Python environment rather than a system
library directory, the installed plugin retains the resolved MuJoCo root as
its runtime library search path.

The MuJoCo root is therefore part of the source-build/install contract. If
that installation is moved or replaced after the ROS workspace is built, the
hardware package must be rebuilt against the new MuJoCo location.

## Runtime MJCF model-path contract

The ros2_control hardware configuration uses the hardware parameter:

    model_path

Example:

    <ros2_control name="PandaMuJoCoSystem" type="system">
      <hardware>
        <plugin>robustlearn_mujoco_hardware/MuJoCoSystem</plugin>
        <param name="model_path">/absolute/path/to/model.xml</param>
      </hardware>
    </ros2_control>

The runtime contract is:

- the parameter name is exactly `model_path`;
- its value identifies the MJCF entry-point file used by the hardware plugin;
- the path must resolve to an existing readable MJCF file;
- missing or invalid model paths must cause hardware initialization to fail
  clearly;
- project-owned MJCF assets remain under `robot_description/mjcf`;
- vendored MuJoCo Menagerie files must not be duplicated or modified.

Issue #59 defines this contract. Actual model loading and validation are part
of Issue #60.

## Panda command and physics-stepping contract

The hardware plugin exposes one position command interface for each Panda arm
joint from `panda_joint1` through `panda_joint7`.

During initialization, the plugin uses the MJCF `home` keyframe when one is
available and advances MuJoCo derived state with `mj_forward()`. The internal
position command buffers are initialized from the resulting Panda joint
positions so activation begins from a hold-current-position command instead of
introducing an initial command jump.

Each `write()` cycle follows this contract:

1. Read all seven ROS position commands.
2. Reject the cycle if any command is non-finite.
3. Validate each command against the compiled MuJoCo actuator control range
   when that actuator is control-limited.
4. If any command is invalid, return an error before changing any Panda
   actuator control value or advancing simulation time.
5. If all commands are valid, copy them to the cached named Panda actuator
   mappings for `actuator1` through `actuator7`.
6. Leave the gripper/task actuator `actuator8` unchanged.
7. Advance MuJoCo by exactly one call to `mj_step()`.

A successful hardware `write()` therefore advances exactly one compiled MuJoCo
physics timestep. The ROS control-loop `period` argument does not scale the
number of MuJoCo physics steps. This makes simulation evolution deterministic
for a fixed initial state and fixed sequence of hardware commands.

For the current Panda insertion MJCF, the compiled MuJoCo timestep is 0.002 s.
A dedicated real-MuJoCo controller-manager bringup can select an appropriate
control-loop rate separately; changing that bringup rate does not change the
one-step-per-successful-write hardware contract.

After a successful `write()`, the next `read()` copies the resulting finite
Panda `qpos` and `qvel` values into the corresponding ros2_control position and
velocity state interfaces.
