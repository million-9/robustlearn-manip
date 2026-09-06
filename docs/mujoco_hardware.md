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

MuJoCo model initialization, Panda joint and actuator mappings, lifecycle
resource ownership, state reads, command writes, and physics stepping are
implemented by later Week 7 issues.

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
