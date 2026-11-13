# Week 11 — Safe Cartesian Peg-Tip Teleoperation (Issue #108)

## Scope and controls

Selectable joint-space and Cartesian XYZ keyboard teleoperation is implemented
for the Panda insertion task. Cartesian jogs use measured Panda joint feedback,
MuJoCo peg-tip forward kinematics and Jacobians, damped least-squares inverse
kinematics, and bounded free-space target checks. W/S command +/-X, A/D command
-/+Y, and R/F command +/-Z; Space refreshes the deadman, X stops, and Escape
aborts. The original joint-space teleoperation mode remains supported.

Motion safety includes joint/workspace limits, feedback freshness, wrist force
and torque checks, controller reference/tracking checks, deadman and stop
behavior, and terminal controller completion tracking. Cartesian motion is
allowed only when the live ROS controller confirms
`interpolate_from_desired_state: true`; the safety gate rejects unverified
configuration and cancels active motion on revocation.

## Launch and RViz

From the repository root, after sourcing ROS and the workspace, run:

    uv run ros2 launch robustlearn_moveit_config moveit_mujoco.launch.py \
      model_path:="$PWD/robot_description/mjcf/insertion/panda_insertion.xml" \
      initial_keyframe:=week9_approach \
      controllers_yaml:="$PWD/ros2_ws/src/robustlearn_description/config/mujoco_trajectory_controllers_cartesian.yaml" \
      use_rviz:=true \
      use_alignment_markers:=true \
      rviz_config_file:="$PWD/ros2_ws/src/robustlearn_moveit_config/config/cartesian_alignment.rviz"

The read-only alignment publisher uses the MuJoCo shadow kinematics reconstructed
from ROS joint feedback. It publishes five world-frame RViz markers for the peg
tip, receptacle center, error line, insertion axis, and XYZ alignment error in
millimetres. Markers expire and publication stops on stale joint feedback.

The live ROS acceptance verified the publisher and all five markers, and the
operator visually confirmed the robot and task-specific alignment display.

## MuJoCo execution evidence

Six directional 1 mm jogs succeeded through the MuJoCo-backed ROS controller:

| Axis | Principal displacement |
| --- | ---: |
| +X | +0.98902 mm |
| -X | -0.98941 mm |
| +Y | +0.99654 mm |
| -Y | -0.99619 mm |
| +Z | +0.99050 mm |
| -Z | -0.99092 mm |

These six jogs preceded the fail-closed runtime interpolation verification gate.
The subsequent separately tested, guarded +X jog confirmed the live controller
parameter and succeeded with X displacement +0.98902 mm, Y drift -0.00474 mm,
Z drift -0.02225 mm, peak tip travel 0.98928 mm, peak tracking error
0.0021516 rad, force 0.5608 N and torque 0.00252 Nm.

## Canonical Cartesian-mode recording acceptance

An automated MuJoCo-backed Cartesian +X jog was captured and reviewed. The
operator-review state accepted it as a **failed**, deliberately incomplete
insertion, not a successful demonstration. Canonical dataset validation passed
with one teleop episode, one frame, zero successes and one accepted failure,
including expected observation/action dimensions and provenance fields.

Temporary smoke-test artifacts:
`/tmp/robustlearn108-recording-x88RWM`

This automated one-frame record is **not** a human demonstration and must not
be included in the production behavior-cloning dataset.

## Regression evidence

- Python: 882 passed; three Gymnasium environment-check warnings.
- Ruff: passed.
- Mypy for three Cartesian modules: passed.
- ROS control: 61 tests, zero failures, eight skipped.
- ROS description: 24 tests, zero failures.
- MoveIt: full CTest passed.
- ROS alignment tests: three unit tests passed, live marker geometry verified,
  and stale-feedback publication stopped.

## Limits

All motion evidence is from MuJoCo simulation, not a physical robot. RViz is
informational and does not enforce motion safety. The automated recording is an
integration smoke test, not a training-quality human demonstration. Manual
collection and dataset v0.1 freezing remain separate acceptance gates.
