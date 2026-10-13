# Week 9 Acceptance: Hybrid MoveIt + Classical Insertion Expert

## Overview

Week 9 completes the first full deployment-oriented classical manipulation
pipeline in RobustLearn-Manip.

The milestone combines:

- MoveIt 2 for global collision-aware approach planning;
- the project-owned MuJoCo Panda hardware backend;
- ros2_control trajectory execution;
- ROS-exposed wrist force/torque feedback;
- a deterministic phase-based classical local insertion expert;
- an explicit MoveIt-to-local-controller handoff;
- deterministic clean-scenario acceptance testing.

The validated architecture is:

    MoveIt 2
        |
        v
    MoveItSimpleControllerManager
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
        |
        +----> /joint_states
        |
        +----> /wrist_ft/wrench
                    |
                    v
            local insertion expert
                    |
                    v
            panda_arm_controller

The high-level task decomposition is:

    global approach
        MoveIt 2
            |
            v
    safe staging / handoff configuration
            |
            v
    local alignment
            |
            v
    contact-aware axial insertion
            |
            v
    task success

MoveIt owns the global collision-aware motion phase.

The local classical expert owns the contact-rich fine-alignment and insertion
phase.

The local expert does not command MuJoCo actuators directly. Both global and
local motion use the existing ros2_control architecture and the same
panda_arm_controller.

---

## Week 9 Issues

Week 9 was divided into five focused GitHub issues.

### Issue #79 — Phase-Based Classical Insertion Expert

GitHub:

https://github.com/million-9/robustlearn-manip/issues/79

Goal:

Turn the existing deterministic scripted insertion controller into an explicit,
reusable classical local insertion expert.

Completed work:

- introduced a project-owned classical insertion expert abstraction;
- introduced explicit local-control phases;
- preserved deterministic behavior;
- preserved bounded Cartesian and joint-space command increments;
- retained the seven-arm-joint command contract;
- kept the gripper outside the expert arm-command path;
- reused the project-owned Panda Jacobian and actuator utilities;
- reused insertion-axis and task-relative geometry;
- introduced explicit expert configuration;
- covered phase transitions and command bounds with automated tests;
- preserved the earlier scripted-insertion behavior and regressions.

The local controller now has a clear stateful control contract suitable for
ROS execution and later expert-data collection.

Status:

    COMPLETED

---

### Issue #80 — Wrist Force/Torque Through ros2_control

GitHub:

https://github.com/million-9/robustlearn-manip/issues/80

Goal:

Expose the project-owned MuJoCo wrist force/torque sensor through the real
ros2_control hardware path.

Completed work:

- resolved the MuJoCo wrist force and torque sensors in MuJoCoSystem;
- validated expected sensor names and dimensions;
- exposed six finite ros2_control sensor state interfaces;
- updated force/torque interfaces during hardware reads;
- configured the ROS 2 force/torque broadcaster;
- published wrist force/torque through:

    /wrist_ft/wrench

- preserved the seven Panda arm position-command interfaces;
- preserved direct trajectory execution;
- preserved the real MuJoCo hardware lifecycle;
- preserved mock workflows;
- kept vendored MuJoCo Menagerie files unchanged.

The real MuJoCo ROS path now exposes both joint feedback and wrist-contact
feedback needed by the local controller.

Status:

    COMPLETED

---

### Issue #81 — Execute the Classical Expert Through ros2_control

GitHub:

https://github.com/million-9/robustlearn-manip/issues/81

Goal:

Run the local classical insertion expert through the real Week 8 ROS/MuJoCo
execution architecture instead of directly controlling MuJoCo actuators.

Completed work:

- implemented a ROS-side local insertion expert runner;
- consumed Panda joint state through ROS feedback;
- consumed wrist force/torque through /wrist_ft/wrench;
- reconstructed task-relative insertion state using project-owned model and
  task references;
- generated bounded local commands at a fixed control rate;
- routed commands through panda_arm_controller;
- avoided direct MuJoCo actuator writes;
- implemented conservative force-aware insertion behavior;
- rejected non-finite control outputs;
- added runtime diagnostics for:
  - expert phase;
  - lateral error;
  - axial offset;
  - insertion depth;
  - force;
  - torque;
  - command magnitude;
  - controller offset;
  - terminal reason;
- implemented deterministic timeout and abort behavior;
- validated canonical insertion through the real MuJoCoSystem path.

The local expert therefore uses the same deployment-style execution stack as
MoveIt.

Status:

    COMPLETED

---

### Issue #82 — MoveIt Pre-Insertion Approach to Local Expert Handoff

GitHub:

https://github.com/million-9/robustlearn-manip/issues/82

Goal:

Integrate the MoveIt global approach with the local contact-aware classical
expert.

Completed work:

- implemented a hybrid insertion coordinator;
- used live MuJoCo joint state as the MoveIt start state;
- implemented project-owned planning-scene geometry;
- attached the fixed peg to the Panda planning representation;
- planned the global approach with MoveIt/OMPL;
- executed the approach through panda_arm_controller;
- required successful trajectory execution before local control;
- implemented explicit measured-state handoff verification;
- retained a strict handoff tolerance of:

    0.002 rad maximum joint error

- armed the local expert only after the handoff contract was satisfied;
- reused the same controller manager, PandaSystem, MuJoCoSystem, and
  panda_arm_controller;
- implemented deterministic failure outcomes for:
  - planning failure;
  - execution failure;
  - handoff failure;
  - expert timeout;
  - local expert failure;
- added headless integration tests.

The complete runtime sequence became:

    waiting for robot state
        |
        v
    apply planning scene
        |
        v
    MoveIt planning
        |
        v
    MoveIt execution
        |
        v
    measured handoff verification
        |
        v
    arm local expert
        |
        v
    local alignment/insertion
        |
        v
    success or bounded failure

Status:

    COMPLETED

---

### Issue #83 — Week 9 Classical Expert Acceptance Workflow

GitHub:

https://github.com/million-9/robustlearn-manip/issues/83

Goal:

Complete Week 9 with one reproducible deterministic acceptance workflow for the
hybrid MoveIt + local classical expert system.

The Week 9 required gate was:

    >= 90% clean-scenario success

The target gate was:

    >= 95% clean-scenario success

The formal 200-episode benchmark remains a Week 10 task.

Completed work:

- added a deterministic clean scenario matrix;
- added 20 distinct valid Panda starting configurations;
- kept noise and domain randomization disabled;
- exposed scenario selection through the hybrid launch workflow;
- added project-owned Week 9 acceptance metrics;
- added a single project-owned acceptance runner;
- recorded per-scenario:
  - MoveIt planning success;
  - MoveIt execution success;
  - measured handoff error;
  - local expert completion;
  - final lateral error;
  - final axial offset;
  - final insertion depth;
  - axial force;
  - lateral force;
  - peak axial force;
  - peak lateral force;
  - failure category;
- rejected non-finite state, wrench, command, and metric data;
- preserved previous Python and ROS regressions.

Status:

    COMPLETED AFTER FINAL ACCEPTANCE

---

## Week 9 Staging/Handoff Refinement

During Week 9 acceptance testing, the original pre-insertion target was found to
leave insufficient margin for reliable global approach execution in some of the
clean scenarios.

The architecture was therefore refined so that MoveIt ends at a higher,
collision-free staging configuration and the local expert owns the additional
fine descent into the insertion region.

This preserves the intended architecture:

    MoveIt:
        global collision-aware motion

    Local expert:
        fine alignment
        close-range descent
        contact-rich insertion

### Collision Validation

The candidate staging configuration was validated using the correctly composed
MuJoCo model containing:

- Panda;
- attached fixed peg;
- insertion workcell.

A dense interpolation from the problematic Week 9 case 08 starting state to
the staging configuration was checked using 4001 sampled configurations.

Ignoring the expected zero-distance finger/finger self-contact artifact:

    staging pose contacts:
        NONE

    case 08 -> staging path:
        no relevant contacts in 4001 sampled configurations

This validation was performed with the project-owned composed insertion model,
not the incomplete standalone MJCF asset declaration.

---

## Validated Staging Command Configuration

The final MoveIt command target is:

    panda_joint1 = -0.000906580138929
    panda_joint2 =  0.022495664431674
    panda_joint3 =  0.000211126909362
    panda_joint4 = -1.506700694310150
    panda_joint5 = -0.000029802001695
    panda_joint6 =  1.513840280027956
    panda_joint7 = -0.785479298665236

The corresponding measured MuJoCo equilibrium was experimentally measured over
a settled one-second window containing 500 joint-state samples.

The measured equilibrium is:

    panda_joint1 = -0.000154117003526
    panda_joint2 =  0.028393605524688
    panda_joint3 =  0.000674241873063
    panda_joint4 = -1.513153749958756
    panda_joint5 =  0.000042615824978
    panda_joint6 =  1.512449322659168
    panda_joint7 = -0.784579660416053

All seven settled joint-state standard deviations and peak-to-peak spans were
effectively zero at the recorded precision.

The command/equilibrium distinction remains intentional because the simulated
position-control equilibrium differs slightly from the commanded trajectory
target under load.

The strict handoff acceptance requirement was not weakened:

    maximum measured joint error <= 0.002 rad

---

## Local Expert From the Higher Staging Pose

The existing local expert successfully handled the additional descent from the
higher staging configuration.

A representative case 08 execution progressed through:

    MoveIt planning
        PASS

    MoveIt execution
        PASS

    handoff verification
        PASS

    expert arm
        PASS

    lateral alignment
        PASS

    axial insertion
        PASS

    canonical insertion success
        PASS

This confirms that the local controller owns the final close-range motion rather
than requiring MoveIt to enter the tight-clearance insertion region.

---

## Deterministic Clean Scenario Matrix

Week 9 acceptance uses 20 deterministic clean starting configurations:

    week9_case_01
    week9_case_02
    week9_case_03
    week9_case_04
    week9_case_05
    week9_case_06
    week9_case_07
    week9_case_08
    week9_case_09
    week9_case_10
    week9_case_11
    week9_case_12
    week9_case_13
    week9_case_14
    week9_case_15
    week9_case_16
    week9_case_17
    week9_case_18
    week9_case_19
    week9_case_20

These scenarios apply small deterministic Panda joint perturbations around the
clean insertion approach region.

Week 9 intentionally does not include:

- sensor noise;
- model randomization;
- domain randomization;
- the Week 10 200-episode benchmark.

---

## Acceptance Runner

The project-owned Week 9 runner is:

    tools/run_week9_acceptance.py

The runner launches a fresh hybrid insertion scenario for every selected case,
collects the structured WEEK9_RESULT, validates required finite fields, and
computes the final success gate.

The full acceptance command is:

    source /opt/ros/jazzy/setup.bash
    source ros2_ws/install/setup.bash
    uv run python tools/run_week9_acceptance.py

Results are written to:

    build/week9_acceptance_results.json

---

## Final Week 9 Acceptance Result

The validated final clean-scenario result is:

    Scenarios:       20
    Successes:       20
    Failures:         0
    Success rate:   100.0%

    Required gate:  >= 90%
    Target gate:    >= 95%

    Required gate:  PASS
    Target gate:    PASS

All 20 successful trials reported:

    MoveIt planning success:   true
    MoveIt execution success:  true
    handoff validation:         pass
    local expert success:       true
    failure category:           none

The final acceptance runner returned:

    exit code: 0

---

## Representative Acceptance Metrics

Across the final 20-case clean matrix:

- all handoff errors remained below the 0.002 rad contract;
- final insertion depths were approximately 0.010 m;
- final lateral errors remained small;
- final force metrics were finite;
- peak axial forces remained approximately 0.63 N;
- peak lateral forces remained approximately 0.34-0.35 N;
- every local expert execution reached the canonical insertion success state.

Representative case 08 result:

    scenario:                 week9_case_08
    success:                  true
    MoveIt planning:          true
    MoveIt execution:         true
    local expert success:     true
    handoff error:            0.001138 rad
    final insertion depth:    0.010168 m
    final lateral error:      0.000281 m
    peak axial force:         0.629213 N
    peak lateral force:       0.348145 N
    failure category:         none

Representative case 18 result:

    scenario:                 week9_case_18
    success:                  true
    MoveIt planning:          true
    MoveIt execution:         true
    local expert success:     true
    handoff error:            0.001443 rad
    final insertion depth:    0.010427 m
    final lateral error:      0.000102 m
    peak axial force:         0.629716 N
    peak lateral force:       0.350331 N
    failure category:         none

---

## Execution-Robustness Investigation

During development, case 18 produced one intermittent trajectory-controller
path-tolerance violation.

The reported controller condition was:

    joint index:           4
    mapped Panda joint:    panda_joint5
    position error:       -0.060273 rad
    trajectory tolerance:  0.060000 rad
    excess:                0.000273 rad

The controller tolerance was not widened further.

Subsequent focused runs passed five consecutive times, and case 18 also passed
the final full 20-scenario acceptance matrix.

This transient observation therefore did not justify weakening the controller
contract.

---

## Failure-Path Regression Coverage

The hybrid coordinator has explicit tests for:

    MoveIt planning failure
    MoveIt execution failure
    handoff tolerance failure
    local expert terminal timeout

After the staging handoff was updated, older mock fixtures were also updated to
use the new measured and commanded staging contracts.

The deliberately invalid handoff fixture remained unchanged.

The failure-path regression result is:

    4 tests
    0 failures

The expert-timeout scenario was explicitly verified to progress through:

    planning
    execution
    handoff verification
    expert arming
    waiting for expert
    bounded terminal timeout

---

## Final ROS Regression Result

After Week 9 acceptance and regression-fixture updates, the complete ROS
workspace test result is:

    165 tests
    0 errors
    0 failures
    13 skipped

The robustlearn_moveit_config package reports:

    100% tests passed
    0 tests failed

The final validation preserves:

- direct trajectory execution;
- Week 8 MoveIt-to-MuJoCo execution;
- real MuJoCo hardware bringup;
- mock Panda bringup;
- mock MoveIt workflows;
- hybrid coordinator failure paths;
- wrist force/torque publication;
- local expert execution;
- Week 9 hybrid acceptance.

---

## Week 9 Milestone Gate

Week 9 is considered complete when:

    MoveIt global approach
        PASS

    measured staging handoff
        PASS

    local classical expert
        PASS

    canonical insertion task
        PASS

    >= 90% deterministic clean scenarios
        PASS

    >= 95% target
        PASS

    20/20 final acceptance
        PASS

    full ROS regression suite
        PASS

Final Week 9 status:

    WEEK 9 ACCEPTANCE: PASS
    20 / 20 deterministic clean scenarios successful
    100.0% clean-scenario success
    165 ROS tests
    0 errors
    0 failures

---

## Week 10 Boundary

Week 9 intentionally stops at deterministic clean hybrid-expert validation.

The following remain outside Week 9:

- formal 200-episode expert benchmarking;
- episode-management infrastructure;
- MCAP/rosbag logging;
- dataset writing;
- demonstration collection;
- keyboard/gamepad teleoperation;
- Behaviour Cloning;
- DAgger;
- reinforcement learning;
- large-scale domain-randomized robustness evaluation.

Those tasks belong to Week 10 and later milestones.

---

## Week 9 Issue Summary

    #79  Phase-based classical insertion expert
         COMPLETED

    #80  MuJoCo wrist force/torque through ros2_control
         COMPLETED

    #81  Classical local expert execution through ros2_control
         COMPLETED

    #82  MoveIt pre-insertion approach -> local expert handoff
         COMPLETED

    #83  Week 9 deterministic hybrid acceptance workflow
         COMPLETED

    Final milestone:
         20/20 clean scenarios
         100% success
         full ROS regression green
