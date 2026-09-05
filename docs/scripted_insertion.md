# Deterministic scripted Panda insertion

Issue #41 adds a lightweight deterministic controller for the clean Week 5
Panda insertion task.

The controller is intentionally separate from future learned policies. Its
purpose is to provide a reproducible task-completion baseline and a regression
test for the canonical insertion scenario.

## Run manually

From the repository root:

```bash
uv run python tools/run_scripted_insertion.py --seed 2026
```

The default maximum episode length is 100 environment steps.

A different limit can be supplied with:

```bash
uv run python tools/run_scripted_insertion.py \
  --seed 2026 \
  --max-steps 100
```

## Expected result

The clean canonical scenario should terminate through the project task
evaluator with:

```text
state=SUCCESS
```

followed by:

```text
Scripted insertion completed successfully.
```

The script does not use a hardcoded completion step. Success is determined by
the same insertion task logic used by the Gymnasium environment.

## Controller

The scripted controller operates through project-defined simulation and
environment interfaces.

At each environment step it:

1. reads the current peg-tip and receptacle geometry;
2. determines the insertion-axis direction;
3. computes axial insertion progress and lateral alignment error;
4. requests a small Cartesian insertion motion while correcting lateral error;
5. computes the Panda peg-tip spatial Jacobian;
6. converts the desired Cartesian motion into a damped joint-space correction;
7. accumulates that correction onto the current actuator position targets;
8. clips the command to the environment action limits;
9. advances the task through `PandaInsertionEnv.step()`.

The controller resolves joints, actuators, and task sites by name rather than
hardcoding raw MuJoCo memory indices.

## Why actuator targets are accumulated

The Panda arm actuators are position-controlled.

The scripted controller therefore updates the previously commanded actuator
targets:

```text
new target = previous target + joint correction
```

rather than repeatedly commanding only a small offset from the measured joint
position.

This allows the position controller to track the desired Cartesian trajectory
while maintaining lateral alignment.

## Task feedback

The controller uses the project task geometry to continuously correct:

- insertion depth;
- lateral peg/receptacle error.

Success remains defined by the task evaluator. The scripted controller does
not contain its own independent success condition.

## Diagnostics

`run_scripted_insertion()` records, for every environment step:

- environment step;
- simulation time;
- lateral insertion error;
- axial offset;
- insertion depth;
- wrist force;
- wrist torque;
- success state;
- failure state;
- truncation state.

The command-line runner prints the main diagnostics while the sequence runs.

## Determinism

The sequence starts from an explicit seeded reset.

The automated regression test executes the same seed twice and verifies that
the scalar task trace is identical and that all recorded state, action, and
wrist-wrench values remain finite.

## Scope

This scripted controller is a Week 5 functional baseline. It is not:

- a learned policy;
- a robust expert benchmark;
- a randomized insertion policy;
- a force-feedback controller;
- a MoveIt-to-MuJoCo execution pipeline.

Those belong to later project milestones.
