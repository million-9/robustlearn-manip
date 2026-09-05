# Week 5 acceptance: sensing and scripted insertion

Week 5 integrates the Panda sensing, camera, task-evaluation, debug, and
scripted-insertion work into one reproducible milestone gate.

## Acceptance gate

Week 5 is accepted when:

> Sensor values are sanity-tested and a scripted sequence can complete the task.

The automated integration coverage lives in:

```text
tests/integration/test_week5_acceptance.py
```

## Acceptance workflow

```text
deterministic Panda insertion environment
                |
                v
          reset(seed=2026)
                |
                v
        joint-state sensing
                |
                v
       wrist wrench sensing
                |
                v
       RGB/depth camera
                |
                v
       task-state evaluation
                |
                v
      scripted insertion
                |
                v
       success detected
```

## 1. Seeded reset and sensing

The acceptance test creates `PandaInsertionEnv` and resets it with:

```text
seed=2026
```

It verifies:

- seven Panda arm joint-position values;
- seven Panda arm joint-velocity values;
- three wrist-force values;
- three wrist-torque values;
- all sensor outputs are finite;
- sensed joint positions match the corresponding simulator joint positions;
- sensed joint velocities match the corresponding simulator joint velocities;
- the canonical reset is not incorrectly classified as success;
- the canonical reset is not incorrectly classified as failure.

## 2. Wrist force/torque response

The acceptance workflow applies a controlled external wrench to the project
`peg_tool` body.

It verifies that both:

- wrist-force output changes measurably;
- wrist-torque output changes measurably.

This demonstrates that the wrist sensing path responds to tool loading rather
than only returning structurally valid values.

## 3. RGB and depth camera

The stable Week 5 workcell camera is:

```text
workcell_oblique
```

with resolution:

```text
320 x 240
```

The acceptance workflow first verifies the camera definition without requiring
a renderer.

Where an offscreen rendering backend is available, it also verifies:

```text
RGB shape:   (240, 320, 3)
RGB dtype:   uint8

depth shape: (240, 320)
depth dtype: float32
```

Depth values must remain finite.

On systems without an available MuJoCo rendering context, only the render-output
portion may be skipped. The rest of the Week 5 acceptance workflow remains
fully headless.

## 4. Scripted insertion

The deterministic scripted controller advances the Panda through the project
Gymnasium environment interface.

During the rollout the acceptance test verifies that:

- actions remain finite;
- observations remain finite;
- simulator time remains finite;
- joint position state remains finite;
- joint velocity state remains finite;
- actuator state remains finite;
- control targets remain finite;
- task-site positions remain finite;
- joint sensor values remain finite;
- wrist force remains finite;
- wrist torque remains finite;
- lateral task error remains finite;
- axial offset remains finite;
- insertion depth remains finite.

The sequence must terminate through the existing insertion task evaluator with:

```text
task_success = True
task_failure = False
truncated = False
```

Success is therefore produced by project task-evaluation logic rather than a
hardcoded scripted completion flag.

## 5. Determinism

The complete scripted insertion is executed twice with the same explicit seed.

The resulting scalar traces are required to match exactly.

The comparison includes:

- environment step;
- simulation time;
- lateral error;
- axial offset;
- insertion depth;
- success state;
- failure state;
- truncation state.

## Run the Week 5 acceptance test

From the repository root:

```bash
uv run pytest tests/integration/test_week5_acceptance.py -q
```

On a machine with working RGB/depth rendering, the expected result is:

```text
5 passed
```

A headless machine without an available rendering backend may report the camera
render-output test as skipped.

## Run the scripted insertion manually

```bash
uv run python tools/run_scripted_insertion.py --seed 2026
```

The canonical clean task should finish with:

```text
Scripted insertion completed successfully.
```

## Run the debug visualization

```bash
uv run python tools/view_insertion.py
```

The viewer exposes the task-reference geometry and prints live task and wrist
wrench diagnostics.

Press:

```text
R
```

to restore the canonical insertion reset state.

## Full Python validation

```bash
uv run ruff check .
uv run mypy robustlearn tools
uv run pytest
```

## Full ROS 2 validation

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
colcon test
colcon test-result --verbose
```

## Week 5 scope completed

The Week 5 milestone now covers:

- Panda arm joint-position sensing;
- Panda arm joint-velocity sensing;
- wrist force sensing;
- wrist torque sensing;
- fixed RGB/depth camera infrastructure;
- insertion success/failure evaluation;
- developer task visualization;
- deterministic scripted insertion;
- integrated Week 5 acceptance coverage.

## Deferred to Week 6

The following remain intentionally outside the Week 5 gate:

- domain randomization framework;
- formal task configuration schema;
- final policy state/action API;
- clean 100-episode evaluation runner.
