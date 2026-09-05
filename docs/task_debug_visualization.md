# Panda insertion task debug visualization

Issue #40 provides an optional developer visualization for inspecting the Panda
insertion task in MuJoCo.

The visualization is intended for debugging and development only. It is not a
dependency of the Gymnasium environment or the headless test suite.

## Launch

From the repository root:

```bash
uv run python tools/view_insertion.py
```

A MuJoCo viewer window opens showing the Panda insertion workcell.

## Task references

The viewer displays MuJoCo site labels so the main insertion-task references can
be inspected directly:

- `peg_tip`
- `receptacle_center`
- `insertion_axis`
- `pre_insertion`

The workcell MJCF already gives these sites distinct visual markers.

## Live diagnostics

While the simulation runs, the terminal displays:

```text
time
state
lateral
axial
depth
force
torque
```

where:

- `state` is `IN_PROGRESS`, `SUCCESS`, or `FAILURE`;
- `lateral` is the peg-tip error perpendicular to the insertion axis;
- `axial` is the signed peg-tip offset along the configured reference axis;
- `depth` is positive insertion progress;
- `force` is the 3-axis wrist force in newtons;
- `torque` is the 3-axis wrist torque in newton-metres.

The task quantities use the same evaluator as the Gymnasium environment, and
the wrench values use the same Panda sensor reader as the simulation API.

## Reset control

Press:

```text
R
```

while the viewer is active to restore the canonical insertion reset state.

The terminal reports:

```text
Reset to canonical insertion state.
```

and simulation time restarts from the beginning.

## Headless architecture

Debug-data generation is separate from the interactive viewer.

```text
MuJoCoSimulation
        |
        +--> task_status()
        |
        +--> sensor_snapshot()
        |
        v
insertion_debug_snapshot()
        |
        +--> headless tests
        |
        v
tools/view_insertion.py
```

`robustlearn/sim/debug.py` can therefore be tested without creating a window or
OpenGL viewer context.

The normal environment API does not import or depend on `mujoco.viewer`.

## Scope

This visualization is intentionally lightweight. It is not:

- a training dashboard;
- a polished end-user GUI;
- an RViz integration;
- a camera-based perception system.

Its purpose is to make the simulation geometry, task state, and sensing signals
easy to inspect during development.
