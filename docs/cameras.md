# Panda insertion cameras

RobustLearn-Manip provides an optional project-owned camera layer for the
MuJoCo Panda insertion task.

The Week 5 control and learning state remains state-based. Camera images are
provided for logging, visualization, debugging, dataset generation, and later
perception extensions. They are not part of the policy observation at this
milestone.

## Fixed workcell camera

The stable camera name is:

```text
workcell_oblique
```

It is defined in:

```text
robot_description/mjcf/insertion/workcell.xml
```

The camera is part of the project-owned insertion workcell. The vendored
MuJoCo Menagerie Panda model is not modified.

Its definition is:

```text
mode          fixed
position      [1.05, -0.75, 0.95] m
field of view 45 degrees
resolution    320 x 240 pixels
outputs       RGB and depth
```

The MJCF orientation is:

```text
xyaxes="0.834354 0.551230 0 -0.265897 0.402468 0.875967"
```

MuJoCo cameras view along their local negative Z axis.

The camera is positioned above and oblique to the insertion fixture and looks
toward the insertion workspace around the receptacle and fixed Panda tool.

At the deterministic home configuration used during development, the camera
successfully contains the receptacle center, pre-insertion target, and peg tip
in front of its image plane.

## Image contract

The project camera contract uses:

```text
width   320 pixels
height  240 pixels
```

RGB images have:

```text
shape   (240, 320, 3)
dtype   uint8
```

Depth images have:

```text
shape   (240, 320)
dtype   float32
```

The MuJoCo model uses metres as its geometric length unit, so depth values are
interpreted in metres.

The current deterministic workcell render produces finite depth values across
the complete image.

## Python API

Camera rendering is centralized in:

```text
robustlearn/sim/rendering.py
```

The main abstraction is:

```python
from robustlearn.sim import MuJoCoSimulation
from robustlearn.sim.rendering import CameraRenderer

sim = MuJoCoSimulation()
sim.reset(seed=2026)

renderer = CameraRenderer(sim.model)

try:
    frame = renderer.render(sim.data)

    print(frame.rgb.shape)
    print(frame.depth.shape)
finally:
    renderer.close()
```

`CameraRenderer.render()` returns an `RGBDepthFrame` containing independent
NumPy copies.

The frame contains:

```text
rgb
depth
```

Individual rendering is also available through:

```text
render_rgb()
render_depth()
```

Application code should use this abstraction instead of constructing and
managing raw `mujoco.Renderer` instances throughout the codebase.

## Optional rendering

Creating `MuJoCoSimulation` does not create an OpenGL context.

Creating `CameraRenderer` also does not create an OpenGL context.

The underlying MuJoCo renderer is created lazily only when an RGB or depth
image is actually requested.

This keeps normal simulation, environment tests, state-based learning, and ROS
2 regression tests independent of rendering.

The renderer can be explicitly released with:

```python
renderer.close()
```

After closing, another render request may lazily construct a new renderer.

## Rendering-backend availability

Offscreen rendering depends on the OpenGL backend available on the host.

On Linux, the project checks for a usable rendering-context source before
constructing `mujoco.Renderer`.

Rendering may proceed when an X11 or Wayland display is available, or when
`MUJOCO_GL` was configured before Python startup for a supported headless
backend such as `egl` or `osmesa`.

When Linux has neither a display nor an explicitly configured headless backend,
`CameraRenderingUnavailableError` is raised before entering MuJoCo's native
renderer construction.

This preflight is important because some OpenGL context failures occur in
native code and can terminate the process instead of producing a catchable
Python exception.

Renderer-construction failures that do surface as Python exceptions are also
wrapped as `CameraRenderingUnavailableError`.

The rendering unit tests may skip GL-dependent image checks when no rendering
backend is available, while camera lookup, resolution, lazy-initialization, and
headless-preflight tests still run normally.

This prevents a missing display or offscreen graphics backend from breaking
otherwise headless simulation functionality.

## Deterministic rendering checks

Issue #38 verifies repeated rendering from the same deterministic reset for
structural consistency.

The tests require repeated RGB images to retain the same:

```text
shape
dtype
finite-value structure
```

and repeated depth images to retain the same:

```text
shape
dtype
finite-value structure
```

Pixel-perfect equality is intentionally not used as a cross-platform contract.

Rendering implementations, graphics drivers, and OpenGL backends can introduce
small platform-dependent rasterization differences even when simulator state is
the same.

The deterministic simulator state remains the authoritative reproducibility
contract.

## Wrist camera decision

Issue #38 permits a wrist camera when practical, but does not require one.

A wrist camera is intentionally not added at this milestone.

The fixed `workcell_oblique` camera already provides the required RGB/depth
infrastructure for logging, visualization, debugging, and future perception
work.

Adding a wrist-mounted visual sensor would increase model and perception
complexity without contributing to the current state-based manipulation policy.

A wrist camera can be introduced later if a perception-conditioned task or
visual policy requires it.

## Policy observation scope

Camera images are not included in the Week 5 learned-policy observation.

The current policy remains state-based.

The following remain outside Issue #38:

- learned visual policies;
- segmentation;
- object detection;
- camera noise;
- camera domain randomization;
- ROS image publishing;
- perception-conditioned control.

## Reproduction

Run the focused rendering tests from the repository root:

```bash
uv run pytest tests/unit/test_rendering.py -v
```

Run the complete Python validation:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

Run the ROS 2 regression validation:

```bash
source /opt/ros/jazzy/setup.bash

cd ros2_ws

rosdep install \
  --from-paths src \
  --ignore-src \
  -r \
  -y \
  --rosdistro jazzy

colcon build --symlink-install

source install/setup.bash

colcon test

colcon test-result --verbose
```

## Issue #38 acceptance mapping

The implementation satisfies the Issue #38 camera requirements as follows:

- a stable fixed workcell camera is defined;
- its position, orientation, field of view, and resolution are documented;
- RGB rendering is supported;
- depth rendering is supported;
- RGB output has fixed shape `(240, 320, 3)` and dtype `uint8`;
- depth output has fixed shape `(240, 320)` and dtype `float32`;
- rendering is optional and lazily initialized;
- headless simulation does not require an OpenGL context;
- project code uses a centralized camera abstraction;
- camera lookup and output structure are covered by tests;
- deterministic resets are checked for repeated render structure;
- camera images remain outside the current policy observation;
- the vendored Panda model remains untouched;
- a wrist camera is intentionally deferred until a visual-control use case
  justifies it.
