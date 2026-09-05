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

### MuJoCo Runtime Dependency

MuJoCo is a runtime dependency of RobustLearn-Manip and is managed through the
normal `uv` dependency workflow.

A separate system-wide MuJoCo installation or manual:

```bash
pip install mujoco
```

step is not required.

MuJoCo is declared in `pyproject.toml`, while the exact resolved dependency
versions are recorded in `uv.lock`.

After installing the locked environment:

```bash
uv sync --locked
```

verify that the MuJoCo Python bindings are available:

```bash
uv run python -c "import mujoco; print(mujoco.__version__)"
```

A successful command prints the installed MuJoCo version.

### Franka Panda MuJoCo Model

The Franka Emika Panda physics model used by RobustLearn-Manip is stored in the
repository under:

```text
robot_description/mjcf/franka_emika_panda/
```

The primary robot MJCF is:

```text
robot_description/mjcf/franka_emika_panda/panda.xml
```

The accompanying `assets/` directory contains the collision and visual meshes
referenced by the MJCF.

The model is vendored into the repository so that simulation does not depend
on an external MuJoCo Menagerie checkout or on machine-specific absolute file
paths.

The reusable Python loader is implemented under:

```text
robustlearn/sim/
```

and can be exercised with:

```bash
uv run python - <<'PY'
from robustlearn.sim import load_panda_model, panda_model_path

model = load_panda_model()

print("Model path:", panda_model_path())
print("nq:", model.nq)
print("nv:", model.nv)
print("njnt:", model.njnt)
print("nu:", model.nu)
PY
```

For the currently vendored Panda model, the complete robot model contains the
seven Panda arm joints together with the two gripper finger joints.

The seven arm joints are identified by the stable names:

```text
joint1
joint2
joint3
joint4
joint5
joint6
joint7
```

The gripper additionally contains:

```text
finger_joint1
finger_joint2
```

The distinction is important: the Panda arm is a seven-joint manipulator, but
the complete MJCF contains additional gripper joints.

The model also contains actuators for the seven arm joints and a coupled
gripper actuator.

### MJCF and ROS Robot Representations

RobustLearn-Manip intentionally maintains separate robot representations for
different responsibilities.

```text
URDF / Xacro
    |
    +---- ROS 2 / MoveIt / ros2_control representation

MJCF
    |
    +---- MuJoCo physics / manipulation-task representation
```

The MuJoCo MJCF must therefore not be treated as merely another visualization
description.

It is the physics-side robot representation used by the manipulation
environment.

The ROS 2 URDF/Xacro representation remains responsible for the ROS software
stack, including MoveIt and `ros2_control`.

The project does not assume that the URDF/Xacro and MJCF models are
numerically identical. Explicit cross-model numerical validation is handled
separately from the initial MuJoCo foundation.

### Model Provenance and Licensing

The Panda MJCF is derived from the Franka Emika Panda model provided by the
MuJoCo Menagerie project.

The exact vendored upstream revision is recorded in:

```text
robot_description/mjcf/franka_emika_panda/UPSTREAM.md
```

The upstream license is preserved in:

```text
robot_description/mjcf/franka_emika_panda/LICENSE
```

Additional upstream documentation retained with the model includes:

```text
robot_description/mjcf/franka_emika_panda/README.md
robot_description/mjcf/franka_emika_panda/CHANGELOG.md
```

These files must remain with the vendored model so that its source,
attribution, and licensing remain traceable.

### MuJoCo Sites

The upstream Panda robot MJCF currently contains no MuJoCo `site` objects.

This is intentional for the RobustLearn-Manip architecture.

Task-specific sites should not be added by modifying the vendored Panda robot
model directly. They belong to the insertion workcell and task description.

Later task models will define stable names for locations such as:

```text
peg tip
receptacle center
insertion axis
pre-insertion target
```

Keeping task geometry separate from the vendored robot model preserves a
clean boundary between:

```text
robot physics model
        |
        v
task / workcell model
        |
        v
learning environment
```

### MuJoCo Foundation Validation

The Panda model foundation can be checked directly with:

```bash
uv run pytest tests/unit/test_panda_model.py -v
```

The tests verify that:

- the committed Panda MJCF path exists;
- the model compiles through the MuJoCo Python API;
- all seven expected Panda arm joints are present;
- joint limits are finite and ordered;
- expected Panda bodies are resolvable by name;
- expected arm actuators are present;
- the model can be stepped for multiple physics steps;
- simulation state remains finite during the smoke rollout.

The standard project-wide Python validation remains:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```