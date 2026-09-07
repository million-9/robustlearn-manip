#!/usr/bin/env bash

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
  pwd
)"

cd "$ROOT_DIR" || exit 1

echo "=== WEEK 8 ACCEPTANCE ==="
echo "repository: $ROOT_DIR"

echo
echo "=== ROS 2 ENVIRONMENT ==="
source /opt/ros/jazzy/setup.bash || exit 1

export ROBUSTLEARN_MUJOCO_ROOT="$(
  uv run python -c \
    'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)'
)" || exit 1

echo "MuJoCo root: $ROBUSTLEARN_MUJOCO_ROOT"

echo
echo "=== PANDA JOINT MAPPING CONTRACT ==="
uv run pytest \
  tests/unit/test_panda_joint_mapping.py \
  -q || exit 1

echo
echo "=== PYTHON QUALITY AND REGRESSION ==="
uv run ruff check . || exit 1
uv run mypy robustlearn || exit 1
uv run pytest || exit 1

echo
echo "=== BUILD ROS WORKSPACE ==="
cd "$ROOT_DIR/ros2_ws" || exit 1

colcon build \
  --symlink-install || exit 1

source install/setup.bash || exit 1

echo
echo "=== CROSS-MODEL FK VALIDATION ==="
ctest \
  --test-dir build/robustlearn_mujoco_hardware \
  -R '^test_cross_model_fk$' \
  --output-on-failure || exit 1

echo
echo "=== MOVEIT -> MUJOCO EXECUTION ==="
ctest \
  --test-dir build/robustlearn_moveit_config \
  -R '^test_test_moveit_mujoco_execution_launch.py$' \
  --output-on-failure || exit 1

echo
echo "=== FULL ROS REGRESSION ==="
colcon test || exit 1

colcon test-result \
  --verbose || exit 1

echo
echo "========================================"
echo "WEEK 8 ACCEPTANCE: PASS"
echo "MoveIt trajectory executes in MuJoCo."
echo "========================================"
