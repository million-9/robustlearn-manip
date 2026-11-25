# CPU and CUDA PyTorch Environments

## Purpose

RobustLearn-Manip supports two mutually exclusive PyTorch
dependency groups managed by uv.

The default environment uses CPU-only PyTorch for development
and continuous integration.

The CUDA environment supports GPU training and inference
on compatible NVIDIA hardware.

## Verified versions

| Component | CPU profile | CUDA profile |
|---|---|---|
| PyTorch | 2.15.0+cpu | 2.14.1+cu130 |
| CUDA runtime | None | 13.0 |
| Python | 3.12 | 3.12 |
| Dependency manager | uv | uv |

The CUDA profile has been verified on an NVIDIA GeForce
RTX 4060 Laptop GPU with driver 580.178.04.

## Default CPU environment

Install the default environment:

    uv sync --locked

Verify:

    uv run --locked python -c "import torch; print(torch.__version__)"

Expected PyTorch version:

    2.15.0+cpu

The default dependency groups are dev and cpu.

GitHub Actions uses the default CPU environment.

## Isolated CUDA environment

Create a dedicated virtual environment:

    uv venv /tmp/robustlearn-cuda --python 3.12

Install the locked CUDA profile:

    UV_PROJECT_ENVIRONMENT=/tmp/robustlearn-cuda uv sync --locked --no-default-groups --group dev --group cuda

Verify CUDA:

    /tmp/robustlearn-cuda/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Expected results:

    2.14.1+cu130
    True

The CUDA group must not be combined with the CPU group.

The --no-default-groups option prevents the default CPU
dependency group from being selected.

The explicit dev group retains pytest, Ruff, and mypy.

## Running tests with the CUDA profile

Use the isolated environment:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_PROJECT_ENVIRONMENT=/tmp/robustlearn-cuda uv run --locked --no-sync --no-default-groups --group dev --group cuda pytest -q tests/integration/test_bc_cuda.py

The CUDA tests exercise:

- GPU inference
- Checkpoint restoration on GPU
- CPU/GPU numerical consistency

The tests skip automatically when CUDA is unavailable.

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 prevents unrelated
system-installed ROS pytest plugins from loading in the
isolated environment.

## Running tests with the CPU profile

Use:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --no-sync pytest -q tests/integration/test_bc_cuda.py

Expected result on CPU-only machines:

    3 skipped

## Repository dependency configuration

The dependency profiles are defined in pyproject.toml.

CPU and CUDA are mutually exclusive dependency groups.

Both environments share uv.lock but resolve to different
PyTorch distributions.

The CPU group is selected by default.

The CUDA group must be explicitly selected.

## Limitations

- CUDA requires a compatible NVIDIA GPU and driver.
- CUDA execution is not available with CPU-only PyTorch.
- GPU execution may differ numerically from CPU execution.
- CUDA-enabled PyTorch does not automatically move models
  and tensors onto the GPU.
- The current CUDA integration tests validate inference
  and checkpoint loading, not long-running GPU training.
- The CUDA installation downloads large NVIDIA dependencies.
- ROS 2 and MuJoCo execution remain separate from CUDA setup.

## Scope

This infrastructure configuration does not change the
frozen v0.1 dataset or the Behavior Cloning architecture.

BC training is handled by subsequent project issues.
