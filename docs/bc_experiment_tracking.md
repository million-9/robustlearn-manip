# BC Experiment Tracking and Offline Evaluation

## Purpose

Issue #114 adds local MLflow experiment tracking and offline
validation metrics to the frozen v0.1 Behavior Cloning baseline.

Offline prediction accuracy does not establish closed-loop
robot manipulation success.

## Environment

Install the optional tracking dependency group:

    uv sync --locked --group tracking

MLflow uses a local SQLite database. No external account
or hosted tracking service is required.

## Run a tracked experiment

From the repository root:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --locked --group tracking \
      python -m robustlearn.bc_train_cli \
      --dataset-root datasets/v0.1 \
      --output-dir /tmp/robustlearn-bc-checkpoints \
      --tracking-dir /tmp/robustlearn-bc-mlflow \
      --experiment-name robustlearn-bc \
      --max-epochs 50 \
      --seed 42 \
      --device cpu \
      --track-mlflow

Use fresh output directories to avoid overwriting checkpoints.

## Inspect experiments

Launch the local MLflow UI:

    uv run --locked --group tracking mlflow ui \
      --backend-store-uri sqlite:////tmp/robustlearn-bc-mlflow/mlflow.db \
      --port 5000

Open http://127.0.0.1:5000 in your browser.

Select the experiment to inspect:

- Training configuration and random seed
- Git revision and dataset SHA-256 fingerprint
- Per-epoch training and validation losses
- Aggregate validation MAE and RMSE
- Per-joint MAE and RMSE
- Best and final checkpoint artifacts
- Run status

The checkpoint paths are also stored as MLflow tags.

## Recorded baseline

Dataset: frozen v0.1

- Training maximum: 50 epochs
- Epochs completed: 50
- Best epoch: 46
- Seed: 42
- Training device: CPU
- Validation samples: 593
- Validation MAE: 0.0007996311 rad
- Validation RMSE: 0.0017378289 rad

The dataset fingerprint is:

    eb28d409db76e18c6a8e6f6f0ccb49429f0271d4bf8a6f16881637ae0dd5db3f

The recorded MLflow run ID is:

    49a059d7c7bc44e4b6f1fe8440873940

This run was stored under /tmp/robustlearn-114-full-mlflow.

## Evaluation scope

The evaluator uses the validation partition and restores
predictions to physical joint-position units (radians).

It reports aggregate and per-dimension MAE and RMSE.

The test partition is not used for checkpoint selection.

## Failure handling

Failures during MLflow logging mark the created run FAILED.

Training failures that occur before MLflow logging begins
do not currently create a failed MLflow run.

Exact checkpoint-based training resumption is not supported.

## Git hygiene

Generated checkpoints, MLflow databases, and experiment
artifacts must remain outside version control.

Prefer storing experiments under /tmp or another dedicated
ignored output directory.
