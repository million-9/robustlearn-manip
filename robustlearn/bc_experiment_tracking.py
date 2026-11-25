"""Local MLflow tracking for reproducible BC experiments."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any


def record_bc_experiment(
    *,
    tracking_dir: Path,
    experiment_name: str,
    parameters: dict[str, str | int | float | bool],
    train_losses: list[float],
    validation_losses: list[float],
    validation_metrics: dict[str, float | int],
    best_checkpoint: Path,
    final_checkpoint: Path,
    git_revision: str,
    dataset_sha256: str,
) -> str:
    """Persist a completed training run in a local MLflow store."""

    if not train_losses or len(train_losses) != len(validation_losses):
        raise ValueError("Training and validation histories must align")

    for loss in train_losses + validation_losses:
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Loss history must be finite and non-negative")

    for name, value in validation_metrics.items():
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Invalid validation metric: {name}")

    for checkpoint in (best_checkpoint, final_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    tracking_dir.mkdir(parents=True, exist_ok=True)

    try:
        tracking_module = importlib.import_module("mlflow.tracking")
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requires the optional tracking dependency group"
        ) from exc

    MlflowClient: Any = tracking_module.MlflowClient

    database = (tracking_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{database}"
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            experiment_name,
            artifact_location=(tracking_dir / "artifacts").resolve().as_uri(),
        )
    )

    run = client.create_run(experiment_id)
    run_id: str = str(run.info.run_id)

    try:
        for name, parameter_value in parameters.items():
            client.log_param(run_id, name, str(parameter_value))

        client.log_param(run_id, "git_revision", git_revision)
        client.log_param(run_id, "dataset_sha256", dataset_sha256)

        for epoch, (train_loss, validation_loss) in enumerate(
            zip(train_losses, validation_losses, strict=True),
            start=1,
        ):
            client.log_metric(run_id, "train_loss", train_loss, step=epoch)
            client.log_metric(
                run_id, "validation_loss", validation_loss, step=epoch
            )

        for name, value in validation_metrics.items():
            client.log_metric(run_id, name, float(value))

        for name, path in (
            ("best_checkpoint", best_checkpoint),
            ("final_checkpoint", final_checkpoint),
        ):
            client.set_tag(run_id, name, str(path.resolve()))
            client.log_artifact(
                run_id,
                str(path),
                artifact_path="checkpoints",
            )

        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise

    return run_id
