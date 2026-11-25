"""End-to-end reproducibility tests for BC training on frozen v0.1."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from robustlearn import bc_train_cli
from robustlearn.bc_training import (
    load_bc_training_checkpoint,
    restore_bc_training_checkpoint,
)
from robustlearn.dataset_freeze import verify_dataset_snapshot

DATASET_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "v0.1"
)


def run_smoke(
    output_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, object]:
    result = bc_train_cli.main(
        [
            "--dataset-root",
            str(DATASET_ROOT),
            "--output-dir",
            str(output_dir),
            "--max-epochs",
            "1",
            "--batch-size",
            "256",
            "--seed",
            "42",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    return json.loads(capsys.readouterr().out)


def test_bc_cli_cpu_checkpoint_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = run_smoke(tmp_path / "run", capsys)

    fingerprint = verify_dataset_snapshot(DATASET_ROOT)[
        "dataset_sha256"
    ]

    assert summary["dataset_sha256"] == fingerprint
    assert summary["seed"] == 42
    assert summary["epochs_completed"] == 1
    assert summary["best_epoch"] == 1

    best_path = Path(str(summary["best_checkpoint"]))
    final_path = Path(str(summary["final_checkpoint"]))

    assert best_path.is_file()
    assert final_path.is_file()

    best = load_bc_training_checkpoint(best_path)
    final = load_bc_training_checkpoint(final_path)

    assert best["dataset_sha256"] == fingerprint
    assert final["dataset_sha256"] == fingerprint
    assert best["epoch"] == 1
    assert final["epoch"] == 1
    assert best["training_config"]["seed"] == 42

    model, normalization, config, history, _ = (
        restore_bc_training_checkpoint(
            best_path,
            expected_dataset_sha256=str(fingerprint),
        )
    )

    assert not model.training
    assert model.config.observation_dim == 23
    assert model.config.action_dim == 7
    assert config.seed == 42
    assert history.epochs_completed == 1
    assert normalization.observation_mean.shape == (23,)
    assert normalization.action_mean.shape == (7,)


def test_bc_cli_deterministic_and_never_loads_test_split(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual_dataset_class = bc_train_cli.BCDataset
    requested_splits: list[str] = []

    def observed_dataset(
        *args: object,
        **kwargs: object,
    ) -> object:
        split = kwargs.get("split")
        assert isinstance(split, str)
        requested_splits.append(split)
        return actual_dataset_class(*args, **kwargs)

    with patch.object(
        bc_train_cli,
        "BCDataset",
        side_effect=observed_dataset,
    ):
        first = run_smoke(tmp_path / "first", capsys)
        second = run_smoke(tmp_path / "second", capsys)

    assert requested_splits == [
        "train",
        "validation",
        "train",
        "validation",
    ]

    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["train_losses"] == second["train_losses"]
    assert first["validation_losses"] == second["validation_losses"]
    assert first["best_epoch"] == second["best_epoch"]

    first_checkpoint = load_bc_training_checkpoint(
        Path(str(first["best_checkpoint"]))
    )
    second_checkpoint = load_bc_training_checkpoint(
        Path(str(second["best_checkpoint"]))
    )

    first_weights = first_checkpoint["model_state_dict"]
    second_weights = second_checkpoint["model_state_dict"]

    assert first_weights.keys() == second_weights.keys()

    for name in first_weights:
        assert torch.equal(
            first_weights[name],
            second_weights[name],
        )

    for name in (
        "observation_mean",
        "observation_std",
        "action_mean",
        "action_std",
    ):
        assert torch.equal(
            first_checkpoint["normalization"][name],
            second_checkpoint["normalization"][name],
        )



def test_bc_cli_mlflow_tracking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    output = tmp_path / "checkpoints"
    tracking = tmp_path / "tracking"

    assert bc_train_cli.main([
        "--dataset-root", str(DATASET_ROOT),
        "--output-dir", str(output),
        "--tracking-dir", str(tracking),
        "--experiment-name", "bc-integration-test",
        "--max-epochs", "1",
        "--batch-size", "256",
        "--seed", "42",
        "--device", "cpu",
        "--track-mlflow",
    ]) == 0

    summary = json.loads(capsys.readouterr().out)
    run_id = summary["mlflow_run_id"]
    metrics = summary["offline_validation_metrics"]

    assert metrics["samples"] == 593
    assert metrics["mae_rad"] >= 0
    assert metrics["rmse_rad"] >= metrics["mae_rad"]
    assert "joint_7_rmse_rad" in metrics

    database = (tracking / "mlflow.db").resolve()
    assert database.is_file()

    client = MlflowClient(
        tracking_uri=f"sqlite:///{database}"
    )
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    assert run.data.params["dataset_sha256"] == (
        summary["dataset_sha256"]
    )
    assert run.data.params["git_revision"]

    assert run.data.metrics["mae_rad"] == pytest.approx(
        metrics["mae_rad"]
    )

    for key in ("train_loss", "validation_loss"):
        history = client.get_metric_history(run_id, key)
        assert len(history) == 1
        assert history[0].step == 1

    artifacts = client.list_artifacts(run_id, "checkpoints")
    assert {item.path for item in artifacts} == {
        "checkpoints/best.pt",
        "checkpoints/final.pt",
    }

    assert (output / "best.pt").is_file()
    assert (output / "final.pt").is_file()



def test_bc_tracked_offline_metrics_are_reproducible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("mlflow")

    summaries = []

    for index in range(2):
        output = tmp_path / f"run-{index}"
        tracking = tmp_path / "tracking"

        assert bc_train_cli.main([
            "--dataset-root", str(DATASET_ROOT),
            "--output-dir", str(output),
            "--tracking-dir", str(tracking),
            "--experiment-name", "bc-reproducibility",
            "--max-epochs", "1",
            "--batch-size", "256",
            "--seed", "42",
            "--device", "cpu",
            "--track-mlflow",
        ]) == 0

        summaries.append(json.loads(capsys.readouterr().out))

    first, second = summaries

    assert first["mlflow_run_id"] != second["mlflow_run_id"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["train_losses"] == second["train_losses"]
    assert first["validation_losses"] == second["validation_losses"]
    assert (
        first["offline_validation_metrics"]
        == second["offline_validation_metrics"]
    )

    assert first["offline_validation_metrics"]["samples"] == 593
