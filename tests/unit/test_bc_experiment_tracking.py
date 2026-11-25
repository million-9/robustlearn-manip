"""Local MLflow experiment tracking regression tests."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient

from robustlearn.bc_experiment_tracking import record_bc_experiment


def test_local_tracking_round_trip(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    final = tmp_path / "final.pt"
    best.write_bytes(b"best checkpoint")
    final.write_bytes(b"final checkpoint")

    tracking_dir = tmp_path / "mlruns"

    run_id = record_bc_experiment(
        tracking_dir=tracking_dir,
        experiment_name="bc-test",
        parameters={"seed": 42, "learning_rate": 3e-4},
        train_losses=[0.5, 0.3],
        validation_losses=[0.4, 0.2],
        validation_metrics={
            "mae_rad": 0.01,
            "joint_1_rmse_rad": 0.02,
        },
        best_checkpoint=best,
        final_checkpoint=final,
        git_revision="abc123",
        dataset_sha256="a" * 64,
    )

    client = MlflowClient(
        tracking_uri=f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}"
    )
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    assert run.data.params["git_revision"] == "abc123"
    assert run.data.params["dataset_sha256"] == "a" * 64
    assert run.data.metrics["mae_rad"] == pytest.approx(0.01)

    history = client.get_metric_history(run_id, "validation_loss")
    assert [metric.step for metric in history] == [1, 2]
    assert [metric.value for metric in history] == [0.4, 0.2]

    artifacts = client.list_artifacts(run_id, "checkpoints")
    assert {item.path for item in artifacts} == {
        "checkpoints/best.pt",
        "checkpoints/final.pt",
    }


@pytest.mark.parametrize(
    "train, validation",
    [
        ([], []),
        ([0.1], []),
        ([float("nan")], [0.1]),
        ([0.1], [-0.5]),
    ],
)
def test_invalid_tracking_histories(
    tmp_path: Path,
    train: list[float],
    validation: list[float],
) -> None:
    with pytest.raises(ValueError):
        record_bc_experiment(
            tracking_dir=tmp_path / "mlruns",
            experiment_name="bc-test",
            parameters={},
            train_losses=train,
            validation_losses=validation,
            validation_metrics={},
            best_checkpoint=tmp_path / "best.pt",
            final_checkpoint=tmp_path / "final.pt",
            git_revision="abc123",
            dataset_sha256="a" * 64,
        )



def test_tracking_failure_records_failed_status(
    tmp_path: Path,
) -> None:
    best = tmp_path / "best.pt"
    final = tmp_path / "final.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")

    tracking_dir = tmp_path / "mlruns"

    with patch.object(
        MlflowClient,
        "log_artifact",
        side_effect=RuntimeError("simulated artifact failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="simulated artifact failure",
        ):
            record_bc_experiment(
                tracking_dir=tracking_dir,
                experiment_name="failure-test",
                parameters={"seed": 42},
                train_losses=[0.5],
                validation_losses=[0.4],
                validation_metrics={"mae_rad": 0.01},
                best_checkpoint=best,
                final_checkpoint=final,
                git_revision="abc123",
                dataset_sha256="a" * 64,
            )

    client = MlflowClient(
        tracking_uri=(
            f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}"
        )
    )
    experiment = client.get_experiment_by_name("failure-test")
    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
