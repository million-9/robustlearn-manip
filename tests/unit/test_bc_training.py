"""Unit tests for Behavior Cloning training utilities."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from robustlearn.bc_dataset import BCNormalization
from robustlearn.bc_policy import BCPolicy, BCPolicyConfig
from robustlearn.bc_training import (
    BCEarlyStopping,
    BCTrainingConfig,
    BCTrainingHistory,
    bc_gradient_step,
    build_bc_training_checkpoint,
    load_bc_training_checkpoint,
    restore_bc_training_checkpoint,
    run_bc_training,
    save_bc_training_checkpoint,
    train_bc_epoch,
    validate_bc_epoch,
)


@pytest.mark.parametrize("loss_name", ["huber", "mse"])
def test_gradient_step_updates_parameters(loss_name: str) -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(32, 32), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    generator = torch.Generator().manual_seed(123)
    observations = torch.randn(16, 23, generator=generator)
    actions = torch.randn(16, 7, generator=generator)

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss = bc_gradient_step(
        model,
        optimizer,
        observations,
        actions,
        loss_name=loss_name,
        max_grad_norm=1.0,
    )

    assert math.isfinite(loss)
    assert loss >= 0.0

    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )

    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "observation_shape, action_shape",
    [
        ((16, 22), (16, 7)),
        ((16, 23), (16, 6)),
        ((16, 23), (15, 7)),
        ((0, 23), (0, 7)),
        ((23,), (7,)),
    ],
)
def test_invalid_training_shapes(
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
) -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    observations = torch.zeros(observation_shape)
    actions = torch.zeros(action_shape)

    with pytest.raises(ValueError):
        bc_gradient_step(
            model,
            optimizer,
            observations,
            actions,
        )


@pytest.mark.parametrize(
    "loss_name, max_grad_norm",
    [
        ("invalid", 1.0),
        ("huber", 0.0),
        ("huber", -1.0),
        ("huber", float("nan")),
        ("huber", float("inf")),
    ],
)
def test_invalid_training_configuration(
    loss_name: str,
    max_grad_norm: float,
) -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with pytest.raises(ValueError):
        bc_gradient_step(
            model,
            optimizer,
            torch.zeros(4, 23),
            torch.zeros(4, 7),
            loss_name=loss_name,
            max_grad_norm=max_grad_norm,
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_training_data_rejected(invalid_value: float) -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    observations = torch.zeros(4, 23)
    actions = torch.zeros(4, 7)
    observations[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        bc_gradient_step(model, optimizer, observations, actions)

    observations[0, 0] = 0.0
    actions[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        bc_gradient_step(model, optimizer, observations, actions)



def test_train_bc_epoch_updates_parameters() -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    generator = torch.Generator().manual_seed(123)
    observations = torch.randn(10, 23, generator=generator)
    actions = torch.randn(10, 7, generator=generator)

    dataset = torch.utils.data.TensorDataset(observations, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    mean_loss = train_bc_epoch(
        model,
        optimizer,
        loader,
        device=torch.device("cpu"),
    )

    assert math.isfinite(mean_loss)
    assert mean_loss >= 0.0
    assert model.training

    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )


def test_validation_epoch_preserves_parameters() -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))

    observations = torch.randn(10, 23)
    actions = torch.randn(10, 7)

    dataset = torch.utils.data.TensorDataset(observations, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    mean_loss = validate_bc_epoch(
        model,
        loader,
        device=torch.device("cpu"),
    )

    assert math.isfinite(mean_loss)
    assert mean_loss >= 0.0
    assert not model.training

    assert all(
        torch.equal(previous, current)
        for previous, current in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )


def test_validation_epoch_sample_weighting() -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))

    observations = torch.randn(10, 23)
    actions = torch.randn(10, 7)

    dataset = torch.utils.data.TensorDataset(observations, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    actual = validate_bc_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        loss_name="mse",
    )

    with torch.inference_mode():
        expected = torch.nn.functional.mse_loss(
            model(observations),
            actions,
        ).item()

    assert actual == pytest.approx(expected, rel=1e-6)


def test_empty_epoch_dataloaders_rejected() -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    empty_dataset = torch.utils.data.TensorDataset(
        torch.empty(0, 23),
        torch.empty(0, 7),
    )
    loader = torch.utils.data.DataLoader(
        empty_dataset,
        batch_size=4,
    )

    with pytest.raises(ValueError, match="no samples"):
        train_bc_epoch(
            model,
            optimizer,
            loader,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="no samples"):
        validate_bc_epoch(
            model,
            loader,
            device=torch.device("cpu"),
        )



def test_training_config_defaults() -> None:
    config = BCTrainingConfig()

    assert config.learning_rate == 3e-4
    assert config.batch_size == 256
    assert config.max_epochs == 100
    assert config.loss_name == "huber"
    assert config.max_grad_norm == 1.0
    assert config.patience == 10
    assert config.min_delta == 1e-4
    assert config.seed == 42
    assert config.device == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"batch_size": True},
        {"max_epochs": -1},
        {"max_epochs": 2.5},
        {"loss_name": "invalid"},
        {"max_grad_norm": 0.0},
        {"max_grad_norm": float("nan")},
        {"patience": 0},
        {"patience": 2.5},
        {"min_delta": -1.0},
        {"min_delta": float("inf")},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_invalid_training_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BCTrainingConfig(**kwargs)  # type: ignore[arg-type]



def test_early_stopping_patience() -> None:
    stopping = BCEarlyStopping(patience=3, min_delta=1e-4)

    assert stopping.update(0.5, epoch=1)
    assert stopping.best_epoch == 1
    assert not stopping.should_stop

    assert not stopping.update(0.49995, epoch=2)
    assert stopping.epochs_without_improvement == 1

    assert not stopping.update(0.5, epoch=3)
    assert not stopping.should_stop

    assert not stopping.update(0.51, epoch=4)
    assert stopping.should_stop
    assert stopping.best_epoch == 1


def test_early_stopping_improvement_resets_patience() -> None:
    stopping = BCEarlyStopping(patience=2, min_delta=1e-4)

    assert stopping.update(0.5, epoch=1)
    assert not stopping.update(0.5, epoch=2)

    assert stopping.update(0.4, epoch=3)
    assert stopping.best_epoch == 3
    assert stopping.best_loss == pytest.approx(0.4)
    assert stopping.epochs_without_improvement == 0
    assert not stopping.should_stop


def test_early_stopping_zero_min_delta() -> None:
    stopping = BCEarlyStopping(patience=2, min_delta=0.0)

    assert stopping.update(0.5, epoch=1)
    assert not stopping.update(0.5, epoch=2)
    assert stopping.update(0.499, epoch=3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patience": 0},
        {"patience": -1},
        {"patience": 1.5},
        {"patience": True},
        {"min_delta": -1.0},
        {"min_delta": float("nan")},
        {"min_delta": float("inf")},
    ],
)
def test_invalid_early_stopping_config(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BCEarlyStopping(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "loss, epoch",
    [
        (float("nan"), 1),
        (float("inf"), 1),
        (float("-inf"), 1),
        (0.5, 0),
        (0.5, -1),
        (0.5, 1.5),
        (0.5, True),
    ],
)
def test_invalid_early_stopping_update(
    loss: float,
    epoch: int,
) -> None:
    stopping = BCEarlyStopping()

    with pytest.raises(ValueError):
        stopping.update(loss, epoch)  # type: ignore[arg-type]



@pytest.mark.parametrize("invalid_epoch", [1, 0, -1])
def test_early_stopping_rejects_out_of_order_epochs(
    invalid_epoch: int,
) -> None:
    stopping = BCEarlyStopping(patience=3)

    assert stopping.update(0.5, epoch=1)

    with pytest.raises(ValueError):
        stopping.update(0.4, epoch=invalid_epoch)

    assert stopping.best_epoch == 1
    assert stopping.last_epoch == 1
    assert stopping.best_loss == pytest.approx(0.5)


def test_early_stopping_accepts_increasing_epochs() -> None:
    stopping = BCEarlyStopping(patience=3)

    assert stopping.update(0.5, epoch=1)
    assert stopping.update(0.4, epoch=2)

    assert stopping.last_epoch == 2
    assert stopping.best_epoch == 2
    assert stopping.best_loss == pytest.approx(0.4)



def test_training_history_records_losses() -> None:
    history = BCTrainingHistory()

    assert history.epochs_completed == 0
    assert history.train_losses == []
    assert history.validation_losses == []

    history.append(0.5, 0.6)
    history.append(0.3, 0.4)

    assert history.epochs_completed == 2
    assert history.train_losses == [0.5, 0.3]
    assert history.validation_losses == [0.6, 0.4]


def test_training_histories_are_independent() -> None:
    first = BCTrainingHistory()
    second = BCTrainingHistory()

    first.append(0.5, 0.6)

    assert second.epochs_completed == 0
    assert second.train_losses == []
    assert second.validation_losses == []


@pytest.mark.parametrize(
    "train_loss, validation_loss",
    [
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (-1.0, 0.5),
        (0.5, float("nan")),
        (0.5, float("inf")),
        (0.5, -1.0),
    ],
)
def test_training_history_rejects_invalid_losses(
    train_loss: float,
    validation_loss: float,
) -> None:
    history = BCTrainingHistory()

    with pytest.raises(ValueError):
        history.append(train_loss, validation_loss)

    assert history.epochs_completed == 0
    assert history.train_losses == []
    assert history.validation_losses == []



def make_checkpoint_test_inputs(
) -> tuple[BCPolicy, BCTrainingConfig, BCNormalization, BCTrainingHistory]:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    config = BCTrainingConfig(max_epochs=2)

    normalization = BCNormalization(
        observation_mean=torch.zeros(23),
        observation_std=torch.ones(23),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )

    history = BCTrainingHistory()
    history.append(0.5, 0.6)
    history.append(0.4, 0.45)

    return model, config, normalization, history


def test_bc_checkpoint_preserves_model_snapshot() -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    weights = checkpoint["model_state_dict"]
    assert isinstance(weights, dict)

    original = {
        name: tensor.clone()
        for name, tensor in weights.items()
    }

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    for name, tensor in weights.items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, original[name])


def test_bc_checkpoint_preserves_normalization_snapshot() -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    saved = checkpoint["normalization"]
    assert isinstance(saved, dict)

    with torch.no_grad():
        normalization.observation_mean.add_(5.0)
        normalization.action_mean.add_(5.0)

    assert torch.equal(
        saved["observation_mean"],
        torch.zeros(23),
    )
    assert torch.equal(
        saved["action_mean"],
        torch.zeros(7),
    )


def test_bc_checkpoint_metadata() -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    assert checkpoint["format_version"] == 1
    assert checkpoint["dataset_sha256"] == "a" * 64
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_epoch"] == 2
    assert checkpoint["best_validation_loss"] == pytest.approx(0.45)

    assert checkpoint["training_config"]["seed"] == 42
    assert checkpoint["model_config"]["observation_dim"] == 23
    assert checkpoint["model_config"]["action_dim"] == 7

    assert checkpoint["history"]["train_losses"] == [0.5, 0.4]
    assert checkpoint["history"]["validation_losses"] == [0.6, 0.45]


@pytest.mark.parametrize(
    "fingerprint, epoch, best_epoch, best_loss",
    [
        ("invalid", 2, 2, 0.45),
        ("z" * 64, 2, 2, 0.45),
        ("a" * 64, 0, 1, 0.45),
        ("a" * 64, 2, 3, 0.45),
        ("a" * 64, 2, 0, 0.45),
        ("a" * 64, 2, 2, float("nan")),
        ("a" * 64, 2, 2, float("inf")),
    ],
)
def test_bc_checkpoint_rejects_invalid_metadata(
    fingerprint: str,
    epoch: int,
    best_epoch: int,
    best_loss: float,
) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    with pytest.raises(ValueError):
        build_bc_training_checkpoint(
            model,
            config,
            normalization,
            history,
            dataset_sha256=fingerprint,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
        )


def test_bc_checkpoint_rejects_history_mismatch() -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    with pytest.raises(ValueError):
        build_bc_training_checkpoint(
            model,
            config,
            normalization,
            history,
            dataset_sha256="a" * 64,
            epoch=3,
            best_epoch=2,
            best_validation_loss=0.45,
        )



def test_bc_checkpoint_atomic_save_round_trip(
    tmp_path: Path,
) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    destination = tmp_path / "nested" / "training.pt"

    save_bc_training_checkpoint(checkpoint, destination)

    assert destination.is_file()

    loaded = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded["dataset_sha256"] == "a" * 64
    assert loaded["epoch"] == 2
    assert loaded["history"]["train_losses"] == [0.5, 0.4]

    assert set(loaded["model_state_dict"]) == set(
        model.state_dict()
    )

    assert not list(destination.parent.glob("*.tmp"))


def test_bc_checkpoint_atomic_save_replaces_existing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "training.pt"
    destination.write_bytes(b"old checkpoint")

    save_bc_training_checkpoint(
        {"format_version": 1, "epoch": 2},
        destination,
    )

    loaded = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded["epoch"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_bc_checkpoint_atomic_save_failure_preserves_existing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "training.pt"
    destination.write_bytes(b"original checkpoint")

    with patch(
        "robustlearn.bc_training.torch.save",
        side_effect=RuntimeError("simulated serialization failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="simulated serialization failure",
        ):
            save_bc_training_checkpoint(
                {"format_version": 1},
                destination,
            )

    assert destination.read_bytes() == b"original checkpoint"
    assert sorted(tmp_path.iterdir()) == [destination]



def test_bc_checkpoint_loader_round_trip(tmp_path: Path) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    destination = tmp_path / "training.pt"
    save_bc_training_checkpoint(checkpoint, destination)

    loaded = load_bc_training_checkpoint(destination)

    assert loaded["format_version"] == 1
    assert loaded["dataset_sha256"] == "a" * 64
    assert loaded["epoch"] == 2
    assert loaded["best_epoch"] == 2
    assert loaded["best_validation_loss"] == pytest.approx(0.45)


@pytest.mark.parametrize(
    "missing_key",
    [
        "format_version",
        "model_config",
        "training_config",
        "model_state_dict",
        "normalization",
        "dataset_sha256",
        "epoch",
        "best_epoch",
        "best_validation_loss",
        "history",
    ],
)
def test_bc_checkpoint_loader_rejects_missing_keys(
    tmp_path: Path,
    missing_key: str,
) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    checkpoint.pop(missing_key)
    destination = tmp_path / "invalid.pt"
    torch.save(checkpoint, destination)

    with pytest.raises(ValueError, match="missing keys"):
        load_bc_training_checkpoint(destination)


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("format_version", 2),
        ("format_version", True),
        ("dataset_sha256", "invalid"),
        ("dataset_sha256", "z" * 64),
        ("epoch", 0),
        ("epoch", True),
        ("best_epoch", 0),
        ("best_epoch", 3),
        ("best_validation_loss", float("nan")),
        ("best_validation_loss", float("inf")),
        ("model_config", None),
        ("normalization", None),
        ("history", None),
    ],
)
def test_bc_checkpoint_loader_rejects_invalid_metadata(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    checkpoint[field] = invalid_value

    destination = tmp_path / "invalid.pt"
    torch.save(checkpoint, destination)

    with pytest.raises(ValueError):
        load_bc_training_checkpoint(destination)



def test_bc_checkpoint_restore_round_trip(tmp_path: Path) -> None:
    model, config, normalization, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    destination = tmp_path / "training.pt"
    save_bc_training_checkpoint(checkpoint, destination)

    restored, norm, training, losses, metadata = (
        restore_bc_training_checkpoint(
            destination,
            expected_dataset_sha256="a" * 64,
        )
    )

    assert not restored.training
    assert training.seed == config.seed
    assert losses.train_losses == history.train_losses
    assert losses.validation_losses == history.validation_losses
    assert metadata["epoch"] == 2

    torch.testing.assert_close(
        norm.observation_mean,
        normalization.observation_mean,
    )

    observations = torch.randn(4, 23)

    with torch.inference_mode():
        expected = model(observations)
        actual = restored(observations)

    torch.testing.assert_close(actual, expected)


def test_bc_checkpoint_restore_rejects_wrong_dataset(
    tmp_path: Path,
) -> None:
    model, config, norm, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        norm,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    destination = tmp_path / "training.pt"
    save_bc_training_checkpoint(checkpoint, destination)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        restore_bc_training_checkpoint(
            destination,
            expected_dataset_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "normalization_shape",
        "normalization_std",
        "history_length",
        "history_nonfinite",
        "best_loss_mismatch",
        "weights_nonfinite",
    ],
)
def test_bc_checkpoint_restore_rejects_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    model, config, norm, history = make_checkpoint_test_inputs()

    checkpoint = build_bc_training_checkpoint(
        model,
        config,
        norm,
        history,
        dataset_sha256="a" * 64,
        epoch=2,
        best_epoch=2,
        best_validation_loss=0.45,
    )

    if corruption == "normalization_shape":
        checkpoint["normalization"]["observation_mean"] = torch.zeros(22)

    elif corruption == "normalization_std":
        checkpoint["normalization"]["action_std"][0] = 0.0

    elif corruption == "history_length":
        checkpoint["history"]["train_losses"].pop()

    elif corruption == "history_nonfinite":
        checkpoint["history"]["validation_losses"][0] = float("nan")

    elif corruption == "best_loss_mismatch":
        checkpoint["best_validation_loss"] = 0.1

    elif corruption == "weights_nonfinite":
        first_weight = next(iter(checkpoint["model_state_dict"].values()))
        first_weight.view(-1)[0] = float("nan")

    destination = tmp_path / "corrupt.pt"
    torch.save(checkpoint, destination)

    with pytest.raises(ValueError):
        restore_bc_training_checkpoint(destination)



def make_coordinator_test_loaders(
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    generator = torch.Generator().manual_seed(123)

    observations = torch.randn(
        12, 23, generator=generator
    )
    actions = torch.randn(
        12, 7, generator=generator
    )

    dataset = torch.utils.data.TensorDataset(
        observations, actions
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    return train_loader, validation_loader


def test_bc_training_coordinator_saves_checkpoints(
    tmp_path: Path,
) -> None:
    model = BCPolicy(
        BCPolicyConfig(hidden_dims=(16,), seed=42)
    )
    _, _, normalization, _ = make_checkpoint_test_inputs()

    train_loader, validation_loader = (
        make_coordinator_test_loaders()
    )

    config = BCTrainingConfig(
        max_epochs=3,
        batch_size=4,
        patience=3,
        seed=42,
    )

    result = run_bc_training(
        model,
        train_loader,
        validation_loader,
        normalization,
        config,
        dataset_sha256="a" * 64,
        output_dir=tmp_path,
    )

    assert result.best_checkpoint.is_file()
    assert result.final_checkpoint.is_file()
    assert result.epochs_completed == 3
    assert result.history.epochs_completed == 3
    assert 1 <= result.best_epoch <= 3
    assert not result.best_model.training

    best = load_bc_training_checkpoint(
        result.best_checkpoint
    )
    final = load_bc_training_checkpoint(
        result.final_checkpoint
    )

    assert best["epoch"] == result.best_epoch
    assert final["epoch"] == result.epochs_completed
    assert final["best_epoch"] == result.best_epoch

    assert len(final["history"]["train_losses"]) == 3
    assert len(final["history"]["validation_losses"]) == 3

    restored, _, _, _, _ = restore_bc_training_checkpoint(
        result.best_checkpoint,
        expected_dataset_sha256="a" * 64,
    )

    observations = torch.randn(2, 23)

    with torch.inference_mode():
        actual = result.best_model(observations)
        expected = restored(observations)

    torch.testing.assert_close(actual, expected)


def test_bc_training_coordinator_early_stops(
    tmp_path: Path,
) -> None:
    model = BCPolicy(
        BCPolicyConfig(hidden_dims=(16,), seed=42)
    )
    _, _, normalization, _ = make_checkpoint_test_inputs()

    train_loader, validation_loader = (
        make_coordinator_test_loaders()
    )

    config = BCTrainingConfig(
        max_epochs=10,
        batch_size=4,
        patience=1,
        min_delta=1e6,
        seed=42,
    )

    result = run_bc_training(
        model,
        train_loader,
        validation_loader,
        normalization,
        config,
        dataset_sha256="a" * 64,
        output_dir=tmp_path,
    )

    assert result.best_epoch == 1
    assert result.epochs_completed == 2

    best = load_bc_training_checkpoint(
        result.best_checkpoint
    )
    final = load_bc_training_checkpoint(
        result.final_checkpoint
    )

    assert best["epoch"] == 1
    assert final["epoch"] == 2
    assert final["best_epoch"] == 1

    assert result.history.epochs_completed == 2



def test_bc_training_coordinator_rejects_overwrite(
    tmp_path: Path,
) -> None:
    model = BCPolicy(BCPolicyConfig(hidden_dims=(16,), seed=42))
    _, _, normalization, _ = make_checkpoint_test_inputs()

    train_loader, validation_loader = make_coordinator_test_loaders()

    config = BCTrainingConfig(
        max_epochs=1,
        batch_size=4,
        patience=1,
        seed=42,
    )

    best_path = tmp_path / "best.pt"
    best_path.write_bytes(b"existing checkpoint")

    with pytest.raises(FileExistsError, match="already exist"):
        run_bc_training(
            model,
            train_loader,
            validation_loader,
            normalization,
            config,
            dataset_sha256="a" * 64,
            output_dir=tmp_path,
        )

    assert best_path.read_bytes() == b"existing checkpoint"
    assert not (tmp_path / "final.pt").exists()
