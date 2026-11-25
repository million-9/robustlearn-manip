"""Reproducible Behavior Cloning training utilities."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from robustlearn.bc_dataset import BCNormalization
from robustlearn.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    resolve_bc_device,
)


def bc_gradient_step(
    model: BCPolicy,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    loss_name: str = "huber",
    max_grad_norm: float = 1.0,
) -> float:
    """Perform one supervised BC update on normalized tensors."""

    if loss_name not in ("huber", "mse"):
        raise ValueError("loss_name must be 'huber' or 'mse'")

    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Training inputs must be batched 2D tensors")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Observation/action batch sizes must match")

    if observations.shape[0] == 0:
        raise ValueError("Training batch must not be empty")

    if observations.shape[1] != model.config.observation_dim:
        raise ValueError("Observation dimension mismatch")

    if actions.shape[1] != model.config.action_dim:
        raise ValueError("Action dimension mismatch")

    if not torch.isfinite(observations).all():
        raise ValueError("Observations must be finite")

    if not torch.isfinite(actions).all():
        raise ValueError("Actions must be finite")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    predictions = model(observations)

    criterion: nn.Module = (
        nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()
    )
    loss = criterion(predictions, actions)

    if not torch.isfinite(loss):
        raise ValueError("BC training loss is not finite")

    loss.backward()

    gradient_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    if not torch.isfinite(gradient_norm):
        raise ValueError("BC gradient norm is not finite")

    optimizer.step()

    return float(loss.detach().item())



def train_bc_epoch(
    model: BCPolicy,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],
    *,
    device: torch.device,
    loss_name: str = "huber",
    max_grad_norm: float = 1.0,
) -> float:
    """Train one epoch and return sample-weighted mean loss."""

    total_loss = 0.0
    total_samples = 0

    for observations, actions in dataloader:
        observations = observations.to(device)
        actions = actions.to(device)

        batch_loss = bc_gradient_step(
            model,
            optimizer,
            observations,
            actions,
            loss_name=loss_name,
            max_grad_norm=max_grad_norm,
        )

        batch_size = observations.shape[0]
        total_loss += batch_loss * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader contains no samples")

    mean_loss = total_loss / total_samples

    if not math.isfinite(mean_loss):
        raise ValueError("Mean training loss is not finite")

    return mean_loss



def validate_bc_epoch(
    model: BCPolicy,
    dataloader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],
    *,
    device: torch.device,
    loss_name: str = "huber",
) -> float:
    """Evaluate one epoch without changing model parameters."""

    if loss_name not in ("huber", "mse"):
        raise ValueError("loss_name must be 'huber' or 'mse'")

    criterion: nn.Module = (
        nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()
    )

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for observations, actions in dataloader:
            observations = observations.to(device)
            actions = actions.to(device)

            if observations.ndim != 2 or actions.ndim != 2:
                raise ValueError("Validation inputs must be batched 2D tensors")

            if observations.shape[0] != actions.shape[0]:
                raise ValueError("Validation batch sizes must match")

            if observations.shape[1] != model.config.observation_dim:
                raise ValueError("Validation observation dimension mismatch")

            if actions.shape[1] != model.config.action_dim:
                raise ValueError("Validation action dimension mismatch")

            if not torch.isfinite(observations).all():
                raise ValueError("Validation observations must be finite")

            if not torch.isfinite(actions).all():
                raise ValueError("Validation actions must be finite")

            predictions = model(observations)
            loss = criterion(predictions, actions)

            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite")

            batch_size = observations.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader contains no samples")

    mean_loss = total_loss / total_samples

    if not math.isfinite(mean_loss):
        raise ValueError("Mean validation loss is not finite")

    return mean_loss



@dataclass(frozen=True)
class BCTrainingConfig:
    """Hyperparameters for reproducible Behavior Cloning training."""

    learning_rate: float = 3e-4
    batch_size: int = 256
    max_epochs: int = 100
    loss_name: str = "huber"
    max_grad_norm: float = 1.0
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

        if self.loss_name not in ("huber", "mse"):
            raise ValueError("loss_name must be 'huber' or 'mse'")

        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")

        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("patience must be positive")

        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be non-negative")



@dataclass
class BCEarlyStopping:
    """Track validation improvement and stopping patience."""

    patience: int = 10
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    last_epoch: int = 0

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("patience must be a positive integer")

        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

    def update(self, validation_loss: float, epoch: int) -> bool:
        """Record an epoch and return True when loss improves."""

        if not math.isfinite(validation_loss):
            raise ValueError("Validation loss must be finite")

        if type(epoch) is not int or epoch <= 0:
            raise ValueError("epoch must be a positive integer")

        if epoch <= self.last_epoch:
            raise ValueError("epoch must be strictly increasing")

        self.last_epoch = epoch

        improved = (
            self.best_epoch == 0
            or validation_loss < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return improved

    @property
    def should_stop(self) -> bool:
        """Return True when patience has been exhausted."""

        return self.epochs_without_improvement >= self.patience



@dataclass
class BCTrainingHistory:
    """Record sample-weighted training and validation losses by epoch."""

    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)

    def append(self, train_loss: float, validation_loss: float) -> None:
        """Append one completed epoch's losses."""

        if not math.isfinite(train_loss) or train_loss < 0:
            raise ValueError("Training loss must be finite and non-negative")

        if not math.isfinite(validation_loss) or validation_loss < 0:
            raise ValueError("Validation loss must be finite and non-negative")

        self.train_losses.append(float(train_loss))
        self.validation_losses.append(float(validation_loss))

    @property
    def epochs_completed(self) -> int:
        """Return number of recorded epochs."""

        return len(self.train_losses)



def build_bc_training_checkpoint(
    model: BCPolicy,
    training_config: BCTrainingConfig,
    normalization: BCNormalization,
    history: BCTrainingHistory,
    *,
    dataset_sha256: str,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
) -> dict[str, object]:
    """Build a reproducible BC training checkpoint payload."""

    if len(dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must contain 64 characters")

    try:
        int(dataset_sha256, 16)
    except ValueError as exc:
        raise ValueError("dataset_sha256 must be hexadecimal") from exc

    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be a positive integer")

    if type(best_epoch) is not int or not 1 <= best_epoch <= epoch:
        raise ValueError("best_epoch must be within completed epochs")

    if not math.isfinite(best_validation_loss):
        raise ValueError("best_validation_loss must be finite")

    if history.epochs_completed != epoch:
        raise ValueError("Checkpoint epoch must match training history")

    if len(history.validation_losses) != epoch:
        raise ValueError("Training and validation histories must align")

    return {
        "format_version": 1,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "normalization": {
            "observation_mean": normalization.observation_mean.clone(),
            "observation_std": normalization.observation_std.clone(),
            "action_mean": normalization.action_mean.clone(),
            "action_std": normalization.action_std.clone(),
        },
        "dataset_sha256": dataset_sha256,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": {
            "train_losses": list(history.train_losses),
            "validation_losses": list(history.validation_losses),
        },
    }



def save_bc_training_checkpoint(
    checkpoint: dict[str, object],
    path: str | Path,
) -> None:
    """Atomically save a BC training checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(checkpoint, temporary)

        os.replace(temporary_path, destination)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)



def load_bc_training_checkpoint(
    path: str | Path,
) -> dict[str, object]:
    """Load and structurally validate a BC training checkpoint."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("BC training checkpoint must be a dictionary")

    required_keys = {
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
    }

    missing = required_keys - checkpoint.keys()

    if missing:
        raise ValueError(
            f"BC training checkpoint missing keys: {sorted(missing)}"
        )

    if type(checkpoint["format_version"]) is not int or (
        checkpoint["format_version"] != 1
    ):
        raise ValueError("Unsupported BC training checkpoint version")

    for key in (
        "model_config",
        "training_config",
        "model_state_dict",
        "normalization",
        "history",
    ):
        if not isinstance(checkpoint[key], dict):
            raise ValueError(f"{key} must be a dictionary")

    fingerprint = checkpoint["dataset_sha256"]

    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Invalid checkpoint dataset fingerprint")

    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise ValueError(
            "Invalid checkpoint dataset fingerprint"
        ) from exc

    epoch = checkpoint["epoch"]
    best_epoch = checkpoint["best_epoch"]

    if type(epoch) is not int or epoch <= 0:
        raise ValueError("Invalid checkpoint epoch")

    if type(best_epoch) is not int or not 1 <= best_epoch <= epoch:
        raise ValueError("Invalid checkpoint best epoch")

    best_loss = checkpoint["best_validation_loss"]

    if not isinstance(best_loss, (int, float)) or not math.isfinite(
        best_loss
    ):
        raise ValueError("Invalid checkpoint validation loss")

    return checkpoint



def restore_bc_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_dataset_sha256: str | None = None,
) -> tuple[
    BCPolicy,
    BCNormalization,
    BCTrainingConfig,
    BCTrainingHistory,
    dict[str, object],
]:
    """Restore a trained BC policy and its reproducibility metadata."""

    resolved_device = resolve_bc_device(device)
    checkpoint = load_bc_training_checkpoint(path)

    if (
        expected_dataset_sha256 is not None
        and checkpoint["dataset_sha256"] != expected_dataset_sha256
    ):
        raise ValueError("Checkpoint dataset fingerprint mismatch")

    model_config_data = cast(
        "dict[str, Any]",
        checkpoint["model_config"],
    )
    training_config_data = cast(
        "dict[str, Any]",
        checkpoint["training_config"],
    )

    model_config = BCPolicyConfig(**model_config_data)
    training_config = BCTrainingConfig(**training_config_data)

    normalization_data = cast(
        "dict[str, object]",
        checkpoint["normalization"],
    )

    normalization_keys = (
        "observation_mean",
        "observation_std",
        "action_mean",
        "action_std",
    )

    expected_dimensions = (
        model_config.observation_dim,
        model_config.observation_dim,
        model_config.action_dim,
        model_config.action_dim,
    )

    normalized: dict[str, torch.Tensor] = {}

    for name, dimension in zip(
        normalization_keys,
        expected_dimensions,
        strict=True,
    ):
        value = normalization_data.get(name)

        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid normalization tensor: {name}")

        if value.ndim != 1 or value.shape[0] != dimension:
            raise ValueError(f"Normalization shape mismatch: {name}")

        if not torch.is_floating_point(value):
            raise ValueError(f"Normalization must be floating point: {name}")

        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite normalization values: {name}")

        if name.endswith("_std") and not bool((value > 0).all()):
            raise ValueError(f"Invalid normalization standard deviation: {name}")

        normalized[name] = value.detach().cpu().clone()

    normalization = BCNormalization(
        observation_mean=normalized["observation_mean"],
        observation_std=normalized["observation_std"],
        action_mean=normalized["action_mean"],
        action_std=normalized["action_std"],
    )

    history_data = cast("dict[str, object]", checkpoint["history"])

    train_losses = history_data.get("train_losses")
    validation_losses = history_data.get("validation_losses")

    def valid_losses(values: object) -> bool:
        return (
            isinstance(values, list)
            and all(
                type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
                for value in values
            )
        )

    if not valid_losses(train_losses) or not valid_losses(
        validation_losses
    ):
        raise ValueError("Invalid training loss history")

    train_values = cast("list[float]", train_losses)
    validation_values = cast("list[float]", validation_losses)

    epoch = cast("int", checkpoint["epoch"])
    best_epoch = cast("int", checkpoint["best_epoch"])
    best_loss = cast("float", checkpoint["best_validation_loss"])

    if len(train_values) != epoch or len(validation_values) != epoch:
        raise ValueError("Checkpoint loss history length mismatch")

    if best_loss < 0:
        raise ValueError("Checkpoint best validation loss is negative")

    if not math.isclose(
        validation_values[best_epoch - 1],
        best_loss,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("Best validation loss does not match history")

    history = BCTrainingHistory(
        train_losses=list(train_values),
        validation_losses=list(validation_values),
    )

    state = checkpoint["model_state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint model state")

    if not all(
        isinstance(key, str)
        and isinstance(value, torch.Tensor)
        and bool(torch.isfinite(value).all())
        for key, value in state.items()
    ):
        raise ValueError("Invalid checkpoint model weights")

    model = BCPolicy(model_config)
    model.load_state_dict(state, strict=True)
    model.to(resolved_device)
    model.eval()

    return model, normalization, training_config, history, checkpoint



@dataclass
class BCTrainingResult:
    """Completed training run and its validation-selected policy."""

    best_model: BCPolicy
    history: BCTrainingHistory
    best_epoch: int
    epochs_completed: int
    best_checkpoint: Path
    final_checkpoint: Path


def run_bc_training(
    model: BCPolicy,
    train_loader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],
    validation_loader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],
    normalization: BCNormalization,
    config: BCTrainingConfig,
    *,
    dataset_sha256: str,
    output_dir: str | Path,
) -> BCTrainingResult:
    """Train BC using validation-only checkpoint selection."""

    device = resolve_bc_device(config.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    best_path = output / "best.pt"
    final_path = output / "final.pt"

    if best_path.exists() or final_path.exists():
        raise FileExistsError(
            "Training checkpoints already exist in output directory"
        )

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    stopping = BCEarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
    )

    history = BCTrainingHistory()

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_bc_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            loss_name=config.loss_name,
            max_grad_norm=config.max_grad_norm,
        )

        validation_loss = validate_bc_epoch(
            model,
            validation_loader,
            device=device,
            loss_name=config.loss_name,
        )

        history.append(train_loss, validation_loss)

        improved = stopping.update(validation_loss, epoch)

        if improved:
            best_checkpoint = build_bc_training_checkpoint(
                model,
                config,
                normalization,
                history,
                dataset_sha256=dataset_sha256,
                epoch=epoch,
                best_epoch=stopping.best_epoch,
                best_validation_loss=stopping.best_loss,
            )
            save_bc_training_checkpoint(best_checkpoint, best_path)

        if stopping.should_stop:
            break

    if stopping.best_epoch == 0:
        raise RuntimeError("Training completed without a best epoch")

    final_checkpoint = build_bc_training_checkpoint(
        model,
        config,
        normalization,
        history,
        dataset_sha256=dataset_sha256,
        epoch=history.epochs_completed,
        best_epoch=stopping.best_epoch,
        best_validation_loss=stopping.best_loss,
    )
    save_bc_training_checkpoint(final_checkpoint, final_path)

    best_model, _, _, _, _ = restore_bc_training_checkpoint(
        best_path,
        device=device,
        expected_dataset_sha256=dataset_sha256,
    )

    return BCTrainingResult(
        best_model=best_model,
        history=history,
        best_epoch=stopping.best_epoch,
        epochs_completed=history.epochs_completed,
        best_checkpoint=best_path,
        final_checkpoint=final_path,
    )
