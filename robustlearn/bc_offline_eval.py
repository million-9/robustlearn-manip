"""Offline behavior-cloning prediction metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from robustlearn.bc_dataset import BCNormalization
from robustlearn.bc_policy import BCPolicy


@dataclass(frozen=True)
class BCOfflineMetrics:
    """Prediction errors in physical action units (radians)."""

    samples: int
    mae: float
    rmse: float
    per_dimension_mae: tuple[float, ...]
    per_dimension_rmse: tuple[float, ...]

    def as_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "samples": self.samples,
            "mae_rad": self.mae,
            "rmse_rad": self.rmse,
        }
        for index, value in enumerate(self.per_dimension_mae):
            result[f"joint_{index + 1}_mae_rad"] = value
        for index, value in enumerate(self.per_dimension_rmse):
            result[f"joint_{index + 1}_rmse_rad"] = value
        return result


def evaluate_bc_predictions(
    model: BCPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    normalization: BCNormalization,
) -> BCOfflineMetrics:
    """Evaluate unnormalized action predictions without training."""

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Observations and actions must be 2D")

    if observations.shape != (
        actions.shape[0],
        model.config.observation_dim,
    ):
        raise ValueError("Observation shape mismatch")

    if actions.shape[1] != model.config.action_dim:
        raise ValueError("Action shape mismatch")

    if actions.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    if not bool(torch.isfinite(observations).all()):
        raise ValueError("Non-finite observations")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("Non-finite actions")

    parameter = next(model.parameters())
    device = parameter.device

    model.eval()
    with torch.inference_mode():
        normalized = normalization.normalize_observation(
            observations.to(device)
        )
        predicted = model(normalized)
        predicted_actions = normalization.denormalize_action(
            predicted.cpu()
        )
        errors = (
            predicted_actions.to(torch.float64)
            - actions.cpu().to(torch.float64)
        )

        if not bool(torch.isfinite(errors).all()):
            raise ValueError("Non-finite prediction errors")

        per_mae = errors.abs().mean(dim=0)
        per_rmse = errors.square().mean(dim=0).sqrt()

        mae = float(errors.abs().mean())
        rmse = float(errors.square().mean().sqrt())

    if not all(math.isfinite(x) for x in (mae, rmse)):
        raise ValueError("Invalid prediction metrics")

    return BCOfflineMetrics(
        samples=actions.shape[0],
        mae=mae,
        rmse=rmse,
        per_dimension_mae=tuple(per_mae.tolist()),
        per_dimension_rmse=tuple(per_rmse.tolist()),
    )
