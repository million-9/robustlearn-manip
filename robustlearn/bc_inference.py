"""Deterministic checkpoint-based BC policy inference."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from robustlearn.bc_dataset import ACTION_DIM, OBSERVATION_DIM
from robustlearn.bc_training import restore_bc_training_checkpoint

FloatArray = NDArray[np.float32]


class BCInference:
    """Load a trained BC checkpoint and predict joint targets."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_dataset_sha256: str,
    ) -> None:
        if not expected_dataset_sha256:
            raise ValueError("Dataset fingerprint is required")

        (
            model,
            normalization,
            _,
            _,
            metadata,
        ) = restore_bc_training_checkpoint(
            checkpoint_path,
            device="cpu",
            expected_dataset_sha256=expected_dataset_sha256,
        )

        if (
            model.config.observation_dim != OBSERVATION_DIM
            or model.config.action_dim != ACTION_DIM
        ):
            raise ValueError("Checkpoint BC dimensions are incompatible")

        self.model = model.eval()
        self.normalization = normalization
        self.metadata = metadata

    def predict(self, observation: NDArray[np.float32]) -> FloatArray:
        """Predict seven absolute joint targets in radians."""
        values = np.asarray(observation, dtype=np.float32)

        if values.shape != (OBSERVATION_DIM,):
            raise ValueError("Expected 23 BC observation features")

        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite BC observation")

        tensor = torch.from_numpy(values.copy())

        with torch.inference_mode():
            normalized = self.normalization.normalize_observation(tensor)
            prediction = self.model(normalized)
            actions = self.normalization.denormalize_action(prediction)

        result = actions.detach().cpu().numpy().astype(
            np.float32,
            copy=True,
        )

        if result.shape != (ACTION_DIM,):
            raise ValueError("BC prediction dimension mismatch")

        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite BC prediction")

        return result
