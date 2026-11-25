"""Portable checkpoint-based BC inference tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robustlearn.bc_dataset import BCNormalization
from robustlearn.bc_inference import BCInference
from robustlearn.bc_policy import BCPolicy
from robustlearn.bc_sim_adapter import bc_observation_from_env
from robustlearn.bc_training import (
    BCTrainingConfig,
    BCTrainingHistory,
    build_bc_training_checkpoint,
    save_bc_training_checkpoint,
)
from robustlearn.envs.panda_insertion import PandaInsertionEnv

DATASET_SHA256 = "a" * 64


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a valid checkpoint without external training artifacts."""
    destination = tmp_path_factory.mktemp("bc-inference") / "model.pt"

    model = BCPolicy()
    normalization = BCNormalization(
        observation_mean=torch.zeros(23),
        observation_std=torch.ones(23),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )

    history = BCTrainingHistory()
    history.append(0.5, 0.4)

    payload = build_bc_training_checkpoint(
        model,
        BCTrainingConfig(),
        normalization,
        history,
        dataset_sha256=DATASET_SHA256,
        epoch=1,
        best_epoch=1,
        best_validation_loss=0.4,
    )

    save_bc_training_checkpoint(payload, destination)
    return destination


@pytest.fixture(scope="module")
def inference(checkpoint: Path) -> BCInference:
    """Restore the temporary checkpoint."""
    return BCInference(
        checkpoint,
        expected_dataset_sha256=DATASET_SHA256,
    )


@pytest.fixture
def observation() -> np.ndarray:
    """Construct an observation from the real simulator."""
    env = PandaInsertionEnv()
    try:
        env.reset(seed=2026)
        return bc_observation_from_env(env)
    finally:
        env.close()


def test_checkpoint_restores(inference: BCInference) -> None:
    assert inference.model.config.observation_dim == 23
    assert inference.model.config.action_dim == 7
    assert not inference.model.training


def test_rejects_wrong_dataset_fingerprint(
    checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        BCInference(
            checkpoint,
            expected_dataset_sha256="b" * 64,
        )


def test_deterministic_prediction(
    inference: BCInference,
    observation: np.ndarray,
) -> None:
    first = inference.predict(observation)
    second = inference.predict(observation)

    assert first.shape == (7,)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_rejects_wrong_observation_shape(
    inference: BCInference,
) -> None:
    with pytest.raises(ValueError, match="23 BC observation"):
        inference.predict(np.zeros(30, dtype=np.float32))


def test_rejects_nonfinite_observation(
    inference: BCInference,
) -> None:
    observation = np.zeros(23, dtype=np.float32)
    observation[0] = np.nan

    with pytest.raises(ValueError, match="Nonfinite"):
        inference.predict(observation)


def test_inference_does_not_modify_observation(
    inference: BCInference,
    observation: np.ndarray,
) -> None:
    original = observation.copy()

    inference.predict(observation)

    np.testing.assert_array_equal(observation, original)
