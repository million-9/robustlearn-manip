"""Regression tests for the Week 12 BC data pipeline."""

from pathlib import Path

import pytest
import torch

from robustlearn.bc_dataset import (
    BCDataset,
    BCNormalization,
    NormalizedBCDataset,
    make_bc_dataloader,
)

DATASET_ROOT = Path("datasets/v0.1")


@pytest.fixture(scope="module")
def train_dataset():
    return BCDataset(DATASET_ROOT, "train")


@pytest.fixture(scope="module")
def normalization(train_dataset):
    return BCNormalization.fit(train_dataset)


@pytest.mark.parametrize(
    ("split", "expected_frames"),
    [
        ("train", 2573),
        ("validation", 593),
        ("test", 561),
    ],
)
def test_canonical_split_shapes(split, expected_frames):
    dataset = BCDataset(DATASET_ROOT, split)

    assert len(dataset) == expected_frames
    assert dataset.observations.shape == (expected_frames, 23)
    assert dataset.actions.shape == (expected_frames, 7)

    assert torch.isfinite(dataset.observations).all()
    assert torch.isfinite(dataset.actions).all()


def test_training_normalization_is_finite(train_dataset, normalization):
    assert normalization.observation_mean.shape == (23,)
    assert normalization.observation_std.shape == (23,)
    assert normalization.action_mean.shape == (7,)
    assert normalization.action_std.shape == (7,)

    assert torch.isfinite(normalization.observation_mean).all()
    assert torch.isfinite(normalization.observation_std).all()
    assert torch.isfinite(normalization.action_mean).all()
    assert torch.isfinite(normalization.action_std).all()

    assert torch.all(normalization.observation_std > 0)
    assert torch.all(normalization.action_std > 0)

    normalized = NormalizedBCDataset(train_dataset, normalization)

    observation, action = normalized[0]

    assert observation.shape == (23,)
    assert action.shape == (7,)
    assert torch.isfinite(observation).all()
    assert torch.isfinite(action).all()


def test_normalization_rejects_validation_split():
    validation = BCDataset(DATASET_ROOT, "validation")

    with pytest.raises(ValueError, match="training split"):
        BCNormalization.fit(validation)


def test_action_normalization_round_trip(train_dataset, normalization):
    original = train_dataset.actions[:32]
    normalized = normalization.normalize_action(original)
    reconstructed = normalization.denormalize_action(normalized)

    assert torch.allclose(original, reconstructed, atol=1e-5)


def test_dataloader_determinism(train_dataset, normalization):
    dataset = NormalizedBCDataset(train_dataset, normalization)

    def first_batch(seed):
        loader = make_bc_dataloader(
            dataset,
            batch_size=32,
            seed=seed,
        )
        return next(iter(loader))

    observation_a, action_a = first_batch(42)
    observation_b, action_b = first_batch(42)
    observation_c, action_c = first_batch(43)

    assert torch.equal(observation_a, observation_b)
    assert torch.equal(action_a, action_b)

    assert not (
        torch.equal(observation_a, observation_c)
        and torch.equal(action_a, action_c)
    )

    assert observation_a.shape == (32, 23)
    assert action_a.shape == (32, 7)


def test_validation_order_is_seed_independent():
    dataset = BCDataset(DATASET_ROOT, "validation")

    batch_a = next(
        iter(make_bc_dataloader(dataset, batch_size=16, seed=1))
    )
    batch_b = next(
        iter(make_bc_dataloader(dataset, batch_size=16, seed=999))
    )

    for first, second in zip(batch_a, batch_b, strict=True):
        assert torch.equal(first, second)


def test_zero_variance_normalization():
    """Constant features must not produce division by zero."""
    from types import SimpleNamespace

    observations = torch.ones((10, 23), dtype=torch.float32)
    actions = torch.ones((10, 7), dtype=torch.float32)

    dataset = SimpleNamespace(
        split="train",
        observations=observations,
        actions=actions,
    )

    normalization = BCNormalization.fit(dataset)

    assert torch.all(normalization.observation_std == 1)
    assert torch.all(normalization.action_std == 1)

    normalized_observations = normalization.normalize_observation(
        observations
    )
    normalized_actions = normalization.normalize_action(actions)

    assert torch.isfinite(normalized_observations).all()
    assert torch.isfinite(normalized_actions).all()
    assert torch.all(normalized_observations == 0)
    assert torch.all(normalized_actions == 0)


def test_missing_joint_velocity_rejected():
    """BC must not silently impute unavailable joint velocities."""
    from dataclasses import replace

    from robustlearn.bc_dataset import frame_to_bc_sample
    from robustlearn.rollout_dataset import read_episode

    episode_path = sorted(
        (DATASET_ROOT / "episodes").glob("*.json")
    )[0]

    episode = read_episode(episode_path)
    invalid_frame = replace(
        episode.frames[0],
        joint_velocity=None,
    )

    with pytest.raises(ValueError, match="missing values"):
        frame_to_bc_sample(invalid_frame)



def test_canonical_split_leakage():
    """Frozen splits must have no episode or scenario overlap."""
    import json

    manifest = json.loads(
        (DATASET_ROOT / "splits.json").read_text(encoding="utf-8")
    )

    splits = manifest["splits"]
    names = ("train", "validation", "test")

    episode_sets = {
        name: set(splits[name]["episode_ids"])
        for name in names
    }

    scenario_sets = {
        name: set(splits[name]["scenarios"])
        for name in names
    }

    for index, first in enumerate(names):
        for second in names[index + 1:]:
            assert episode_sets[first].isdisjoint(episode_sets[second])
            assert scenario_sets[first].isdisjoint(scenario_sets[second])

    assert len(set.union(*episode_sets.values())) == 94

    assert sum(
        splits[name]["frame_count"]
        for name in names
    ) == 3727


def test_normalization_matches_training_statistics(
    train_dataset,
    normalization,
):
    """Normalization must match independently calculated train statistics."""
    observations = train_dataset.observations.double()
    actions = train_dataset.actions.double()

    expected_observation_mean = observations.mean(dim=0).float()
    expected_action_mean = actions.mean(dim=0).float()

    expected_observation_std = observations.std(
        dim=0, unbiased=False
    )
    expected_action_std = actions.std(
        dim=0, unbiased=False
    )

    expected_observation_std = torch.where(
        expected_observation_std > 1e-8,
        expected_observation_std,
        torch.ones_like(expected_observation_std),
    ).float()

    expected_action_std = torch.where(
        expected_action_std > 1e-8,
        expected_action_std,
        torch.ones_like(expected_action_std),
    ).float()

    assert torch.allclose(
        normalization.observation_mean,
        expected_observation_mean,
    )
    assert torch.allclose(
        normalization.action_mean,
        expected_action_mean,
    )
    assert torch.allclose(
        normalization.observation_std,
        expected_observation_std,
    )
    assert torch.allclose(
        normalization.action_std,
        expected_action_std,
    )
