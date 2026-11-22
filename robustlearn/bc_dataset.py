"""Behavior Cloning data representation for frozen rollout demonstrations.

Observations contain:
- 7 joint positions in radians
- 7 joint velocities in radians/second
- 3 task-relative position values
- 6 wrist wrench values

Actions are 7 absolute joint-position targets in radians.

This module does not train a policy or modify the frozen dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from robustlearn.dataset_freeze import verify_dataset_snapshot
from robustlearn.dataset_splits import validate_split_manifest
from robustlearn.rollout_dataset import (
    PANDA_JOINT_COUNT,
    TASK_RELATIVE_STATE_SIZE,
    WRIST_WRENCH_SIZE,
    RolloutFrame,
    read_episode,
)

OBSERVATION_DIM = (
    2 * PANDA_JOINT_COUNT
    + TASK_RELATIVE_STATE_SIZE
    + WRIST_WRENCH_SIZE
)
ACTION_DIM = PANDA_JOINT_COUNT

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class BCSample:
    """One validated observation/action pair."""

    observation: FloatArray
    action: FloatArray

    def __post_init__(self) -> None:
        if self.observation.shape != (OBSERVATION_DIM,):
            raise ValueError("Invalid BC observation shape")

        if self.action.shape != (ACTION_DIM,):
            raise ValueError("Invalid BC action shape")

        if not np.all(np.isfinite(self.observation)):
            raise ValueError("BC observation contains nonfinite values")

        if not np.all(np.isfinite(self.action)):
            raise ValueError("BC action contains nonfinite values")


def frame_to_bc_sample(frame: RolloutFrame) -> BCSample:
    """Convert a rollout frame into a BC observation/action pair."""

    if frame.joint_velocity is None:
        raise ValueError(
            "BC requires measured joint velocities; "
            "missing values cannot be silently imputed"
        )

    observation = np.asarray(
        (
            *frame.joint_position,
            *frame.joint_velocity,
            *frame.task_relative_state,
            *frame.wrist_wrench,
        ),
        dtype=np.float32,
    )

    action = np.asarray(
        frame.expert_action,
        dtype=np.float32,
    )

    return BCSample(
        observation=observation,
        action=action,
    )


@dataclass(frozen=True)
class BCSampleMetadata:
    """Traceability metadata for one supervised sample."""

    episode_index: int
    frame_index: int
    source: str
    scenario: str
    timestamp_s: float


class BCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch-compatible dataset using a canonical frozen split.

    Samples retain the original episode/frame ordering. No reshuffling,
    normalization, resampling or timestamp interpolation occurs here.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        verify_snapshot: bool = True,
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown BC split: {split}")

        root = Path(dataset_root)

        if verify_snapshot:
            verify_dataset_snapshot(root)

        manifest = json.loads(
            (root / "splits.json").read_text(encoding="utf-8")
        )

        validate_split_manifest(root / "episodes", manifest)

        split_record = manifest["splits"][split]
        episode_files = split_record["episode_files"]

        observations: list[FloatArray] = []
        actions: list[FloatArray] = []
        metadata: list[BCSampleMetadata] = []

        for filename in episode_files:
            episode = read_episode(root / "episodes" / filename)

            for frame in episode.frames:
                sample = frame_to_bc_sample(frame)

                observations.append(sample.observation)
                actions.append(sample.action)
                metadata.append(
                    BCSampleMetadata(
                        episode_index=episode.episode_index,
                        frame_index=frame.frame_index,
                        source=episode.source,
                        scenario=episode.scenario,
                        timestamp_s=frame.timestamp_s,
                    )
                )

        if not observations:
            raise ValueError(f"BC split contains no frames: {split}")

        self.split = split
        self.metadata = tuple(metadata)
        self.observations = torch.from_numpy(
            np.stack(observations)
        )
        self.actions = torch.from_numpy(
            np.stack(actions)
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]


@dataclass(frozen=True)
class BCNormalization:
    """Training-derived normalization for BC observations and actions."""

    observation_mean: torch.Tensor
    observation_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @classmethod
    def fit(cls, dataset: BCDataset) -> BCNormalization:
        """Fit normalization statistics exclusively on training samples."""

        if dataset.split != "train":
            raise ValueError(
                "BC normalization must be fitted on the training split"
            )

        observations = dataset.observations.to(torch.float64)
        actions = dataset.actions.to(torch.float64)

        observation_mean = observations.mean(dim=0)
        action_mean = actions.mean(dim=0)

        observation_std = observations.std(dim=0, unbiased=False)
        action_std = actions.std(dim=0, unbiased=False)

        observation_std = torch.where(
            observation_std > 1e-8,
            observation_std,
            torch.ones_like(observation_std),
        )

        action_std = torch.where(
            action_std > 1e-8,
            action_std,
            torch.ones_like(action_std),
        )

        return cls(
            observation_mean=observation_mean.float(),
            observation_std=observation_std.float(),
            action_mean=action_mean.float(),
            action_std=action_std.float(),
        )

    def normalize_observation(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize observations with training-only statistics."""
        return (value - self.observation_mean) / self.observation_std

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize actions with training-only statistics."""
        return (value - self.action_mean) / self.action_std

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        """Convert normalized action predictions back to joint radians."""
        return value * self.action_std + self.action_mean


class NormalizedBCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read-only normalized view over an existing BC dataset."""

    def __init__(
        self,
        dataset: BCDataset,
        normalization: BCNormalization,
    ) -> None:
        self.dataset = dataset
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        observation, action = self.dataset[index]

        return (
            self.normalization.normalize_observation(observation),
            self.normalization.normalize_action(action),
        )


def make_bc_dataloader(
    dataset: BCDataset | NormalizedBCDataset,
    *,
    batch_size: int = 256,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a reproducible single-process BC DataLoader.

    Only training samples are shuffled. Validation and test preserve
    their canonical episode/frame ordering.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    base = dataset.dataset if isinstance(dataset, NormalizedBCDataset) else dataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=base.split == "train",
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
