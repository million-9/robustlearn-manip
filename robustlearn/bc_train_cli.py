"""Reproducible Behavior Cloning training CLI."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from robustlearn.bc_dataset import (
    BCDataset,
    BCNormalization,
    NormalizedBCDataset,
    make_bc_dataloader,
)
from robustlearn.bc_policy import BCPolicy, BCPolicyConfig
from robustlearn.bc_training import BCTrainingConfig, run_bc_training
from robustlearn.dataset_freeze import verify_dataset_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a BC policy on frozen dataset v0.1."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/v0.1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--loss",
        choices=("huber", "mse"),
        default="huber",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = BCTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        loss_name=args.loss,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    snapshot = verify_dataset_snapshot(args.dataset_root)
    fingerprint = snapshot["dataset_sha256"]

    if not isinstance(fingerprint, str):
        raise ValueError("Verified dataset fingerprint is invalid")

    train = BCDataset(args.dataset_root, split="train")
    validation = BCDataset(
        args.dataset_root,
        split="validation",
    )

    normalization = BCNormalization.fit(train)

    train_loader = make_bc_dataloader(
        NormalizedBCDataset(train, normalization),
        batch_size=config.batch_size,
        seed=config.seed,
    )
    validation_loader = make_bc_dataloader(
        NormalizedBCDataset(validation, normalization),
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = BCPolicy(
        BCPolicyConfig(seed=config.seed)
    )

    result = run_bc_training(
        model,
        train_loader,
        validation_loader,
        normalization,
        config,
        dataset_sha256=fingerprint,
        output_dir=args.output_dir,
    )

    summary = {
        "dataset_sha256": fingerprint,
        "seed": config.seed,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_checkpoint": str(result.best_checkpoint),
        "final_checkpoint": str(result.final_checkpoint),
        "train_losses": result.history.train_losses,
        "validation_losses": result.history.validation_losses,
    }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
