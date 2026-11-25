# Behavior Cloning Training — Frozen Dataset v0.1

## Overview

This baseline trains a feedforward Behavior Cloning (BC) policy using the frozen v0.1 manipulation dataset.

The policy maps 23 normalized observation features to seven normalized absolute Panda joint-position targets.

This is supervised imitation learning. It is not evidence of successful closed-loop robot manipulation.

## Dataset

- Version: v0.1
- Episodes: 94
- Frames: 3,727
- Training: 65 episodes, 2,573 frames
- Validation: 15 episodes, 593 frames
- Test: 14 episodes, 561 frames

Dataset SHA-256:

`eb28d409db76e18c6a8e6f6f0ccb49429f0271d4bf8a6f16881637ae0dd5db3f`

Normalization statistics are calculated exclusively from the training split.

The validation split is used for checkpoint selection and early stopping. The test split is not loaded by the training CLI.

## Model and training configuration

| Parameter | Value |
|---|---|
| Model | Feedforward MLP |
| Input features | 23 |
| Output dimensions | 7 |
| Hidden layers | 256, 256, 256 |
| Activation | ReLU |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Batch size | 256 |
| Loss | Huber |
| Gradient clipping | 1.0 |
| Maximum epochs | 50 |
| Early-stopping patience | 10 |
| Minimum improvement | 0.0001 |
| Seed | 42 |
| Device | CPU |

## Reproducing the baseline

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python -m robustlearn.bc_train_cli \
  --dataset-root datasets/v0.1 \
  --output-dir /tmp/robustlearn-bc-baseline \
  --max-epochs 50 \
  --batch-size 256 \
  --learning-rate 0.0003 \
  --max-grad-norm 1.0 \
  --loss huber \
  --patience 10 \
  --min-delta 0.0001 \
  --seed 42 \
  --device cpu
```

The training coordinator rejects an output directory containing `best.pt` or `final.pt`, preventing accidental checkpoint overwrites.

## Recorded CPU baseline results

| Metric | Result |
|---|---:|
| Completed epochs | 50 |
| Selected best epoch | 46 |
| Initial training loss | 0.33601037 |
| Final training loss | 0.00213607 |
| Initial validation loss | 0.12567321 |
| Final validation loss | 0.00841964 |
| Selected best validation loss | 0.00569806 |

The run reached 50 epochs without triggering early stopping.

The selected best checkpoint performed better on validation loss than the final checkpoint.

## Checkpoints

The CLI produces `best.pt` and `final.pt`.

Both checkpoint formats include model configuration and weights, training configuration, normalization statistics, dataset fingerprint, seed, epoch metadata, and training/validation loss histories.

The best checkpoint contains history up to its selected epoch. The final checkpoint contains history through the last completed epoch.

Checkpoints support model restoration for inference. Exact interruption-and-resume training is not supported because optimizer and RNG states are not stored.

Load checkpoints only from trusted sources.

## Verification

The training implementation has passed 114 unit tests and two real-dataset integration tests, totaling 116 tests.

The integration tests verify CPU training, checkpoint restoration, same-seed reproducibility, and that training never loads the test partition.

## Limitations

This baseline has not yet been evaluated for closed-loop manipulation success, rollout stability, or compounding error.

Multi-seed performance characterization, held-out test evaluation, and exact checkpoint-based training resumption remain outside the scope of this training baseline.