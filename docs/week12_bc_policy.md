# Week 12 — Behavior Cloning MLP Policy

## Scope

Issue #112 implements a configurable PyTorch multilayer perceptron
(MLP) for Behavior Cloning.

Implementation: robustlearn/bc_policy.py

Training, controller execution, Cartesian action conversion, and
hyperparameter optimization are outside this issue.

## Observation contract

The policy accepts 23 normalized float32 observation features.

| Indices | Features | Units before normalization |
|---|---|---|
| 0-6 | Panda joint positions | radians |
| 7-13 | Panda joint velocities | radians/second |
| 14-16 | Task-relative position | meters |
| 17-22 | Wrist force and torque | newtons and newton-meters |

Input shapes:
- Single observation: (23,)
- Batch: (batch_size, 23)

Normalization is performed by the BC dataset pipeline.

## Action contract

The policy predicts seven normalized absolute joint-position targets.

Output shapes:
- Single prediction: (7,)
- Batch: (batch_size, 7)

After denormalization, the action represents seven Panda joint
position targets in radians.

IMPORTANT: The original Week 12 issue requests Cartesian delta
actions, but the verified frozen v0.1 dataset contains absolute
joint-position labels.

This policy matches the actual dataset schema. It does not
predict Cartesian deltas.

## Architecture

Default configuration:
- Input: 23 features
- Hidden layers: 256, 256, 256
- Activation: ReLU
- Output: 7 features
- Initialization seed: 42

Configurable options:
- Hidden-layer dimensions
- ReLU or SiLU activation
- Initialization seed
- Observation and action dimensions

The output layer is linear, without clipping or bounded activation.

## Normalization and deployment safety

The network receives normalized observations and predicts
normalized actions.

The matching training-set statistics must be used to denormalize
predictions before controller execution.

Deployment must also validate:
- Finite action values
- Robot-specific joint limits
- Motion and rate constraints
- Controller compatibility

The model does not enforce physical robot safety.

Normalized outputs must not be interpreted as radians.

## Checkpoint persistence

Checkpoints contain:
- Format version 1
- Model configuration
- PyTorch model state dictionary

Example usage:

    import torch
    from robustlearn.bc_policy import BCPolicy, save_bc_policy, load_bc_policy

    model = BCPolicy().eval()
    save_bc_policy(model, "bc_policy.pt")

    restored = load_bc_policy("bc_policy.pt", device="cpu")

    with torch.inference_mode():
        prediction = restored(torch.zeros(1, 23))

The loader validates required checkpoint fields and restores
model parameters with strict state-dictionary loading.

Only load checkpoints from trusted sources.

## Device handling

CPU inference is supported.

CUDA can be selected when a compatible CUDA-enabled PyTorch
installation and GPU are available.

The current CPU-only installation does not support CUDA inference.

GPU execution has not yet been validated.

## Reproducibility

Model initialization uses an isolated PyTorch RNG context.

Identical configurations and seeds produce identical initial
parameters without advancing the caller's CPU RNG state.

This does not guarantee deterministic training or identical
execution across different hardware and PyTorch versions.

## Validation

The policy unit tests cover:
- Forward-pass dimensions
- Finite smoke-test predictions
- ReLU and SiLU configurations
- Default and custom architectures
- Deterministic initialization
- Global RNG preservation
- CPU inference
- Unsupported and unavailable devices
- Save/load equivalence
- Checkpoint structural validation

Policy training and closed-loop evaluation belong to
subsequent Week 12 issues.
