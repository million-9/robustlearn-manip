# Week 12 — First BC Closed-Loop Insertion Acceptance

## Scope

Issue #115 evaluates the first trained Behavior Cloning policy in closed-loop MuJoCo Panda insertion simulation.

This is one deterministic baseline evaluation, not a multi-seed benchmark.

## Dataset and checkpoint

- Dataset: frozen v0.1
- Episodes: 94
- Frames: 3,727
- Dataset SHA-256: eb28d409db76e18c6a8e6f6f0ccb49429f0271d4bf8a6f16881637ae0dd5db3f
- Checkpoint: best validation-selected Week 12 BC checkpoint
- Checkpoint SHA-256: d3497a3438e183130a66646856fc59c3bc406b4663add17b450791558f2c21b9
- Policy input: 23 normalized observation features
- Policy output: seven normalized absolute joint-position targets
- Inference: CPU

The checkpoint includes training-derived normalization.

## Evaluation configuration

- MuJoCo Panda insertion environment
- Evaluation seed: 2026
- Domain randomization: disabled
- Maximum environment steps: 100
- Maximum wrist-force magnitude: 50 N
- Environment frame skip: 5
- Physics timestep: 0.002 seconds
- Maximum nominal episode duration: 1.0 simulation second

Seed 2026 does not appear among the 94 frozen demonstration episodes. This establishes an unseen demonstration seed, not necessarily an unseen scenario.

## Reproduction command

Restore the identified trained checkpoint before running the evaluation.

Run from the repository root:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --locked python -m robustlearn.bc_rollout \
      --checkpoint /tmp/robustlearn-114-full-baseline/best.pt \
      --dataset-manifest datasets/v0.1/dataset_manifest.json \
      --seed 2026 \
      --max-steps 100 \
      --max-force-n 50 \
      --output /tmp/robustlearn-115-bc-rollout-seed2026-diagnostic.json

The output records policy targets, applied actuator commands, task geometry, wrist measurements, terminal outcome, and rejected-action diagnostics.

## Measured result

| Metric | Value |
|---|---|
| Task success | False |
| Terminal outcome | unsafe_action |
| Environment steps executed | 1 |
| Elapsed simulation time | 0.01 s |
| Maximum measured wrist-force magnitude | 2.3844 N |
| Mean action change | Not available |
| Rejected action step | 2 |
| Violating joint | Panda joint 6 |
| Predicted joint-6 target | -7.325173 rad |
| Joint-6 lower limit | -0.017500 rad |
| Joint-6 upper limit | +3.752500 rad |

The first policy command passed validation and advanced MuJoCo.

The second policy command was rejected before application because its joint-6 target exceeded the actuator limits.

The insertion task did not succeed.

## Failure classification

Primary category: unsafe policy action / actuator-limit violation.

The rejected joint-6 prediction was approximately 7.3077 radians below the permitted minimum.

There is insufficient evidence to classify this rollout as rim sticking, alignment failure, oscillation, or excessive contact force.

## Normalization diagnostics

Maximum absolute normalized observation deviation:

- Before the first action: approximately 184.8.
- After the first action: approximately 1190.2.

Selected second-observation deviations:

- Wrist torque Z: +1190.2
- Wrist force Z: -916.8
- Wrist force Y: -814.4
- Wrist force X: -748.1
- Joint velocity 7: -571.9

The training standard deviation for wrist torque Z is approximately 4.66e-7 Nm.

These measurements demonstrate substantial disagreement between live observations and training normalization statistics.

They do not prove that normalization alone caused the invalid prediction.

The sensor-frame relationship between the demonstration collection pipeline and MuJoCo requires further independent verification.

## Safety and compatibility

The implementation:

- Restores the trained checkpoint with dataset fingerprint validation.
- Constructs 23-feature observations from live MuJoCo state.
- Runs deterministic policy inference.
- Denormalizes predictions into absolute joint-position targets.
- Maps seven arm targets into the eight-actuator environment command.
- Preserves the gripper control target.
- Rejects nonfinite and out-of-range commands.
- Enforces environment step limits and wrist-force safety checks.
- Records objective task outcomes and rejected-action diagnostics.

Portable tests construct temporary synthetic checkpoints and do not require locally generated training artifacts.

## Limitations

1. Only one clean evaluation was performed.
2. The scenario is not established as unseen.
3. The trained policy did not complete insertion.
4. Force checking occurs at environment-step boundaries, not at every MuJoCo physics substep.
5. The trajectory records pre-action observations and submitted controls; rejected-action diagnostics separately record the rejected prediction.
6. Action smoothness cannot be estimated from one accepted action.
7. Extreme normalized observations were measured, but their causal role remains unproven.
8. The trained checkpoint and rollout JSON are local artifacts, not Git-tracked files.

## Acceptance conclusion

A real trained BC policy-controlled MuJoCo transition was executed, and an objective safety failure was recorded.

The Week 12 acceptance gate permits measured failed rollouts.

This evaluation establishes closed-loop integration and actuator safety termination, not successful manipulation.

Multi-seed evaluation and policy tuning belong to Week 13. DAgger belongs to Week 14.
