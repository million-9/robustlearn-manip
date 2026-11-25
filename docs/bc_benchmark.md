# Week 13 — Multi-seed BC benchmark

## Scope

Issue #123 extends the Week 12 closed-loop evaluation into a
reproducible multi-seed benchmark.

The original trained checkpoint, observation adapter, normalization,
action mapping and safety rejection behavior remain unchanged.

## Reproduction

Run from the repository root:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --locked python -m robustlearn.bc_benchmark \
      --checkpoint /tmp/robustlearn-114-full-baseline/best.pt \
      --dataset-manifest datasets/v0.1/dataset_manifest.json \
      --seeds $(seq 2026 2045) \
      --max-steps 100 \
      --max-force-n 50 \
      --output /tmp/robustlearn-123-baseline-benchmark.json

The benchmark defaults to 20 explicit evaluation seeds.

These are evaluation seeds, not independently trained policy seeds.

## Recorded evidence

- Checkpoint and dataset fingerprints.
- Software versions and evaluation configuration.
- Per-episode outcomes, trajectories and rejected actions.
- Success rates and failure reason counts.
- Executed steps and zero-step failures.
- Successful completion times.
- Recorded peak forces and action smoothness.
- Exact unique initial observation counts.

All failed episodes remain in the success-rate denominator.

Unavailable measurements are represented as JSON null.

Nonfinite diagnostic values are serialized as JSON null without
changing the associated episode outcome.

## Statistical interpretation

Domain randomization remains disabled for this baseline.

Different evaluation seeds do not necessarily produce distinct
initial conditions or independent scenarios.

Exact initial observation diversity is measured where observations
are available. This is not a complete simulator-state comparison.

Binomial confidence intervals are not reported until the
independence and scenario sampling assumptions are justified.

The initial benchmark is a repeatability and failure-characterization
experiment, not a population-level generalization estimate.

## Measurement limitations

Peak force is inherited from the Week 12 rollout implementation.
It includes recorded post-action measurements only.

Episodes that execute no actions do not contribute a synthetic
zero-force measurement to the aggregate.

Force is sampled at environment-step boundaries rather than every
physics substep.

Action smoothness is unavailable when fewer than two accepted
policy actions exist.

## Acceptance evidence

### Measured nominal-condition benchmark

- Evaluation seeds: 2026 through 2045.
- Evaluation attempts: 20.
- Successful episodes: 0.
- Failed episodes: 20.
- Observed success rate: 0.0%.
- Unsafe-action terminations: 20.
- Total executed steps: 20.
- Mean executed steps: 1.0.
- Mean recorded peak force: 2.384399 N.
- Maximum recorded peak force: 2.384399 N.
- Recorded initial observations: 20.
- Unique initial observations: 1.

Every episode executed one environment step before its next
policy prediction was rejected by the existing actuator-limit check.

No successful completion times or action smoothness measurements
were available.

### Week 12 failure reproduction

Seed 2026 reproduced the original measured failure:

- Outcome: unsafe_action.
- Executed steps: 1.
- Rejected action step: 2.
- Joint 6 prediction: -7.325173378 rad.
- Joint 6 lower limit: -0.017500 rad.
- Joint 6 upper limit: 3.752500 rad.

The existing safety rejection behavior remains unchanged.

### Repeatability

Two complete benchmark executions produced byte-identical JSON
artifacts, including all episode trajectories and diagnostics.

Benchmark JSON SHA-256:

    e9bbb932a32903e890ea2ac5bf0b67caee38b782ed44ad1243c92b0d18d176b0

Checkpoint SHA-256:

    d3497a3438e183130a66646856fc59c3bc406b4663add17b450791558f2c21b9

Dataset fingerprint:

    eb28d409db76e18c6a8e6f6f0ccb49429f0271d4bf8a6f16881637ae0dd5db3f

Both complete benchmark artifacts are local experiment outputs:

    /tmp/robustlearn-123-baseline-benchmark.json
    /tmp/robustlearn-123-baseline-repeat.json

A compact machine-readable acceptance record is retained in
`docs/bc_benchmark_baseline.json`.

### Interpretation

The nominal-condition failure is reproducible.

All 20 seeds produced the same recorded initial observation.
These executions do not establish 20 independent scenarios.

The observed 0% success rate describes the executed benchmark,
not a statistically established population-level failure rate.

No independent-scenario confidence interval is reported.

The baseline does not establish whether observation normalization,
sensor-frame disagreement, controller dynamics or other factors
caused the unsafe prediction.

Those hypotheses require controlled investigation in issue #124
and subsequent ablation experiments.

See also: week12_bc_closed_loop_acceptance.md.
