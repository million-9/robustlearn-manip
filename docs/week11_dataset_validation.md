# Week 11 Dataset Validation and Statistics

## Purpose

GitHub Issue #101 introduces validation of canonical rollout episodes before
using them as a learning dataset.

The tool supports scripted expert and reviewed teleoperation episodes using
the existing `robustlearn.rollout_dataset` schema.

The validator does not modify, repair, or delete source episodes.

## Validate a candidate dataset

Run from the repository root:

    uv run python -m robustlearn.dataset_validation PATH_TO_EPISODES

The input must be a directory containing canonical episode JSON files.

The tool returns:

- Exit code 0 for a valid dataset.
- Exit code 1 for invalid datasets.
- Exit code 2 for invalid CLI usage or an unsafe report destination.

Validation errors identify the affected file and the detected problem.

## Generate deterministic statistics

    uv run python -m robustlearn.dataset_validation PATH_TO_EPISODES --json-output PATH_TO_REPORTS/statistics.json

The JSON report must be outside the episode directory.

Statistics include:

- Total episodes and frames.
- Scripted expert and teleoperation episode counts.
- Successes, failures, and success rate.
- Failure reason, scenario, task, and source distributions.
- Episode length minimum, maximum, mean, and median.
- Fixed observation and action dimensions.
- Per-component minimum, maximum, mean, and population standard deviation.

Numeric statistics are available for joint position, joint velocity,
task-relative state, wrist wrench, and expert action.

Statistics include accepted failures as well as successful demonstrations.
They are not filtered to successful episodes only.

Episodes are processed in deterministic filename order. Repeating the
command on identical input files produces reproducible JSON content.

## Validation requirements

Each episode must:

- Conform to canonical rollout schema version 1.
- Contain required episode and frame fields.
- Use source `scripted` or `teleop`.
- Include nonempty scenario, task, commit, and configuration metadata.
- Have a unique episode ID within the candidate dataset.
- Contain at least one frame.
- Have contiguous, zero-based frame indices.
- Have finite, chronologically ordered timestamps.
- Have valid observation and action dimensions.
- Contain finite numeric observations and actions.
- Have consistent success/failure metadata.
- End in a terminal frame with outcome matching the episode.

The validator reports corrupt records and exits nonzero. It does not
silently exclude them.

## Empty and incomplete episode policy

The Week 10 canonical schema permits failed or incomplete archival
episodes containing zero frames.

That does not make those records suitable for behavioral cloning.

For this validator, a candidate training dataset must contain complete,
nonempty trajectories with explicit terminal outcome metadata.

Therefore, zero-frame archival episodes are reported as invalid rather
than silently skipped.

Such episodes may remain in the raw collection/archive, but must not be
placed in a training-ready dataset without an explicit curation decision.

This distinction preserves archival evidence while preventing accidental
training on incomplete trajectories.

## Limitations

The validator checks schema, numeric integrity, terminal consistency,
and dataset-level identity uniqueness. It cannot establish whether a
demonstration is physically correct, safe to execute on hardware,
or suitable for training a particular policy.

Dataset splitting and immutable dataset v0.1 manifests are addressed by
GitHub Issues #102 and #103 respectively.
