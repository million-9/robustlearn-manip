# Week 11 Deterministic Dataset Splits

## Purpose

GitHub Issue #102 creates deterministic train, validation, and test
assignments for complete canonical episodes.

The tool uses the validated episode dataset from Issue #101.

It does not copy, modify, or split individual episode frames.

## Generate a split manifest

Run from the repository root:

    uv run python -m robustlearn.dataset_splits PATH_TO_EPISODES --manifest PATH_TO_REPORTS/splits.json --dataset-ref week11-candidate --seed 42

The default proportions are:

- Train: 70%
- Validation: 15%
- Test: 15%

Custom proportions can be supplied:

    --train 0.8 --validation 0.1 --test 0.1

All three proportions must be positive and sum to 1.

## Grouping and leakage protection

The default grouping mode is `scenario_initial`.

It keeps episodes together whenever they share either:

- The same scenario label.
- The same recorded initial joint configuration.

The grouping is transitive. Connections through other episodes
cannot cause a correlated group to be split.

Available grouping modes:

- scenario_initial: protect both known correlations.
- scenario: keep identical scenario labels together.
- initial_condition: keep identical recorded starting poses together.
- episode: enforce episode-only grouping.

The alternate modes may permit leakage along grouping dimensions
they do not protect.

When initial-position metadata is missing, the combined mode still
protects scenario equality. It cannot detect an unrecorded shared
initial condition across differently labeled scenarios.

Exact joint configurations are compared. Physically similar but
numerically different configurations are not automatically grouped.

The combined mode is conservative. If fewer than three independent
groups remain, splitting fails rather than violating the grouping policy.

## Proportion limitations

Group assignments are indivisible.

Requested ratios are optimization targets, not guaranteed exact
episode fractions.

Scenario grouping and initial-condition grouping can prevent exact
proportional assignments, especially for small datasets.

The manifest records actual counts and distributions so deviations
remain visible.

## Verify an existing manifest

    uv run python -m robustlearn.dataset_splits PATH_TO_EPISODES --manifest PATH_TO_REPORTS/splits.json --check

Verification regenerates the deterministic assignment and rejects
differences in dataset fingerprint, split configuration, or manifest.

Existing manifests are not silently overwritten.

## Manifest contents

The JSON manifest records:

- Manifest version.
- Dataset reference.
- Dataset SHA256 fingerprint.
- Splitting seed and proportions.
- Grouping mode.
- Total episode and frame counts.
- Train, validation, and test episode IDs and filenames.
- Per-split episode and frame counts.
- Per-split scenario, task, and source distributions.
- Per-split group counts.

Dataset fingerprints include canonical filenames and exact contents.

The output must be stored outside the episode directory.

## Limitations

The algorithm uses deterministic greedy balancing. It does not
guarantee globally optimal class coverage or exact proportions.

It cannot establish semantic independence between demonstrations
with distinct recorded scenario labels and initial configurations.

No BC dataloader or policy training is included.

Dataset version v0.1 will be frozen under Issue #103.
