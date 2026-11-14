# Dataset v0.1 — Release Preparation

GitHub Issue: #103

Status: DRAFT_NOT_FROZEN

## 1. Dataset Inventory

The current release candidate contains:

- 93 scripted expert demonstrations
- 0 teleoperation demonstrations
- 3,694 recorded frames
- 93 successful episodes
- 0 failed episodes
- 20 recorded scenarios

Dataset location:

    datasets/v0.1_staging/

This is a verified candidate, not an approved frozen release.

## 2. Dataset Fingerprint

Canonical episode dataset SHA-256:

    fd609550ad3bb6743a60dabe99652e26899afd0f53b64ddcf65109b237a16252

Artifact SHA-256 fingerprints:

Statistics:

    49cf0fe7875ff64b45a4f5e432f11e17008fe547eaff89a6b74cf58c3be87b41

Splits:

    26e1095943f10971d501762e4f3bad997aefc88c999760b662eb429e26c30dc5

Provenance:

    925e338e44a45b8951872a171faa37f2d8207c9667b988a522db99cf14dc5fd0

The draft dataset manifest contains individual episode checksums.

## 3. Dataset Splits

The canonical deterministic split assignment is:

- Train: 64 episodes
- Validation: 15 episodes
- Test: 14 episodes
- Seed: 42
- Grouping: scenario_initial

The split checker verifies deterministic assignment and
protects against shared scenario and recorded initial-condition
groups crossing dataset partitions.

The existing split reference remains marked as a candidate.

## 4. Validation Instructions

Run from the repository root.

Verify the complete staging snapshot:

    uv run python -m robustlearn.dataset_freeze datasets/v0.1_staging

Validate the canonical dataset schema:

    uv run python -m robustlearn.dataset_validation datasets/v0.1_staging/episodes

Verify deterministic dataset splits:

    uv run python -m robustlearn.dataset_splits datasets/v0.1_staging/episodes --manifest datasets/v0.1_staging/splits.json --check

Run freeze-verifier regression tests:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/unit/test_week11_dataset_freeze.py

The freeze verifier checks:

- Exact episode membership
- Individual episode checksums
- Dataset fingerprint
- Report file checksums
- Canonical dataset statistics
- Provenance metadata
- Deterministic split assignments
- Missing, additional, or modified artifacts

Source-to-curated integrity was separately verified against:

    build/week11_v01_eligible_1000_1099/episodes

## 5. Teleoperation Demonstrations

Human-operated Cartesian teleoperation has been demonstrated.

The recording pipeline supports:

- Operator-controlled Cartesian movements
- Explicit deadman controls
- Controller completion verification
- Operator review and approval
- Canonical episode export
- Dataset schema validation

A verified two-frame accepted-failure pilot exists at:

    build/week11_manual_retry_ZUt6No/episodes/

This pilot is excluded from the current release candidate.

It demonstrates recording functionality but not successful
peg insertion or a meaningful complete manipulation trajectory.

Earlier Week 11 acceptance documentation references a
22-frame human-operated recording.

That artifact was not found in the searched local dataset
directories and is not included in this candidate.

The current candidate is therefore SCRIPTED ONLY.

Mixed-source acceptance requirements remain unresolved.

## 6. Reproducibility

The scripted episodes were collected from recorded
configuration identifiers and seeds.

The dataset provenance contains:

- Source episode filenames
- Source episode checksums
- Curated episode checksums
- Source Git commits
- Collection configuration identifiers
- Episode seeds
- Scenario identifiers
- Transformation records

The exact scripted regeneration commands must be verified
against the existing collection tooling before release.

Matching provenance checksums establishes the identity
of existing artifacts, not guaranteed byte-for-byte replay
from a new simulation run.

The final dataset release must identify the Git revision
containing the validated release artifacts and tooling.

The draft manifest's tooling commit is not a final
release commit.

### Verified production selection policy

The original production collection contains 100 scripted expert
episodes with seeds 1000 through 1099.

Collection provenance:

- Source Git commit: e2c9a265766cdeffb95f092c8a9a4cb6ea035abb
- Configuration ID: week11-v01-production-v1
- Original source label: expert
- Collection task: peg_insertion

Selection rule:

Include an original episode if and only if its recorded
success field is true.

The audited production results were:

- 100 original episodes
- 93 successful episodes retained
- 7 failed episodes excluded

Excluded episode IDs and reasons:

- 1007: moveit_execution_failure
- 1014: acceptance_probe_failure
- 1017: moveit_execution_failure
- 1027: moveit_execution_failure
- 1057: moveit_execution_failure
- 1067: moveit_execution_failure
- 1097: moveit_execution_failure

The selected eligible source episodes are stored at:

    build/week11_v01_eligible_1000_1099/episodes

The original complete collection is stored at:

    build/week11_v01_production_raw/episodes

Every retained source file was verified against its original
production episode using SHA-256.

The curated dataset was created by changing the top-level
source label from expert to scripted.

The curation process preserves observations, actions,
outcomes, timestamps, and recorded randomization metadata.

The existing curation command accepts the eligible source
directory as its source root and requires a new output root.

The exact source files and curation transformations are
recorded in provenance.json.

This is a success-only expert dataset. It is not representative
of the complete distribution of scripted task outcomes.

The seven excluded failures remain in the original production
collection. They have not been deleted or relabeled.

### Scripted regeneration outline

The collection interface supports count, start-seed,
timeout, dataset-root, capture-root, summary-path,
config-id, and source Git commit arguments.

The recorded collection parameters establish:

- Count: 100
- Start seed: 1000
- Configuration ID: week11-v01-production-v1
- Source revision: e2c9a265766cdeffb95f092c8a9a4cb6ea035abb

The precise production timeout, environment configuration,
and complete original invocation have not yet been
independently recovered.

To reproduce the collection, use the original Git revision
in a clean environment, configure ROS2 and MuJoCo, and run
the scripted collector with the recorded parameters and
isolated output paths.

Apply the verified success-only eligibility rule, then run
the canonical dataset curation command on the selected files.

Regenerated trajectories are not guaranteed byte-identical
because simulation execution and ROS2 timing may differ.

The frozen dataset checksums refer to the original verified
artifacts, not newly regenerated trajectories.

## 7. Dataset Versioning Policy

Dataset v0.1 must not be silently modified after freezing.

Any change to episode contents, episode membership,
split assignments, or accepted dataset metadata requires
a new versioned release.

The current staging snapshot is not immutable.

## 8. Outstanding Acceptance Gates

Before closing GitHub Issue #103:

- Resolve scripted-only versus mixed-source release scope
- Verify scripted regeneration instructions
- Finalize the versioned dataset manifest
- Finalize the release directory and version identity
- Pin the release Git revision
- Run full Python regression tests
- Run ROS2 build, tests, and lint checks
- Review and commit release artifacts
- Create and merge the GitHub pull request
- Confirm reproducibility from the released revision

## 9. Current Status

The 93-episode dataset candidate has passed:

- Canonical dataset validation
- Statistics verification
- Episode checksum verification
- Source-to-curated integrity verification
- Canonical deterministic split validation
- Draft snapshot verification
- Freeze-verifier regression tests
- A semantic split-corruption rejection test

Final status:

    DRAFT_NOT_FROZEN

GitHub Issue #103 remains open.
