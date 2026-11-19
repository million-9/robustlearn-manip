# Dataset v0.1 — Release Preparation

GitHub Issue: #103

Status: FROZEN

## 1. Dataset Inventory

The current mixed-source release candidate contains:

- 93 scripted expert demonstrations
- 1 human teleoperation demonstration
- 94 total episodes
- 3,727 recorded frames
- 94 successful episodes
- 0 failed episodes
- 21 recorded scenarios

Dataset location:

    datasets/v0.1/

Status: FROZEN

This is an integrity-verified frozen dataset snapshot.
Publication remains pending Git and PR acceptance.

## 2. Dataset Fingerprint

Canonical episode dataset SHA-256:

    eb28d409db76e18c6a8e6f6f0ccb49429f0271d4bf8a6f16881637ae0dd5db3f

Artifact SHA-256 fingerprints:

Statistics:

    6e31a96dba564592454291fd99258ec65d1ddcefbf6e8552e2617438ce18c5c0

Splits:

    fa45836a7727ed783a316589ea48c310c4ac71501b21c8cfe2727dc35454869c

Provenance:

    d133219e78e196ac7c36b0a200fdb757d90fe707babf9b9325be8ef03cd17c9e

The draft manifest contains individual checksums for all
94 episodes.

The frozen snapshot passes the dataset freeze verifier.

## 3. Dataset Splits

The canonical deterministic split assignment is:

- Train: 65 episodes
- Validation: 15 episodes
- Test: 14 episodes
- Seed: 42
- Grouping: scenario_initial
- Dataset reference: week11-v01-94-mixed-candidate-not-frozen

The split checker verifies deterministic assignment and
protects against shared scenario and recorded initial-condition
groups crossing dataset partitions.

The split manifest is frozen with the v0.1 snapshot.

## 4. Validation Instructions

Run from the repository root.

Verify the complete staging snapshot:

    uv run python -m robustlearn.dataset_freeze datasets/v0.1_staging

Validate the canonical dataset schema:

    uv run python -m robustlearn.dataset_validation datasets/v0.1/episodes

Verify deterministic dataset splits:

    uv run python -m robustlearn.dataset_splits datasets/v0.1/episodes --manifest datasets/v0.1/splits.json --check

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

One human-operated Cartesian insertion demonstration is
included in the mixed-source candidate.

Episode:

    episode_002004__seed_2004__manual_home_insertion_retry_01.json

Collection evidence:

- Source: teleop
- Scenario: manual_home_insertion_retry_01
- Recorded frames: 33
- Saved outcome: success
- Final-frame task_success: true
- Final-frame geometric success: verified
- Final lateral error: approximately 0.535 mm
- Final insertion depth: approximately 12.586 mm
- Maximum consecutive recorded displacement: 0.997 mm
- No consecutive recorded displacement above 2 mm
- Canonical dataset validation: passed

The recording was captured through the ROS2 Cartesian
teleoperation controller with operator deadman controls,
controller-feedback observations, explicit review, and
canonical episode export.

The trajectory uses command-triggered recording rather
than uniform-rate sampling. The maximum observed timestamp
gap was 19.041 seconds, consistent with deliberate operator
pauses. Original timestamps have been preserved.

Source-specific Git revision, configuration ID, episode
seed, checksums, and transformation information are recorded
in provenance.json.

The scripted demonstrations retain the task label
peg_insertion; the teleoperation demonstration uses
panda_insertion. These distinct recorded labels are preserved.

### Mixed-source limitations

- Only one human teleoperation demonstration is included.
- All 94 selected episodes are successful; failure coverage
  is absent from the released candidate.
- Teleoperation sampling is command-triggered, not fixed-rate.
- No mixed-source policy-training benchmark has been established.
- Dataset integrity does not establish downstream policy quality.
- The dataset is frozen locally but has not yet been published.

The earlier incomplete teleoperation pilots and deleted
recordings are not included.

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

The historical collector interface and reconstructed
seed-to-scenario schedule have been verified. The complete
original production invocation remains unverified.

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

The frozen dataset checksums identify the verified
release artifacts, not newly regenerated trajectories.

### Verified collection schedule reconstruction

The original collector interface was inspected at source
revision:

    e2c9a265766cdeffb95f092c8a9a4cb6ea035abb

Its command-line arguments match the current collection
interface.

The following command reconstructs the planned schedule
without launching ROS2 simulations:

    uv run python -m tools.collect_scripted_demos \
      --count 100 \
      --start-seed 1000 \
      --timeout-s 90 \
      --config-id week11-v01-production-v1 \
      --git-commit e2c9a265766cdeffb95f092c8a9a4cb6ea035abb \
      --dry-run

Verification results:

- 100 episodes scheduled with seeds 1000 through 1099
- All 93 retained scripted episodes match the reconstructed
  episode indices and scenario assignments
- Seven scheduled episodes were excluded by the documented
  success-only selection policy
- The schedule matches the retained scripted provenance

The 90-second timeout is the historical collector's
default and the value used for this reconstruction.
It is not independently verified as the timeout used
during the original production execution.

### Full trajectory regeneration limitations

A new production collection would require:

- The recorded source revision
- A configured ROS2 Jazzy, MoveIt2, and MuJoCo environment
- Isolated dataset, capture, and summary output directories
- The recorded configuration ID and seed range
- Independent validation and success-only curation

The precise original timeout, complete runtime environment,
and full production command remain unverified.

Reconstructed schedule agreement does not establish
trajectory-level numerical reproducibility.

The source checksums and curated checksums in provenance.json
identify the preserved original artifacts. A new collection
must not overwrite the versioned dataset.

## 7. Dataset Versioning Policy

Dataset v0.1 must not be silently modified after freezing.

Any change to episode contents, episode membership,
split assignments, or accepted dataset metadata requires
a new versioned release.

The frozen snapshot must remain unchanged after publication.
Its durable Git release identity is still pending.

## 8. Outstanding Acceptance Gates

Before closing GitHub Issue #103:

- Review the final frozen release artifacts
- Stage and commit the release and tooling changes
- Pin the release identity to a Git commit or tag
- Re-run validation against the committed release
- Push the feature branch and open the GitHub pull request
- Confirm CI passes and merge the pull request
- Verify snapshot integrity from the released revision

The original scripted production runtime parameters remain
partially unverified, as documented in Section 6.

## 9. Current Status

The 94-episode mixed-source frozen snapshot has passed:

- Canonical dataset validation
- Statistics verification
- Episode checksum verification
- Scripted source-to-curated integrity verification (93 episodes)
- Canonical deterministic split validation
- Draft snapshot verification
- Freeze-verifier and mixed-source metadata regression tests
- A semantic split-corruption rejection test
- Full Python regression suite (928 passed)
- ROS2 build (5 packages)
- ROS2 tests (180 tests, 0 failures, 13 skipped)
- Ruff lint validation
- Git diff integrity checks

Final status:

    FROZEN

GitHub Issue #103 remains open.
