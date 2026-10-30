# Week 11 Scripted Dataset Curation

## Purpose

The raw scripted-expert collector records `source="expert"`.
The canonical training dataset validator accepts `source="scripted"`.

This curation tool bridges those contracts without modifying the
original demonstration records.

## Generate a candidate

From the repository root:

    uv run python -m robustlearn.dataset_curation \
      --source-root PATH_TO_RAW_EXPERT_EPISODES \
      --output-root PATH_TO_NEW_CURATED_CANDIDATE

The output directory must not already exist.

Generated artifacts:

- `episodes/`: canonical scripted demonstration episodes
- `statistics.json`: deterministic dataset statistics
- `provenance.json`: original and curated checksums and metadata

## Preservation rules

- Only the `source` field changes from `expert` to `scripted`.
- Observations, actions, seeds, timestamps, outcome labels,
  scenario labels, and configuration metadata remain unchanged.
- Original dataset files are never overwritten.
- Duplicate episode IDs are rejected.
- Invalid episode contents are rejected.
- Source SHA256 and curated SHA256 are recorded separately.
- Existing output directories are never overwritten.
- A complete candidate is validated before publication.
- Dirty Git revisions remain explicitly visible in provenance.

The output is a **candidate**, not dataset version `v0.1`.

Collection provenance and structural validation do not independently
prove that a demonstration is high quality or genuinely human-operated.

Production acceptance requires separately reviewing collected runs,
reconciling duplicates, validating teleoperation recordings, and
generating the final versioned freeze manifest and integrity checks.

The original scripted collection may be reproducible in seeded initial
conditions without producing byte-identical ROS2 trajectories.
