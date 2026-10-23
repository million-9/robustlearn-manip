# Week 11 — Scripted Expert Demonstration Collection

## Purpose

Collect reproducible, seeded expert demonstrations for the
RobustLearn-Manip Panda peg-insertion task.

The collection pipeline uses:

- ROS2 Jazzy and MoveIt2 for motion planning and execution
- MuJoCo for simulated robot execution
- The classical local insertion expert
- The Week 10 seeded benchmark runner
- The Week 11 rollout dataset writer

Behavior cloning and learned-policy training are outside
the scope of this collection step.

## Prerequisites

Run commands from the repository root.

Source the ROS2 environment:

    source /opt/ros/jazzy/setup.bash
    source ros2_ws/install/setup.bash

Use the project's uv-managed Python environment.

Ensure the necessary ROS2 packages have been built.

## Collect demonstrations

Example: collect three episodes starting at seed 2.

    uv run python -m tools.collect_scripted_demos \
      --count 3 \
      --start-seed 2 \
      --timeout-s 90 \
      --dataset-root "$PWD/build/week11_dataset" \
      --capture-root "$PWD/build/week11_captures" \
      --summary-path "$PWD/build/week11_summary.json" \
      --config-id week11-collection-v1 \
      --git-commit "$(git rev-parse HEAD)"

Each seed selects a deterministic benchmark scenario and
seeded initial configuration.

Use a distinct dataset directory when changing the
experiment configuration or source revision.

Do not reuse the same configuration identifier for
materially different collection configurations.

For production collection, use a clean Git revision.
A revision labelled "-dirty" identifies uncommitted
development and is not sufficient to reproduce the exact
source code independently.

## Dataset contents

Each recorded episode contains:

- Episode index, seed, scenario, task, and source
- Success or failure outcome and failure reason
- Episode timing boundaries
- Git revision and configuration identifier
- Initial base joint positions, when available
- Seeded joint perturbations, when available
- Resulting starting joint positions, when available
- Time-ordered expert demonstration frames

Each frame contains:

- Seven Panda joint positions
- Seven joint velocities, when available
- Three-dimensional task-relative position
- Six-dimensional wrist wrench
- Seven-dimensional absolute expert joint targets
- Success, failure, and termination labels

The expert action represents a controller-accepted joint
target. It is not proof that the entire trajectory action
has completed.

The final terminal label is attached to the last captured
action frame, not to a separately sampled final observation.

Missing measured velocities are represented as null.
They must not be replaced with invented values.

Existing schema-version-1 episodes without randomization
metadata remain readable.

## Dataset validation

Successful demonstrations should have:

- At least one frame
- Contiguous frame indices
- Finite observations and actions
- Correct observation and action dimensions
- Non-decreasing frame timestamps
- A valid final terminal success label
- Matching raw capture and persisted dataset frames

When randomization metadata is present, all three vectors
must be seven-dimensional, finite, and satisfy:

    start_positions = base_positions + perturbations

within the dataset validation tolerance.

## Safe resumption

Rerun the original collection command with the same
seed range, Git revision, and configuration identifier.

The collector validates existing dataset episodes and
skips those that match the requested configuration.

Existing datasets are not overwritten.

The summary tracks:

- Requested episodes
- Newly collected successes
- Newly collected failures
- Skipped episodes
- Collection errors

The collector checkpoints its summary after each processed
episode.

A task failure and a collection error are different:
a failed task can have a valid dataset record, while a
collection error may interrupt the batch.

## Interrupted simulations and orphaned captures

If ROS2 or MuJoCo is interrupted, the expert may have
written a capture without a completed dataset record.

The collector intentionally refuses to overwrite an
existing capture file.

Recovery procedure:

1. Stop the interrupted simulation processes.
2. Inspect the affected capture and dataset paths.
3. Preserve existing artifacts in a separate archive.
4. Ensure the intended output paths are clear.
5. Restart only the interrupted seed or seed range.
6. Validate the new episode and collection summary.

Do not delete an orphaned capture automatically.

Do not modify a previously validated demonstration merely
to make a resume operation succeed.

A collection error is reported in the checkpoint summary.
Correct the underlying problem before restarting.

## Verified development milestones

During Week 11 development:

- Seed 0: successful capture with 40 frames
- Seed 1: successful capture with 38 frames
- Seeds 2–4: three successful pilot episodes
- Safe resume: three existing episodes skipped
- Seed 5: 41-frame successful demonstration after
  recovering an interrupted run
- Seed 5: complete joint velocities and validated
  starting-state randomization metadata

These development runs do not constitute the planned
large-scale production dataset.

## Seeded replay verification

An independent seed-5 replay was performed using the same
scenario, configuration identifier, and development revision.

The following fields matched exactly:

- Initial base joint positions
- Seed-specific joint perturbations
- Resulting starting joint positions

Both executions completed the insertion task successfully.

The original demonstration contained 41 frames; the replay
contained 39 frames.

This demonstrates deterministic seeded initialization,
not bit-for-bit deterministic ROS2 execution.

Controller timing, simulation scheduling, and action
acceptance timing may cause different recorded trajectories
and frame counts.

Reproducibility claims must distinguish starting-state
reproducibility from trajectory reproducibility.

## Known limitations

- Interrupted captures require explicit recovery.
- Older demonstrations may lack measured joint velocities.
- Older demonstrations may lack starting-state metadata.
- Terminal labels are attached to the final action frame.
- Controller acceptance does not imply completed execution.
- Development runs labelled "-dirty" are not fully
  reproducible from a Git commit alone.

Keep development datasets separate from future production
collection artifacts.
