# Week 11 — Manual Teleoperation Demonstrations

GitHub Issue: #100

## Goal

Collect operator-controlled Panda manipulation demonstrations
using the canonical rollout dataset schema.

## Recording configuration

Recording is disabled by default.

Enable recording explicitly using ROS2 parameters:

- record_demonstrations=true
- teleop_output_root: absolute episode directory
- teleop_episode_index: non-negative index
- teleop_seed: non-negative seed
- teleop_scenario: scenario identifier
- teleop_git_commit: Git commit
- teleop_config_id: configuration identifier
- teleop_task: task identifier
- teleop_failure_reason: required for accepted failures

## Example command

After building and sourcing the ROS2 workspace:

    ros2 run robustlearn_control keyboard_teleop.py --ros-args \
      -p record_demonstrations:=true \
      -p teleop_output_root:=/tmp/robustlearn_week11/episodes \
      -p teleop_episode_index:=0 \
      -p teleop_seed:=42 \
      -p teleop_scenario:=nominal \
      -p teleop_git_commit:=example-commit \
      -p teleop_config_id:=manual-v1 \
      -p teleop_task:=panda_insertion \
      -p teleop_failure_reason:="Insertion incomplete"

Replace example metadata with actual collection values.

## Operator controls

- Space: refresh deadman
- Joint jog keys: bounded keyboard controls
- x: stop progression
- v: enter review
- 0: resume after safety checks
- 1: accept successful demonstration
- 2: accept failed demonstration
- 3: reject demonstration
- s: retry a pending summary update
- Escape: abort

## Safety and dataset integrity

Controller completion and fresh stationary joint feedback
must be verified before accepting a recording.

Accepted episodes use the canonical rollout schema with
source=teleop.

Rejected and aborted recordings never export training episodes.

Episode writes are exclusive and do not overwrite existing data.

Collection summaries are stored outside the episode directory.

Failed episode writes support controlled retry using the
original outcome and metadata.

Summary-only retries must not rewrite the saved episode.

## Verification

Automated tests cover:

- Dataset schema compatibility
- Episode metadata
- Operator review decisions
- Controller completion tracking
- Cancellation handling
- Persistence failures and retries
- Collection summary accounting
- Scripted/teleop dataset coexistence

ROS2 build, tests, and linting must also pass.

A supervised end-to-end robot demonstration remains necessary.

Abrupt process termination may leave an attempt without a
final summary record.

Concurrent collection writers are not supported.

The long-term dataset target is approximately 50–100
human/gamepad demonstrations. Automated tests do not fulfill
that collection target.


## Final acceptance evidence

GitHub Issue: #100

### Automated validation

- Full Python test suite passed.
- Ruff checks passed.
- ROS2 workspace built successfully.
- ROS2 validation: 169 tests, 0 failures, 13 skipped.
- ROS2 Flake8 passed.
- Canonical scripted/teleop dataset coexistence verified.

### Live ROS2/MuJoCo acceptance

- MuJoCo simulation hardware verified.
- Automated accepted-failure recording persisted successfully.
- Review and rejection confirmed without exporting an episode.
- Unexpected termination before approval exported no episode.
- Recording restarted successfully after interruption.
- Explicit abort recorded without dataset contamination.
- Human-operated accepted-failure demonstration verified.
- Human-operated episode captured 22 frames.
- Saved episode source: teleop.
- Saved outcome: accepted_failure.
- Collection summary matched the saved episode.

### Scope and limitations

The acceptance trial used the fixed MuJoCo initial configuration,
not randomized starting poses or domain randomization.

The saved demonstration represents a deliberately incomplete
insertion, not successful task execution.

The interruption trial covered termination before approval.
Recovery from a crash during summary persistence remains
a known limitation.

The project target of 50-100 manual demonstrations will be
collected incrementally.

BC training, policy evaluation, and DAgger remain out of scope.

## Verified manual acceptance

- Human-operated accepted-failure trial: 22 captured frames.
- MuJoCo simulation with canonical source=teleop.
- Episode and collection summary validated successfully.
- Rejection and interruption/restart probes passed.
- Crash-during-persistence recovery remains unverified.
