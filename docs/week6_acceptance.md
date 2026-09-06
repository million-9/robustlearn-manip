# Week 6 Acceptance Workflow

Week 6 completes the configuration, randomization, policy-interface, and
reproducible evaluation foundations for RobustLearn-Manip.

The milestone gate is:

```text
100 seeded episodes execute automatically.
```

## Architecture

The Week 6 evaluation path is:

```text
PandaInsertionTaskConfig
        |
        +---- task thresholds
        |
        +---- randomization configuration
        |
        v
PandaInsertionEnv
        |
        v
MuJoCoSimulation
        |
        +---- deterministic reset
        +---- seeded randomization
        +---- named task/sensor references
        |
        v
scripted insertion controller
        |
        v
evaluation runner
        |
        +---- per-episode records
        +---- aggregate statistics
        |
        v
reproducible seeded evaluation report
```

The learned-policy interface is defined separately from the existing low-level
MuJoCo actuator controls.

## Configuration

The project-owned task configuration is:

```text
PandaInsertionTaskConfig
```

It contains:

- frame skip;
- maximum episode steps;
- success lateral tolerance;
- success insertion depth;
- failure lateral threshold;
- randomization configuration.

The default configuration preserves the clean Week 5 task behavior.

Randomization is disabled by default.

## Randomization

The initial Week 6 randomization scope supports:

- receptacle X offset;
- receptacle Y offset;
- receptacle yaw offset.

Each range is represented using:

```text
FloatRange(low, high)
```

When randomization is enabled:

```text
same seed + same configuration = same sampled randomization
```

Samples are applied relative to the canonical task geometry and remain within
their configured bounds.

## Policy observation API

The state-based learned-policy observation contains 38 `float64` values:

```text
7 joint positions
7 joint velocities
3 end-effector position values
4 end-effector quaternion values
3 target-relative position values
4 target-relative quaternion values
3 wrist-force values
3 wrist-torque values
4 previous-action values
```

The policy observation does not require RGB or depth rendering.

The task end-effector reference is:

```text
peg_tip
```

The task target reference is:

```text
receptacle_center
```

All required task and sensor objects are resolved through named references
rather than exposed raw MuJoCo indices.

## Policy action API

The policy action is:

```text
[delta_x, delta_y, delta_z, delta_yaw]
```

with bounds:

```text
delta_x:   [-0.002, 0.002] m
delta_y:   [-0.002, 0.002] m
delta_z:   [-0.002, 0.002] m
delta_yaw: [-0.05, 0.05] rad
```

The learned-policy action remains separate from the existing low-level
MuJoCo actuator-control representation.

Gripper behavior is outside the initial learned insertion policy.

## Evaluation result schema

Each episode records:

```text
seed
success
failure
truncated
step_count
simulation_duration_s
final_lateral_error_m
final_insertion_depth_m
task_config
randomization_sample
```

The aggregate report records:

```text
episode_count
success_count
failure_count
truncated_count
success_rate
failure_categories
```

Non-finite episode observations, actions, final metrics, and randomization
metadata are rejected.

## CI-sized acceptance run

Normal CI uses a small representative acceptance subset:

```bash
uv run pytest tests/integration/test_week6_acceptance.py -q
```

Expected result:

```text
3 passed
```

This verifies:

- clean/default configuration;
- randomization disabled by default;
- finite policy observation construction;
- stable policy action bounds;
- deterministic clean multi-episode evaluation;
- deterministic randomized evaluation;
- configured randomization bounds;
- finite evaluation records.

## Full 100-episode milestone run

Run the complete Week 6 milestone gate from the repository root:

```bash
uv run python - <<'PY'
import numpy as np

from robustlearn.evaluation import evaluate_scripted_episodes

report = evaluate_scripted_episodes(
    base_seed=2026,
    episode_count=100,
)

assert report.summary.episode_count == 100

for episode in report.episodes:
    assert np.isfinite(episode.simulation_duration_s)
    assert np.isfinite(episode.final_lateral_error_m)
    assert np.isfinite(episode.final_insertion_depth_m)

print("Week 6 milestone: 100 seeded episodes executed automatically")
print(f"seed range: {report.episodes[0].seed}..{report.episodes[-1].seed}")
print(f"episodes: {report.summary.episode_count}")
print(f"successes: {report.summary.success_count}")
print(f"failures: {report.summary.failure_count}")
print(f"truncated: {report.summary.truncated_count}")
print(f"success rate: {report.summary.success_rate:.3f}")
print("all recorded scalar results finite: yes")
PY
```

Validated milestone result:

```text
Week 6 milestone: 100 seeded episodes executed automatically
seed range: 2026..2125
episodes: 100
successes: 100
failures: 0
truncated: 0
success rate: 1.000
all recorded scalar results finite: yes
```

## Full reproducibility check

The exact 100-episode result records were generated twice using the same
ordered seed set.

Validated result:

```text
Week 6 reproducibility: PASS
episodes compared: 100
seed range: 2026..2125
result records identical: yes
```

Therefore:

```text
same task configuration
+ same ordered seed set
+ same scripted controller
=
same evaluation records
```

## Full project validation

Before completing Week 6, run:

```bash
uv run ruff check .
uv run mypy robustlearn
uv run pytest
```

and the ROS 2 regression:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
colcon test
colcon test-result --verbose
```

## Week 6 milestone status

The Week 6 milestone gate is satisfied:

```text
100 seeded episodes execute automatically.
```

The validated clean evaluation achieved:

```text
100 / 100 successful episodes
0 task failures
0 truncations
1.000 success rate
```

and reproduced the same complete episode records on an independent rerun.

## Deferred beyond Week 6

The following remain intentionally outside this milestone:

- ros2_control MuJoCo hardware plugin;
- MoveIt-to-MuJoCo integration;
- classical insertion expert benchmark;
- demonstration collection;
- Behaviour Cloning;
- DAgger;
- SAC;
- large-scale robustness experiments;
- vision-based learned-policy inputs;
- ONNX deployment.
