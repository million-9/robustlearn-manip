# Week 6 Seeded Panda Insertion Evaluation

Issue #52 provides a reusable, headless evaluation runner for executing
multiple Panda insertion episodes with explicit deterministic seeds.

The runner currently evaluates the existing scripted insertion controller.
Learned-policy evaluation is intentionally outside this issue.

## Command-line usage

Run a clean evaluation using consecutive seeds:

```bash
uv run python -m robustlearn.evaluate \
  --base-seed 2026 \
  --episodes 10
```

This evaluates seeds:

```text
2026, 2027, ..., 2035
```

Run an explicit seed list:

```bash
uv run python -m robustlearn.evaluate \
  --seeds 17,42,2026
```

The CLI is headless by default and does not create a rendering context.

## Seed contract

The Python API supports two seed modes.

### Consecutive seed mode

Provide:

- `base_seed`
- `episode_count`

The runner generates consecutive seeds beginning at `base_seed`.

Example:

```python
evaluate_scripted_episodes(
    base_seed=2026,
    episode_count=3,
)
```

uses:

```text
2026, 2027, 2028
```

### Explicit seed mode

Provide an explicit ordered seed sequence:

```python
evaluate_scripted_episodes(
    seeds=[17, 3, 99],
)
```

The supplied order is preserved exactly.

Explicit seeds cannot be combined with `base_seed` or `episode_count`.

Seeds must be non-negative integers.

## Configuration

The runner accepts an optional `PandaInsertionTaskConfig`.

If no configuration is supplied, the canonical clean/default task
configuration is used.

The same runner also supports randomized configurations:

```python
config = PandaInsertionTaskConfig(
    randomization=PandaInsertionRandomizationConfig(
        enabled=True,
        receptacle_x_offset_m=FloatRange(-0.0005, 0.0005),
        receptacle_y_offset_m=FloatRange(-0.0005, 0.0005),
        receptacle_yaw_offset_rad=FloatRange(-0.02, 0.02),
    )
)

report = evaluate_scripted_episodes(
    base_seed=2026,
    episode_count=10,
    config=config,
)
```

Randomization therefore changes configuration, not evaluation-runner
architecture.

## Episode result

Each completed episode records:

| Field | Meaning |
| --- | --- |
| `seed` | explicit episode seed |
| `success` | task success flag |
| `failure` | geometric task failure flag |
| `truncated` | episode ended at the configured step limit |
| `step_count` | number of environment steps executed |
| `simulation_duration_s` | final MuJoCo simulation time in seconds |
| `final_lateral_error_m` | final peg/receptacle lateral error in metres |
| `final_insertion_depth_m` | final insertion depth in metres |
| `task_config` | serialized task configuration used by the episode |
| `randomization_sample` | actual sampled randomization values |

Non-finite observations, actions, final task metrics, or randomization
metadata are rejected rather than written into a result record.

## Aggregate summary

A completed evaluation report contains:

- episode count;
- success count;
- task-failure count;
- truncation count;
- success rate;
- failure-category counts.

The current failure categories are:

```text
task_failure
truncated
```

Task failure and truncation remain distinct because truncation represents the
episode time/step limit rather than geometric insertion failure.

## Output structure

The CLI prints JSON with this top-level structure:

```json
{
  "episodes": [
    {
      "seed": 2026,
      "success": true,
      "failure": false,
      "truncated": false,
      "step_count": 34,
      "simulation_duration_s": 0.34,
      "final_lateral_error_m": 0.000057,
      "final_insertion_depth_m": 0.010737,
      "task_config": {},
      "randomization_sample": {}
    }
  ],
  "summary": {
    "episode_count": 1,
    "success_count": 1,
    "failure_count": 0,
    "truncated_count": 0,
    "success_rate": 1.0,
    "failure_categories": {
      "task_failure": 0,
      "truncated": 0
    }
  }
}
```

The real `task_config` and `randomization_sample` objects contain the complete
serialized configuration and sampled values.

## Reproducibility

Re-running the same ordered seed set with the same configuration and controller
produces the same episode result records.

Each episode creates and resets its own environment through the same evaluation
path. A normal task failure in one episode therefore does not prevent later
seeds from being evaluated.

## Week 6 milestone relationship

Issue #52 provides the reusable evaluation mechanism.

The final Week 6 acceptance issue performs the milestone gate:

```text
100 seeded episodes execute automatically.
```

That larger acceptance run is intentionally separate from the lightweight
episode counts used by normal CI tests.
