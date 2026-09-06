# Week 6 Panda Insertion Configuration

Issue #49 introduces the project-owned configuration contract used by later
Week 6 randomization, policy-interface, and evaluation work.

The configuration layer is intentionally separate from MuJoCo model mutation.
This issue defines and validates configuration; applying domain randomization
is handled by the later randomization issue.

## Task configuration

`PandaInsertionTaskConfig` defines the clean Panda insertion runtime settings.

Default values preserve the Week 5 baseline:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `frame_skip` | `5` | MuJoCo physics steps per environment step |
| `max_episode_steps` | `200` | Environment truncation limit |
| `success_lateral_tolerance_m` | `0.001` | Maximum lateral error for success |
| `success_insertion_depth_m` | `0.010` | Minimum insertion depth for success |
| `failure_lateral_error_m` | `0.0022` | Lateral error threshold for inserted failure |

All task thresholds must be finite and positive.

The failure lateral threshold must be strictly greater than the success
lateral tolerance.

## Randomization configuration

`PandaInsertionRandomizationConfig` is nested inside the task configuration.

Randomization is explicitly disabled by default:

```python
PandaInsertionRandomizationConfig(enabled=False)
```

The initial schema defines bounded episode-level perturbations for:

- receptacle X offset in metres;
- receptacle Y offset in metres;
- receptacle yaw offset in radians.

Their default ranges are all `[0.0, 0.0]`, so the default configuration does
not modify the clean Week 5 task.

Actual sampling and application of these values is intentionally deferred to
the Week 6 randomization implementation.

## Validated ranges

`FloatRange` represents an inclusive finite range:

```python
FloatRange(low=-0.002, high=0.002)
```

A range is rejected when:

- either bound is non-finite;
- `low > high`.

This makes invalid or inverted randomization ranges fail deterministically at
configuration construction time.

## Inspecting configuration

Each configuration object provides `to_dict()`.

For example:

```python
from robustlearn.config import PandaInsertionTaskConfig

config = PandaInsertionTaskConfig()

print(config.to_dict())
```

The returned dictionary contains only plain serializable configuration values
and nested dictionaries. It is suitable for later evaluation metadata and
reproducibility records.

## Environment integration

`PandaInsertionEnv` accepts an explicit configuration:

```python
from robustlearn.config import PandaInsertionTaskConfig
from robustlearn.envs import PandaInsertionEnv

config = PandaInsertionTaskConfig(
    frame_skip=5,
    max_episode_steps=200,
)

env = PandaInsertionEnv(config=config)
```

For compatibility with the existing Week 5 API, the legacy `frame_skip=` and
`max_episode_steps=` keyword arguments remain available when no explicit
configuration object is supplied.

An explicit `config=` cannot be combined with those legacy overrides.

The environment passes its effective task configuration to geometric task
evaluation, and the scripted insertion controller reads the configured success
depth from the environment.

## Clean baseline guarantee

With `PandaInsertionTaskConfig()`:

- randomization is disabled;
- all randomization ranges are zero-width;
- canonical Week 5 task thresholds are preserved;
- canonical reset behavior remains unchanged;
- canonical scripted insertion behavior remains unchanged.

This clean configuration is the reference baseline for later Week 6 work.

## Out of scope

Issue #49 does not:

- sample randomization values;
- mutate MuJoCo model or state parameters for randomization;
- define the final learned-policy observation;
- define the final Cartesian policy action;
- implement multi-episode evaluation.

Those capabilities are introduced by subsequent Week 6 issues.
