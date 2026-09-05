# Panda insertion task evaluation

Issue #39 adds explicit geometric task semantics for the Panda insertion
environment.

The evaluator classifies the current task state as:

```text
success
failure
in progress
```

This task evaluation is intentionally separate from the final learned-policy
reward function.

## Geometric quantities

The evaluator uses the world-frame positions of:

```text
peg_tip
receptacle_center
insertion_axis
```

A reference axis is derived from the task geometry:

```text
axis_vector = insertion_axis - receptacle_center
axis_direction = axis_vector / norm(axis_vector)
```

The `insertion_axis` marker lies on the pre-insertion side of the receptacle.
Therefore `axis_direction` points outward toward the approach side, while
positive insertion progress occurs in the opposite direction.

The insertion-axis marker must differ from the receptacle center.

Define the peg-tip displacement from the receptacle center as:

```text
delta = peg_tip - receptacle_center
```

The signed offset along the insertion axis is:

```text
axial_offset = dot(delta, axis_direction)
```

Its sign convention is:

```text
positive   peg tip is toward the pre-insertion marker
zero       peg tip is at the entrance plane
negative   peg tip has progressed in the insertion direction
```

The component perpendicular to the insertion axis is:

```text
lateral_vector = delta - axial_offset * axis_direction
```

and the lateral alignment error is:

```text
lateral_error = norm(lateral_vector)
```

This value is always non-negative and measures misalignment perpendicular to
the configured insertion axis.

Insertion depth is defined as:

```text
insertion_depth = -axial_offset
```

Therefore:

```text
negative   peg tip has not reached the entrance plane
zero       peg tip is at the entrance plane
positive   peg tip has progressed along the insertion direction
```

This formulation does not assume that insertion is aligned with world Z. If
the fixture and its task sites are rotated later, the same evaluator follows
the configured insertion axis.

All geometric values are expressed in metres.

## Success condition

The initial success condition is:

```text
lateral_error <= 0.001
AND
insertion_depth >= 0.010
```

This corresponds to:

```text
maximum lateral error     1.0 mm
minimum insertion depth  10.0 mm
```

The workcell geometry provides approximately 1.5 mm clearance per lateral axis
between the peg and receptacle opening.

The 1.0 mm success tolerance is deliberately more conservative than the
available physical clearance.

The 10 mm insertion-depth requirement prevents simple contact with the entrance
plane from being classified as successful insertion.

## Failure condition

The initial geometric failure condition is:

```text
insertion_depth > 0.0
AND
lateral_error >= 0.0022
```

This corresponds to a lateral error of 2.2 mm after the peg tip has moved below
the entrance plane.

The square opening provides approximately 1.5 mm clearance per lateral axis.
The diagonal equivalent is approximately:

```text
sqrt(2) * 1.5 mm = 2.12 mm
```

The 2.2 mm failure threshold therefore represents a clearly incompatible
inserted state rather than a boundary case.

A peg that is laterally misaligned while still above the receptacle is not
immediately classified as failed. It remains in progress because it may still
be corrected before insertion.

## Canonical reset state

The deterministic reset geometry is approximately:

```text
lateral_error    0.000000522 m
axial_offset  0.019502429 m
insertion_depth -0.019502429 m
```

The reset state is therefore:

```text
success = false
failure = false
```

and remains in progress.

## Gymnasium termination semantics

Task termination and episode truncation are separate.

Task termination is:

```text
terminated = success OR failure
```

The configured episode step limit is:

```text
truncated = elapsed_steps >= max_episode_steps
```

Reaching the episode step limit does not imply task failure.

A time-limited episode may therefore report:

```text
terminated = false
truncated = true
task_failure = false
```

## Environment diagnostics

The Gymnasium `info` dictionary exposes:

```text
task_lateral_error
task_axial_offset
task_insertion_depth
task_success
task_failure
```

alongside the existing:

```text
simulation_time
elapsed_steps
```

These values are intended for debugging, evaluation, scripted task logic, and
later benchmark reporting.

## Reward scope

Issue #39 does not define the final reinforcement-learning reward.

The environment reward remains:

```text
0.0
```

for this milestone.

This keeps geometric task evaluation independent from later reward shaping,
force-based control logic, and reinforcement-learning design.
