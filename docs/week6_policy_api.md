# Week 6 Panda Insertion Policy API

Issue #51 defines the stable state/action contract that later Behaviour Cloning,
DAgger, and SAC policies will consume.

The policy API is intentionally separate from the existing low-level MuJoCo
actuator-control interface.

## Policy observation

The policy observation is a flat `float64` vector with 38 elements.

| Slice | Size | Component | Units / frame |
| --- | ---: | --- | --- |
| `0:7` | 7 | Panda joint positions | radians |
| `7:14` | 7 | Panda joint velocities | radians/second |
| `14:17` | 3 | end-effector position | metres, world frame |
| `17:21` | 4 | end-effector quaternion | `[w, x, y, z]`, world orientation |
| `21:24` | 3 | target-relative position | metres, expressed in `peg_tip` frame |
| `24:28` | 4 | target-relative quaternion | `[w, x, y, z]`, target orientation relative to `peg_tip` |
| `28:31` | 3 | wrist force | newtons, `panda_wrist_ft` local frame |
| `31:34` | 3 | wrist torque | newton-metres, `panda_wrist_ft` local frame |
| `34:38` | 4 | previous policy action | same ordering/units as the policy action |

The end-effector reference for the insertion policy is the project-owned
`peg_tip` site.

The target reference is the project-owned `receptacle_center` site.

All robot and task references are resolved by name rather than exposed as raw
MuJoCo indices.

## Policy action

The learned-policy action is:

```text
[delta_x, delta_y, delta_z, delta_yaw]
```

with the following ordering and limits:

| Component | Units | Minimum | Maximum |
| --- | --- | ---: | ---: |
| `delta_x` | metres | -0.002 | 0.002 |
| `delta_y` | metres | -0.002 | 0.002 |
| `delta_z` | metres | -0.002 | 0.002 |
| `delta_yaw` | radians | -0.05 | 0.05 |

These are Cartesian policy commands, not MuJoCo actuator targets.

By default, finite out-of-range actions are clipped component-wise to these
limits. Callers can request strict validation instead, in which case
out-of-range values are rejected.

Non-finite actions and actions with any shape other than `(4,)` are rejected.

## Previous action

The observation includes the previous 4D policy action explicitly.

At the beginning of an episode, callers should use the zero action:

```text
[0.0, 0.0, 0.0, 0.0]
```

This keeps the observation definition fixed across the first and later policy
steps.

## Cameras

RGB and depth cameras are not part of the Week 6 learned-policy observation.

They remain available for:

- logging;
- debugging;
- visualization;
- later perception extensions.

The state-based policy therefore does not require a rendering context.

## Gripper

Gripper control is outside the initial learned insertion-policy action.

The fixed insertion task continues to use the existing gripper/tool setup while
the learned-policy API controls only Cartesian insertion motion.

## Separation from low-level control

The existing environment and scripted controller use low-level MuJoCo actuator
targets.

The Week 6 policy API does not replace or reinterpret those actuator controls.
It defines the higher-level state/action contract that later policy-control
integration will consume.

The conversion from the 4D Cartesian policy action to low-level Panda actuator
commands is a separate control-layer responsibility.

## Determinism

For an identical simulator state and identical previous action,
`build_policy_observation()` returns the same observation exactly.

The canonical clean task observation contains only finite values.
