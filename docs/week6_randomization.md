# Week 6 Panda Insertion Randomization

Issue #50 introduces deterministic, configuration-driven domain randomization for
the Panda insertion environment.

The clean Week 5 task remains the reference baseline.

## Design principles

Randomization is:

- disabled by default;
- controlled through `PandaInsertionRandomizationConfig`;
- sampled exclusively from the simulation-owned NumPy RNG;
- reproducible from the same explicit seed and configuration;
- inspectable after every reset;
- applied relative to the canonical workcell pose rather than accumulated across
  episodes.

Vendored MuJoCo Menagerie files are not modified.

## Supported parameters

The initial Week 6 implementation randomizes only the receptacle fixture pose:

- receptacle X offset in metres;
- receptacle Y offset in metres;
- receptacle yaw offset in radians.

Each parameter uses the validated `FloatRange` schema introduced in Issue #49.

Mass, friction, sensor-noise, and other uncertainty families are intentionally
deferred until the basic framework is established and tested.

## Reset lifecycle

When `PandaInsertionEnv.reset(seed=...)` is called:

1. the environment passes its configured randomization settings to
   `MuJoCoSimulation.reset()`;
2. an explicit seed recreates the simulation RNG;
3. the randomization sampler draws the episode parameters from that RNG;
4. disabled randomization produces exact zero perturbations and consumes no
   random draws;
5. the insertion fixture is restored from its cached canonical pose;
6. the sampled X, Y, and yaw perturbations are applied;
7. MuJoCo performs the normal reset and forward computation;
8. the sampled parameters are stored as `last_randomization_sample`;
9. the environment exposes the sample through the reset and step `info`
   dictionaries.

This order prevents perturbations from accumulating between episodes.

## Reproducibility contract

For a fixed configuration:

- same seed produces the same sampled parameters;
- same seed produces the same randomized simulator reset state;
- different seeds can produce different episodes when configured ranges have
  non-zero width;
- every sampled value remains inside its configured bounds.

The sampled episode parameters provide explicit metadata for the later Week 6
evaluation runner.

## Clean mode

`PandaInsertionRandomizationConfig()` has `enabled=False`.

In clean mode:

- the sampled X offset is exactly `0.0`;
- the sampled Y offset is exactly `0.0`;
- the sampled yaw offset is exactly `0.0`;
- the sampler does not advance the RNG;
- the canonical Week 5 workcell geometry is restored.

This preserves the deterministic Week 5 baseline and allows randomized runs to
be followed by clean resets without state leakage.
