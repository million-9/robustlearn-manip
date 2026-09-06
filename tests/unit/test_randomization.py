"""Tests for deterministic Panda insertion randomization sampling."""

import numpy as np

from robustlearn.config import (
    FloatRange,
    PandaInsertionRandomizationConfig,
)
from robustlearn.sim.insertion import load_insertion_model
from robustlearn.sim.randomization import (
    PandaInsertionRandomizationSample,
    PandaInsertionRandomizer,
    sample_panda_insertion_randomization,
)


def enabled_config() -> PandaInsertionRandomizationConfig:
    """Return a non-zero randomization configuration for tests."""
    return PandaInsertionRandomizationConfig(
        enabled=True,
        receptacle_x_offset_m=FloatRange(-0.002, 0.002),
        receptacle_y_offset_m=FloatRange(-0.003, 0.003),
        receptacle_yaw_offset_rad=FloatRange(-0.05, 0.05),
    )


def test_disabled_randomization_returns_exact_zero_sample() -> None:
    rng = np.random.default_rng(2026)

    sample = sample_panda_insertion_randomization(
        rng,
        PandaInsertionRandomizationConfig(),
    )

    assert sample == PandaInsertionRandomizationSample(
        receptacle_x_offset_m=0.0,
        receptacle_y_offset_m=0.0,
        receptacle_yaw_offset_rad=0.0,
    )


def test_disabled_randomization_does_not_consume_rng() -> None:
    randomized_rng = np.random.default_rng(2026)
    reference_rng = np.random.default_rng(2026)

    sample_panda_insertion_randomization(
        randomized_rng,
        PandaInsertionRandomizationConfig(),
    )

    assert randomized_rng.random() == reference_rng.random()


def test_same_seed_and_config_produce_same_sample() -> None:
    config = enabled_config()

    sample_a = sample_panda_insertion_randomization(
        np.random.default_rng(42),
        config,
    )
    sample_b = sample_panda_insertion_randomization(
        np.random.default_rng(42),
        config,
    )

    assert sample_a == sample_b


def test_different_seeds_can_produce_different_samples() -> None:
    config = enabled_config()

    sample_a = sample_panda_insertion_randomization(
        np.random.default_rng(1),
        config,
    )
    sample_b = sample_panda_insertion_randomization(
        np.random.default_rng(2),
        config,
    )

    assert sample_a != sample_b


def test_samples_remain_inside_configured_ranges() -> None:
    config = enabled_config()

    sample = sample_panda_insertion_randomization(
        np.random.default_rng(123),
        config,
    )

    assert (
        config.receptacle_x_offset_m.low
        <= sample.receptacle_x_offset_m
        <= config.receptacle_x_offset_m.high
    )
    assert (
        config.receptacle_y_offset_m.low
        <= sample.receptacle_y_offset_m
        <= config.receptacle_y_offset_m.high
    )
    assert (
        config.receptacle_yaw_offset_rad.low
        <= sample.receptacle_yaw_offset_rad
        <= config.receptacle_yaw_offset_rad.high
    )


def test_enabled_zero_width_ranges_return_exact_bounds() -> None:
    config = PandaInsertionRandomizationConfig(
        enabled=True,
        receptacle_x_offset_m=FloatRange(0.001, 0.001),
        receptacle_y_offset_m=FloatRange(-0.002, -0.002),
        receptacle_yaw_offset_rad=FloatRange(0.03, 0.03),
    )

    sample = sample_panda_insertion_randomization(
        np.random.default_rng(2026),
        config,
    )

    assert sample == PandaInsertionRandomizationSample(
        receptacle_x_offset_m=0.001,
        receptacle_y_offset_m=-0.002,
        receptacle_yaw_offset_rad=0.03,
    )


def test_sample_serialization_is_stable() -> None:
    sample = PandaInsertionRandomizationSample(
        receptacle_x_offset_m=0.001,
        receptacle_y_offset_m=-0.002,
        receptacle_yaw_offset_rad=0.03,
    )

    assert sample.to_dict() == {
        "receptacle_x_offset_m": 0.001,
        "receptacle_y_offset_m": -0.002,
        "receptacle_yaw_offset_rad": 0.03,
    }



def test_randomizer_applies_xy_offsets_without_accumulation() -> None:
    model = load_insertion_model()
    randomizer = PandaInsertionRandomizer(model)

    fixture = model.body("insertion_fixture")
    base_position = np.asarray(
        fixture.pos,
        dtype=np.float64,
    ).copy()

    randomizer.apply(
        PandaInsertionRandomizationSample(
            receptacle_x_offset_m=0.002,
            receptacle_y_offset_m=-0.003,
            receptacle_yaw_offset_rad=0.0,
        )
    )

    np.testing.assert_allclose(
        fixture.pos,
        base_position + np.asarray([0.002, -0.003, 0.0]),
    )

    randomizer.apply(
        PandaInsertionRandomizationSample(
            receptacle_x_offset_m=-0.001,
            receptacle_y_offset_m=0.0015,
            receptacle_yaw_offset_rad=0.0,
        )
    )

    np.testing.assert_allclose(
        fixture.pos,
        base_position + np.asarray([-0.001, 0.0015, 0.0]),
    )


def test_randomizer_applies_fixture_yaw_from_canonical_pose() -> None:
    model = load_insertion_model()
    randomizer = PandaInsertionRandomizer(model)

    fixture = model.body("insertion_fixture")
    base_quaternion = np.asarray(
        fixture.quat,
        dtype=np.float64,
    ).copy()

    yaw = 0.2

    randomizer.apply(
        PandaInsertionRandomizationSample(
            receptacle_x_offset_m=0.0,
            receptacle_y_offset_m=0.0,
            receptacle_yaw_offset_rad=yaw,
        )
    )

    half_yaw = yaw / 2.0
    yaw_quaternion = np.asarray(
        [
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        ],
        dtype=np.float64,
    )

    expected = np.empty(4, dtype=np.float64)

    import mujoco

    mujoco.mju_mulQuat(
        expected,
        yaw_quaternion,
        base_quaternion,
    )

    np.testing.assert_allclose(
        fixture.quat,
        expected,
    )
