"""Test fail-closed Cartesian wrist-wrench validation."""

import math

import pytest

from robustlearn.cartesian_wrench_safety import (
    CartesianWrenchSafetyConfig,
    validate_cartesian_wrench,
)


def check(**overrides):
    values = {
        "force_xyz": (0.0, 0.0, 0.0),
        "torque_xyz": (0.0, 0.0, 0.0),
        "received_at_s": 10.0,
        "now_s": 10.1,
    }
    values.update(overrides)
    validate_cartesian_wrench(**values)


def test_valid_wrench_passes():
    check()


def test_nonzero_safe_wrench_passes():
    check(
        force_xyz=(1.0, 2.0, 0.0),
        torque_xyz=(0.1, 0.1, 0.0),
    )


def test_exact_freshness_boundary_passes():
    check(now_s=10.25)


def test_zero_age_passes():
    check(now_s=10.0)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"force_xyz": None}, "force feedback unavailable"),
        ({"torque_xyz": None}, "torque feedback unavailable"),
        ({"received_at_s": None}, "feedback unavailable"),
        ({"now_s": 10.251}, "feedback stale"),
        ({"now_s": 9.999}, "feedback stale"),
        ({"now_s": math.nan}, "current time"),
        ({"received_at_s": math.inf}, "timestamp"),
        ({"force_xyz": (5.0, 0.0, 0.0)}, "force limit"),
        ({"force_xyz": (3.0, 4.0, 0.0)}, "force limit"),
        ({"torque_xyz": (1.0, 0.0, 0.0)}, "torque limit"),
        ({"torque_xyz": (0.6, 0.8, 0.0)}, "torque limit"),
        ({"force_xyz": (math.nan, 0.0, 0.0)}, "non-finite"),
        ({"torque_xyz": (math.inf, 0.0, 0.0)}, "non-finite"),
        ({"force_xyz": (1.0, 2.0)}, "three components"),
        ({"torque_xyz": (1.0, 2.0)}, "three components"),
        (
            {"force_xyz": (1e308, 1e308, 1e308)},
            "force limit",
        ),
    ],
)
def test_unsafe_wrench_rejected(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        check(**overrides)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("feedback_timeout_s", 0.0),
        ("feedback_timeout_s", math.nan),
        ("max_force_norm_n", -1.0),
        ("max_force_norm_n", math.inf),
        ("max_torque_norm_nm", 0.0),
        ("max_torque_norm_nm", math.nan),
    ],
)
def test_invalid_configuration_rejected(field, invalid):
    with pytest.raises(ValueError):
        CartesianWrenchSafetyConfig(**{field: invalid})
