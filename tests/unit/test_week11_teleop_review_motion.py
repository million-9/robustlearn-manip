"""Tests for fail-closed teleoperation review readiness."""

import numpy as np
import pytest

from robustlearn.teleop_demo_collection import (
    TeleopReviewMotionGuard,
)

STATIONARY = (0.0,) * 7


def settled_guard():
    guard = TeleopReviewMotionGuard()

    for timestamp in (10.0, 10.15, 10.30):
        guard.observe(
            timestamp_s=timestamp,
            joint_velocities=STATIONARY,
        )

    return guard


def ready(guard, **overrides):
    conditions = {
        "now_s": 10.31,
        "in_review": True,
        "controller_request_pending": False,
        "controller_goal_active": False,
        "cancellation_confirmed": True,
    }
    conditions.update(overrides)
    return guard.ready(**conditions)


def test_sustained_stationary_feedback_is_ready():
    assert ready(settled_guard())


def test_review_mode_is_required():
    assert not ready(
        settled_guard(), in_review=False
    )


def test_pending_controller_request_blocks_approval():
    assert not ready(
        settled_guard(),
        controller_request_pending=True,
    )


def test_active_controller_goal_blocks_approval():
    assert not ready(
        settled_guard(),
        controller_goal_active=True,
    )


def test_unconfirmed_cancellation_blocks_approval():
    assert not ready(
        settled_guard(),
        cancellation_confirmed=False,
    )


def test_stale_feedback_blocks_approval():
    assert not ready(
        settled_guard(), now_s=10.60
    )


def test_single_stationary_sample_is_insufficient():
    guard = TeleopReviewMotionGuard()

    guard.observe(
        timestamp_s=10.30,
        joint_velocities=STATIONARY,
    )

    assert not ready(guard)


def test_moving_joint_resets_stationary_window():
    guard = settled_guard()

    guard.observe(
        timestamp_s=10.35,
        joint_velocities=(
            0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ),
    )

    guard.observe(
        timestamp_s=10.40,
        joint_velocities=STATIONARY,
    )

    assert not ready(
        guard, now_s=10.41
    )


def test_feedback_gap_resets_stationary_window():
    guard = TeleopReviewMotionGuard()

    guard.observe(
        timestamp_s=10.0,
        joint_velocities=STATIONARY,
    )

    guard.observe(
        timestamp_s=10.50,
        joint_velocities=STATIONARY,
    )

    assert not ready(
        guard, now_s=10.51
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        (0.0,) * 6,
        (np.nan,) * 7,
        (np.inf,) * 7,
    ],
)
def test_missing_or_invalid_feedback_fails_closed(invalid):
    guard = settled_guard()

    guard.observe(
        timestamp_s=10.35,
        joint_velocities=invalid,
    )

    assert not ready(
        guard, now_s=10.36
    )


def test_reset_invalidates_existing_stationary_evidence():
    guard = settled_guard()

    guard.reset()

    assert not ready(guard)


def test_non_monotonic_feedback_invalidates_evidence():
    guard = settled_guard()

    with pytest.raises(ValueError):
        guard.observe(
            timestamp_s=10.10,
            joint_velocities=STATIONARY,
        )

    assert not ready(guard)


def test_future_feedback_is_not_accepted_as_current():
    guard = settled_guard()

    assert not ready(
        guard, now_s=10.20
    )
