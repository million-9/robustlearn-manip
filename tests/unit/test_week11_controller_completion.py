"""Verify fail-closed ROS2 controller completion accounting."""

import pytest

from robustlearn.teleop_demo_collection import (
    TeleopControllerCompletionTracker,
)


def test_new_tracker_has_no_outstanding_requests():
    tracker = TeleopControllerCompletionTracker()

    assert tracker.settled
    assert tracker.outstanding_request_ids == ()


def test_submitted_goal_blocks_readiness():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)


def test_rejected_goal_resolves_without_terminal_result():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=False)

    assert tracker.settled


def test_accepted_goal_requires_terminal_result():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)

    assert not tracker.settled

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_cancellation_acknowledgment_is_not_completion():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)
    tracker.cancellation_requested(1)
    tracker.cancellation_response(
        1, acknowledged=True
    )

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_rejected_cancellation_keeps_goal_outstanding():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)
    tracker.cancellation_requested(1)
    tracker.cancellation_response(
        1, acknowledged=False
    )

    assert not tracker.settled
    assert tracker.cancellation_rejected_ids == (1,)

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_cancel_before_goal_response_keeps_request_tracked():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.cancellation_requested(1)

    assert not tracker.settled

    tracker.goal_response(1, accepted=True)

    assert not tracker.settled

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_late_rejected_goal_finishes_pending_request():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.cancellation_requested(1)
    tracker.goal_response(1, accepted=False)

    assert tracker.settled


def test_terminal_result_can_precede_cancel_reply():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)
    tracker.cancellation_requested(1)
    tracker.terminal_result(1, confirmed=True)

    tracker.cancellation_response(
        1, acknowledged=True
    )

    assert tracker.settled


def test_unverified_result_permanently_blocks_review():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)
    tracker.terminal_result(1, confirmed=False)

    assert tracker.uncertain
    assert not tracker.settled
    assert tracker.outstanding_request_ids == ()


def test_multiple_overlapping_requests_are_tracked():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.submitted(2)

    tracker.cancellation_requested(1)
    tracker.goal_response(1, accepted=True)
    tracker.goal_response(2, accepted=False)

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)

    tracker.terminal_result(1, confirmed=True)

    assert tracker.settled


def test_unknown_stale_callback_cannot_complete_request():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)

    with pytest.raises(ValueError):
        tracker.goal_response(999, accepted=True)

    assert not tracker.settled
    assert tracker.outstanding_request_ids == (1,)


def test_request_ids_cannot_be_reused():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=False)

    with pytest.raises(ValueError):
        tracker.submitted(1)


def test_terminal_result_requires_accepted_goal():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)

    with pytest.raises(ValueError):
        tracker.terminal_result(1, confirmed=True)

    assert not tracker.settled


def test_explicit_uncertainty_blocks_readiness():
    tracker = TeleopControllerCompletionTracker()

    tracker.invalidate()

    assert tracker.uncertain
    assert not tracker.settled


@pytest.mark.parametrize("invalid_id", [0, -1, True, "1"])
def test_invalid_request_ids_are_rejected(invalid_id):
    tracker = TeleopControllerCompletionTracker()

    with pytest.raises(ValueError):
        tracker.submitted(invalid_id)


def test_invalid_cancellation_reply_is_rejected():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)

    with pytest.raises(ValueError):
        tracker.cancellation_response(
            1, acknowledged=True
        )


def test_non_boolean_terminal_confirmation_is_rejected():
    tracker = TeleopControllerCompletionTracker()

    tracker.submitted(1)
    tracker.goal_response(1, accepted=True)

    with pytest.raises(TypeError):
        tracker.terminal_result(1, confirmed=1)

    assert not tracker.settled
