"""Verify safe keyboard mapping for teleoperation review."""

import pytest

from robustlearn.teleop import KEYBOARD_JOINT_BINDINGS
from robustlearn.teleop_demo_collection import (
    teleop_review_key_action,
)


def test_review_entry_requires_explicit_key():
    assert teleop_review_key_action(
        "v", in_review=False
    ) == "enter_review"

    assert teleop_review_key_action(
        "1", in_review=False
    ) is None

    assert teleop_review_key_action(
        "2", in_review=False
    ) is None

    assert teleop_review_key_action(
        "3", in_review=False
    ) is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("0", "resume"),
        ("1", "accept_success"),
        ("2", "accept_failure"),
        ("3", "reject"),
    ],
)
def test_review_decision_keys(key, expected):
    assert teleop_review_key_action(
        key, in_review=True
    ) == expected


def test_review_keys_do_not_overlap_joint_jog_keys():
    review_keys = {"v", "0", "1", "2", "3"}

    assert review_keys.isdisjoint(
        KEYBOARD_JOINT_BINDINGS
    )


def test_jog_keys_cannot_trigger_review_decisions():
    for key in KEYBOARD_JOINT_BINDINGS:
        assert teleop_review_key_action(
            key, in_review=True
        ) is None


def test_space_and_stop_remain_separate():
    assert teleop_review_key_action(
        " ", in_review=True
    ) is None

    assert teleop_review_key_action(
        "x", in_review=True
    ) is None


def test_escape_remains_global_abort_control():
    assert teleop_review_key_action(
        "\x1b", in_review=True
    ) is None


def test_review_entry_key_does_not_finalize_episode():
    assert teleop_review_key_action(
        "v", in_review=True
    ) is None


@pytest.mark.parametrize(
    "invalid",
    ["", "accept", "12", None],
)
def test_invalid_input_is_rejected(invalid):
    with pytest.raises(ValueError):
        teleop_review_key_action(
            invalid, in_review=True
        )


def test_review_state_requires_boolean():
    with pytest.raises(TypeError):
        teleop_review_key_action(
            "1", in_review=1
        )
