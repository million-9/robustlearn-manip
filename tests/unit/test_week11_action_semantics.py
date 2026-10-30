"""Regression tests for Week 11 scripted command target semantics."""

from robustlearn.scripted_demo_collection import ExpertFrameBuffer


def arguments():
    return {
        "timestamp_s": 1.0,
        "joint_position": (0.0, 0.0, 0.0, -1.5, 0.0, 1.5, -0.8),
        "joint_velocity": (0.0,) * 7,
        "task_relative_state": (0.01, 0.02, 0.03),
        "wrist_wrench": (0.0,) * 6,
        "submitted_joint_target": (
            0.01, 0.02, 0.03, -1.49, 0.04, 1.51, -0.79
        ),
    }


def test_recorded_action_is_absolute_submitted_target():
    buffer = ExpertFrameBuffer()
    values = arguments()

    buffer.record_submitted_command(**values)

    assert len(buffer.frames) == 1
    frame = buffer.frames[0]

    assert frame.joint_position == values["joint_position"]
    assert frame.expert_action == values["submitted_joint_target"]

    delta = tuple(
        target - position
        for target, position in zip(
            values["submitted_joint_target"],
            values["joint_position"],
            strict=True,
        )
    )

    assert frame.expert_action != delta


def test_staged_target_is_recorded_only_after_acceptance():
    buffer = ExpertFrameBuffer()
    values = arguments()

    buffer.stage_submitted_command(**values)

    assert buffer.frames == ()

    buffer.accept_pending_command()

    assert len(buffer.frames) == 1
    assert buffer.frames[0].expert_action == values["submitted_joint_target"]
