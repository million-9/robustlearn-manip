"""Tests for the expert rollout dataset schema."""

import json
import math

import pytest

from robustlearn.rollout_dataset import (
    ROLLOUT_SCHEMA_VERSION,
    RolloutEpisode,
    RolloutFrame,
)


def make_frame(
    *,
    frame_index: int = 0,
    timestamp_s: float = 1.0,
    joint_velocity: tuple[float, ...] | None = None,
    task_success: bool = False,
    task_failure: bool = False,
    terminated: bool = False,
) -> RolloutFrame:
    return RolloutFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        joint_position=(0.0,) * 7,
        joint_velocity=joint_velocity,
        task_relative_state=(
            0.0001,
            -0.005,
            0.005,
        ),
        wrist_wrench=(0.0,) * 6,
        expert_action=(0.001,) * 7,
        task_success=task_success,
        task_failure=task_failure,
        terminated=terminated,
    )


def make_episode(
    *,
    success: bool = True,
    failure_reason: str | None = None,
    frames: tuple[RolloutFrame, ...] | None = None,
) -> RolloutEpisode:
    if frames is None:
        frames = (
            make_frame(
                task_success=success,
                terminated=success,
            ),
        )

    return RolloutEpisode(
        episode_index=3,
        seed=42,
        scenario="week9_case_03",
        source="classical_expert",
        task="panda_insertion",
        success=success,
        failure_reason=failure_reason,
        start_timestamp_s=0.5,
        end_timestamp_s=2.0,
        git_commit="abcdef123456",
        config_id="default",
        frames=frames,
    )


def test_schema_version_is_explicit() -> None:
    episode = make_episode()

    assert episode.schema_version == ROLLOUT_SCHEMA_VERSION
    assert ROLLOUT_SCHEMA_VERSION == 1


def test_frame_accepts_optional_joint_velocity() -> None:
    without_velocity = make_frame()
    with_velocity = make_frame(
        joint_velocity=(0.0,) * 7,
    )

    assert without_velocity.joint_velocity is None
    assert with_velocity.joint_velocity == (0.0,) * 7


@pytest.mark.parametrize(
    ("field", "values", "size"),
    (
        ("joint_position", (0.0,) * 6, 6),
        ("joint_velocity", (0.0,) * 6, 6),
        ("task_relative_state", (0.0,) * 2, 2),
        ("wrist_wrench", (0.0,) * 5, 5),
        ("expert_action", (0.0,) * 6, 6),
    ),
)
def test_frame_rejects_invalid_vector_shapes(
    field: str,
    values: tuple[float, ...],
    size: int,
) -> None:
    kwargs = {
        "frame_index": 0,
        "timestamp_s": 1.0,
        "joint_position": (0.0,) * 7,
        "joint_velocity": None,
        "task_relative_state": (0.0,) * 3,
        "wrist_wrench": (0.0,) * 6,
        "expert_action": (0.0,) * 7,
        "task_success": False,
        "task_failure": False,
        "terminated": False,
    }

    kwargs[field] = values

    with pytest.raises(
        ValueError,
        match=rf"{field} must have shape",
    ):
        RolloutFrame(**kwargs)


@pytest.mark.parametrize(
    "field",
    (
        "joint_position",
        "joint_velocity",
        "task_relative_state",
        "wrist_wrench",
        "expert_action",
    ),
)
def test_frame_rejects_non_finite_vectors(
    field: str,
) -> None:
    kwargs = {
        "frame_index": 0,
        "timestamp_s": 1.0,
        "joint_position": (0.0,) * 7,
        "joint_velocity": (0.0,) * 7,
        "task_relative_state": (0.0,) * 3,
        "wrist_wrench": (0.0,) * 6,
        "expert_action": (0.0,) * 7,
        "task_success": False,
        "task_failure": False,
        "terminated": False,
    }

    values = list(kwargs[field])
    values[0] = math.nan
    kwargs[field] = tuple(values)

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        RolloutFrame(**kwargs)


def test_frame_rejects_non_finite_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="timestamp_s must be finite",
    ):
        make_frame(
            timestamp_s=math.inf,
        )


def test_frame_rejects_conflicting_task_terminal_state() -> None:
    with pytest.raises(
        ValueError,
        match="both task-success and task-failure",
    ):
        make_frame(
            task_success=True,
            task_failure=True,
            terminated=True,
        )


def test_failed_episode_requires_failure_reason() -> None:
    with pytest.raises(
        ValueError,
        match="failed episode must have",
    ):
        make_episode(
            success=False,
            failure_reason=None,
            frames=(),
        )


def test_failed_episode_can_have_no_frames() -> None:
    episode = make_episode(
        success=False,
        failure_reason="launch failure",
        frames=(),
    )

    assert not episode.success
    assert episode.failure_reason == "launch failure"
    assert episode.frames == ()


def test_episode_preserves_explicit_boundaries() -> None:
    frames = (
        make_frame(
            frame_index=0,
            timestamp_s=1.0,
        ),
        make_frame(
            frame_index=1,
            timestamp_s=1.5,
            task_success=True,
            terminated=True,
        ),
    )

    episode = make_episode(
        frames=frames,
    )

    assert tuple(
        frame.frame_index
        for frame in episode.frames
    ) == (0, 1)


def test_episode_rejects_non_contiguous_frame_indices() -> None:
    frames = (
        make_frame(
            frame_index=0,
            timestamp_s=1.0,
        ),
        make_frame(
            frame_index=2,
            timestamp_s=1.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="frame indices must be contiguous",
    ):
        make_episode(
            frames=frames,
        )


def test_episode_rejects_decreasing_frame_timestamps() -> None:
    frames = (
        make_frame(
            frame_index=0,
            timestamp_s=1.5,
        ),
        make_frame(
            frame_index=1,
            timestamp_s=1.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="timestamps must be non-decreasing",
    ):
        make_episode(
            frames=frames,
        )


def test_episode_rejects_frame_outside_episode_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="within episode timestamps",
    ):
        make_episode(
            frames=(
                make_frame(
                    timestamp_s=3.0,
                ),
            ),
        )


def test_episode_dict_round_trip_is_lossless() -> None:
    from robustlearn.rollout_dataset import (
        episode_from_dict,
        episode_to_dict,
    )

    episode = make_episode(
        frames=(
            make_frame(
                frame_index=0,
                timestamp_s=1.0,
                joint_velocity=(0.1,) * 7,
            ),
            make_frame(
                frame_index=1,
                timestamp_s=1.5,
                joint_velocity=(0.2,) * 7,
                task_success=True,
                terminated=True,
            ),
        ),
    )

    raw = episode_to_dict(
        episode,
    )

    restored = episode_from_dict(
        raw,
    )

    assert restored == episode
    assert episode_to_dict(restored) == raw


def test_failed_episode_dict_round_trip_preserves_failure() -> None:
    from robustlearn.rollout_dataset import (
        episode_from_dict,
        episode_to_dict,
    )

    episode = make_episode(
        success=False,
        failure_reason="expert timeout",
        frames=(),
    )

    restored = episode_from_dict(
        episode_to_dict(episode)
    )

    assert not restored.success
    assert restored.failure_reason == "expert timeout"
    assert restored.frames == ()


def test_episode_from_dict_rejects_wrong_schema_version() -> None:
    from robustlearn.rollout_dataset import (
        episode_from_dict,
        episode_to_dict,
    )

    raw = episode_to_dict(
        make_episode()
    )
    raw["schema_version"] = 999

    with pytest.raises(
        ValueError,
        match="unsupported rollout schema version",
    ):
        episode_from_dict(raw)


def test_episode_from_dict_rejects_non_finite_frame_value() -> None:
    from robustlearn.rollout_dataset import (
        episode_from_dict,
        episode_to_dict,
    )

    raw = episode_to_dict(
        make_episode()
    )

    frames = raw["frames"]
    assert isinstance(frames, list)

    frame = frames[0]
    assert isinstance(frame, dict)

    frame["wrist_wrench"] = [
        math.nan,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        episode_from_dict(raw)


def test_episode_to_dict_is_deterministic() -> None:
    from robustlearn.rollout_dataset import episode_to_dict

    first = episode_to_dict(
        make_episode()
    )
    second = episode_to_dict(
        make_episode()
    )

    assert first == second


def test_episode_filename_is_deterministic() -> None:
    from robustlearn.rollout_dataset import episode_filename

    episode = make_episode()

    assert episode_filename(episode) == (
        "episode_000003__seed_42__week9_case_03.json"
    )

    assert (
        episode_filename(episode)
        == episode_filename(episode)
    )


def test_episode_filename_sanitizes_scenario() -> None:
    from dataclasses import replace

    from robustlearn.rollout_dataset import episode_filename

    episode = replace(
        make_episode(),
        scenario="case with spaces/slash",
    )

    assert episode_filename(episode) == (
        "episode_000003__seed_42__case_with_spaces_slash.json"
    )


def test_write_and_read_episode_round_trip(
    tmp_path,
) -> None:
    from robustlearn.rollout_dataset import (
        read_episode,
        write_episode,
    )

    episode = make_episode(
        frames=(
            make_frame(
                frame_index=0,
                timestamp_s=1.0,
                joint_velocity=(0.1,) * 7,
            ),
            make_frame(
                frame_index=1,
                timestamp_s=1.5,
                joint_velocity=(0.2,) * 7,
                task_success=True,
                terminated=True,
            ),
        ),
    )

    path = write_episode(
        tmp_path,
        episode,
    )

    assert path.name == (
        "episode_000003__seed_42__week9_case_03.json"
    )

    restored = read_episode(
        path,
    )

    assert restored == episode


def test_write_episode_output_is_deterministic(
    tmp_path,
) -> None:
    from robustlearn.rollout_dataset import write_episode

    episode = make_episode()

    first_path = write_episode(
        tmp_path / "first",
        episode,
    )
    second_path = write_episode(
        tmp_path / "second",
        episode,
    )

    assert (
        first_path.read_bytes()
        == second_path.read_bytes()
    )


def test_write_failed_episode_preserves_failure(
    tmp_path,
) -> None:
    from robustlearn.rollout_dataset import (
        read_episode,
        write_episode,
    )

    episode = make_episode(
        success=False,
        failure_reason="MoveIt planning failed",
        frames=(),
    )

    path = write_episode(
        tmp_path,
        episode,
    )

    restored = read_episode(
        path,
    )

    assert not restored.success
    assert restored.failure_reason == "MoveIt planning failed"
    assert restored.frames == ()


def test_read_episode_rejects_malformed_json(
    tmp_path,
) -> None:
    from robustlearn.rollout_dataset import read_episode

    path = tmp_path / "bad.json"
    path.write_text(
        "{not-json}",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        read_episode(path)


def test_rollout_episode_from_success_result() -> None:
    from robustlearn.episode_manager import EpisodeResult
    from robustlearn.rollout_dataset import rollout_episode_from_result

    result = EpisodeResult(
        episode_index=7,
        seed=123,
        scenario="week9_case_08",
        start_time_s=10.0,
        end_time_s=12.5,
        success=True,
        failure_category=None,
        failure_reason=None,
    )

    episode = rollout_episode_from_result(
        result,
        source="classical_expert",
        task="panda_insertion",
        git_commit="abc123",
        config_id="default",
        frames=(),
    )

    assert episode.episode_index == 7
    assert episode.seed == 123
    assert episode.scenario == "week9_case_08"
    assert episode.success
    assert episode.failure_reason is None
    assert episode.start_timestamp_s == 10.0
    assert episode.end_timestamp_s == 12.5


def test_rollout_episode_from_failed_result_preserves_reason() -> None:
    from robustlearn.episode_manager import (
        EpisodeFailureCategory,
        EpisodeResult,
    )
    from robustlearn.rollout_dataset import rollout_episode_from_result

    result = EpisodeResult(
        episode_index=2,
        seed=5,
        scenario="week9_case_03",
        start_time_s=1.0,
        end_time_s=2.0,
        success=False,
        failure_category=EpisodeFailureCategory.EXPERT,
        failure_reason="local expert failed",
    )

    episode = rollout_episode_from_result(
        result,
        source="classical_expert",
        task="panda_insertion",
        git_commit="abc123",
        config_id="default",
        frames=(),
    )

    assert not episode.success
    assert episode.failure_reason == "local expert failed"
    assert episode.frames == ()
