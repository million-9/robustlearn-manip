"""Versioned structured records for expert rollout datasets.

Week 10 defines the stable writer/schema boundary only. Dataset v0.1
collection, splitting, and statistics belong to later milestones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from robustlearn.episode_manager import EpisodeResult

ROLLOUT_SCHEMA_VERSION = 1
PANDA_JOINT_COUNT = 7
WRIST_WRENCH_SIZE = 6
TASK_RELATIVE_STATE_SIZE = 3


def _validate_finite_vector(
    name: str,
    values: tuple[float, ...],
    *,
    size: int,
) -> None:
    if len(values) != size:
        raise ValueError(
            f"{name} must have shape ({size},), got ({len(values)},)"
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"{name} must contain only finite values"
        )


@dataclass(frozen=True)
class RolloutFrame:
    """One time-aligned expert rollout frame."""

    frame_index: int
    timestamp_s: float
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...] | None
    task_relative_state: tuple[float, ...]
    wrist_wrench: tuple[float, ...]
    expert_action: tuple[float, ...]
    task_success: bool
    task_failure: bool
    terminated: bool

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(
                "frame_index must be non-negative"
            )

        if not math.isfinite(self.timestamp_s):
            raise ValueError(
                "timestamp_s must be finite"
            )

        _validate_finite_vector(
            "joint_position",
            self.joint_position,
            size=PANDA_JOINT_COUNT,
        )

        if self.joint_velocity is not None:
            _validate_finite_vector(
                "joint_velocity",
                self.joint_velocity,
                size=PANDA_JOINT_COUNT,
            )

        _validate_finite_vector(
            "task_relative_state",
            self.task_relative_state,
            size=TASK_RELATIVE_STATE_SIZE,
        )

        _validate_finite_vector(
            "wrist_wrench",
            self.wrist_wrench,
            size=WRIST_WRENCH_SIZE,
        )

        _validate_finite_vector(
            "expert_action",
            self.expert_action,
            size=PANDA_JOINT_COUNT,
        )

        if self.task_success and self.task_failure:
            raise ValueError(
                "frame cannot be both task-success "
                "and task-failure"
            )

        if (
            self.task_success or self.task_failure
        ) and not self.terminated:
            raise ValueError(
                "terminal task state requires terminated=True"
            )


@dataclass(frozen=True)
class RolloutEpisode:
    """One complete or failed expert rollout."""

    episode_index: int
    seed: int
    scenario: str
    source: str
    task: str
    success: bool
    failure_reason: str | None
    start_timestamp_s: float
    end_timestamp_s: float
    git_commit: str
    config_id: str
    frames: tuple[RolloutFrame, ...]
    schema_version: int = ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROLLOUT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported rollout schema version: "
                f"{self.schema_version}"
            )

        if self.episode_index < 0:
            raise ValueError(
                "episode_index must be non-negative"
            )

        for name, value in (
            ("scenario", self.scenario),
            ("source", self.source),
            ("task", self.task),
            ("git_commit", self.git_commit),
            ("config_id", self.config_id),
        ):
            if not value.strip():
                raise ValueError(
                    f"{name} must be non-empty"
                )

        if not math.isfinite(self.start_timestamp_s):
            raise ValueError(
                "start_timestamp_s must be finite"
            )

        if not math.isfinite(self.end_timestamp_s):
            raise ValueError(
                "end_timestamp_s must be finite"
            )

        if self.end_timestamp_s < self.start_timestamp_s:
            raise ValueError(
                "end_timestamp_s must not precede "
                "start_timestamp_s"
            )

        if self.success:
            if self.failure_reason is not None:
                raise ValueError(
                    "successful episode must not have "
                    "a failure_reason"
                )
        elif (
            self.failure_reason is None
            or not self.failure_reason.strip()
        ):
            raise ValueError(
                "failed episode must have a non-empty "
                "failure_reason"
            )

        expected_indices = tuple(
            range(len(self.frames))
        )

        actual_indices = tuple(
            frame.frame_index
            for frame in self.frames
        )

        if actual_indices != expected_indices:
            raise ValueError(
                "frame indices must be contiguous and "
                "start at zero"
            )

        previous_timestamp: float | None = None

        for frame in self.frames:
            if (
                frame.timestamp_s
                < self.start_timestamp_s
                or frame.timestamp_s
                > self.end_timestamp_s
            ):
                raise ValueError(
                    "frame timestamp must lie within "
                    "episode timestamps"
                )

            if (
                previous_timestamp is not None
                and frame.timestamp_s
                < previous_timestamp
            ):
                raise ValueError(
                    "frame timestamps must be "
                    "non-decreasing"
                )

            previous_timestamp = frame.timestamp_s


def frame_to_dict(
    frame: RolloutFrame,
) -> dict[str, object]:
    """Serialize one rollout frame to JSON-compatible data."""
    return {
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "joint_position": list(frame.joint_position),
        "joint_velocity": (
            None
            if frame.joint_velocity is None
            else list(frame.joint_velocity)
        ),
        "task_relative_state": list(frame.task_relative_state),
        "wrist_wrench": list(frame.wrist_wrench),
        "expert_action": list(frame.expert_action),
        "task_success": frame.task_success,
        "task_failure": frame.task_failure,
        "terminated": frame.terminated,
    }


def episode_to_dict(
    episode: RolloutEpisode,
) -> dict[str, object]:
    """Serialize one rollout episode to JSON-compatible data."""
    return {
        "schema_version": episode.schema_version,
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "scenario": episode.scenario,
        "source": episode.source,
        "task": episode.task,
        "success": episode.success,
        "failure_reason": episode.failure_reason,
        "start_timestamp_s": episode.start_timestamp_s,
        "end_timestamp_s": episode.end_timestamp_s,
        "git_commit": episode.git_commit,
        "config_id": episode.config_id,
        "frames": [
            frame_to_dict(frame)
            for frame in episode.frames
        ],
    }


def _require_dict(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be an object"
        )

    return value


def _require_int(
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer"
        )

    return value


def _require_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite"
        )

    return result


def _require_bool(
    value: object,
    *,
    name: str,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be a boolean"
        )

    return value


def _require_string(
    value: object,
    *,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string"
        )

    return value


def _require_optional_string(
    value: object,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _require_string(
        value,
        name=name,
    )


def _require_float_tuple(
    value: object,
    *,
    name: str,
    size: int,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"{name} must be an array"
        )

    result = tuple(
        _require_float(
            element,
            name=f"{name}[{index}]",
        )
        for index, element in enumerate(value)
    )

    _validate_finite_vector(
        name,
        result,
        size=size,
    )

    return result


def frame_from_dict(
    raw: object,
) -> RolloutFrame:
    """Deserialize and validate one rollout frame."""
    data = _require_dict(
        raw,
        name="frame",
    )

    raw_joint_velocity = data.get(
        "joint_velocity"
    )

    joint_velocity = (
        None
        if raw_joint_velocity is None
        else _require_float_tuple(
            raw_joint_velocity,
            name="joint_velocity",
            size=PANDA_JOINT_COUNT,
        )
    )

    return RolloutFrame(
        frame_index=_require_int(
            data.get("frame_index"),
            name="frame_index",
        ),
        timestamp_s=_require_float(
            data.get("timestamp_s"),
            name="timestamp_s",
        ),
        joint_position=_require_float_tuple(
            data.get("joint_position"),
            name="joint_position",
            size=PANDA_JOINT_COUNT,
        ),
        joint_velocity=joint_velocity,
        task_relative_state=_require_float_tuple(
            data.get("task_relative_state"),
            name="task_relative_state",
            size=TASK_RELATIVE_STATE_SIZE,
        ),
        wrist_wrench=_require_float_tuple(
            data.get("wrist_wrench"),
            name="wrist_wrench",
            size=WRIST_WRENCH_SIZE,
        ),
        expert_action=_require_float_tuple(
            data.get("expert_action"),
            name="expert_action",
            size=PANDA_JOINT_COUNT,
        ),
        task_success=_require_bool(
            data.get("task_success"),
            name="task_success",
        ),
        task_failure=_require_bool(
            data.get("task_failure"),
            name="task_failure",
        ),
        terminated=_require_bool(
            data.get("terminated"),
            name="terminated",
        ),
    )


def episode_from_dict(
    raw: object,
) -> RolloutEpisode:
    """Deserialize and validate one rollout episode."""
    data = _require_dict(
        raw,
        name="episode",
    )

    raw_frames = data.get("frames")

    if not isinstance(raw_frames, list):
        raise ValueError(
            "frames must be an array"
        )

    return RolloutEpisode(
        schema_version=_require_int(
            data.get("schema_version"),
            name="schema_version",
        ),
        episode_index=_require_int(
            data.get("episode_index"),
            name="episode_index",
        ),
        seed=_require_int(
            data.get("seed"),
            name="seed",
        ),
        scenario=_require_string(
            data.get("scenario"),
            name="scenario",
        ),
        source=_require_string(
            data.get("source"),
            name="source",
        ),
        task=_require_string(
            data.get("task"),
            name="task",
        ),
        success=_require_bool(
            data.get("success"),
            name="success",
        ),
        failure_reason=_require_optional_string(
            data.get("failure_reason"),
            name="failure_reason",
        ),
        start_timestamp_s=_require_float(
            data.get("start_timestamp_s"),
            name="start_timestamp_s",
        ),
        end_timestamp_s=_require_float(
            data.get("end_timestamp_s"),
            name="end_timestamp_s",
        ),
        git_commit=_require_string(
            data.get("git_commit"),
            name="git_commit",
        ),
        config_id=_require_string(
            data.get("config_id"),
            name="config_id",
        ),
        frames=tuple(
            frame_from_dict(frame)
            for frame in raw_frames
        ),
    )


def episode_filename(
    episode: RolloutEpisode,
) -> str:
    """Return the deterministic filename for one rollout episode."""
    scenario = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in episode.scenario.strip()
    ).strip("._")

    if not scenario:
        raise ValueError(
            "scenario must contain a usable filename character"
        )

    return (
        f"episode_{episode.episode_index:06d}"
        f"__seed_{episode.seed}"
        f"__{scenario}.json"
    )


def episode_path(
    output_root: Path,
    episode: RolloutEpisode,
) -> Path:
    """Return the deterministic path for one rollout episode."""
    return output_root / episode_filename(episode)


def write_episode(
    output_root: Path,
    episode: RolloutEpisode,
) -> Path:
    """Write one validated rollout episode deterministically as JSON."""
    import json

    path = episode_path(
        output_root,
        episode,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        episode_to_dict(episode),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    path.write_text(
        payload + "\n",
        encoding="utf-8",
    )

    return path


def read_episode(
    path: Path,
) -> RolloutEpisode:
    """Read and validate one rollout episode JSON file."""
    import json

    raw = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    return episode_from_dict(raw)



def rollout_episode_from_result(
    result: EpisodeResult,
    *,
    source: str,
    task: str,
    git_commit: str,
    config_id: str,
    frames: tuple[RolloutFrame, ...] = (),
) -> RolloutEpisode:
    """Build one rollout episode from an episode-manager result."""
    failure_reason = (
        None
        if result.success
        else result.failure_reason
    )

    return RolloutEpisode(
        episode_index=result.episode_index,
        seed=result.seed,
        scenario=result.scenario,
        source=source,
        task=task,
        success=result.success,
        failure_reason=failure_reason,
        start_timestamp_s=result.start_time_s,
        end_timestamp_s=result.end_time_s,
        git_commit=git_commit,
        config_id=config_id,
        frames=frames,
    )
