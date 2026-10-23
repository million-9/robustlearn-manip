"""Safe, reproducible storage for scripted expert demonstrations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from robustlearn.episode_manager import EpisodeResult
from robustlearn.week10_benchmark import BenchmarkEpisodeRecord
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    episode_path,
    episode_to_dict,
    frame_from_dict,
    frame_to_dict,
    rollout_episode_from_result,
)


def persist_scripted_expert_episode(
    episode: RolloutEpisode,
    output_root: Path,
) -> Path:
    """Persist an expert episode without overwriting existing data."""

    if episode.source != "expert":
        raise ValueError(
            "scripted collection requires source='expert'"
        )

    if episode.success and not episode.frames:
        raise ValueError(
            "successful expert episodes must contain recorded frames"
        )

    path = episode_path(output_root, episode)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        json.dumps(
            episode_to_dict(episode),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        try:
            # Atomic creation: never replace an existing episode.
            os.link(temporary_path, path)

        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"conflicting demonstration already exists: {path}"
                ) from None

        return path

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class ExpertFrameBuffer:
    """Collect validated expert observation-action frames."""

    def __init__(self) -> None:
        self._frames: list[RolloutFrame] = []
        self._pending_frame: RolloutFrame | None = None

    @property
    def frames(self) -> tuple[RolloutFrame, ...]:
        """Return an immutable snapshot of recorded frames."""
        return tuple(self._frames)

    def record_submitted_command(
        self,
        *,
        timestamp_s: float,
        joint_position: tuple[float, ...],
        joint_velocity: tuple[float, ...] | None,
        task_relative_state: tuple[float, ...],
        wrist_wrench: tuple[float, ...],
        submitted_joint_target: tuple[float, ...],
    ) -> None:
        """Record one observation and submitted expert joint target."""

        if (
            self._frames
            and timestamp_s < self._frames[-1].timestamp_s
        ):
            raise ValueError(
                "frame timestamp must be non-decreasing"
            )

        frame = RolloutFrame(
            frame_index=len(self._frames),
            timestamp_s=timestamp_s,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            task_relative_state=task_relative_state,
            wrist_wrench=wrist_wrench,
            expert_action=submitted_joint_target,
            task_success=False,
            task_failure=False,
            terminated=False,
        )

        # Append only after the complete frame passes validation.
        self._frames.append(frame)

    def stage_submitted_command(
        self,
        *,
        timestamp_s: float,
        joint_position: tuple[float, ...],
        joint_velocity: tuple[float, ...] | None,
        task_relative_state: tuple[float, ...],
        wrist_wrench: tuple[float, ...],
        submitted_joint_target: tuple[float, ...],
    ) -> None:
        """Validate and stage a command awaiting ROS2 acceptance."""

        if self._pending_frame is not None:
            raise ValueError(
                "a pending expert command already exists"
            )

        if self._frames and self._frames[-1].terminated:
            raise ValueError(
                "cannot stage commands after terminal state"
            )

        if (
            self._frames
            and timestamp_s < self._frames[-1].timestamp_s
        ):
            raise ValueError(
                "frame timestamp must be non-decreasing"
            )

        # Constructing a RolloutFrame validates all observation
        # and action dimensions and finite numerical values.
        pending = RolloutFrame(
            frame_index=len(self._frames),
            timestamp_s=timestamp_s,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            task_relative_state=task_relative_state,
            wrist_wrench=wrist_wrench,
            expert_action=submitted_joint_target,
            task_success=False,
            task_failure=False,
            terminated=False,
        )

        self._pending_frame = pending

    def accept_pending_command(self) -> None:
        """Commit the observation-action pair after goal acceptance."""

        pending = self._pending_frame

        if pending is None:
            raise ValueError(
                "no pending expert command to accept"
            )

        if self._frames and self._frames[-1].terminated:
            raise ValueError(
                "cannot accept commands after terminal state"
            )

        self._frames.append(pending)
        self._pending_frame = None

    def reject_pending_command(self) -> None:
        """Discard a submitted command rejected by the controller."""

        if self._pending_frame is None:
            raise ValueError(
                "no pending expert command to reject"
            )

        self._pending_frame = None

    def mark_terminal(self, *, success: bool) -> None:
        """Label the last recorded action with the episode outcome."""
        from dataclasses import replace

        if not isinstance(success, bool):
            raise TypeError("success must be boolean")

        if not self._frames:
            if success:
                raise ValueError(
                    "successful expert capture requires recorded frames"
                )
            return

        if self._frames[-1].terminated:
            raise ValueError(
                "expert capture is already terminal"
            )

        self._frames[-1] = replace(
            self._frames[-1],
            task_success=success,
            task_failure=not success,
            terminated=True,
        )


EXPERT_CAPTURE_SCHEMA_VERSION = 1


def _validate_frame_sequence(
    frames: tuple[RolloutFrame, ...],
) -> None:
    """Validate contiguous indices and ordered timestamps."""
    previous_timestamp: float | None = None

    for index, frame in enumerate(frames):
        if frame.frame_index != index:
            raise ValueError(
                "frame indices must be contiguous and start at zero"
            )

        if (
            previous_timestamp is not None
            and frame.timestamp_s < previous_timestamp
        ):
            raise ValueError(
                "frame timestamps must be non-decreasing"
            )

        previous_timestamp = frame.timestamp_s


def write_expert_frame_capture(
    path: Path,
    frames: tuple[RolloutFrame, ...],
) -> Path:
    """Write validated expert frames without replacing existing data."""
    _validate_frame_sequence(frames)

    payload = (
        json.dumps(
            {
                "capture_schema_version": EXPERT_CAPTURE_SCHEMA_VERSION,
                "frames": [
                    frame_to_dict(frame)
                    for frame in frames
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"conflicting expert capture already exists: {path}"
                ) from None

        return path

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_expert_frame_capture(
    path: Path,
) -> tuple[RolloutFrame, ...]:
    """Read and validate a versioned expert frame capture."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("expert capture must be a JSON object")

    version = data.get("capture_schema_version")

    if type(version) is not int or version != EXPERT_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported expert capture schema version")

    raw_frames = data.get("frames")

    if not isinstance(raw_frames, list):
        raise ValueError("expert capture frames must be a list")

    frames = tuple(
        frame_from_dict(frame)
        for frame in raw_frames
    )

    _validate_frame_sequence(frames)

    return frames



def assemble_scripted_expert_episode(
    result: EpisodeResult,
    *,
    capture_path: Path,
    git_commit: str,
    config_id: str,
    task: str,
) -> RolloutEpisode:
    """Assemble a validated scripted-expert demonstration."""

    if capture_path.is_file():
        frames = read_expert_frame_capture(capture_path)

    elif capture_path.exists():
        raise ValueError(
            f"expert capture path is not a file: {capture_path}"
        )

    elif result.success:
        raise ValueError(
            f"successful expert episode has no capture: {capture_path}"
        )

    else:
        # Preserve failed attempts even when recording never started.
        frames = ()

    if result.success and not frames:
        raise ValueError(
            "successful expert episode requires recorded frames"
        )

    return rollout_episode_from_result(
        result,
        source="expert",
        task=task,
        git_commit=git_commit,
        config_id=config_id,
        frames=frames,
    )



def benchmark_record_to_episode_result(
    record: BenchmarkEpisodeRecord,
    *,
    start_time_s: float,
    end_time_s: float,
) -> EpisodeResult:
    """Convert a benchmark record into a validated episode result.

    Timestamps must come from the collection runner.
    The benchmark completion duration is not an absolute timestamp.
    """
    return EpisodeResult(
        episode_index=record.episode_index,
        seed=record.seed,
        scenario=record.scenario,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        success=record.final_task_success,
        failure_category=record.failure_category,
        failure_reason=record.failure_reason,
    )
