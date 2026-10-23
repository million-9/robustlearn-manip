"""Safe recording lifecycle for manual teleoperation demonstrations."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robustlearn.episode_manager import EpisodeFailureCategory, EpisodeResult
from robustlearn.rollout_dataset import (
    RolloutFrame,
    episode_path,
    episode_to_dict,
    rollout_episode_from_result,
)
from robustlearn.sim.shadow_kinematics import ShadowInsertionKinematics


class TeleopRecordingSession:
    """Manage operator acceptance of recorded teleoperation frames."""

    def __init__(self) -> None:
        self._status = "recording"
        self._frames: list[RolloutFrame] = []
        self._pending_frame: RolloutFrame | None = None

    @property
    def status(self) -> str:
        """Return the current recording lifecycle state."""
        return self._status

    @property
    def frames(self) -> tuple[RolloutFrame, ...]:
        """Return an immutable snapshot of recorded frames."""
        return tuple(self._frames)

    @property
    def has_pending_goal(self) -> bool:
        """Whether a controller response is still outstanding."""
        return self._pending_frame is not None

    def _require_recording(self) -> None:
        if self._status != "recording":
            raise ValueError(
                "teleoperation recording is already finalized"
            )

    def _validate_frame(self, frame: RolloutFrame) -> None:
        """Validate a frame before staging or recording."""
        if not isinstance(frame, RolloutFrame):
            raise TypeError("frame must be a RolloutFrame")

        if frame.frame_index != len(self._frames):
            raise ValueError(
                "teleop frame indices must be contiguous"
            )

        if frame.terminated:
            raise ValueError(
                "terminal state must be assigned by the session"
            )

        if self._frames:
            if frame.timestamp_s < self._frames[-1].timestamp_s:
                raise ValueError(
                    "teleop timestamps must be non-decreasing"
                )

    def record_frame(self, frame: RolloutFrame) -> None:
        """Append a frame only when no controller goal is pending."""
        self._require_recording()

        if self._pending_frame is not None:
            raise ValueError(
                "cannot record while a controller goal is pending"
            )

        self._validate_frame(frame)
        self._frames.append(frame)

    def stage_frame(self, frame: RolloutFrame) -> None:
        """Stage an observation before controller acceptance."""
        self._require_recording()

        if self._pending_frame is not None:
            raise ValueError(
                "a teleop controller goal is already pending"
            )

        self._validate_frame(frame)
        self._pending_frame = frame

    def accept_pending_frame(self) -> None:
        """Record a staged frame after controller goal acceptance."""
        self._require_recording()

        if self._pending_frame is None:
            raise ValueError(
                "no pending teleop frame to accept"
            )

        self._frames.append(self._pending_frame)
        self._pending_frame = None

    def reject_pending_frame(self) -> None:
        """Discard a frame when controller submission is rejected."""
        self._require_recording()

        if self._pending_frame is None:
            raise ValueError(
                "no pending teleop frame to reject"
            )

        self._pending_frame = None

    def accept(self, *, success: bool) -> None:
        """Approve the recording and assign its terminal task label."""
        self._require_recording()

        if self._pending_frame is not None:
            raise ValueError(
                "cannot accept a recording with a pending goal"
            )

        if not isinstance(success, bool):
            raise TypeError("success must be boolean")

        if not self._frames:
            raise ValueError(
                "cannot accept an empty teleoperation recording"
            )

        self._frames[-1] = replace(
            self._frames[-1],
            task_success=success,
            task_failure=not success,
            terminated=True,
        )

        self._status = "accepted"

    def reject(self) -> None:
        """Reject this recording without exporting any frames."""
        self._require_recording()
        self._pending_frame = None
        self._status = "rejected"

    def abort(self) -> None:
        """Abort this recording without exporting any frames."""
        self._require_recording()
        self._pending_frame = None
        self._status = "aborted"

    def export_accepted_frames(self) -> tuple[RolloutFrame, ...]:
        """Expose frames only after explicit operator acceptance."""
        if self._status != "accepted":
            raise ValueError(
                "only accepted teleoperation recordings can be exported"
            )

        return tuple(self._frames)


def persist_accepted_teleop_episode(
    session: TeleopRecordingSession,
    result: EpisodeResult,
    *,
    output_root: Path,
    git_commit: str,
    config_id: str,
    task: str,
) -> Path:
    """Persist only explicitly accepted manual demonstrations.

    Uses the canonical rollout schema and never replaces
    an existing episode file.
    """
    if session.status != "accepted":
        raise ValueError(
            "teleoperation recording must be accepted before persistence"
        )

    frames = session.export_accepted_frames()

    if not frames:
        raise ValueError(
            "accepted teleoperation recording cannot be empty"
        )

    final_frame = frames[-1]

    if not final_frame.terminated:
        raise ValueError(
            "accepted recording must have a terminal frame"
        )

    if (
        final_frame.task_success != result.success
        or final_frame.task_failure != (not result.success)
    ):
        raise ValueError(
            "operator outcome does not match episode result"
        )

    episode = rollout_episode_from_result(
        result,
        source="teleop",
        task=task,
        git_commit=git_commit,
        config_id=config_id,
        frames=frames,
    )

    path = episode_path(output_root, episode)

    payload = (
        json.dumps(
            episode_to_dict(episode),
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

        # Atomic creation: never replace an existing episode.
        os.link(temporary_path, path)

        return path

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required_observation_vector(
    name: str,
    values: tuple[float, ...] | None,
    *,
    size: int,
) -> np.ndarray:
    """Reject missing, incorrectly shaped, or non-finite feedback."""
    if values is None:
        raise ValueError(f"{name} feedback is unavailable")

    vector = np.asarray(values, dtype=np.float64)

    if vector.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},)"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain finite values"
        )

    return vector


def build_teleop_observation_frame(
    *,
    kinematics: ShadowInsertionKinematics,
    frame_index: int,
    timestamp_s: float,
    joint_position: tuple[float, ...] | None,
    joint_velocity: tuple[float, ...] | None,
    wrist_force_sensor: tuple[float, ...] | None,
    wrist_torque_sensor: tuple[float, ...] | None,
    submitted_joint_target: tuple[float, ...],
) -> RolloutFrame:
    """Construct a validated teleoperation observation-action frame.

    Joint positions and wrist F/T must come from actual ROS feedback.
    Task geometry is reconstructed using the shared MuJoCo model.

    The caller must record this frame only after the ROS2 trajectory
    goal has been accepted by the controller.
    """
    positions = _required_observation_vector(
        "joint_position", joint_position, size=7
    )

    velocities = (
        None
        if joint_velocity is None
        else _required_observation_vector(
            "joint_velocity", joint_velocity, size=7
        )
    )

    force = _required_observation_vector(
        "wrist_force_sensor", wrist_force_sensor, size=3
    )

    torque = _required_observation_vector(
        "wrist_torque_sensor", wrist_torque_sensor, size=3
    )

    target = _required_observation_vector(
        "submitted_joint_target",
        submitted_joint_target,
        size=7,
    )

    snapshot = kinematics.update(positions)
    rotation_world = kinematics.wrist_rotation_world()

    relative_position = (
        snapshot.receptacle_center_position
        - snapshot.peg_tip_position
    )

    world_wrench = np.concatenate(
        (
            rotation_world @ force,
            rotation_world @ torque,
        )
    )

    return RolloutFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        joint_position=tuple(float(x) for x in positions),
        joint_velocity=(
            tuple(float(x) for x in velocities)
            if velocities is not None
            else None
        ),
        task_relative_state=tuple(
            float(x) for x in relative_position
        ),
        wrist_wrench=tuple(float(x) for x in world_wrench),
        expert_action=tuple(float(x) for x in target),
        task_success=False,
        task_failure=False,
        terminated=False,
    )


def create_teleop_recording_components(
    enabled: bool,
) -> tuple[
    TeleopRecordingSession | None,
    ShadowInsertionKinematics | None,
]:
    """Create recording infrastructure only when explicitly enabled."""
    if type(enabled) is not bool:
        raise TypeError("recording enabled flag must be boolean")

    if not enabled:
        return None, None

    session = TeleopRecordingSession()
    kinematics = ShadowInsertionKinematics()

    return session, kinematics


def apply_teleop_operator_decision(
    session: TeleopRecordingSession,
    decision: str,
    *,
    result: EpisodeResult | None = None,
    output_root: Path | None = None,
    git_commit: str | None = None,
    config_id: str | None = None,
    task: str | None = None,
) -> Path | None:
    """Apply an explicit operator decision to one demonstration.

    Rejected and aborted recordings never reach the dataset writer.
    Accepted recordings require validated episode metadata.

    An already accepted session may retry persistence after an I/O
    failure without requiring another acceptance transition.
    """
    if decision == "reject":
        session.reject()
        return None

    if decision == "abort":
        session.abort()
        return None

    if decision not in ("accept_success", "accept_failure"):
        raise ValueError(
            f"unsupported teleop operator decision: {decision!r}"
        )

    if not isinstance(result, EpisodeResult):
        raise ValueError(
            "acceptance requires a validated EpisodeResult"
        )

    if not isinstance(output_root, Path):
        raise ValueError(
            "acceptance requires a dataset output Path"
        )

    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("acceptance requires non-empty git_commit")

    if not isinstance(config_id, str) or not config_id.strip():
        raise ValueError("acceptance requires non-empty config_id")

    if not isinstance(task, str) or not task.strip():
        raise ValueError("acceptance requires non-empty task")

    expected_success = decision == "accept_success"

    if result.success != expected_success:
        raise ValueError(
            "operator decision conflicts with episode outcome"
        )

    if session.status == "recording":
        # Preflight validation must happen before the recording
        # is finalized. The final atomic write remains exclusive.
        if session.has_pending_goal:
            raise ValueError(
                "cannot approve while a controller goal is pending"
            )

        frames = session.frames

        if not frames:
            raise ValueError(
                "cannot approve an empty teleop recording"
            )

        final_frame = replace(
            frames[-1],
            task_success=expected_success,
            task_failure=not expected_success,
            terminated=True,
        )

        candidate = rollout_episode_from_result(
            result,
            source="teleop",
            task=task,
            git_commit=git_commit,
            config_id=config_id,
            frames=(*frames[:-1], final_frame),
        )

        candidate_path = episode_path(output_root, candidate)

        if candidate_path.exists():
            raise FileExistsError(
                f"teleop episode already exists: {candidate_path}"
            )

        session.accept(success=expected_success)
    elif session.status != "accepted":
        raise ValueError(
            "rejected or aborted recordings cannot be persisted"
        )

    return persist_accepted_teleop_episode(
        session,
        result,
        output_root=output_root,
        git_commit=git_commit,
        config_id=config_id,
        task=task,
    )


def teleop_review_key_action(
    key: str,
    *,
    in_review: bool,
) -> str | None:
    """Interpret reserved demonstration-review keys.

    This function never sends robot commands or persists data.
    Approval keys are inactive outside explicit review mode.
    """
    if not isinstance(key, str) or len(key) != 1:
        raise ValueError(
            "review input must be a single character"
        )

    if not isinstance(in_review, bool):
        raise TypeError(
            "in_review must be boolean"
        )

    if not in_review:
        return "enter_review" if key == "v" else None

    actions = {
        "0": "resume",
        "1": "accept_success",
        "2": "accept_failure",
        "3": "reject",
    }

    return actions.get(key)


class TeleopReviewMotionGuard:
    """Require sustained fresh stationary feedback before review approval."""

    def __init__(
        self,
        *,
        max_feedback_age_s: float = 0.25,
        min_stationary_s: float = 0.25,
        speed_limit_rad_s: float = 0.01,
    ) -> None:
        for name, value in (
            ("max_feedback_age_s", max_feedback_age_s),
            ("min_stationary_s", min_stationary_s),
            ("speed_limit_rad_s", speed_limit_rad_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        self._max_feedback_age_s = max_feedback_age_s
        self._min_stationary_s = min_stationary_s
        self._speed_limit_rad_s = speed_limit_rad_s
        self.reset()

    def reset(self) -> None:
        """Invalidate previously observed stationary evidence."""
        self._last_feedback_s: float | None = None
        self._stationary_since_s: float | None = None

    def observe(
        self,
        *,
        timestamp_s: float,
        joint_velocities: tuple[float, ...] | None,
    ) -> None:
        """Consume real joint feedback using a monotonic clock."""
        if not np.isfinite(timestamp_s):
            self.reset()
            raise ValueError("feedback timestamp must be finite")

        previous = self._last_feedback_s

        if previous is not None and timestamp_s < previous:
            self.reset()
            raise ValueError(
                "joint feedback timestamps must be non-decreasing"
            )

        if joint_velocities is None:
            self.reset()
            return

        try:
            velocities = np.asarray(
                joint_velocities, dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return

        if (
            velocities.shape != (7,)
            or not np.all(np.isfinite(velocities))
        ):
            self.reset()
            return

        feedback_gap = (
            previous is None
            or timestamp_s - previous > self._max_feedback_age_s
        )

        moving = bool(
            np.any(
                np.abs(velocities) > self._speed_limit_rad_s
            )
        )

        if moving:
            self._stationary_since_s = None
        elif feedback_gap or self._stationary_since_s is None:
            self._stationary_since_s = timestamp_s

        self._last_feedback_s = timestamp_s

    def ready(
        self,
        *,
        now_s: float,
        in_review: bool,
        controller_request_pending: bool,
        controller_goal_active: bool,
        cancellation_confirmed: bool,
    ) -> bool:
        """Fail closed unless review and motion conditions are satisfied."""
        if (
            not in_review
            or controller_request_pending
            or controller_goal_active
            or not cancellation_confirmed
        ):
            return False

        last = self._last_feedback_s
        stationary_since = self._stationary_since_s

        if last is None or stationary_since is None:
            return False

        if not np.isfinite(now_s):
            return False

        feedback_age_s = now_s - last

        if (
            feedback_age_s < 0.0
            or feedback_age_s > self._max_feedback_age_s
        ):
            return False

        return (
            now_s - stationary_since
            >= self._min_stationary_s
        )

class TeleopControllerCompletionTracker:
    """Track controller requests until their outcome is definitive.

    Cancellation acknowledgment alone never completes a goal.
    An accepted goal requires a confirmed terminal action result.

    An uncertain result permanently blocks readiness for this
    tracker instance. The operator must abort that collection.
    """

    def __init__(self) -> None:
        self._outstanding: dict[int, str] = {}
        self._used_ids: set[int] = set()
        self._cancel_requests: set[int] = set()
        self._cancel_responses: dict[int, bool] = {}
        self._uncertain = False

    @property
    def outstanding_request_ids(self) -> tuple[int, ...]:
        """Return requests not yet rejected or confirmed terminal."""
        return tuple(sorted(self._outstanding))

    @property
    def settled(self) -> bool:
        """Whether all tracked goals have definitive outcomes."""
        return not self._outstanding and not self._uncertain

    @property
    def uncertain(self) -> bool:
        """Whether controller completion could not be verified."""
        return self._uncertain

    @property
    def cancellation_rejected_ids(self) -> tuple[int, ...]:
        """Report rejected cancellations still awaiting a result."""
        return tuple(
            sorted(
                request_id
                for request_id, accepted
                in self._cancel_responses.items()
                if (
                    not accepted
                    and request_id in self._outstanding
                )
            )
        )

    def submitted(self, request_id: int) -> None:
        """Register a request before asynchronous goal submission."""
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id <= 0
        ):
            raise ValueError(
                "controller request ID must be a positive integer"
            )

        if request_id in self._used_ids:
            raise ValueError(
                "controller request ID cannot be reused"
            )

        self._used_ids.add(request_id)
        self._outstanding[request_id] = "submitted"

    def goal_response(
        self,
        request_id: int,
        *,
        accepted: bool,
    ) -> None:
        """Resolve goal submission without losing accepted goals."""
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be boolean")

        if self._outstanding.get(request_id) != "submitted":
            raise ValueError(
                "goal response does not match a pending request"
            )

        if accepted:
            self._outstanding[request_id] = "active"
        else:
            del self._outstanding[request_id]

    def cancellation_requested(self, request_id: int) -> None:
        """Remember cancellation intent, even before goal acceptance."""
        if request_id not in self._outstanding:
            raise ValueError(
                "cannot cancel an unknown or completed request"
            )

        self._cancel_requests.add(request_id)

    def cancellation_response(
        self,
        request_id: int,
        *,
        acknowledged: bool,
    ) -> None:
        """Record cancellation response without declaring completion."""
        if not isinstance(acknowledged, bool):
            raise TypeError(
                "acknowledged must be boolean"
            )

        if request_id not in self._cancel_requests:
            raise ValueError(
                "cancellation response has no matching request"
            )

        # A terminal result may arrive before the cancellation reply.
        if request_id not in self._outstanding:
            return

        if self._outstanding[request_id] != "active":
            raise ValueError(
                "cancellation response requires an accepted goal"
            )

        self._cancel_responses[request_id] = acknowledged

    def terminal_result(
        self,
        request_id: int,
        *,
        confirmed: bool,
    ) -> None:
        """Resolve an accepted goal only after its terminal result."""
        if not isinstance(confirmed, bool):
            raise TypeError("confirmed must be boolean")

        if self._outstanding.get(request_id) != "active":
            raise ValueError(
                "terminal result does not match an accepted goal"
            )

        del self._outstanding[request_id]

        if not confirmed:
            self._uncertain = True

    def invalidate(self) -> None:
        """Permanently block review after an ambiguous controller event."""
        self._uncertain = True

def build_teleop_episode_result(
    session: TeleopRecordingSession,
    *,
    episode_index: int,
    seed: int,
    scenario: str,
    success: bool,
    failure_reason: str | None = None,
) -> EpisodeResult:
    """Build canonical metadata from a pending operator review.

    Timestamps come from captured frames. Identity, scenario,
    and outcome must be explicitly supplied by collection setup
    and operator review rather than inferred from controller state.
    """
    if not isinstance(session, TeleopRecordingSession):
        raise TypeError(
            "session must be a TeleopRecordingSession"
        )

    if session.status != "recording":
        raise ValueError(
            "metadata can only be prepared before finalization"
        )

    if session.has_pending_goal:
        raise ValueError(
            "cannot build metadata with a pending controller goal"
        )

    frames = session.frames

    if not frames:
        raise ValueError(
            "cannot build metadata for an empty recording"
        )

    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise ValueError(
            "episode_index must be a non-negative integer"
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError(
            "scenario must be a non-empty string"
        )

    if not isinstance(success, bool):
        raise TypeError(
            "success must be boolean"
        )

    if success:
        if failure_reason is not None:
            raise ValueError(
                "successful demonstrations cannot have a failure reason"
            )
        failure_category = None
    else:
        if (
            not isinstance(failure_reason, str)
            or not failure_reason.strip()
        ):
            raise ValueError(
                "failed demonstrations require a failure reason"
            )
        failure_category = EpisodeFailureCategory.TASK

    return EpisodeResult(
        episode_index=episode_index,
        seed=seed,
        scenario=scenario,
        start_time_s=frames[0].timestamp_s,
        end_time_s=frames[-1].timestamp_s,
        success=success,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )

@dataclass(frozen=True)
class TeleopCollectionConfig:
    """Explicit provenance and destination for one teleop episode."""

    output_root: Path
    episode_index: int
    seed: int
    scenario: str
    git_commit: str
    config_id: str
    task: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate collection identity before recording begins."""
        if (
            not isinstance(self.output_root, Path)
            or not self.output_root.is_absolute()
        ):
            raise ValueError(
                "teleop output_root must be an absolute Path"
            )

        for name, number in (
            ("episode_index", self.episode_index),
            ("seed", self.seed),
        ):
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer"
                )

        for name, value in (
            ("scenario", self.scenario),
            ("git_commit", self.git_commit),
            ("config_id", self.config_id),
            ("task", self.task),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{name} must be a non-empty string"
                )

        if self.failure_reason is not None:
            if (
                not isinstance(self.failure_reason, str)
                or not self.failure_reason.strip()
            ):
                raise ValueError(
                    "failure_reason must be non-empty when supplied"
                )
