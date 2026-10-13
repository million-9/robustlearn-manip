"""MCAP logging contracts for hybrid manipulation episodes.

This module defines logging configuration, deterministic paths, and metadata.
It deliberately does not start ROS 2 recorder processes.
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from robustlearn.episode_manager import (
    EpisodeBackend,
    EpisodeExecutionOutcome,
    EpisodeSpec,
)

DEFAULT_MCAP_TOPICS: tuple[str, ...] = (
    "/joint_states",
    "/wrist_ft/wrench",
    "/local_insertion_expert/status",
    "/panda_arm_controller/controller_state",
    "/execute_trajectory/_action/feedback",
    "/execute_trajectory/_action/status",
    "/panda_arm_controller/follow_joint_trajectory/_action/feedback",
    "/panda_arm_controller/follow_joint_trajectory/_action/status",
)


_SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    """Return a filesystem-safe non-empty path component."""
    stripped = value.strip()

    if not stripped:
        raise ValueError("path component must be non-empty")

    safe = _SAFE_COMPONENT_PATTERN.sub("_", stripped).strip("._")

    if not safe:
        raise ValueError("path component must contain a usable character")

    return safe


@dataclass(frozen=True)
class McapLoggingConfig:
    """Configuration controlling per-episode MCAP recording."""

    enabled: bool
    output_root: Path
    run_id: str
    topics: tuple[str, ...] = DEFAULT_MCAP_TOPICS

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")

        if not self.topics:
            raise ValueError("topics must not be empty")

        if len(set(self.topics)) != len(self.topics):
            raise ValueError("topics must not contain duplicates")

        for topic in self.topics:
            if not topic.startswith("/"):
                raise ValueError(
                    f"ROS topic must be absolute: {topic!r}"
                )

            if topic == "/":
                raise ValueError("ROS topic must not be root only")


@dataclass(frozen=True)
class EpisodeLogMetadata:
    """Metadata associated with one recorded hybrid episode."""

    episode_index: int
    seed: int
    scenario: str
    git_commit: str
    timestamp_s: float
    ros_domain_id: int | None = None

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        if not self.scenario.strip():
            raise ValueError("scenario must be non-empty")

        if not self.git_commit.strip():
            raise ValueError("git_commit must be non-empty")

        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")

        if self.ros_domain_id is not None and self.ros_domain_id < 0:
            raise ValueError("ros_domain_id must be non-negative")


def episode_directory_name(
    metadata: EpisodeLogMetadata,
) -> str:
    """Return the deterministic directory name for one episode."""
    scenario = _safe_component(metadata.scenario)

    return (
        f"episode_{metadata.episode_index:06d}"
        f"__seed_{metadata.seed}"
        f"__{scenario}"
    )


def episode_bag_directory(
    config: McapLoggingConfig,
    metadata: EpisodeLogMetadata,
) -> Path:
    """Return the deterministic MCAP bag directory for one episode."""
    run_id = _safe_component(config.run_id)

    return (
        config.output_root
        / run_id
        / episode_directory_name(metadata)
        / "mcap"
    )


def metadata_dict(
    metadata: EpisodeLogMetadata,
) -> dict[str, str]:
    """Return rosbag2 custom-data metadata values."""
    values = {
        "episode_index": str(metadata.episode_index),
        "seed": str(metadata.seed),
        "scenario": metadata.scenario,
        "git_commit": metadata.git_commit,
        "timestamp_s": repr(metadata.timestamp_s),
    }

    if metadata.ros_domain_id is not None:
        values["ros_domain_id"] = str(metadata.ros_domain_id)

    return values


def build_rosbag_record_command(
    config: McapLoggingConfig,
    metadata: EpisodeLogMetadata,
) -> tuple[str, ...]:
    """Build the ros2 bag record command for one episode."""
    if not config.enabled:
        raise ValueError("MCAP logging is disabled")

    bag_directory = episode_bag_directory(
        config,
        metadata,
    )

    command: list[str] = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(bag_directory),
        "--disable-keyboard-controls",
        "--include-hidden-topics",
        "--topics",
        *config.topics,
        "--custom-data",
    ]

    for key, value in metadata_dict(metadata).items():
        command.append(f"{key}={value}")

    return tuple(command)


class RecorderProcess(Protocol):
    """Minimal subprocess interface required by the MCAP recorder."""

    def poll(self) -> int | None:
        """Return process status without blocking."""
        ...

    def send_signal(self, sig: int) -> None:
        """Send a signal to the recorder process."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for recorder termination."""
        ...

    def terminate(self) -> None:
        """Request process termination."""
        ...

    def kill(self) -> None:
        """Force process termination."""
        ...


class McapRecorder:
    """Manage one rosbag2 MCAP recorder process."""

    def __init__(
        self,
        config: McapLoggingConfig,
        metadata: EpisodeLogMetadata,
        *,
        stop_timeout_s: float = 10.0,
    ) -> None:
        if stop_timeout_s <= 0.0 or not math.isfinite(stop_timeout_s):
            raise ValueError(
                "stop_timeout_s must be finite and positive"
            )

        self._config = config
        self._metadata = metadata
        self._stop_timeout_s = stop_timeout_s
        self._process: RecorderProcess | None = None

    @property
    def running(self) -> bool:
        """Return whether a recorder process is currently alive."""
        return (
            self._process is not None
            and self._process.poll() is None
        )

    def start(self) -> None:
        """Start rosbag2 recording for this episode."""
        if not self._config.enabled:
            raise ValueError("MCAP logging is disabled")

        if self._process is not None:
            raise RuntimeError("MCAP recorder has already been started")

        bag_directory = episode_bag_directory(
            self._config,
            self._metadata,
        )

        bag_directory.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = build_rosbag_record_command(
            self._config,
            self._metadata,
        )

        environment = os.environ.copy()

        if self._metadata.ros_domain_id is not None:
            environment["ROS_DOMAIN_ID"] = str(
                self._metadata.ros_domain_id
            )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )

        if process.poll() is not None:
            raise RuntimeError(
                "rosbag2 recorder exited immediately "
                f"with return code {process.poll()}"
            )

        self._process = process

    def stop(self) -> None:
        """Stop rosbag2 cleanly, escalating only if necessary."""
        process = self._process

        if process is None:
            return

        if process.poll() is not None:
            self._process = None
            return

        process.send_signal(signal.SIGINT)

        try:
            process.wait(
                timeout=self._stop_timeout_s,
            )
        except subprocess.TimeoutExpired:
            process.terminate()

            try:
                process.wait(
                    timeout=self._stop_timeout_s,
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(
                    timeout=self._stop_timeout_s,
                )

        self._process = None


class LoggedEpisodeBackend:
    """Wrap an episode backend with per-episode MCAP recording."""

    def __init__(
        self,
        backend: EpisodeBackend,
        config: McapLoggingConfig,
        *,
        git_commit: str,
        timestamp_factory: Callable[[], float],
        ros_domain_id_factory: Callable[[EpisodeSpec], int] | None = None,
        recorder_factory: Callable[
            [McapLoggingConfig, EpisodeLogMetadata],
            McapRecorder,
        ] = McapRecorder,
    ) -> None:
        if not git_commit.strip():
            raise ValueError("git_commit must be non-empty")

        self._backend = backend
        self._config = config
        self._git_commit = git_commit
        self._timestamp_factory = timestamp_factory
        self._ros_domain_id_factory = ros_domain_id_factory
        self._recorder_factory = recorder_factory

    def reset(self, spec: EpisodeSpec) -> None:
        """Delegate reset without changing existing backend behavior."""
        self._backend.reset(spec)

    def execute(
        self,
        spec: EpisodeSpec,
    ) -> EpisodeExecutionOutcome:
        """Execute one episode with optional MCAP recording."""
        if not self._config.enabled:
            return self._backend.execute(spec)

        metadata = EpisodeLogMetadata(
            episode_index=spec.episode_index,
            seed=spec.seed,
            scenario=spec.scenario,
            git_commit=self._git_commit,
            timestamp_s=self._timestamp_factory(),
            ros_domain_id=(
                self._ros_domain_id_factory(spec)
                if self._ros_domain_id_factory is not None
                else None
            ),
        )

        recorder = self._recorder_factory(
            self._config,
            metadata,
        )

        recorder.start()

        try:
            return self._backend.execute(spec)
        finally:
            recorder.stop()


@dataclass(frozen=True)
class McapValidationResult:
    """Validation summary for one completed MCAP recording."""

    bag_directory: Path
    topics: tuple[str, ...]
    message_count: int
    storage_id: str


def parse_rosbag_info(
    output: str,
) -> McapValidationResult:
    """Parse the subset of `ros2 bag info` needed for validation."""
    storage_id = ""
    message_count = -1
    topics: list[str] = []
    bag_directory = Path(".")

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("Storage id:"):
            storage_id = line.split(":", 1)[1].strip()

        elif line.startswith("Messages:"):
            message_count = int(
                line.split(":", 1)[1].strip()
            )

        elif "Topic: " in line:
            topic_part = line.split("Topic: ", 1)[1]
            topic = topic_part.split(" | ", 1)[0].strip()
            topics.append(topic)

    return McapValidationResult(
        bag_directory=bag_directory,
        topics=tuple(topics),
        message_count=message_count,
        storage_id=storage_id,
    )


def validate_mcap_bag(
    bag_directory: Path,
    *,
    required_topics: tuple[str, ...] = DEFAULT_MCAP_TOPICS,
) -> McapValidationResult:
    """Validate that a bag is readable and contains required topics."""
    if not bag_directory.is_dir():
        raise ValueError(
            f"bag directory does not exist: {bag_directory}"
        )

    completed = subprocess.run(
        [
            "ros2",
            "bag",
            "info",
            str(bag_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = parse_rosbag_info(
        completed.stdout,
    )

    if parsed.storage_id != "mcap":
        raise ValueError(
            f"expected MCAP storage, got {parsed.storage_id!r}"
        )

    if parsed.message_count <= 0:
        raise ValueError(
            "MCAP bag contains no messages"
        )

    missing_topics = tuple(
        topic
        for topic in required_topics
        if topic not in parsed.topics
    )

    if missing_topics:
        raise ValueError(
            "MCAP bag missing required topics: "
            + ", ".join(missing_topics)
        )

    return McapValidationResult(
        bag_directory=bag_directory,
        topics=parsed.topics,
        message_count=parsed.message_count,
        storage_id=parsed.storage_id,
    )
