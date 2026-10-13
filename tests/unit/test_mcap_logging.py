"""Unit tests for Week 10 MCAP logging contracts."""

import math
from pathlib import Path

import pytest

from robustlearn.mcap_logging import (
    DEFAULT_MCAP_TOPICS,
    EpisodeLogMetadata,
    McapLoggingConfig,
    episode_bag_directory,
    episode_directory_name,
    metadata_dict,
)


def test_default_topics_cover_core_hybrid_runtime_streams() -> None:
    assert DEFAULT_MCAP_TOPICS == (
        "/joint_states",
        "/wrist_ft/wrench",
        "/local_insertion_expert/status",
        "/panda_arm_controller/controller_state",
        "/execute_trajectory/_action/feedback",
        "/execute_trajectory/_action/status",
        "/panda_arm_controller/follow_joint_trajectory/_action/feedback",
        "/panda_arm_controller/follow_joint_trajectory/_action/status",
    )


def test_logging_can_be_disabled_explicitly() -> None:
    config = McapLoggingConfig(
        enabled=False,
        output_root=Path("build/episode_logs"),
        run_id="week10",
    )

    assert not config.enabled


def test_episode_directory_name_is_deterministic() -> None:
    metadata = EpisodeLogMetadata(
        episode_index=7,
        seed=1234,
        scenario="week9_case_15",
        git_commit="abcdef123456",
        timestamp_s=10.0,
    )

    assert episode_directory_name(metadata) == (
        "episode_000007__seed_1234__week9_case_15"
    )

    assert (
        episode_directory_name(metadata)
        == episode_directory_name(metadata)
    )


def test_episode_directory_sanitizes_scenario() -> None:
    metadata = EpisodeLogMetadata(
        episode_index=1,
        seed=2,
        scenario="case with spaces/slash",
        git_commit="abc",
        timestamp_s=1.0,
    )

    assert episode_directory_name(metadata) == (
        "episode_000001__seed_2__case_with_spaces_slash"
    )


def test_bag_directory_has_run_and_episode_structure() -> None:
    config = McapLoggingConfig(
        enabled=True,
        output_root=Path("build/episode_logs"),
        run_id="benchmark_a",
    )

    metadata = EpisodeLogMetadata(
        episode_index=3,
        seed=22,
        scenario="week9_case_03",
        git_commit="deadbeef",
        timestamp_s=100.0,
    )

    assert episode_bag_directory(
        config,
        metadata,
    ) == Path(
        "build/episode_logs"
        "/benchmark_a"
        "/episode_000003__seed_22__week9_case_03"
        "/mcap"
    )


def test_metadata_dict_contains_required_fields() -> None:
    metadata = EpisodeLogMetadata(
        episode_index=4,
        seed=-10,
        scenario="week9_case_11",
        git_commit="1234567890abcdef",
        timestamp_s=12.5,
    )

    assert metadata_dict(metadata) == {
        "episode_index": "4",
        "seed": "-10",
        "scenario": "week9_case_11",
        "git_commit": "1234567890abcdef",
        "timestamp_s": "12.5",
    }


@pytest.mark.parametrize(
    "timestamp_s",
    (math.nan, math.inf, -math.inf),
)
def test_metadata_rejects_non_finite_timestamp(
    timestamp_s: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timestamp_s must be finite",
    ):
        EpisodeLogMetadata(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            git_commit="abc",
            timestamp_s=timestamp_s,
        )


def test_config_rejects_relative_topic_name() -> None:
    with pytest.raises(
        ValueError,
        match="ROS topic must be absolute",
    ):
        McapLoggingConfig(
            enabled=True,
            output_root=Path("build/logs"),
            run_id="run",
            topics=("joint_states",),
        )


def test_config_rejects_duplicate_topics() -> None:
    with pytest.raises(
        ValueError,
        match="topics must not contain duplicates",
    ):
        McapLoggingConfig(
            enabled=True,
            output_root=Path("build/logs"),
            run_id="run",
            topics=(
                "/joint_states",
                "/joint_states",
            ),
        )


def test_build_rosbag_record_command_uses_mcap_and_required_topics() -> None:
    from robustlearn.mcap_logging import build_rosbag_record_command

    config = McapLoggingConfig(
        enabled=True,
        output_root=Path("build/episode_logs"),
        run_id="week10",
    )

    metadata = EpisodeLogMetadata(
        episode_index=2,
        seed=42,
        scenario="week9_case_03",
        git_commit="abc123",
        timestamp_s=123.5,
    )

    command = build_rosbag_record_command(
        config,
        metadata,
    )

    assert command[:6] == (
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
    )

    assert (
        "build/episode_logs/"
        "week10/"
        "episode_000002__seed_42__week9_case_03/"
        "mcap"
    ) in command

    topics_index = command.index("--topics")
    custom_data_index = command.index("--custom-data")

    assert command[
        topics_index + 1:custom_data_index
    ] == DEFAULT_MCAP_TOPICS


def test_build_rosbag_record_command_includes_metadata() -> None:
    from robustlearn.mcap_logging import build_rosbag_record_command

    config = McapLoggingConfig(
        enabled=True,
        output_root=Path("build/logs"),
        run_id="run_a",
    )

    metadata = EpisodeLogMetadata(
        episode_index=5,
        seed=-3,
        scenario="week9_case_06",
        git_commit="deadbeef",
        timestamp_s=9.25,
    )

    command = build_rosbag_record_command(
        config,
        metadata,
    )

    assert "episode_index=5" in command
    assert "seed=-3" in command
    assert "scenario=week9_case_06" in command
    assert "git_commit=deadbeef" in command
    assert "timestamp_s=9.25" in command


def test_build_rosbag_record_command_rejects_disabled_logging() -> None:
    from robustlearn.mcap_logging import build_rosbag_record_command

    config = McapLoggingConfig(
        enabled=False,
        output_root=Path("build/logs"),
        run_id="run",
    )

    metadata = EpisodeLogMetadata(
        episode_index=0,
        seed=0,
        scenario="week9_case_01",
        git_commit="abc",
        timestamp_s=0.0,
    )

    with pytest.raises(
        ValueError,
        match="MCAP logging is disabled",
    ):
        build_rosbag_record_command(
            config,
            metadata,
        )


class FakeRecorderProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.signals = []
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_mcap_recorder_starts_and_stops_cleanly(
    monkeypatch,
    tmp_path,
) -> None:
    import signal

    from robustlearn.mcap_logging import McapRecorder

    process = FakeRecorderProcess()

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    config = McapLoggingConfig(
        enabled=True,
        output_root=tmp_path,
        run_id="run",
    )

    metadata = EpisodeLogMetadata(
        episode_index=0,
        seed=1,
        scenario="week9_case_01",
        git_commit="abc",
        timestamp_s=1.0,
    )

    recorder = McapRecorder(
        config,
        metadata,
    )

    assert not recorder.running

    recorder.start()

    assert recorder.running

    recorder.stop()

    assert process.signals == [signal.SIGINT]
    assert process.wait_calls == [10.0]
    assert not process.terminated
    assert not process.killed
    assert not recorder.running


def test_mcap_recorder_rejects_double_start(
    monkeypatch,
    tmp_path,
) -> None:
    from robustlearn.mcap_logging import McapRecorder

    process = FakeRecorderProcess()

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    recorder = McapRecorder(
        McapLoggingConfig(
            enabled=True,
            output_root=tmp_path,
            run_id="run",
        ),
        EpisodeLogMetadata(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            git_commit="abc",
            timestamp_s=1.0,
        ),
    )

    recorder.start()

    with pytest.raises(
        RuntimeError,
        match="already been started",
    ):
        recorder.start()


def test_mcap_recorder_stop_before_start_is_safe(
    tmp_path,
) -> None:
    from robustlearn.mcap_logging import McapRecorder

    recorder = McapRecorder(
        McapLoggingConfig(
            enabled=True,
            output_root=tmp_path,
            run_id="run",
        ),
        EpisodeLogMetadata(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            git_commit="abc",
            timestamp_s=1.0,
        ),
    )

    recorder.stop()

    assert not recorder.running


def test_mcap_recorder_rejects_invalid_stop_timeout(
    tmp_path,
) -> None:
    from robustlearn.mcap_logging import McapRecorder

    with pytest.raises(
        ValueError,
        match="stop_timeout_s must be finite and positive",
    ):
        McapRecorder(
            McapLoggingConfig(
                enabled=True,
                output_root=tmp_path,
                run_id="run",
            ),
            EpisodeLogMetadata(
                episode_index=0,
                seed=0,
                scenario="week9_case_01",
                git_commit="abc",
                timestamp_s=1.0,
            ),
            stop_timeout_s=0.0,
        )


def test_logged_backend_leaves_disabled_execution_unchanged() -> None:
    from robustlearn.mcap_logging import LoggedEpisodeBackend

    events = []

    class Backend:
        def reset(self, spec):
            events.append(("reset", spec.episode_index))

        def execute(self, spec):
            events.append(("execute", spec.episode_index))
            return "result"

    class Spec:
        episode_index = 1
        seed = 2
        scenario = "week9_case_03"

    wrapped = LoggedEpisodeBackend(
        backend=Backend(),
        config=McapLoggingConfig(
            enabled=False,
            output_root=Path("build/logs"),
            run_id="run",
        ),
        git_commit="abc",
        timestamp_factory=lambda: 1.0,
    )

    wrapped.reset(Spec())
    result = wrapped.execute(Spec())

    assert result == "result"
    assert events == [
        ("reset", 1),
        ("execute", 1),
    ]


def test_logged_backend_starts_before_execute_and_stops_after() -> None:
    from robustlearn.mcap_logging import LoggedEpisodeBackend

    events = []

    class Backend:
        def reset(self, spec):
            events.append("reset")

        def execute(self, spec):
            events.append("execute")
            return "ok"

    class Recorder:
        def __init__(self, config, metadata):
            assert metadata.episode_index == 4
            assert metadata.seed == 99
            assert metadata.scenario == "week9_case_05"
            assert metadata.git_commit == "deadbeef"
            assert metadata.timestamp_s == 123.0

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class Spec:
        episode_index = 4
        seed = 99
        scenario = "week9_case_05"

    wrapped = LoggedEpisodeBackend(
        backend=Backend(),
        config=McapLoggingConfig(
            enabled=True,
            output_root=Path("build/logs"),
            run_id="run",
        ),
        git_commit="deadbeef",
        timestamp_factory=lambda: 123.0,
        recorder_factory=Recorder,
    )

    wrapped.reset(Spec())
    result = wrapped.execute(Spec())

    assert result == "ok"
    assert events == [
        "reset",
        "start",
        "execute",
        "stop",
    ]


def test_logged_backend_stops_recorder_when_execution_raises() -> None:
    from robustlearn.mcap_logging import LoggedEpisodeBackend

    events = []

    class Backend:
        def reset(self, spec):
            pass

        def execute(self, spec):
            events.append("execute")
            raise RuntimeError("episode failed")

    class Recorder:
        def __init__(self, config, metadata):
            pass

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class Spec:
        episode_index = 0
        seed = 0
        scenario = "week9_case_01"

    wrapped = LoggedEpisodeBackend(
        backend=Backend(),
        config=McapLoggingConfig(
            enabled=True,
            output_root=Path("build/logs"),
            run_id="run",
        ),
        git_commit="abc",
        timestamp_factory=lambda: 1.0,
        recorder_factory=Recorder,
    )

    with pytest.raises(
        RuntimeError,
        match="episode failed",
    ):
        wrapped.execute(Spec())

    assert events == [
        "start",
        "execute",
        "stop",
    ]


def test_logged_backend_rejects_blank_git_commit() -> None:
    from robustlearn.mcap_logging import LoggedEpisodeBackend

    class Backend:
        def reset(self, spec):
            pass

        def execute(self, spec):
            return None

    with pytest.raises(
        ValueError,
        match="git_commit must be non-empty",
    ):
        LoggedEpisodeBackend(
            backend=Backend(),
            config=McapLoggingConfig(
                enabled=True,
                output_root=Path("build/logs"),
                run_id="run",
            ),
            git_commit=" ",
            timestamp_factory=lambda: 1.0,
        )


def test_metadata_can_record_ros_domain_id() -> None:
    metadata = EpisodeLogMetadata(
        episode_index=0,
        seed=0,
        scenario="week9_case_01",
        git_commit="abc",
        timestamp_s=1.0,
        ros_domain_id=190,
    )

    assert metadata_dict(metadata)["ros_domain_id"] == "190"


def test_metadata_rejects_negative_ros_domain_id() -> None:
    with pytest.raises(
        ValueError,
        match="ros_domain_id must be non-negative",
    ):
        EpisodeLogMetadata(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            git_commit="abc",
            timestamp_s=1.0,
            ros_domain_id=-1,
        )


def test_recorder_uses_episode_ros_domain(
    monkeypatch,
    tmp_path,
) -> None:
    from robustlearn.mcap_logging import McapRecorder

    process = FakeRecorderProcess()
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.Popen",
        fake_popen,
    )

    recorder = McapRecorder(
        McapLoggingConfig(
            enabled=True,
            output_root=tmp_path,
            run_id="run",
        ),
        EpisodeLogMetadata(
            episode_index=0,
            seed=0,
            scenario="week9_case_01",
            git_commit="abc",
            timestamp_s=1.0,
            ros_domain_id=190,
        ),
    )

    recorder.start()

    assert captured["env"]["ROS_DOMAIN_ID"] == "190"

    recorder.stop()


def test_logged_backend_propagates_ros_domain_id() -> None:
    from robustlearn.episode_manager import (
        EpisodeExecutionOutcome,
        EpisodeSpec,
    )
    from robustlearn.mcap_logging import LoggedEpisodeBackend

    captured = {}

    class Backend:
        def reset(self, spec):
            pass

        def execute(self, spec):
            return EpisodeExecutionOutcome(
                success=True,
                failure_category=None,
                failure_reason=None,
            )

    class Recorder:
        def __init__(self, config, metadata):
            captured["domain"] = metadata.ros_domain_id

        def start(self):
            pass

        def stop(self):
            pass

    spec = EpisodeSpec(
        episode_index=3,
        seed=3,
        scenario="week9_case_04",
        timeout_s=90.0,
    )

    wrapped = LoggedEpisodeBackend(
        backend=Backend(),
        config=McapLoggingConfig(
            enabled=True,
            output_root=Path("build/logs"),
            run_id="run",
        ),
        git_commit="abc",
        timestamp_factory=lambda: 1.0,
        ros_domain_id_factory=lambda episode: 190 + episode.episode_index,
        recorder_factory=Recorder,
    )

    wrapped.execute(spec)

    assert captured["domain"] == 193


def test_parse_rosbag_info_extracts_storage_messages_and_topics() -> None:
    from robustlearn.mcap_logging import parse_rosbag_info

    output = """
Files:             mcap_0.mcap
Bag size:          6.5 MiB
Storage id:        mcap
Messages:          12189
Topic information: Topic: /joint_states | Type: A | Count: 4584
                   Topic: /local_insertion_expert/status | Type: B | Count: 3
                   Topic: /panda_arm_controller/controller_state | Type: C | Count: 4584
                   Topic: /wrist_ft/wrench | Type: D | Count: 3018
"""

    result = parse_rosbag_info(output)

    assert result.storage_id == "mcap"
    assert result.message_count == 12189
    assert result.topics == (
        "/joint_states",
        "/local_insertion_expert/status",
        "/panda_arm_controller/controller_state",
        "/wrist_ft/wrench",
    )


def test_validate_mcap_bag_accepts_required_topics(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from robustlearn.mcap_logging import validate_mcap_bag

    bag_directory = tmp_path / "bag"
    bag_directory.mkdir()

    output = """
Storage id:        mcap
Messages:          18
Topic information: Topic: /joint_states | Type: A | Count: 3
                   Topic: /wrist_ft/wrench | Type: B | Count: 3
                   Topic: /local_insertion_expert/status | Type: C | Count: 1
                   Topic: /panda_arm_controller/controller_state | Type: D | Count: 3
                   Topic: /execute_trajectory/_action/feedback | Type: E | Count: 1
                   Topic: /execute_trajectory/_action/status | Type: F | Count: 2
                   Topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback | G
                   Topic: /panda_arm_controller/follow_joint_trajectory/_action/status | H
"""

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=output,
        ),
    )

    result = validate_mcap_bag(
        bag_directory,
    )

    assert result.storage_id == "mcap"
    assert result.message_count == 18


def test_validate_mcap_bag_rejects_empty_recording(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from robustlearn.mcap_logging import validate_mcap_bag

    bag_directory = tmp_path / "bag"
    bag_directory.mkdir()

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="""
Storage id:        mcap
Messages:          0
""",
        ),
    )

    with pytest.raises(
        ValueError,
        match="contains no messages",
    ):
        validate_mcap_bag(
            bag_directory,
        )


def test_validate_mcap_bag_rejects_missing_topics(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from robustlearn.mcap_logging import validate_mcap_bag

    bag_directory = tmp_path / "bag"
    bag_directory.mkdir()

    monkeypatch.setattr(
        "robustlearn.mcap_logging.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="""
Storage id:        mcap
Messages:          5
Topic information: Topic: /joint_states | Type: A | Count: 5
""",
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing required topics",
    ):
        validate_mcap_bag(
            bag_directory,
        )


def test_rosbag_command_enables_hidden_action_topics() -> None:
    from robustlearn.mcap_logging import build_rosbag_record_command

    config = McapLoggingConfig(
        enabled=True,
        output_root=Path("build/logs"),
        run_id="run",
    )

    metadata = EpisodeLogMetadata(
        episode_index=0,
        seed=0,
        scenario="week9_case_01",
        git_commit="abc",
        timestamp_s=1.0,
    )

    command = build_rosbag_record_command(
        config,
        metadata,
    )

    assert "--include-hidden-topics" in command

    assert (
        "/execute_trajectory/_action/feedback"
        in command
    )
    assert (
        "/execute_trajectory/_action/status"
        in command
    )
    assert (
        "/panda_arm_controller/"
        "follow_joint_trajectory/_action/feedback"
        in command
    )
    assert (
        "/panda_arm_controller/"
        "follow_joint_trajectory/_action/status"
        in command
    )
