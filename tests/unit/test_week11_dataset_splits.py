"""Issue 102: deterministic episode grouping and leakage prevention."""

import json
from dataclasses import replace

import pytest

from robustlearn.dataset_splits import (
    SplitError,
    build_split_manifest,
    main,
    validate_split_manifest,
)
from robustlearn.rollout_dataset import (
    RolloutEpisode,
    RolloutFrame,
    read_episode,
    write_episode,
)


def episode(index, scenario, source="scripted", start=None):
    frames = tuple(
        RolloutFrame(
            frame_index=i,
            timestamp_s=float(i + 1),
            joint_position=(0.1,) * 7,
            joint_velocity=(0.0,) * 7,
            task_relative_state=(0.01, 0.02, 0.03),
            wrist_wrench=(0.0,) * 6,
            expert_action=(0.2,) * 7,
            task_success=i == 1,
            task_failure=False,
            terminated=i == 1,
        )
        for i in range(2)
    )

    return RolloutEpisode(
        episode_index=index,
        seed=index,
        scenario=scenario,
        source=source,
        task="panda_insertion",
        success=True,
        failure_reason=None,
        start_timestamp_s=1.0,
        end_timestamp_s=2.0,
        git_commit="test-commit",
        config_id="test-config",
        frames=frames,
        start_base_joint_positions_rad=(0.0,) * 7 if start else None,
        start_perturbation_rad=(start,) * 7 if start else None,
        start_joint_positions_rad=(start,) * 7 if start else None,
    )


def sample_dataset(tmp_path, scenario_count=6, per_scenario=3):
    root = tmp_path / "episodes"
    index = 0

    for scenario_index in range(scenario_count):
        for _ in range(per_scenario):
            source = "scripted" if index % 2 == 0 else "teleop"
            write_episode(
                root,
                episode(
                    index,
                    f"scenario_{scenario_index:02d}",
                    source,
                ),
            )
            index += 1

    return root


def test_deterministic_manifest_and_complete_assignment(tmp_path):
    root = sample_dataset(tmp_path)

    first = build_split_manifest(root, seed=123)
    second = build_split_manifest(root, seed=123)

    assert first == second
    assert first["total_episodes"] == 18
    assert first["total_frames"] == 36
    assert len(first["dataset_sha256"]) == 64

    splits = first["splits"]

    id_sets = {
        name: set(splits[name]["episode_ids"])
        for name in ("train", "validation", "test")
    }

    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])
    assert set.union(*id_sets.values()) == set(range(18))

    for name in id_sets:
        detail = splits[name]

        assert detail["episode_count"] > 0
        assert detail["frame_count"] == detail["episode_count"] * 2
        assert sum(detail["scenarios"].values()) == detail["episode_count"]
        assert sum(detail["sources"].values()) == detail["episode_count"]

    validate_split_manifest(root, first)


def test_default_scenario_grouping_prevents_scenario_overlap(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = build_split_manifest(root)

    scenarios = [
        set(manifest["splits"][name]["scenarios"])
        for name in ("train", "validation", "test")
    ]

    assert scenarios[0].isdisjoint(scenarios[1])
    assert scenarios[0].isdisjoint(scenarios[2])
    assert scenarios[1].isdisjoint(scenarios[2])


def test_frames_of_episode_never_cross_splits(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = build_split_manifest(root, group_by="episode")

    seen = set()

    for name in ("train", "validation", "test"):
        split = manifest["splits"][name]

        for episode_id, filename in zip(
            split["episode_ids"],
            split["episode_files"],
            strict=True,
        ):
            recorded = read_episode(root / filename)

            assert recorded.episode_index == episode_id
            assert episode_id not in seen
            assert [f.frame_index for f in recorded.frames] == [0, 1]

            seen.add(episode_id)

    assert seen == set(range(18))


def test_initial_condition_grouping(tmp_path):
    root = tmp_path / "episodes"

    write_episode(root, episode(0, "scenario_a", start=0.1))
    write_episode(root, episode(1, "scenario_b", start=0.1))
    write_episode(root, episode(2, "scenario_c", start=0.2))
    write_episode(root, episode(3, "scenario_d", start=0.3))

    manifest = build_split_manifest(root, group_by="initial_condition")

    for split in manifest["splits"].values():
        ids = set(split["episode_ids"])
        assert (0 in ids) == (1 in ids)


def test_reject_insufficient_scenario_groups(tmp_path):
    root = sample_dataset(tmp_path, scenario_count=2)

    with pytest.raises(SplitError, match="need at least 3"):
        build_split_manifest(root)

    # Episode-level grouping is an explicit alternative.
    manifest = build_split_manifest(root, group_by="episode")
    assert manifest["total_episodes"] == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train": 0.9, "validation": 0.15, "test": 0.15},
        {"train": 0.0},
        {"validation": -0.1},
        {"test": float("nan")},
        {"seed": -1},
        {"dataset_reference": ""},
        {"group_by": "frame"},
    ],
)
def test_reject_invalid_configuration(tmp_path, kwargs):
    root = sample_dataset(tmp_path)

    with pytest.raises(SplitError):
        build_split_manifest(root, **kwargs)


def test_cli_manifest_reproducibility_and_verification(tmp_path):
    root = sample_dataset(tmp_path)
    path = tmp_path / "splits.json"

    arguments = [
        str(root),
        "--manifest",
        str(path),
        "--dataset-ref",
        "candidate-102",
        "--seed",
        "7",
    ]

    assert main(arguments) == 0
    initial = path.read_bytes()

    assert main(arguments) == 0
    assert path.read_bytes() == initial

    assert main([
        str(root),
        "--manifest",
        str(path),
        "--check",
    ]) == 0

    manifest = json.loads(path.read_text())
    assert manifest["dataset_reference"] == "candidate-102"
    assert manifest["seed"] == 7


def test_manifest_tampering_is_detected(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = build_split_manifest(root)

    manifest["splits"]["train"]["episode_ids"].append(999)

    with pytest.raises(SplitError, match="manifest does not match"):
        validate_split_manifest(root, manifest)


def test_dataset_changes_invalidate_manifest(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = build_split_manifest(root)

    original_path = sorted(root.glob("*.json"))[0]
    original = read_episode(original_path)

    write_episode(root, replace(original, config_id="changed-config"))

    with pytest.raises(SplitError, match="manifest does not match"):
        validate_split_manifest(root, manifest)


def test_cli_rejects_manifest_inside_dataset(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = root / "splits.json"

    assert main([
        str(root),
        "--manifest",
        str(manifest),
    ]) == 1

    assert not manifest.exists()


def test_cli_refuses_to_overwrite_changed_manifest(tmp_path):
    root = sample_dataset(tmp_path)
    manifest = tmp_path / "splits.json"

    assert main([
        str(root),
        "--manifest",
        str(manifest),
    ]) == 0

    original = manifest.read_bytes()

    assert main([
        str(root),
        "--manifest",
        str(manifest),
        "--seed",
        "1234",
    ]) == 1

    assert manifest.read_bytes() == original


def test_combined_grouping_protects_transitive_correlations(tmp_path):
    """Scenario and initial-condition connections must remain together."""
    root = tmp_path / "episodes"

    configurations = [
        (0, "scenario_a", 0.1),
        (1, "scenario_b", 0.1),
        (2, "scenario_b", 0.2),
        (3, "scenario_c", 0.3),
        (4, "scenario_d", 0.4),
        (5, "scenario_e", 0.5),
        (6, "scenario_b", None),
    ]

    for index, scenario, start in configurations:
        write_episode(root, episode(index, scenario, start=start))

    manifest = build_split_manifest(root)

    assert manifest["group_by"] == "scenario_initial"

    connected = {0, 1, 2, 6}

    for detail in manifest["splits"].values():
        assigned = set(detail["episode_ids"])
        assert connected.issubset(assigned) or connected.isdisjoint(assigned)

    validate_split_manifest(root, manifest)


def test_combined_grouping_rejects_insufficient_independent_groups(tmp_path):
    """Never break a correlated group merely to achieve three splits."""
    root = tmp_path / "episodes"

    for index in range(3):
        write_episode(
            root,
            episode(index, f"scenario_{index}", start=0.1),
        )

    with pytest.raises(SplitError, match="need at least 3"):
        build_split_manifest(root)

    # Less conservative behavior requires an explicit choice.
    manifest = build_split_manifest(root, group_by="scenario")
    assert manifest["group_by"] == "scenario"
