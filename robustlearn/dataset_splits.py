"""Deterministic, group-aware train/validation/test episode splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from robustlearn.dataset_validation import validate_dataset
from robustlearn.rollout_dataset import RolloutEpisode, read_episode

SPLITS = ("train", "validation", "test")
GROUP_MODES = ("scenario_initial", "scenario", "initial_condition", "episode")
MANIFEST_VERSION = 1


class SplitError(ValueError):
    """Invalid split configuration, dataset, or manifest."""


def _validate_configuration(
    dataset_reference: str,
    seed: int,
    ratios: dict[str, float],
    group_by: str,
) -> None:
    if not isinstance(dataset_reference, str) or not dataset_reference.strip():
        raise SplitError("dataset_reference must be nonempty")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("split seed must be a non-negative integer")

    if group_by not in GROUP_MODES:
        raise SplitError(f"unsupported grouping mode: {group_by!r}")

    if set(ratios) != set(SPLITS):
        raise SplitError("ratios must define train, validation, and test")

    for name, value in ratios.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise SplitError(f"{name} ratio must be finite and between 0 and 1")

    if not math.isclose(
        sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise SplitError("split ratios must sum to 1.0")


def _group_key(episode: RolloutEpisode, mode: str) -> str:
    """Keep correlated episodes together according to grouping policy."""
    if mode == "episode":
        return f"episode:{episode.episode_index}"

    if mode == "scenario":
        return f"scenario:{episode.scenario}"

    # Explicit initial configurations group even across scenario labels.
    # Older records without this metadata fall back to scenario grouping.
    start = episode.start_joint_positions_rad

    if start is None:
        return f"scenario:{episode.scenario}"

    return "initial:" + json.dumps(list(start), separators=(",", ":"))



def _group_episodes(
    episodes: list[RolloutEpisode],
    mode: str,
) -> dict[str, list[RolloutEpisode]]:
    """Build indivisible groups, optionally protecting both correlations."""
    groups: dict[str, list[RolloutEpisode]] = {}

    if mode != "scenario_initial":
        for episode in episodes:
            key = _group_key(episode, mode)
            groups.setdefault(key, []).append(episode)
        return groups

    # Transitive grouping: if A shares a scenario with B and B shares
    # a recorded initial pose with C, all three must stay together.
    parent = {
        episode.episode_index: episode.episode_index
        for episode in episodes
    }

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        a = find(first)
        b = find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    scenario_owner: dict[str, int] = {}
    initial_owner: dict[tuple[float, ...], int] = {}

    for episode in episodes:
        index = episode.episode_index

        previous_scenario = scenario_owner.setdefault(
            episode.scenario, index
        )
        union(index, previous_scenario)

        initial = episode.start_joint_positions_rad

        if initial is not None:
            previous_initial = initial_owner.setdefault(initial, index)
            union(index, previous_initial)

    for episode in episodes:
        key = f"scenario_initial:{find(episode.episode_index)}"
        groups.setdefault(key, []).append(episode)

    return groups


def _dataset_fingerprint(paths: list[Path]) -> str:
    """Hash canonical episode filenames and exact file contents."""
    digest = hashlib.sha256()

    for path in paths:
        name = path.name.encode("utf-8")
        payload = path.read_bytes()

        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    return digest.hexdigest()


def _frequency(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def build_split_manifest(
    dataset_root: Path,
    *,
    dataset_reference: str = "week11-candidate",
    seed: int = 42,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    group_by: str = "scenario_initial",
) -> dict[str, object]:
    """Build a reproducible manifest without splitting individual frames."""
    ratios = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    _validate_configuration(dataset_reference, seed, ratios, group_by)

    # Reuse Issue #101's strict dataset validation.
    validate_dataset(dataset_root)

    paths = sorted(dataset_root.glob("*.json"), key=lambda p: p.name)

    episode_paths = {
        episode.episode_index: (episode, path.name)
        for path in paths
        for episode in (read_episode(path),)
    }

    episodes = [item[0] for item in episode_paths.values()]

    groups = _group_episodes(episodes, group_by)

    if len(groups) < len(SPLITS):
        raise SplitError(
            f"cannot create three nonempty splits from {len(groups)} "
            f"indivisible {group_by} groups; need at least 3. "
            "Collect more groups or explicitly choose another grouping mode."
        )

    # Stable seeded ordering, independent of input directory enumeration.
    def ordering(item: tuple[str, list[RolloutEpisode]]) -> tuple[int, str]:
        key, members = item
        tie_break = hashlib.sha256(
            f"{seed}:{key}".encode()
        ).hexdigest()
        return (-len(members), tie_break)

    ordered_groups = sorted(groups.items(), key=ordering)

    assigned: dict[str, list[RolloutEpisode]] = {
        name: [] for name in SPLITS
    }
    assigned_groups: dict[str, list[str]] = {
        name: [] for name in SPLITS
    }

    total_episodes = len(episodes)

    # Balance both episode totals and task/source representation.
    categories = Counter((e.task, e.source) for e in episodes)

    current_categories: dict[str, Counter[tuple[str, str]]] = {
        name: Counter() for name in SPLITS
    }

    def score(
        candidate: str,
        members: list[RolloutEpisode],
    ) -> float:
        result = 0.0
        member_categories = Counter(
            (episode.task, episode.source) for episode in members
        )

        for name in SPLITS:
            target = total_episodes * ratios[name]
            projected = len(assigned[name])
            if name == candidate:
                projected += len(members)

            result += (projected - target) ** 2 / max(1.0, target)

            for category, count in categories.items():
                target_category = count * ratios[name]
                projected_category = current_categories[name][category]

                if name == candidate:
                    projected_category += member_categories[category]

                result += (
                    0.15
                    * (projected_category - target_category) ** 2
                    / max(1.0, target_category)
                )

        return result

    for index, (key, members) in enumerate(ordered_groups):
        remaining = len(ordered_groups) - index
        empty = [name for name in SPLITS if not assigned_groups[name]]

        # Reserve enough groups for every requested nonempty split.
        candidates = empty if len(empty) == remaining else SPLITS

        chosen = min(candidates, key=lambda name: score(name, members))

        assigned[chosen].extend(members)
        assigned_groups[chosen].append(key)

        current_categories[chosen].update(
            (episode.task, episode.source) for episode in members
        )

    manifest_splits: dict[str, object] = {}

    for name in SPLITS:
        members = sorted(
            assigned[name], key=lambda episode: episode.episode_index
        )

        manifest_splits[name] = {
            "episode_ids": [episode.episode_index for episode in members],
            "episode_files": [
                episode_paths[episode.episode_index][1]
                for episode in members
            ],
            "episode_count": len(members),
            "frame_count": sum(len(episode.frames) for episode in members),
            "scenarios": _frequency([e.scenario for e in members]),
            "tasks": _frequency([e.task for e in members]),
            "sources": _frequency([e.source for e in members]),
            "group_count": len(assigned_groups[name]),
        }

    all_ids = [
        episode.episode_index
        for name in SPLITS
        for episode in assigned[name]
    ]

    expected_ids = {episode.episode_index for episode in episodes}

    if (
        len(all_ids) != len(expected_ids)
        or set(all_ids) != expected_ids
    ):
        raise SplitError("episode leakage or incomplete split assignment")

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_reference": dataset_reference,
        "dataset_sha256": _dataset_fingerprint(paths),
        "seed": seed,
        "ratios": ratios,
        "group_by": group_by,
        "total_episodes": total_episodes,
        "total_frames": sum(len(e.frames) for e in episodes),
        "splits": manifest_splits,
    }


def validate_split_manifest(
    dataset_root: Path,
    manifest: object,
) -> None:
    """Reconstruct the expected assignment and reject any differences."""
    if not isinstance(manifest, dict):
        raise SplitError("split manifest must be a JSON object")

    ratios = manifest.get("ratios")

    if not isinstance(ratios, dict):
        raise SplitError("split manifest has missing or invalid ratios")

    values: dict[str, float] = {}

    for name in SPLITS:
        value = ratios.get(name)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitError(f"manifest has invalid {name} ratio")

        values[name] = float(value)

    dataset_reference = manifest.get("dataset_reference")
    seed = manifest.get("seed")
    group_by = manifest.get("group_by")

    if not isinstance(dataset_reference, str):
        raise SplitError("manifest has invalid dataset reference")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("manifest has invalid split seed")

    if not isinstance(group_by, str):
        raise SplitError("manifest has invalid grouping mode")

    expected = build_split_manifest(
        dataset_root,
        dataset_reference=dataset_reference,
        seed=seed,
        train=values["train"],
        validation=values["validation"],
        test=values["test"],
        group_by=group_by,
    )

    if manifest != expected:
        raise SplitError(
            "manifest does not match the current dataset or deterministic "
            "split assignment; files or manifest may have changed"
        )


def _serialize(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Generate or verify one persistent canonical split manifest."""
    parser = argparse.ArgumentParser(
        description="Generate or validate leakage-free dataset splits."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-ref", default="week11-candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--group-by", choices=GROUP_MODES, default="scenario_initial")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing manifest against the current dataset.",
    )

    args = parser.parse_args(argv)

    try:
        if args.manifest.resolve().parent == args.dataset.resolve():
            raise SplitError(
                "manifest must be outside the canonical episode directory"
            )

        if args.check:
            raw = json.loads(args.manifest.read_text(encoding="utf-8"))
            validate_split_manifest(args.dataset, raw)
            print("SPLIT MANIFEST VALIDATION PASSED")
            return 0

        manifest = build_split_manifest(
            args.dataset,
            dataset_reference=args.dataset_ref,
            seed=args.seed,
            train=args.train,
            validation=args.validation,
            test=args.test,
            group_by=args.group_by,
        )

        payload = _serialize(manifest)

        if args.manifest.exists():
            if args.manifest.read_text(encoding="utf-8") != payload:
                raise SplitError(
                    "existing manifest differs; refusing to overwrite it"
                )
        else:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents accidental manifest replacement.
            with args.manifest.open("x", encoding="utf-8") as handle:
                handle.write(payload)

        print("SPLIT MANIFEST GENERATED")
        print(f"Dataset: {manifest['dataset_reference']}")
        print(f"Grouping: {manifest['group_by']}")
        print(f"Manifest: {args.manifest}")

        split_data = manifest["splits"]

        if isinstance(split_data, dict):
            for name in SPLITS:
                detail = split_data[name]
                print(f"{name}: {detail['episode_count']} episodes")

        return 0

    except (ValueError, OSError) as exc:
        print(f"SPLIT ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
