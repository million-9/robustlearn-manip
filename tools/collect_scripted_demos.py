#!/usr/bin/env python3
"""Collect seeded scripted-expert demonstrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import monotonic

from robustlearn.episode_manager import EpisodeSpec
from robustlearn.rollout_dataset import RolloutEpisode, episode_path
from robustlearn.scripted_demo_collection import (
    assemble_scripted_expert_episode,
    benchmark_record_to_episode_result,
    persist_scripted_expert_episode,
)
from tools.run_week10_benchmark import run_episode


def collect_one_episode(
    spec: EpisodeSpec,
    *,
    dataset_root: Path,
    capture_root: Path,
    git_commit: str,
    config_id: str,
    task: str,
    base_domain_id: int,
    domain_pool_size: int,
    clock: Callable[[], float] = monotonic,
) -> RolloutEpisode:
    """Execute, assemble, and persist one seeded expert episode."""

    scenario_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in spec.scenario.strip()
    ).strip("._")

    if not scenario_name:
        raise ValueError("scenario has no usable filename characters")

    existing_dataset_path = episode_path(dataset_root, spec)
    if existing_dataset_path.exists():
        raise FileExistsError(
            f'dataset episode already exists: {existing_dataset_path}'
        )

    capture_root.mkdir(parents=True, exist_ok=True)

    capture_path = capture_root / (
        f"episode_{spec.episode_index:06d}"
        f"__seed_{spec.seed}"
        f"__{scenario_name}.json"
    )

    if capture_path.exists():
        raise FileExistsError(
            f"expert capture already exists: {capture_path}"
        )

    start_time_s = clock()

    record = run_episode(
        spec,
        base_domain_id=base_domain_id,
        domain_pool_size=domain_pool_size,
        expert_capture_path=capture_path,
    )

    end_time_s = clock()

    if (
        record.episode_index != spec.episode_index
        or record.seed != spec.seed
        or record.scenario != spec.scenario
    ):
        raise ValueError(
            "benchmark result does not match requested episode"
        )

    result = benchmark_record_to_episode_result(
        record,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )

    episode = assemble_scripted_expert_episode(
        result,
        capture_path=capture_path,
        git_commit=git_commit,
        config_id=config_id,
        task=task,
    )

    # Preserve actual seeded initial conditions from Week 10.
    # Dataclass validation rejects incomplete or inconsistent vectors.
    episode = replace(
        episode,
        start_base_joint_positions_rad=(
            record.start_base_joint_positions_rad
        ),
        start_perturbation_rad=(
            record.start_perturbation_rad
        ),
        start_joint_positions_rad=(
            record.start_joint_positions_rad
        ),
    )

    persist_scripted_expert_episode(
        episode,
        output_root=dataset_root,
    )

    return episode


def collect_episodes(
    specs,
    *,
    dataset_root: Path,
    capture_root: Path,
    summary_path: Path,
    git_commit: str,
    config_id: str,
    task: str,
    base_domain_id: int,
    domain_pool_size: int,
) -> dict[str, object]:
    """Collect a batch of expert episodes with safe resume support."""
    import json
    import os
    import tempfile

    from robustlearn.rollout_dataset import read_episode

    requested = tuple(specs)

    if not requested:
        raise ValueError("at least one episode is required")

    paths = [episode_path(dataset_root, spec) for spec in requested]

    if len(set(paths)) != len(paths):
        raise ValueError("duplicate dataset episode paths in request")

    entries = []
    skipped_count = 0
    success_count = 0
    failure_count = 0

    collection_error: Exception | None = None

    def _save_summary() -> dict[str, object]:
        summary = {
            "schema_version": 1,
            "requested_count": len(requested),
            "skipped_count": skipped_count,
            "collected_success_count": success_count,
            "collected_failure_count": failure_count,
            "collection_error_count": int(collection_error is not None),
            "git_commit": git_commit,
            "config_id": config_id,
            "task": task,
            "episodes": entries,
        }

        # Atomically replace the summary, never individual episode data.
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=summary_path.parent,
                prefix=f".{summary_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary_path, summary_path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return summary

    for spec, path in zip(requested, paths, strict=True):
        try:
            if path.exists():
                # Existing data must be valid before being skipped.
                episode = read_episode(path)

                if (
                    episode.episode_index != spec.episode_index
                    or episode.seed != spec.seed
                    or episode.scenario != spec.scenario
                ):
                    raise ValueError(
                        f"existing dataset identity mismatch: {path}"
                    )

                if (
                    episode.source != "expert"
                    or episode.task != task
                    or episode.git_commit != git_commit
                    or episode.config_id != config_id
                ):
                    raise ValueError(
                        f"existing dataset provenance mismatch: {path}"
                    )

                if episode.success and not episode.frames:
                    raise ValueError(
                        f"successful dataset contains no frames: {path}"
                    )

                status = "skipped"
                skipped_count += 1

            else:
                episode = collect_one_episode(
                    spec,
                    dataset_root=dataset_root,
                    capture_root=capture_root,
                    git_commit=git_commit,
                    config_id=config_id,
                    task=task,
                    base_domain_id=base_domain_id,
                    domain_pool_size=domain_pool_size,
                )

                if (
                    episode.episode_index != spec.episode_index
                    or episode.seed != spec.seed
                    or episode.scenario != spec.scenario
                ):
                    raise ValueError(
                        "collected episode identity mismatch"
                    )

                if not path.is_file():
                    raise ValueError(
                        f"collected episode was not persisted: {path}"
                    )

                if episode.success:
                    status = "success"
                    success_count += 1
                else:
                    status = "failure"
                    failure_count += 1

            entries.append({
                "episode_index": spec.episode_index,
                "seed": spec.seed,
                "scenario": spec.scenario,
                "source": episode.source,
                "status": status,
                "success": episode.success,
                "failure_reason": episode.failure_reason,
                "dataset_path": str(path),
                "frame_count": len(episode.frames),
            })

        except Exception as exc:
            collection_error = exc

            entries.append({
                "episode_index": spec.episode_index,
                "seed": spec.seed,
                "scenario": spec.scenario,
                "source": "expert",
                "status": "error",
                "success": False,
                "failure_reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "dataset_path": str(path),
                "frame_count": 0,
            })

            _save_summary()
            break

        _save_summary()

    summary = _save_summary()

    if collection_error is not None:
        raise collection_error.with_traceback(
            collection_error.__traceback__
        )

    return summary


def build_collection_specs(
    *,
    count: int,
    start_seed: int,
    timeout_s: float,
) -> tuple[EpisodeSpec, ...]:
    """Generate deterministic seeded expert collection specifications."""
    import math

    from robustlearn.week10_benchmark import (
        benchmark_scenario_for_seed,
    )

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError("count must be a positive integer")

    if (
        isinstance(start_seed, bool)
        or not isinstance(start_seed, int)
        or start_seed < 0
    ):
        raise ValueError(
            "start_seed must be a non-negative integer"
        )

    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError(
            "timeout_s must be finite and positive"
        )

    return tuple(
        EpisodeSpec(
            episode_index=seed,
            seed=seed,
            scenario=benchmark_scenario_for_seed(seed),
            timeout_s=float(timeout_s),
        )
        for seed in range(
            start_seed,
            start_seed + count,
        )
    )


def parse_args(argv: list[str] | None = None):
    """Parse scripted-expert collection CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Collect reproducible Week 11 scripted-expert demonstrations."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First deterministic seed and episode index.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=90.0,
        help="Timeout per episode in seconds.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("build/week11_dataset"),
        help="Directory containing validated dataset episodes.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=Path("build/week11_captures"),
        help="Directory containing raw expert captures.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("build/week11_collection_summary.json"),
        help="Output path for the machine-readable collection summary.",
    )
    parser.add_argument(
        "--config-id",
        required=True,
        help="Reproducible configuration identifier.",
    )
    parser.add_argument(
        "--git-commit",
        required=True,
        help="Git commit identifying the source implementation.",
    )
    parser.add_argument(
        "--task",
        default="peg_insertion",
        help="Task name stored in each episode.",
    )
    parser.add_argument(
        "--base-domain-id",
        type=int,
        default=190,
        help="Base ROS_DOMAIN_ID.",
    )
    parser.add_argument(
        "--domain-pool-size",
        type=int,
        default=40,
        help="Number of reusable ROS domain IDs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the episode schedule without launching ROS2.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the scripted-expert collection CLI."""
    import json

    args = parse_args(argv)

    specs = build_collection_specs(
        count=args.count,
        start_seed=args.start_seed,
        timeout_s=args.timeout_s,
    )

    if args.base_domain_id < 0:
        raise ValueError(
            "base-domain-id must be non-negative"
        )

    if args.domain_pool_size <= 0:
        raise ValueError(
            "domain-pool-size must be positive"
        )

    if args.dry_run:
        schedule = {
            "mode": "dry-run",
            "requested_count": len(specs),
            "episodes": [
                {
                    "episode_index": spec.episode_index,
                    "seed": spec.seed,
                    "scenario": spec.scenario,
                    "timeout_s": spec.timeout_s,
                }
                for spec in specs
            ],
        }

        print(json.dumps(schedule, indent=2))
        return 0

    summary = collect_episodes(
        specs,
        dataset_root=args.dataset_root,
        capture_root=args.capture_root,
        summary_path=args.summary_path,
        git_commit=args.git_commit,
        config_id=args.config_id,
        task=args.task,
        base_domain_id=args.base_domain_id,
        domain_pool_size=args.domain_pool_size,
    )

    print(json.dumps({
        "requested_count": summary["requested_count"],
        "skipped_count": summary["skipped_count"],
        "collected_success_count": summary[
            "collected_success_count"
        ],
        "collected_failure_count": summary[
            "collected_failure_count"
        ],
        "collection_error_count": summary[
            "collection_error_count"
        ],
        "summary_path": str(args.summary_path),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
