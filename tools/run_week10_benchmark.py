#!/usr/bin/env python3
"""Run the reproducible Week 10 classical-expert benchmark."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

from robustlearn.episode_manager import EpisodeSpec
from robustlearn.week10_benchmark import (
    BENCHMARK_EPISODE_COUNT,
    BenchmarkEpisodeRecord,
    benchmark_domain_for_episode,
    benchmark_keyframe_name,
    benchmark_record_from_week9_result,
    benchmark_scenario_for_seed,
    build_benchmark_config,
    build_benchmark_report,
    build_benchmark_start_state,
    build_seeded_benchmark_mjcf,
    load_benchmark_base_joint_positions,
    write_benchmark_checkpoint,
    write_benchmark_report,
)
from tools.run_week9_acceptance import run_scenario


def repository_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Week 10 reproducible classical-expert benchmark."
        )
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=90.0,
        help="Wall-clock timeout for each isolated episode.",
    )
    parser.add_argument(
        "--base-domain-id",
        type=int,
        default=190,
        help="Base ROS domain ID.",
    )
    parser.add_argument(
        "--domain-pool-size",
        type=int,
        default=40,
        help="Number of reusable ROS domain IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N episodes for smoke validation. "
            "Omit for the formal 200-episode benchmark."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root()
            / "build"
            / "week10_benchmark_results.json"
        ),
        help="Output JSON artifact path.",
    )

    return parser.parse_args()


def run_episode(
    spec: EpisodeSpec,
    *,
    base_domain_id: int,
    domain_pool_size: int,
    expert_capture_path: Path | None = None,
) -> BenchmarkEpisodeRecord:
    """Run one seeded real hybrid episode and preserve rich Week 9 metrics."""
    domain_id = benchmark_domain_for_episode(
        episode_index=spec.episode_index,
        base_domain_id=base_domain_id,
        domain_pool_size=domain_pool_size,
    )

    source_model = (
        repository_root()
        / "robot_description"
        / "mjcf"
        / "insertion"
        / "panda_insertion.xml"
    )

    base_joint_positions = load_benchmark_base_joint_positions(
        source_path=source_model,
        scenario=spec.scenario,
    )

    start_state = build_benchmark_start_state(
        seed=spec.seed,
        scenario=spec.scenario,
        base_joint_positions_rad=base_joint_positions,
    )

    initial_keyframe = benchmark_keyframe_name(
        spec.seed
    )

    with tempfile.TemporaryDirectory(
        prefix="robustlearn_week10_"
    ) as temporary_directory:
        temporary_model = (
            Path(temporary_directory)
            / "panda_insertion_seeded.xml"
        )

        build_seeded_benchmark_mjcf(
            source_path=source_model,
            output_path=temporary_model,
            start_state=start_state,
        )

        start = time.monotonic()

        result = run_scenario(
            spec.scenario,
            domain_id=domain_id,
            timeout_s=spec.timeout_s,
            model_path=str(temporary_model),
            initial_keyframe=initial_keyframe,
            expert_capture_path=(
                str(expert_capture_path)
                if expert_capture_path is not None
                else None
            ),
        )

        completion_time_s = time.monotonic() - start

    return benchmark_record_from_week9_result(
        episode_index=spec.episode_index,
        seed=spec.seed,
        completion_time_s=completion_time_s,
        result=result,
        start_state=start_state,
    )


def main() -> int:
    """Run smoke or full Week 10 benchmark."""
    args = parse_args()

    if (
        not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0.0
    ):
        raise ValueError(
            "--timeout-s must be finite and positive"
        )

    if args.base_domain_id < 0:
        raise ValueError(
            "--base-domain-id must be non-negative"
        )

    if args.domain_pool_size <= 0:
        raise ValueError(
            "--domain-pool-size must be positive"
        )

    if (
        args.limit is not None
        and (
            args.limit <= 0
            or args.limit > BENCHMARK_EPISODE_COUNT
        )
    ):
        raise ValueError(
            "--limit must be between 1 and 200"
        )

    config = build_benchmark_config(
        timeout_s=args.timeout_s,
        base_domain_id=args.base_domain_id,
        domain_pool_size=args.domain_pool_size,
    )

    run_count = (
        BENCHMARK_EPISODE_COUNT
        if args.limit is None
        else args.limit
    )

    specs = tuple(
        EpisodeSpec(
            episode_index=index,
            seed=config.seeds[index],
            scenario=benchmark_scenario_for_seed(
                config.seeds[index]
            ),
            timeout_s=config.timeout_s,
        )
        for index in range(run_count)
    )

    records = []

    for spec in specs:
        print(
            f"\n=== WEEK 10 EPISODE "
            f"{spec.episode_index + 1}/{run_count} "
            f"seed={spec.seed} "
            f"scenario={spec.scenario} ===",
            flush=True,
        )

        record = run_episode(
            spec,
            base_domain_id=config.base_domain_id,
            domain_pool_size=config.domain_pool_size,
        )

        records.append(record)

        print(
            "WEEK10_EPISODE "
            + json.dumps(
                record.to_dict(),
                sort_keys=True,
            ),
            flush=True,
        )

        checkpoint_path = write_benchmark_checkpoint(
            args.output.resolve(),
            config=config,
            episodes=records,
            requested_episode_count=run_count,
        )

        print(
            f"WEEK10_CHECKPOINT "
            f"{len(records)}/{run_count} "
            f"{checkpoint_path}",
            flush=True,
        )

    if args.limit is not None:
        smoke_output = {
            "mode": "smoke",
            "requested_episode_count": run_count,
            "config": config.to_dict(),
            "episodes": [
                record.to_dict()
                for record in records
            ],
        }

        output_path = args.output.resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                smoke_output,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        print(
            f"\nSmoke artifact written to {output_path}"
        )
        return 0

    report = build_benchmark_report(
        config=config,
        episodes=records,
    )

    output_path = write_benchmark_report(
        args.output.resolve(),
        report,
    )

    print("\n=== WEEK 10 BENCHMARK SUMMARY ===")
    print(
        json.dumps(
            report.summary.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"\nBenchmark artifact written to {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
