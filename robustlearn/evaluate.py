"""Command-line entry point for seeded Panda insertion evaluation."""

import argparse
import json
from collections.abc import Sequence

from robustlearn.evaluation import evaluate_scripted_episodes


def _parse_seed_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated explicit seed list."""
    try:
        seeds = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc

    if not seeds:
        raise argparse.ArgumentTypeError(
            "seeds must contain at least one integer"
        )

    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "seeds must be non-negative"
        )

    return seeds


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run headless seeded Panda insertion evaluation.",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=2026,
        help="first seed for consecutive evaluation episodes",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="number of consecutive seeded episodes to run",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        help="explicit comma-separated seed list; overrides base/count mode",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation CLI and print a JSON report."""
    args = build_parser().parse_args(argv)

    if args.seeds is not None:
        report = evaluate_scripted_episodes(
            seeds=args.seeds,
        )
    else:
        report = evaluate_scripted_episodes(
            base_seed=args.base_seed,
            episode_count=args.episodes,
        )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
