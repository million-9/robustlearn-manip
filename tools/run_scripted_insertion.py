"""Run the deterministic Panda insertion script from the command line."""

import argparse
from typing import cast

import numpy as np

from robustlearn.scripted_insertion import run_scripted_insertion


def _format_vector(value: object) -> str:
    """Format one recorded 3-vector."""
    vector = np.asarray(
        value,
        dtype=np.float64,
    )

    return np.array2string(
        vector,
        precision=3,
        suppress_small=True,
    )


def main() -> None:
    """Run the scripted insertion sequence and print diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run the deterministic Panda insertion sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="deterministic environment seed",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="maximum environment steps",
    )

    args = parser.parse_args()

    diagnostics = run_scripted_insertion(
        seed=args.seed,
        max_episode_steps=args.max_steps,
    )

    for record in diagnostics:
        success = cast(bool, record["success"])
        failure = cast(bool, record["failure"])
        truncated = cast(bool, record["truncated"])

        state = (
            "SUCCESS"
            if success
            else "FAILURE"
            if failure
            else "TRUNCATED"
            if truncated
            else "IN_PROGRESS"
        )

        print(
            f"step={cast(int, record['step']):03d}  "
            f"time={cast(float, record['simulation_time']):7.3f}s  "
            f"state={state:11s}  "
            f"lateral="
            f"{cast(float, record['lateral_error']) * 1000:7.3f} mm  "
            f"depth="
            f"{cast(float, record['insertion_depth']) * 1000:7.3f} mm  "
            f"force={_format_vector(record['wrist_force'])} N  "
            f"torque={_format_vector(record['wrist_torque'])} Nm"
        )

    if not diagnostics:
        raise RuntimeError("Scripted insertion produced no diagnostics")

    final = diagnostics[-1]

    if final["success"] is True:
        print("\nScripted insertion completed successfully.")
    elif final["failure"] is True:
        raise RuntimeError("Scripted insertion ended in task failure")
    else:
        raise RuntimeError(
            "Scripted insertion ended without reaching task success"
        )


if __name__ == "__main__":
    main()
