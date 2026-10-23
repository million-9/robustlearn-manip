#!/usr/bin/env python3
"""Run the deterministic Week 9 clean hybrid-acceptance scenario matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCENARIOS = tuple(
    f'week9_case_{index:02d}'
    for index in range(1, 21)
)

RESULT_PREFIX = 'WEEK9_RESULT '

MINIMUM_PASS_COUNT = 18
TARGET_PASS_COUNT = 19

REQUIRED_FINITE_FIELDS = (
    'handoff_error_rad',
    'final_lateral_error_m',
    'final_axial_offset_m',
    'final_insertion_depth_m',
    'final_axial_force_n',
    'final_lateral_force_n',
    'peak_axial_force_n',
    'peak_lateral_force_n',
)


def repository_root() -> Path:
    """Return the repository root containing this runner."""
    return Path(__file__).resolve().parents[1]


def scenario_test_path() -> Path:
    """Return the Week 9 single-scenario launch-test path."""
    return (
        repository_root()
        / 'ros2_ws'
        / 'src'
        / 'robustlearn_moveit_config'
        / 'test'
        / 'test_week9_acceptance_scenario_launch.py'
    )


def parse_result(
    output: str,
) -> dict[str, Any] | None:
    """Parse the final structured Week 9 result from launch-test output."""
    result = None

    for line in output.splitlines():
        stripped = line.strip()

        if not stripped.startswith(
            RESULT_PREFIX
        ):
            continue

        payload = stripped[
            len(RESULT_PREFIX):
        ]

        parsed = json.loads(
            payload
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                'WEEK9_RESULT payload must be a JSON object'
            )

        result = parsed

    return result


def classify_failure(
    output: str,
    *,
    timed_out: bool,
) -> str:
    """Classify a scenario that did not produce a passing result."""
    if timed_out:
        return 'scenario_timeout'

    if 'Hybrid insertion failed:' in output:
        if (
            'phase=planning'
            in output
            and 'phase=executing'
            not in output
        ):
            return 'moveit_planning_failure'

        if (
            'phase=executing'
            in output
            and 'phase=verifying_handoff'
            not in output
        ):
            return 'moveit_execution_failure'

        if (
            'phase=verifying_handoff'
            in output
            and 'phase=arming_expert'
            not in output
        ):
            return 'handoff_failure'

        if (
            'phase=waiting_for_expert'
            in output
        ):
            return 'local_expert_failure'

        return 'hybrid_coordinator_failure'

    if (
        'failed|'
        in output
        or 'Local insertion expert aborted:'
        in output
    ):
        return 'local_expert_failure'

    if (
        'FAILED'
        in output
        or 'AssertionError'
        in output
    ):
        return 'acceptance_probe_failure'

    return 'launch_failure'


def failure_record(
    scenario: str,
    output: str,
    *,
    returncode: int | None,
    timed_out: bool,
) -> dict[str, Any]:
    """Create a structured result for a failed scenario."""
    planning_seen = (
        'Hybrid insertion phase=planning'
        in output
    )

    execution_seen = (
        'Hybrid insertion phase=executing'
        in output
    )

    handoff_seen = (
        'Hybrid insertion phase=verifying_handoff'
        in output
    )

    expert_success = (
        'Hybrid insertion completed: '
        'local insertion expert reported success: '
        'canonical insertion task reached success'
        in output
    )

    return {
        'scenario': scenario,
        'success': False,
        'moveit_planning_success': (
            planning_seen
            and execution_seen
        ),
        'moveit_execution_success': (
            execution_seen
            and handoff_seen
        ),
        'handoff_error_rad': None,
        'local_expert_success': expert_success,
        'final_lateral_error_m': None,
        'final_axial_offset_m': None,
        'final_insertion_depth_m': None,
        'final_axial_force_n': None,
        'final_lateral_force_n': None,
        'peak_axial_force_n': None,
        'peak_lateral_force_n': None,
        'failure_category': classify_failure(
            output,
            timed_out=timed_out,
        ),
        'returncode': returncode,
    }


def validate_result(
    result: dict[str, Any],
) -> None:
    """Validate the structured result contract for one scenario."""
    scenario = result.get(
        'scenario'
    )

    if scenario not in SCENARIOS:
        raise ValueError(
            f'Unexpected Week 9 scenario in result: {scenario!r}'
        )

    success = result.get(
        'success'
    )

    if not isinstance(
        success,
        bool,
    ):
        raise ValueError(
            f'{scenario}: success must be boolean'
        )

    if not success:
        return

    for field in REQUIRED_FINITE_FIELDS:
        value = result.get(
            field
        )

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f'{scenario}: {field} must be numeric'
            )

        if not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f'{scenario}: {field} must be finite'
            )

    boolean_fields = (
        'moveit_planning_success',
        'moveit_execution_success',
        'local_expert_success',
    )

    for field in boolean_fields:
        if result.get(
            field
        ) is not True:
            raise ValueError(
                f'{scenario}: successful result requires {field}=true'
            )

    if result.get(
        'failure_category'
    ) is not None:
        raise ValueError(
            f'{scenario}: successful result must not have a failure category'
        )


def run_scenario(
    scenario: str,
    *,
    domain_id: int,
    timeout_s: float,
    model_path: str | None = None,
    initial_keyframe: str | None = None,
    expert_capture_path: str | None = None,
) -> dict[str, Any]:
    """Run one isolated Week 9 scenario and return its result."""
    test_path = scenario_test_path()

    if not test_path.is_file():
        raise RuntimeError(
            f'Week 9 scenario test not found: {test_path}'
        )

    environment = os.environ.copy()

    environment[
        'ROBUSTLEARN_WEEK9_SCENARIO'
    ] = scenario

    environment[
        'ROS_DOMAIN_ID'
    ] = str(
        domain_id
    )

    if model_path is not None:
        environment[
            'ROBUSTLEARN_WEEK10_MODEL_PATH'
        ] = model_path

    if initial_keyframe is not None:
        environment[
            'ROBUSTLEARN_WEEK10_INITIAL_KEYFRAME'
        ] = initial_keyframe

    if expert_capture_path is not None:
        if not expert_capture_path.strip():
            raise ValueError(
                "expert_capture_path must be non-empty"
            )

        environment[
            'ROBUSTLEARN_WEEK11_EXPERT_CAPTURE_PATH'
        ] = expert_capture_path
    else:
        environment.pop(
            'ROBUSTLEARN_WEEK11_EXPERT_CAPTURE_PATH',
            None,
        )

    command = (
        sys.executable,
        '-m',
        'launch_testing.launch_test',
        str(test_path),
    )

    print(
        f'\n=== {scenario} '
        f'(ROS_DOMAIN_ID={domain_id}) ===',
        flush=True,
    )

    timed_out = False

    try:
        completed = subprocess.run(
            command,
            cwd=repository_root(),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )

        output = completed.stdout
        returncode = completed.returncode

    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None

        if isinstance(
            exc.stdout,
            bytes,
        ):
            output = exc.stdout.decode(
                errors='replace'
            )

        elif exc.stdout is None:
            output = ''

        else:
            output = exc.stdout

    parsed = None

    try:
        parsed = parse_result(
            output
        )
    except (
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(
            f'{scenario}: invalid structured result: {exc}',
            flush=True,
        )

    if (
        parsed is not None
        and returncode == 0
        and not timed_out
    ):
        try:
            validate_result(
                parsed
            )
        except ValueError as exc:
            print(
                f'{scenario}: result validation failed: {exc}',
                flush=True,
            )
        else:
            print(
                RESULT_PREFIX
                + json.dumps(
                    parsed,
                    sort_keys=True,
                ),
                flush=True,
            )

            return parsed

    failed = failure_record(
        scenario,
        output,
        returncode=returncode,
        timed_out=timed_out,
    )

    print(
        RESULT_PREFIX
        + json.dumps(
            failed,
            sort_keys=True,
        ),
        flush=True,
    )

    print(
        f'--- {scenario} launch-test tail ---',
        flush=True,
    )

    output_lines = output.splitlines()

    for line in output_lines[
        -40:
    ]:
        print(
            line,
            flush=True,
        )

    return failed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Run the deterministic Week 9 clean hybrid-acceptance matrix.'
        )
    )

    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=SCENARIOS,
        default=list(
            SCENARIOS
        ),
        help=(
            'Scenario names to run. Defaults to all 20 Week 9 cases.'
        ),
    )

    parser.add_argument(
        '--base-domain-id',
        type=int,
        default=190,
        help=(
            'ROS domain ID used for the first scenario. '
            'Each following scenario receives the next integer.'
        ),
    )

    parser.add_argument(
        '--timeout-s',
        type=float,
        default=90.0,
        help=(
            'Wall-clock timeout for each isolated scenario.'
        ),
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=(
            repository_root()
            / 'build'
            / 'week9_acceptance_results.json'
        ),
        help=(
            'Path for the complete JSON acceptance record.'
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the requested Week 9 scenario matrix."""
    args = parse_args()

    if (
        not math.isfinite(
            args.timeout_s
        )
        or args.timeout_s <= 0.0
    ):
        raise ValueError(
            '--timeout-s must be finite and positive'
        )

    if args.base_domain_id < 0:
        raise ValueError(
            '--base-domain-id must be non-negative'
        )

    scenarios = tuple(
        args.scenarios
    )

    results = []

    for offset, scenario in enumerate(
        scenarios
    ):
        result = run_scenario(
            scenario,
            domain_id=(
                args.base_domain_id
                + offset
            ),
            timeout_s=args.timeout_s,
        )

        results.append(
            result
        )

    success_count = sum(
        1
        for result in results
        if result.get(
            'success'
        ) is True
    )

    scenario_count = len(
        results
    )

    success_rate = (
        success_count
        / scenario_count
        if scenario_count
        else 0.0
    )

    full_matrix = (
        scenarios
        == SCENARIOS
    )

    minimum_gate_passed = (
        success_count
        >= MINIMUM_PASS_COUNT
        if full_matrix
        else None
    )

    target_gate_passed = (
        success_count
        >= TARGET_PASS_COUNT
        if full_matrix
        else None
    )

    summary = {
        'scenario_count': scenario_count,
        'success_count': success_count,
        'failure_count': (
            scenario_count
            - success_count
        ),
        'success_rate': success_rate,
        'full_matrix': full_matrix,
        'minimum_required_successes': (
            MINIMUM_PASS_COUNT
        ),
        'target_successes': (
            TARGET_PASS_COUNT
        ),
        'minimum_gate_passed': (
            minimum_gate_passed
        ),
        'target_gate_passed': (
            target_gate_passed
        ),
        'results': results,
    }

    output_path = args.output.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + '\n'
    )

    print(
        '\n=== WEEK 9 ACCEPTANCE SUMMARY ==='
    )

    print(
        f'Scenarios: {scenario_count}'
    )

    print(
        f'Successes: {success_count}'
    )

    print(
        f'Failures: {scenario_count - success_count}'
    )

    print(
        f'Success rate: {success_rate:.1%}'
    )

    if full_matrix:
        print(
            'Minimum gate: '
            f'{success_count}/20 '
            f'({"PASS" if minimum_gate_passed else "FAIL"})'
        )

        print(
            'Target gate: '
            f'{success_count}/20 '
            f'({"PASS" if target_gate_passed else "NOT MET"})'
        )

    else:
        print(
            'Gate enforcement skipped: '
            'partial scenario selection.'
        )

    print(
        f'Results: {output_path}'
    )

    if full_matrix:
        return (
            0
            if minimum_gate_passed
            else 1
        )

    return (
        0
        if success_count == scenario_count
        else 1
    )


if __name__ == '__main__':
    raise SystemExit(
        main()
    )
