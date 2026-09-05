"""Launch the RobustLearn Panda insertion task-debug viewer."""

import time

import mujoco
import mujoco.viewer  # type: ignore[import-untyped]
import numpy as np

from robustlearn.sim.debug import insertion_debug_snapshot
from robustlearn.sim.simulation import MuJoCoSimulation


def _format_vector(vector: np.ndarray) -> str:
    """Format a 3-vector compactly for terminal diagnostics."""
    return np.array2string(
        vector,
        precision=3,
        suppress_small=True,
    )


def main() -> None:
    """Launch the interactive Panda insertion debug viewer."""
    simulation = MuJoCoSimulation()
    reset_requested = False

    def key_callback(keycode: int) -> None:
        """Handle developer controls from the MuJoCo viewer."""
        nonlocal reset_requested

        if chr(keycode).lower() == "r":
            reset_requested = True

    print("Panda insertion debug viewer")
    print("Controls: R = reset")

    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.lookat[:] = (0.55, 0.0, 0.45)
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

        next_diagnostic_time = 0.0

        while viewer.is_running():
            step_start = time.time()

            if reset_requested:
                simulation.reset()
                reset_requested = False
                next_diagnostic_time = 0.0
                print("\nReset to canonical insertion state.")

            simulation.step()

            if simulation.data.time >= next_diagnostic_time:
                debug = insertion_debug_snapshot(simulation)

                state = (
                    "SUCCESS"
                    if debug.success
                    else "FAILURE"
                    if debug.failure
                    else "IN_PROGRESS"
                )

                print(
                    "\r"
                    f"time={debug.simulation_time:7.3f}s  "
                    f"state={state:11s}  "
                    f"lateral={debug.lateral_error * 1000:7.3f} mm  "
                    f"axial={debug.axial_offset * 1000:7.3f} mm  "
                    f"depth={debug.insertion_depth * 1000:7.3f} mm  "
                    f"force={_format_vector(debug.wrist_force)} N  "
                    f"torque={_format_vector(debug.wrist_torque)} Nm",
                    end="",
                    flush=True,
                )

                next_diagnostic_time += 0.1

            viewer.sync()

            elapsed = time.time() - step_start
            remaining = simulation.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

    print()


if __name__ == "__main__":
    main()
