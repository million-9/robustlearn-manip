"""Read-only MuJoCo insertion viewer synchronized with ROS joint states."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import rclpy
from sensor_msgs.msg import JointState

from robustlearn.sim.shadow_kinematics import (
    ShadowInsertionKinematics,
)
from robustlearn.sim.viewer_joint_states import extract_panda_joint_positions

STALE_TIMEOUT_S = 1.0
STARTUP_TIMEOUT_S = 30.0


def main() -> None:
    """Visualize observed ROS robot motion without stepping physics."""
    shadow = ShadowInsertionKinematics()
    model = shadow.model
    data = shadow.data

    addresses = []

    for index in range(1, 8):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"joint{index}",
        )
        if joint_id < 0:
            raise RuntimeError(f"Missing MuJoCo joint{index}")

        addresses.append(int(model.jnt_qposadr[joint_id]))

    rclpy.init()
    node = rclpy.create_node("robustlearn_live_mujoco_viewer")

    latest = None
    last_received = 0.0

    def on_joint_state(message: JointState) -> None:
        nonlocal latest, last_received

        positions = extract_panda_joint_positions(
            message.name, message.position
        )

        if positions is None:
            return

        latest = positions
        last_received = time.monotonic()

    _subscription = node.create_subscription(
        JointState,
        "/joint_states",
        on_joint_state,
        10,
    )

    try:
        print("Waiting for live ROS joint feedback...", flush=True)

        deadline = time.monotonic() + STARTUP_TIMEOUT_S

        while latest is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)

        if latest is None:
            raise RuntimeError(
                "No valid Panda joint feedback received"
            )

        print("PASS: Live joint feedback available", flush=True)
        print("Loaded geometries:", model.ngeom, flush=True)
        print("Opening synchronized MuJoCo viewer...", flush=True)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (0.55, 0.0, 0.45)
                viewer.cam.distance = 1.2
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

            stale_reported = False

            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(node, timeout_sec=0.05)

                age = time.monotonic() - last_received

                if latest is not None and age <= STALE_TIMEOUT_S:
                    with viewer.lock():
                        for address, position in zip(
                            addresses, latest, strict=True
                        ):
                            data.qpos[address] = position

                        data.qvel.fill(0.0)

                        # Kinematic reconstruction only.
                        # Physics is owned by the ROS hardware plugin.
                        mujoco.mj_forward(model, data)

                    viewer.sync()
                    stale_reported = False

                elif not stale_reported:
                    print(
                        "WARNING: ROS feedback stale; "
                        "display is not live",
                        flush=True,
                    )
                    stale_reported = True

                time.sleep(0.02)

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Viewer closed")


if __name__ == "__main__":
    main()
