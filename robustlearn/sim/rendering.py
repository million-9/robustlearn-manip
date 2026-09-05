"""Optional RGB and depth rendering for the Panda insertion task."""

import os
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]

WORKCELL_CAMERA_NAME = "workcell_oblique"

WORKCELL_CAMERA_WIDTH = 320
WORKCELL_CAMERA_HEIGHT = 240


class CameraRenderingUnavailableError(RuntimeError):
    """Raised when MuJoCo cannot create an offscreen renderer."""


def _known_linux_headless_context_unavailable() -> bool:
    """Detect a Linux process with no configured rendering context source."""
    if sys.platform != "linux":
        return False

    backend = os.environ.get(
        "MUJOCO_GL",
        "",
    ).strip().lower()

    if backend in {
        "egl",
        "osmesa",
    }:
        return False

    return not (
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )


@dataclass(frozen=True)
class RGBDepthFrame:
    """Independent RGB and depth images from one camera."""

    rgb: UInt8Array
    depth: Float32Array


class CameraRenderer:
    """Render project cameras without exposing raw MuJoCo renderer state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str = WORKCELL_CAMERA_NAME,
        width: int = WORKCELL_CAMERA_WIDTH,
        height: int = WORKCELL_CAMERA_HEIGHT,
    ) -> None:
        """Resolve a camera while deferring OpenGL creation until rendering."""
        if width <= 0:
            raise ValueError("Camera render width must be positive")

        if height <= 0:
            raise ValueError("Camera render height must be positive")

        camera_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
        )

        if camera_id < 0:
            raise RuntimeError(
                f"MuJoCo model does not contain camera {camera_name!r}"
            )

        stored_resolution = np.asarray(
            model.cam_resolution[camera_id],
            dtype=np.int32,
        )

        expected_resolution = np.asarray(
            [width, height],
            dtype=np.int32,
        )

        if not np.array_equal(
            stored_resolution,
            expected_resolution,
        ):
            raise RuntimeError(
                f"Camera {camera_name!r} stores resolution "
                f"{stored_resolution.tolist()}, expected "
                f"{expected_resolution.tolist()}"
            )

        self._model = model
        self._camera_name = camera_name
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._renderer: mujoco.Renderer | None = None

    @property
    def camera_name(self) -> str:
        """Return the stable MuJoCo camera name."""
        return self._camera_name

    @property
    def camera_id(self) -> int:
        """Return the resolved compiled camera identifier."""
        return self._camera_id

    @property
    def width(self) -> int:
        """Return the RGB/depth image width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Return the RGB/depth image height in pixels."""
        return self._height

    @property
    def renderer_initialized(self) -> bool:
        """Report whether an offscreen rendering context has been created."""
        return self._renderer is not None

    def _get_renderer(self) -> mujoco.Renderer:
        """Create the MuJoCo renderer only when an image is requested."""
        if self._renderer is None:
            if _known_linux_headless_context_unavailable():
                raise CameraRenderingUnavailableError(
                    "No Linux display is available and MUJOCO_GL is not "
                    "configured for a headless backend. Configure "
                    "MUJOCO_GL=egl or MUJOCO_GL=osmesa before starting "
                    "Python, or provide an X11/Wayland display."
                )

            try:
                self._renderer = mujoco.Renderer(
                    self._model,
                    height=self._height,
                    width=self._width,
                )
            except (mujoco.FatalError, RuntimeError) as exc:
                raise CameraRenderingUnavailableError(
                    "MuJoCo could not create an offscreen rendering context"
                ) from exc

        return self._renderer

    def render_rgb(
        self,
        data: mujoco.MjData,
    ) -> UInt8Array:
        """Render an independent RGB image."""
        renderer = self._get_renderer()

        renderer.disable_depth_rendering()

        renderer.update_scene(
            data,
            camera=self._camera_name,
        )

        image = renderer.render()

        expected_shape = (
            self._height,
            self._width,
            3,
        )

        if image.shape != expected_shape:
            raise RuntimeError(
                f"RGB image has shape {image.shape}; "
                f"expected {expected_shape}"
            )

        if image.dtype != np.uint8:
            raise RuntimeError(
                f"RGB image has dtype {image.dtype}; expected uint8"
            )

        return np.asarray(
            image,
            dtype=np.uint8,
        ).copy()

    def render_depth(
        self,
        data: mujoco.MjData,
    ) -> Float32Array:
        """Render an independent metric depth image."""
        renderer = self._get_renderer()

        renderer.enable_depth_rendering()

        try:
            renderer.update_scene(
                data,
                camera=self._camera_name,
            )

            image = renderer.render()
        finally:
            renderer.disable_depth_rendering()

        expected_shape = (
            self._height,
            self._width,
        )

        if image.shape != expected_shape:
            raise RuntimeError(
                f"Depth image has shape {image.shape}; "
                f"expected {expected_shape}"
            )

        if image.dtype != np.float32:
            raise RuntimeError(
                f"Depth image has dtype {image.dtype}; expected float32"
            )

        return np.asarray(
            image,
            dtype=np.float32,
        ).copy()

    def render(
        self,
        data: mujoco.MjData,
    ) -> RGBDepthFrame:
        """Render independent RGB and depth images."""
        return RGBDepthFrame(
            rgb=self.render_rgb(data),
            depth=self.render_depth(data),
        )

    def close(self) -> None:
        """Release the optional MuJoCo rendering context."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
