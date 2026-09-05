"""Tests for optional RGB and depth camera rendering."""

import sys

import mujoco
import numpy as np
import pytest

from robustlearn.sim import MuJoCoSimulation, load_insertion_model
from robustlearn.sim.rendering import (
    WORKCELL_CAMERA_HEIGHT,
    WORKCELL_CAMERA_NAME,
    WORKCELL_CAMERA_WIDTH,
    CameraRenderer,
    CameraRenderingUnavailableError,
    RGBDepthFrame,
)


def test_workcell_camera_definition() -> None:
    model = load_insertion_model()

    camera_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            WORKCELL_CAMERA_NAME,
        )
    )

    assert camera_id >= 0

    np.testing.assert_array_equal(
        model.cam_resolution[camera_id],
        np.asarray(
            [
                WORKCELL_CAMERA_WIDTH,
                WORKCELL_CAMERA_HEIGHT,
            ],
            dtype=np.int32,
        ),
    )

    np.testing.assert_allclose(
        model.cam_pos[camera_id],
        np.asarray(
            [1.05, -0.75, 0.95],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )

    assert model.cam_fovy[camera_id] == pytest.approx(45.0)


def test_camera_renderer_resolves_camera_without_creating_gl_context() -> None:
    model = load_insertion_model()

    renderer = CameraRenderer(model)

    assert renderer.camera_name == WORKCELL_CAMERA_NAME
    assert renderer.camera_id >= 0
    assert renderer.width == WORKCELL_CAMERA_WIDTH
    assert renderer.height == WORKCELL_CAMERA_HEIGHT
    assert not renderer.renderer_initialized


def test_camera_renderer_rejects_missing_camera() -> None:
    model = load_insertion_model()

    with pytest.raises(
        RuntimeError,
        match="does not contain camera",
    ):
        CameraRenderer(
            model,
            camera_name="missing_camera",
        )


def test_camera_renderer_rejects_resolution_mismatch() -> None:
    model = load_insertion_model()

    with pytest.raises(
        RuntimeError,
        match="stores resolution",
    ):
        CameraRenderer(
            model,
            width=160,
            height=120,
        )


def test_unconfigured_linux_headless_rendering_fails_before_gl_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "linux":
        pytest.skip("Linux-specific rendering preflight test")

    monkeypatch.delenv(
        "DISPLAY",
        raising=False,
    )
    monkeypatch.delenv(
        "WAYLAND_DISPLAY",
        raising=False,
    )
    monkeypatch.delenv(
        "MUJOCO_GL",
        raising=False,
    )

    sim = MuJoCoSimulation()
    sim.reset(seed=2026)

    renderer = CameraRenderer(sim.model)

    with pytest.raises(
        CameraRenderingUnavailableError,
        match="No Linux display is available",
    ):
        renderer.render(sim.data)

    assert not renderer.renderer_initialized


def render_or_skip(
    renderer: CameraRenderer,
    sim: MuJoCoSimulation,
) -> RGBDepthFrame:
    """Render when a GL backend exists; otherwise skip this GL-only check."""
    try:
        return renderer.render(sim.data)
    except CameraRenderingUnavailableError as exc:
        pytest.skip(
            f"MuJoCo rendering backend unavailable: {exc}"
        )


def test_rgb_and_depth_output_structure() -> None:
    sim = MuJoCoSimulation()
    sim.reset(seed=2026)

    renderer = CameraRenderer(sim.model)

    try:
        frame = render_or_skip(
            renderer,
            sim,
        )

        assert renderer.renderer_initialized

        assert frame.rgb.shape == (
            WORKCELL_CAMERA_HEIGHT,
            WORKCELL_CAMERA_WIDTH,
            3,
        )
        assert frame.rgb.dtype == np.uint8
        assert np.all(np.isfinite(frame.rgb))

        assert frame.depth.shape == (
            WORKCELL_CAMERA_HEIGHT,
            WORKCELL_CAMERA_WIDTH,
        )
        assert frame.depth.dtype == np.float32
        assert np.all(np.isfinite(frame.depth))
        assert np.all(frame.depth > 0.0)

    finally:
        renderer.close()

    assert not renderer.renderer_initialized


def test_repeated_rendering_from_same_reset_is_structurally_consistent() -> None:
    sim = MuJoCoSimulation()
    sim.reset(seed=2026)

    renderer = CameraRenderer(sim.model)

    try:
        first = render_or_skip(
            renderer,
            sim,
        )
        second = render_or_skip(
            renderer,
            sim,
        )

        assert first.rgb.shape == second.rgb.shape
        assert first.rgb.dtype == second.rgb.dtype

        assert first.depth.shape == second.depth.shape
        assert first.depth.dtype == second.depth.dtype

        assert np.all(np.isfinite(first.rgb))
        assert np.all(np.isfinite(second.rgb))

        assert np.all(np.isfinite(first.depth))
        assert np.all(np.isfinite(second.depth))

    finally:
        renderer.close()
