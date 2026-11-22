"""CUDA integration tests for the Behavior Cloning policy."""

from pathlib import Path

import pytest
import torch

from robustlearn.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    load_bc_policy,
    resolve_bc_device,
    save_bc_policy,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


def test_bc_cuda_inference() -> None:
    """Verify that BC inference executes on a real CUDA device."""
    device = resolve_bc_device("cuda")
    model = BCPolicy(BCPolicyConfig(seed=42)).to(device).eval()

    observations = torch.randn(8, 23, device=device)

    with torch.inference_mode():
        predictions = model(observations)

    torch.cuda.synchronize()

    assert predictions.shape == (8, 7)
    assert predictions.device.type == "cuda"
    assert torch.isfinite(predictions).all()


def test_bc_cuda_checkpoint_round_trip(tmp_path: Path) -> None:
    """Verify checkpoint restoration preserves CUDA predictions."""
    config = BCPolicyConfig(seed=42)
    model = BCPolicy(config).to("cuda").eval()

    observations = torch.randn(8, 23, device="cuda")

    with torch.inference_mode():
        expected = model(observations).clone()

    checkpoint = tmp_path / "bc_cuda_policy.pt"
    save_bc_policy(model, checkpoint)

    restored = load_bc_policy(checkpoint, device="cuda")

    with torch.inference_mode():
        actual = restored(observations)

    torch.cuda.synchronize()

    assert restored.config == config
    assert next(restored.parameters()).device.type == "cuda"
    assert torch.equal(expected, actual)


def test_bc_cpu_cuda_consistency() -> None:
    """Verify CPU and CUDA predictions agree within tolerance."""
    config = BCPolicyConfig(seed=42)

    cpu_model = BCPolicy(config).eval()
    gpu_model = BCPolicy(config).to("cuda").eval()

    observations = torch.randn(8, 23)

    with torch.inference_mode():
        cpu_output = cpu_model(observations)
        gpu_output = gpu_model(observations.to("cuda"))

    torch.cuda.synchronize()

    assert torch.allclose(
        cpu_output,
        gpu_output.cpu(),
        atol=1e-5,
        rtol=1e-5,
    )
