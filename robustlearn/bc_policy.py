"""PyTorch MLP policy for normalized Behavior Cloning observations.

Input: 23 normalized observation features.
Output: 7 normalized absolute joint-position targets.

This module defines the policy architecture only.
Training and controller safety checks are separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from robustlearn.bc_dataset import ACTION_DIM, OBSERVATION_DIM


@dataclass(frozen=True)
class BCPolicyConfig:
    """Configuration for the Behavior Cloning MLP."""

    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    activation: str = "relu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")

        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")

        if not self.hidden_dims or any(
            dimension <= 0 for dimension in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive dimensions")

        if self.activation not in ("relu", "silu"):
            raise ValueError("activation must be 'relu' or 'silu'")

        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class BCPolicy(nn.Module):
    """Configurable feedforward Behavior Cloning policy."""

    def __init__(self, config: BCPolicyConfig | None = None) -> None:
        super().__init__()

        self.config = config or BCPolicyConfig()

        activation_type: type[nn.Module] = (
            nn.ReLU if self.config.activation == "relu" else nn.SiLU
        )

        # Isolate both layer construction and explicit initialization
        # from the caller's global PyTorch RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)

            layers: list[nn.Module] = []
            input_dim = self.config.observation_dim

            for hidden_dim in self.config.hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activation_type())
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, self.config.action_dim))
            self.network = nn.Sequential(*layers)

            for module in self.network:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        nonlinearity="relu",
                    )
                    nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Predict normalized action targets from normalized observations."""

        if observation.ndim not in (1, 2):
            raise ValueError("observation must be a 1D or 2D tensor")

        if observation.shape[-1] != self.config.observation_dim:
            raise ValueError("observation feature dimension mismatch")

        prediction: torch.Tensor = self.network(observation)
        return prediction


def resolve_bc_device(device: str | torch.device = "cpu") -> torch.device:
    """Resolve a requested compute device, rejecting unavailable CUDA."""

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported BC device: {resolved}")

    return resolved


def save_bc_policy(model: BCPolicy, path: str | Path) -> None:
    """Save the architecture configuration and model weights."""

    from dataclasses import asdict

    checkpoint = {
        "format_version": 1,
        "config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint, path)


def load_bc_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> BCPolicy:
    """Restore a BC policy checkpoint onto the selected device."""

    resolved_device = resolve_bc_device(device)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("BC policy checkpoint must be a dictionary")

    required_keys = {
        "format_version",
        "config",
        "model_state_dict",
    }
    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise ValueError(
            f"BC policy checkpoint missing keys: {sorted(missing_keys)}"
        )

    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported BC policy checkpoint version")

    if not isinstance(checkpoint["config"], dict):
        raise ValueError("BC policy checkpoint config must be a dictionary")

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError("BC policy checkpoint state must be a dictionary")

    config = BCPolicyConfig(**checkpoint["config"])
    model = BCPolicy(config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()

    return model
