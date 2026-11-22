"""Unit tests for the PyTorch Behavior Cloning MLP."""

from pathlib import Path

import pytest
import torch
from torch import nn

from robustlearn.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    load_bc_policy,
    resolve_bc_device,
    save_bc_policy,
)


@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_forward_shapes_and_finite_outputs(activation: str) -> None:
    model = BCPolicy(BCPolicyConfig(activation=activation))

    single = model(torch.zeros(23))
    batch = model(torch.randn(8, 23))
    empty = model(torch.empty(0, 23))

    assert single.shape == (7,)
    assert batch.shape == (8, 7)
    assert empty.shape == (0, 7)
    assert torch.isfinite(single).all()
    assert torch.isfinite(batch).all()


def test_default_architecture() -> None:
    model = BCPolicy()

    linear_layers = [
        layer
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (23, 256),
        (256, 256),
        (256, 256),
        (256, 7),
    ]


def test_custom_architecture() -> None:
    config = BCPolicyConfig(hidden_dims=(32, 16), activation="silu")
    model = BCPolicy(config)

    assert model(torch.zeros(4, 23)).shape == (4, 7)
    assert sum(isinstance(layer, nn.SiLU) for layer in model.network) == 2


def test_deterministic_initialization() -> None:
    model_a = BCPolicy(BCPolicyConfig(seed=42))
    model_b = BCPolicy(BCPolicyConfig(seed=42))
    model_c = BCPolicy(BCPolicyConfig(seed=43))

    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    params_c = list(model_c.parameters())

    assert all(
        torch.equal(a, b)
        for a, b in zip(params_a, params_b, strict=True)
    )
    assert any(
        not torch.equal(a, c)
        for a, c in zip(params_a, params_c, strict=True)
    )


def test_initialization_preserves_global_rng() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()

        BCPolicy()

        after = torch.random.get_rng_state()
        assert torch.equal(before, after)


@pytest.mark.parametrize(
    "shape",
    [(22,), (4, 22), (2, 3, 23)],
)
def test_invalid_observation_shape(shape: tuple[int, ...]) -> None:
    model = BCPolicy()

    with pytest.raises(ValueError):
        model(torch.zeros(shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_dim": 0},
        {"action_dim": 0},
        {"hidden_dims": ()},
        {"hidden_dims": (32, -1)},
        {"activation": "tanh"},
        {"seed": -1},
    ],
)
def test_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BCPolicyConfig(**kwargs)  # type: ignore[arg-type]


def test_cpu_inference() -> None:
    model = BCPolicy().to("cpu")
    observation = torch.zeros(2, 23, device="cpu")

    prediction = model(observation)

    assert prediction.device.type == "cpu"
    assert prediction.shape == (2, 7)
    assert torch.isfinite(prediction).all()



def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = BCPolicyConfig(
        hidden_dims=(32, 16),
        activation="silu",
        seed=17,
    )
    model = BCPolicy(config).eval()
    observation = torch.randn(6, 23)

    with torch.no_grad():
        expected = model(observation).clone()

    checkpoint_path = tmp_path / "bc_policy.pt"
    save_bc_policy(model, checkpoint_path)

    restored = load_bc_policy(checkpoint_path, device="cpu")

    with torch.no_grad():
        actual = restored(observation)

    assert restored.config == config
    assert restored.training is False
    assert torch.equal(expected, actual)
    assert next(restored.parameters()).device.type == "cpu"


def test_checkpoint_restores_modified_weights(tmp_path: Path) -> None:
    model = BCPolicy()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)

    checkpoint_path = tmp_path / "modified.pt"
    save_bc_policy(model, checkpoint_path)
    restored = load_bc_policy(checkpoint_path)

    for original, loaded in zip(
        model.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)


def test_cpu_device_resolution() -> None:
    assert resolve_bc_device("cpu") == torch.device("cpu")
    assert resolve_bc_device(torch.device("cpu")) == torch.device("cpu")


def test_unsupported_device_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_bc_device("meta")


def test_unavailable_cuda_rejected() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")

    with pytest.raises(ValueError, match="unavailable"):
        resolve_bc_device("cuda")


def test_unsupported_checkpoint_version(tmp_path: Path) -> None:
    path = tmp_path / "invalid_version.pt"

    torch.save(
        {
            "format_version": 999,
            "config": {},
            "model_state_dict": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        load_bc_policy(path)



@pytest.mark.parametrize(
    "checkpoint",
    [
        {},
        {"format_version": 1},
        {"format_version": 1, "config": {}},
    ],
)
def test_checkpoint_missing_keys(
    tmp_path: Path,
    checkpoint: dict[str, object],
) -> None:
    path = tmp_path / "missing.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="missing keys"):
        load_bc_policy(path)


@pytest.mark.parametrize(
    "checkpoint, error",
    [
        ([], "must be a dictionary"),
        (
            {
                "format_version": 1,
                "config": [],
                "model_state_dict": {},
            },
            "config must be a dictionary",
        ),
        (
            {
                "format_version": 1,
                "config": {},
                "model_state_dict": [],
            },
            "state must be a dictionary",
        ),
    ],
)
def test_checkpoint_invalid_structure(
    tmp_path: Path,
    checkpoint: object,
    error: str,
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=error):
        load_bc_policy(path)
