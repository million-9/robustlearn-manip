"""Regression tests for offline BC prediction evaluation."""

import math

import pytest
import torch

from robustlearn.bc_dataset import BCNormalization
from robustlearn.bc_offline_eval import evaluate_bc_predictions
from robustlearn.bc_policy import BCPolicy, BCPolicyConfig


def make_inputs():
    model = BCPolicy(BCPolicyConfig(hidden_dims=(8,), seed=42))
    norm = BCNormalization(
        observation_mean=torch.zeros(23),
        observation_std=torch.ones(23),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    observations = torch.zeros(5, 23)
    actions = torch.zeros(5, 7)
    return model, norm, observations, actions


def test_offline_metrics_are_deterministic():
    model, norm, observations, actions = make_inputs()
    first = evaluate_bc_predictions(model, observations, actions, norm)
    second = evaluate_bc_predictions(model, observations, actions, norm)

    assert first == second
    assert first.samples == 5
    assert len(first.per_dimension_mae) == 7
    assert len(first.per_dimension_rmse) == 7
    assert first.rmse >= first.mae >= 0
    assert all(math.isfinite(x) for x in first.per_dimension_mae)
    assert "joint_7_rmse_rad" in first.as_dict()


def test_zero_error_when_actions_match_predictions():
    model, norm, observations, _ = make_inputs()
    with torch.inference_mode():
        actions = model(observations).clone()

    result = evaluate_bc_predictions(model, observations, actions, norm)
    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)


def test_physical_action_scaling():
    model, norm, observations, _ = make_inputs()
    norm.action_std[:] = 2.0

    with torch.inference_mode():
        predicted = model(observations) * 2.0

    actions = predicted + 1.0
    result = evaluate_bc_predictions(model, observations, actions, norm)

    assert result.mae == pytest.approx(1.0)
    assert result.rmse == pytest.approx(1.0)


@pytest.mark.parametrize("problem", ["empty", "wrong_dim", "nan"])
def test_invalid_offline_inputs(problem):
    model, norm, observations, actions = make_inputs()

    if problem == "empty":
        observations = observations[:0]
        actions = actions[:0]
    elif problem == "wrong_dim":
        actions = actions[:, :6]
    else:
        observations[0, 0] = float("nan")

    with pytest.raises(ValueError):
        evaluate_bc_predictions(model, observations, actions, norm)
