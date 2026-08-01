from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_models as adapters
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_observations,
    transformed_sv_observations,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _predator_prey_dataset,
    _sir_dataset,
    _sv_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json"
DTYPE = tf.float64


def _registry_rows() -> dict[str, dict[str, object]]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {row["row_id"]: row for row in payload["rows"]}


def _fixture(adapter: adapters.ContractETPModelAdapter):
    if "actual_nongaussian" in adapter.row_id:
        data = _sv_dataset(81101)
        theta = tf.constant(data["truth_theta"], DTYPE)
        y = exact_transformed_sv_observations(data["observations"])[0]
        parent = tf.constant([[-0.4], [0.3]], DTYPE)
        noise = tf.constant([[0.15], [-0.2]], DTYPE)
    elif "ksc_gaussian_mixture" in adapter.row_id:
        data = _sv_dataset(81101)
        theta = tf.constant(data["truth_theta"], DTYPE)
        y = transformed_sv_observations(data["observations"], offset=1.0e-8)[0]
        parent = tf.constant([[-0.4], [0.3]], DTYPE)
        noise = tf.constant([[0.15], [-0.2]], DTYPE)
    elif "generalized_sv" in adapter.row_id:
        data = _generalized_sv_prior_mean_dataset(81105)
        theta = tf.constant(data["truth_theta"], DTYPE)
        y = tf.convert_to_tensor(data["observations"][0], DTYPE)
        parent = tf.constant([[-0.4], [0.3]], DTYPE)
        noise = tf.constant([[0.15], [-0.2]], DTYPE)
    elif "predator_prey" in adapter.row_id:
        data = _predator_prey_dataset(81104)
        theta = tf.constant(data["truth_theta"], DTYPE)
        y = tf.convert_to_tensor(data["observations"][0], DTYPE)
        parent = tf.constant([[50.0, 5.0], [52.0, 4.5]], DTYPE)
        noise = tf.constant([[0.05, -0.05], [-0.1, 0.1]], DTYPE)
    else:
        data = _sir_dataset(81103)
        theta = tf.constant(data["truth_theta"], DTYPE)
        y = tf.convert_to_tensor(data["observations"][0], DTYPE)
        parent = tf.convert_to_tensor(data["states"][:2], DTYPE)
        noise = tf.zeros_like(parent)
    return theta, parent, noise, y


def _autodiff_rows(function, theta: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        values = function(theta)
    return tape.jacobian(
        values, theta, unconnected_gradients=tf.UnconnectedGradients.ZERO
    )


def test_adapter_registry_matches_frozen_rows_and_target_policies() -> None:
    frozen = _registry_rows()
    actual = adapters.contract_e_tp_model_adapters()
    assert len(actual) == 5
    assert len({adapter.row_id for adapter in actual}) == 5
    for adapter in actual:
        row = frozen[adapter.row_id]
        assert adapter.parameter_dimension == row["parameter_dim"]
        assert list(adapter.parameter_order) == row["parameter_order"]
        assert adapter.theta_coordinate_system == row["theta_coordinate_system"]
        if "generalized_sv" not in adapter.row_id:
            assert adapter.target_observation_policy == row["target_observation_policy"]
        else:
            assert "raw_observation" in adapter.target_observation_policy
            assert adapter.target_observation_policy.startswith(
                row["target_observation_policy"]
            )
            assert row["dataset"]["target_observations"] == row["dataset"]["raw_observations"]
        assert "requires_tp" in adapter.proposal_flow_status


@pytest.mark.parametrize("adapter", adapters.contract_e_tp_model_adapters())
def test_adapter_one_step_push_and_target_density_are_finite(adapter) -> None:
    theta, parent, noise, observation = _fixture(adapter)
    pushed = adapter.transition_push(theta, parent, noise, 0)
    assert pushed.shape == parent.shape
    assert bool(tf.reduce_all(tf.math.is_finite(pushed)).numpy())
    assert bool(adapter.support_valid(pushed).numpy())
    transition = adapter.model.transition_log_density(theta, parent, pushed, 0)
    observed = adapter.model.observation_log_density(theta, pushed, observation, 0)
    assert transition.shape == (2,)
    assert observed.shape == (2,)
    assert bool(tf.reduce_all(tf.math.is_finite(transition)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(observed)).numpy())


@pytest.mark.parametrize("adapter", adapters.contract_e_tp_model_adapters())
def test_adapter_density_parameter_scores_match_autodiff(adapter) -> None:
    theta, parent, noise, observation = _fixture(adapter)
    pushed = tf.stop_gradient(adapter.transition_push(theta, parent, noise, 0))
    transition_ad = _autodiff_rows(
        lambda value: adapter.model.transition_log_density(value, parent, pushed, 0),
        theta,
    )
    observation_ad = _autodiff_rows(
        lambda value: adapter.model.observation_log_density(value, pushed, observation, 0),
        theta,
    )
    transition_manual = adapter.model.transition_log_density_parameter_score(
        theta, parent, pushed, 0
    )
    observation_manual = adapter.model.observation_log_density_parameter_score(
        theta, pushed, observation, 0
    )
    np.testing.assert_allclose(transition_manual, transition_ad, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(observation_manual, observation_ad, rtol=3e-11, atol=3e-11)


def test_predator_and_sir_transition_push_parameter_tangents_match_manual_jacobians() -> None:
    for adapter in (
        adapters.make_predator_prey_contract_e_tp_adapter(),
        adapters.make_sir_contract_e_tp_adapter(),
    ):
        theta, parent, noise, _ = _fixture(adapter)
        automatic = _autodiff_rows(
            lambda value: adapter.transition_push(value, parent, noise, 0), theta
        )
        _, manual_parameter_first = adapter.model.transition_mean_parameter_jacobian(
            theta, parent
        )
        manual = tf.transpose(manual_parameter_first, [1, 2, 0])
        np.testing.assert_allclose(manual, automatic, rtol=2e-10, atol=2e-10)


def test_actual_ksc_and_generalized_observation_targets_are_distinct() -> None:
    actual = adapters.make_actual_sv_contract_e_tp_adapter()
    ksc = adapters.make_ksc_sv_contract_e_tp_adapter()
    generalized = adapters.make_generalized_sv_contract_e_tp_adapter()
    data = _sv_dataset(81101)
    theta = tf.constant(data["truth_theta"], DTYPE)
    state = tf.constant([[0.2]], DTYPE)
    exact_y = exact_transformed_sv_observations(data["observations"])[0]
    ksc_y = transformed_sv_observations(data["observations"], offset=1.0e-8)[0]
    exact_value = actual.model.observation_log_density(theta, state, exact_y, 0)
    ksc_value = ksc.model.observation_log_density(theta, state, ksc_y, 0)
    assert not np.allclose(exact_value.numpy(), ksc_value.numpy())
    assert "raw_observation" in generalized.target_observation_policy


def test_predator_adapter_preserves_additive_gaussian_real_support() -> None:
    adapter = adapters.make_predator_prey_contract_e_tp_adapter()
    assert bool(adapter.support_valid(tf.constant([[1.0, -0.1]], DTYPE)).numpy())
    assert adapter.model.domain_policy == "diagnose_negative_after_noise"
