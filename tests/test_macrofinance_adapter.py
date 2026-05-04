from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bayesfilter.adapters.macrofinance import (
    evaluate_macrofinance_hmc_readiness,
    evaluate_macrofinance_provider_derivatives,
    evaluate_macrofinance_provider_likelihood,
    macrofinance_lgssm_to_bayesfilter,
)
from bayesfilter.filters import kalman_log_likelihood
from bayesfilter.structural import StatePartition


MACROFINANCE_ROOT = Path("/home/chakwong/MacroFinance")


@dataclass(frozen=True)
class FakeMacroFinanceLGSSM:
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    transition_offset: np.ndarray
    transition_matrix: np.ndarray
    transition_covariance: np.ndarray
    observation_offset: np.ndarray
    observation_matrix: np.ndarray
    observation_covariance: np.ndarray


class FakeProvider:
    parameter_dim = 1
    requested_orders: list[int]

    def __init__(self):
        self.requested_orders = []

    def parameter_names(self):
        return ["offset_shift"]

    def build_state_space(self, theta):
        theta = np.asarray(theta, dtype=float)
        return FakeMacroFinanceLGSSM(
            initial_mean=np.array([0.0]),
            initial_covariance=np.array([[0.2]]),
            transition_offset=np.array([theta[0]]),
            transition_matrix=np.array([[0.5]]),
            transition_covariance=np.array([[0.1]]),
            observation_offset=np.array([0.0]),
            observation_matrix=np.array([[1.0]]),
            observation_covariance=np.array([[0.05]]),
        )

    def build_state_space_with_derivatives(self, theta, order):
        self.requested_orders.append(order)
        return self.build_state_space(theta), {"order": order}


def fake_first_order_backend(observations, model, derivatives, *, jitter):
    assert derivatives["order"] == 1
    assert jitter == 1e-9
    return -1.25, np.array([0.5])


def fake_second_order_backend(observations, model, derivatives, *, jitter):
    assert derivatives["order"] == 2
    assert jitter == 1e-9
    return -1.25, np.array([0.5]), np.array([[-2.0]])


class FakePosteriorAdapter:
    analytic_backend = "fake_analytic"

    def parameter_names(self):
        return ["offset_shift"]

    def initial_position(self):
        return np.array([0.02])

    def log_prob_and_grad(self, theta):
        theta = np.asarray(theta, dtype=float)
        return -0.5 * float(theta[0] ** 2), np.array([-theta[0]])

    def negative_log_prob_and_gradient(self, theta):
        value, grad = self.log_prob_and_grad(theta)
        return -value, -grad

    def negative_log_prob_hessian(self, theta):
        return np.array([[1.0]])


def test_macrofinance_lgssm_conversion_preserves_arrays_and_partition():
    provider = FakeProvider()
    partition = StatePartition(
        state_names=("x",),
        stochastic_indices=(0,),
        deterministic_indices=(),
        innovation_dim=1,
    )

    converted = macrofinance_lgssm_to_bayesfilter(
        provider.build_state_space(np.array([0.1])),
        partition=partition,
    )

    assert converted.partition is partition
    np.testing.assert_allclose(converted.transition_offset, np.array([0.1]))
    np.testing.assert_allclose(converted.transition_matrix, np.array([[0.5]]))
    np.testing.assert_allclose(converted.observation_covariance, np.array([[0.05]]))


def test_value_wrapper_matches_direct_bayesfilter_kalman():
    provider = FakeProvider()
    theta = np.array([0.02])
    observations = np.array([[0.1], [0.0], [0.2]])

    wrapped = evaluate_macrofinance_provider_likelihood(
        provider,
        theta,
        observations,
        jitter=1e-9,
    )
    direct = kalman_log_likelihood(
        observations,
        macrofinance_lgssm_to_bayesfilter(provider.build_state_space(theta)),
        jitter=1e-9,
    )

    np.testing.assert_allclose(wrapped.log_likelihood, direct.log_likelihood)
    assert wrapped.parameter_names == ("offset_shift",)
    assert wrapped.kalman_result.metadata.filter_name == "covariance_kalman"
    assert wrapped.kalman_result.metadata.differentiability_status == "value_only"


def test_derivative_bridge_preserves_first_and_second_order_workloads():
    provider = FakeProvider()
    theta = np.array([0.02])
    observations = np.array([[0.1], [0.0], [0.2]])

    first = evaluate_macrofinance_provider_derivatives(
        provider,
        theta,
        observations,
        fake_first_order_backend,
        derivative_order=1,
        backend_name="fake_first_order",
        jitter=1e-9,
        initial_mean_policy="zero",
    )
    second = evaluate_macrofinance_provider_derivatives(
        provider,
        theta,
        observations,
        fake_second_order_backend,
        derivative_order=2,
        backend_name="fake_second_order",
        jitter=1e-9,
        initial_mean_policy="zero",
    )

    assert provider.requested_orders == [1, 2]
    assert first.hessian is None
    np.testing.assert_allclose(first.score, np.array([0.5]))
    assert first.derivative_order == 1
    assert first.backend == "fake_first_order"
    assert first.initial_mean_policy == "zero"
    np.testing.assert_allclose(second.hessian, np.array([[-2.0]]))
    assert second.derivative_order == 2


def test_hmc_readiness_conformance_is_finite_without_convergence_claim():
    result = evaluate_macrofinance_hmc_readiness(FakePosteriorAdapter())

    assert result.parameter_names == ("offset_shift",)
    assert result.backend == "fake_analytic"
    assert result.readiness_label == "hmc_contract_ready_smoke"
    assert result.convergence_claim == "not_claimed"
    np.testing.assert_allclose(result.negative_hessian, np.array([[1.0]]))


def _macrofinance_module(name: str):
    if not MACROFINANCE_ROOT.exists():
        pytest.skip("MacroFinance checkout is not available")
    root = str(MACROFINANCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def test_optional_one_country_value_matches_macrofinance_numpy_reference():
    domain_types = _macrofinance_module("domain.types")
    derivative_provider = _macrofinance_module("one_country_derivative_provider")
    differentiated_kalman = _macrofinance_module("filters.differentiated_kalman")
    helper = _macrofinance_module("tests.helpers_one_country_analytic_hmc")

    parameters, data, _prior_mean, _prior_scale, true_vector = (
        helper.one_country_analytic_hmc_case(n_steps=8)
    )
    provider = derivative_provider.OneCountryAFNSDerivativeProvider(
        data=data,
        fixed_parameters=parameters,
        run_config=domain_types.RunConfig(),
    )
    model, derivatives = provider.build_state_space_with_derivatives(true_vector, order=2)
    macrofinance_value, _score, _hessian = differentiated_kalman.differentiated_kalman_loglik(
        np.asarray(data.yields, dtype=float),
        model,
        derivatives,
        jitter=domain_types.RunConfig().jitter,
    )

    bayesfilter_value = evaluate_macrofinance_provider_likelihood(
        provider,
        true_vector,
        np.asarray(data.yields, dtype=float),
        jitter=domain_types.RunConfig().jitter,
    )

    np.testing.assert_allclose(
        bayesfilter_value.log_likelihood,
        macrofinance_value,
        rtol=1e-10,
        atol=1e-10,
    )
    assert bayesfilter_value.parameter_names == tuple(provider.parameter_names())


def test_optional_one_country_derivative_bridge_matches_macrofinance_numpy_reference():
    domain_types = _macrofinance_module("domain.types")
    derivative_provider = _macrofinance_module("one_country_derivative_provider")
    differentiated_kalman = _macrofinance_module("filters.differentiated_kalman")
    helper = _macrofinance_module("tests.helpers_one_country_analytic_hmc")

    parameters, data, _prior_mean, _prior_scale, true_vector = (
        helper.one_country_analytic_hmc_case(n_steps=8)
    )
    provider = derivative_provider.OneCountryAFNSDerivativeProvider(
        data=data,
        fixed_parameters=parameters,
        run_config=domain_types.RunConfig(),
    )
    model, derivatives = provider.build_state_space_with_derivatives(true_vector, order=2)
    macrofinance_value, macrofinance_score, macrofinance_hessian = (
        differentiated_kalman.differentiated_kalman_loglik(
            np.asarray(data.yields, dtype=float),
            model,
            derivatives,
            jitter=domain_types.RunConfig().jitter,
        )
    )

    bridged = evaluate_macrofinance_provider_derivatives(
        provider,
        true_vector,
        np.asarray(data.yields, dtype=float),
        differentiated_kalman.differentiated_kalman_loglik,
        derivative_order=2,
        backend_name="macrofinance_numpy_differentiated_kalman",
        jitter=domain_types.RunConfig().jitter,
        initial_mean_policy="zero",
    )

    np.testing.assert_allclose(bridged.log_likelihood, macrofinance_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bridged.score, macrofinance_score, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bridged.hessian, macrofinance_hessian, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bridged.hessian, bridged.hessian.T, atol=1e-10)
    assert bridged.parameter_names == tuple(provider.parameter_names())
    assert bridged.backend == "macrofinance_numpy_differentiated_kalman"


def test_optional_one_country_hmc_readiness_conformance_is_finite():
    domain_types = _macrofinance_module("domain.types")
    hmc = _macrofinance_module("inference.hmc")
    helper = _macrofinance_module("tests.helpers_one_country_analytic_hmc")

    adapter = hmc.OneCountryAnalyticThetaNoisePosteriorAdapter(
        **helper.one_country_adapter_kwargs(n_steps=8),
        run_config=domain_types.RunConfig(),
        analytic_backend="tf_covariance_analytic",
    )
    theta = adapter.initial_position() + np.array([0.0007, -0.0004, 0.0002, 0.05])

    result = evaluate_macrofinance_hmc_readiness(adapter, theta)

    assert result.parameter_names == tuple(adapter.parameter_names())
    assert result.backend == "tf_covariance_analytic"
    assert result.convergence_claim == "not_claimed"
    assert np.all(np.isfinite(result.score))
    assert np.all(np.isfinite(result.negative_gradient))
    assert np.all(np.isfinite(result.negative_hessian))
    np.testing.assert_allclose(result.negative_hessian, result.negative_hessian.T, atol=1e-10)
