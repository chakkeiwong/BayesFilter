from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import numpy as np
import pytest

from bayesfilter.adapters.macrofinance import (
    DerivativeCoverageMetadata,
    FiniteDifferenceOracleMetadata,
    ObservationMaskMetadata,
    ParameterUnitMetadata,
    ReadinessBlockerMetadata,
    evaluate_macrofinance_hmc_gate,
    evaluate_macrofinance_hmc_readiness,
    evaluate_macrofinance_provider_derivatives,
    evaluate_macrofinance_provider_likelihood,
    extract_derivative_coverage_metadata,
    extract_finite_difference_oracle_metadata,
    extract_identification_evidence_metadata,
    extract_observation_mask_metadata,
    extract_parameter_unit_metadata,
    extract_readiness_blocker_metadata,
    extract_sparse_backend_policy_metadata,
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

    def parameter_units(self):
        return np.array([0.1])

    def observations(self):
        return np.array([[1.0], [np.nan]])

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

    def derivative_coverage_matrix(self):
        return (
            {
                "block_name": "offset",
                "parameter_names": ["offset_shift"],
                "reduced_form_objects": ["b"],
                "derivative_status": "implemented_first_order",
            },
        )

    def finite_difference_oracle(self):
        return FakeFiniteDifferenceOracle()


class FakeFiniteDifferenceOracle:
    finite_difference_step = 1e-5


class FakeEvidenceRow(NamedTuple):
    block_name: str
    trust_status: str
    notes: list[str]


class FakePolicyRow(NamedTuple):
    backend: str
    status: str
    supported_order: int


class FakeProductionProvider(FakeProvider):
    @property
    def blockers(self):
        return ("missing_final_panel",)

    def blocker_summary(self):
        return "missing_final_panel"

    def readiness_blocker_table(self):
        return (("final_country_panel", "missing"),)

    def validate_final_ten_country_ready(self):
        raise ValueError("requires 10 countries")

    def identification_evidence_status(self):
        return (FakeEvidenceRow("offset", "Weakly identified", ["synthetic fixture only"]),)

    def sparse_derivative_backend_policy(self):
        return (FakePolicyRow("masked_covariance_reference", "implemented_order_2", 2),)


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


def test_metadata_primitives_are_immutable_and_fail_closed():
    units = ParameterUnitMetadata(
        parameter_names=("a", "b"),
        units=np.array([1.0, 0.1]),
    )
    mask = ObservationMaskMetadata(
        shape=(2, 3),
        observed_count=5,
        missing_count=1,
        all_observed=False,
    )
    oracle = FiniteDifferenceOracleMetadata(
        available=True,
        step="1e-5",
        notes=["central_difference"],
    )
    readiness = ReadinessBlockerMetadata(
        blockers=["missing_final_panel"],
        blocker_table=[["data", "missing_final_panel"]],
        blocker_summary="missing_final_panel",
        final_ready=False,
        final_readiness_error="blocked",
    )
    coverage = DerivativeCoverageMetadata(
        rows=[(("block_name", "offset"), ("parameter_names", ["a", "b"]))],
    )

    assert units.parameter_names == ("a", "b")
    assert mask.convention == "true_means_observed"
    assert oracle.notes == ("central_difference",)
    assert readiness.final_ready is False
    assert readiness.blockers == ("missing_final_panel",)
    assert coverage.rows[0][1][1] == ("a", "b")
    with pytest.raises(ValueError):
        units.units[0] = 2.0
    with pytest.raises(Exception):
        readiness.final_ready = True


def test_large_scale_metadata_extractors_work_with_fake_provider():
    provider = FakeProvider()

    units = extract_parameter_unit_metadata(provider)
    mask = extract_observation_mask_metadata(provider)

    assert units.parameter_names == ("offset_shift",)
    np.testing.assert_allclose(units.units, np.array([0.1]))
    assert mask.shape == (2, 1)
    assert mask.observed_count == 1
    assert mask.missing_count == 1
    assert mask.all_observed is False


def test_cross_currency_metadata_extractors_work_with_fake_provider():
    provider = FakeProvider()

    coverage = extract_derivative_coverage_metadata(provider)
    oracle = extract_finite_difference_oracle_metadata(provider)

    row = dict(coverage.rows[0])
    assert row["block_name"] == "offset"
    assert row["parameter_names"] == ("offset_shift",)
    assert row["reduced_form_objects"] == ("b",)
    assert row["derivative_status"] == "implemented_first_order"
    assert oracle.available is True
    assert oracle.step == 1e-5
    assert oracle.notes == ("FakeFiniteDifferenceOracle",)


def test_production_readiness_metadata_fails_closed_with_fake_provider():
    provider = FakeProductionProvider()

    readiness = extract_readiness_blocker_metadata(provider)
    evidence = extract_identification_evidence_metadata(provider)
    sparse_policy = extract_sparse_backend_policy_metadata(provider)

    assert readiness.blockers == ("missing_final_panel",)
    assert readiness.blocker_table == (("final_country_panel", "missing"),)
    assert readiness.blocker_summary == "missing_final_panel"
    assert readiness.final_ready is False
    assert "requires 10 countries" in readiness.final_readiness_error
    evidence_row = dict(evidence.rows[0])
    assert evidence_row["trust_status"] == "Weakly identified"
    assert evidence_row["notes"] == ("synthetic fixture only",)
    policy_row = dict(sparse_policy.rows[0])
    assert policy_row["backend"] == "masked_covariance_reference"
    assert policy_row["supported_order"] == 2


def test_hmc_gate_checks_finiteness_symmetry_and_parity_without_convergence_claim():
    adapter = FakePosteriorAdapter()

    result = evaluate_macrofinance_hmc_gate(
        adapter,
        compiled_log_prob_and_grad=adapter.log_prob_and_grad,
    )

    assert result.value_finite is True
    assert result.score_finite is True
    assert result.negative_hessian_finite is True
    assert result.negative_hessian_symmetric is True
    assert result.eager_compiled_parity is True
    assert result.max_abs_value_diff == 0.0
    assert result.max_abs_score_diff == 0.0
    assert result.target_ready is True
    assert result.convergence_claim == "not_claimed"


def test_hmc_gate_records_parity_failure_without_claiming_target_ready():
    adapter = FakePosteriorAdapter()

    def shifted_compiled(theta):
        value, score = adapter.log_prob_and_grad(theta)
        return value + 1.0, score

    result = evaluate_macrofinance_hmc_gate(
        adapter,
        compiled_log_prob_and_grad=shifted_compiled,
        parity_tolerance=1e-10,
    )

    assert result.eager_compiled_parity is False
    assert result.target_ready is False
    assert result.convergence_claim == "not_claimed"


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


def test_optional_large_scale_metadata_extracts_units_and_mask_policy():
    large_provider_module = _macrofinance_module("large_scale_lgssm_derivative_provider")
    scenarios = _macrofinance_module("tests.helpers_large_scale_lgssm")

    provider = large_provider_module.LargeScaleLGSSMDerivativeProvider(
        scenarios.scenario_by_name("baseline_10x3x5"),
        parameter_count_per_block=1,
    )

    units = extract_parameter_unit_metadata(provider)
    mask = extract_observation_mask_metadata(provider)

    assert units.parameter_names == tuple(provider.parameter_names())
    assert units.units.shape == (provider.parameter_dim,)
    assert np.all(np.isfinite(units.units))
    assert mask.shape == provider.observations().shape
    assert mask.all_observed is True


def test_optional_cross_currency_metadata_extracts_coverage_and_oracle_provenance():
    structural_provider = _macrofinance_module("cross_currency_structural_derivative_provider")

    provider = structural_provider.CrossCurrencyStructuralDerivativeProvider.from_synthetic_fixture(
        n_steps=4,
    )

    coverage = extract_derivative_coverage_metadata(provider)
    oracle = extract_finite_difference_oracle_metadata(provider)

    block_names = {dict(row)["block_name"] for row in coverage.rows}
    assert "physical_dynamics" in block_names
    assert "measurement_error_blocks" in block_names
    assert oracle.available is True
    assert oracle.step == provider.finite_difference_step
    assert oracle.notes == ("FiniteDifferenceStructuralDerivativeOracle",)


def test_optional_production_metadata_preserves_blockers_and_policy():
    production_provider = _macrofinance_module("production_cross_currency_derivative_provider")

    provider = production_provider.ProductionCrossCurrencyDerivativeProvider.from_synthetic_fixture(
        n_steps=4,
    )

    readiness = extract_readiness_blocker_metadata(provider)
    coverage = extract_derivative_coverage_metadata(provider)
    evidence = extract_identification_evidence_metadata(provider)
    sparse_policy = extract_sparse_backend_policy_metadata(provider)

    coverage_rows = [dict(row) for row in coverage.rows]
    assert readiness.blockers == provider.blockers
    assert readiness.blocker_table == provider.readiness_blocker_table()
    assert readiness.final_ready is False
    assert "requires 10 countries" in readiness.final_readiness_error
    assert any(row["derivative_status"] == "blocked_final_provider" for row in coverage_rows)
    assert all(dict(row)["trust_status"] != "Identified" for row in evidence.rows)
    assert any(dict(row)["backend"] == "masked_covariance_reference" for row in sparse_policy.rows)


def test_optional_one_country_hmc_gate_records_target_readiness_without_convergence_claim():
    domain_types = _macrofinance_module("domain.types")
    hmc = _macrofinance_module("inference.hmc")
    helper = _macrofinance_module("tests.helpers_one_country_analytic_hmc")

    adapter = hmc.OneCountryAnalyticThetaNoisePosteriorAdapter(
        **helper.one_country_adapter_kwargs(n_steps=8),
        run_config=domain_types.RunConfig(),
        analytic_backend="tf_covariance_analytic",
    )
    theta = adapter.initial_position() + np.array([0.0007, -0.0004, 0.0002, 0.05])

    result = evaluate_macrofinance_hmc_gate(
        adapter,
        theta,
        compiled_log_prob_and_grad=adapter.log_prob_and_grad,
    )

    assert result.value_finite is True
    assert result.score_finite is True
    assert result.negative_hessian_finite is True
    assert result.negative_hessian_symmetric is True
    assert result.eager_compiled_parity is True
    assert result.target_ready is True
    assert result.convergence_claim == "not_claimed"
