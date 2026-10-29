"""Scope-bound batch-native SVX-ZC posterior target for NeuTra and HMC.

The default adapter in this module preserves the reviewed UKF-frozen initializer
identity and emits the active same-program batch-native finite target.

The serious/common HMC route uses the analytic score backend built below, which
keeps that same frozen UKF initializer identity and exposes the same-program
adjacent-state analytic derivative backend with XLA admission enabled after
parity checks.
"""



from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.filtering import legendre_gauss_nodes_weights
from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
    BATCH_ROUTE_ID,
    COORDINATE_HALF_WIDTH,
    DEGREE,
    ORDER,
    RANK,
    batched_fixed_tt_likelihood_analytic_score_status,
    batched_fixed_tt_likelihood_value_score_status,
    source_two_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    stable_ssm_target_signature,
)


PARAMETER_NAMES = ("gamma_source_probit", "beta_source_probit")
TARGET_SCOPE = "SVX-ZC-T10-d10-r2-o25-center-frozen-ukf-v1"
NONCLAIMS = (
    "fixed adjacent-state squared-TT extension, not exact filtering",
    "center-frozen UKF initializer, not runtime retuning",
    "score is a diagnostic derivative of the same finite program",
    "no posterior correctness, HMC convergence, or production-readiness claim",
)
SCORE_BACKEND_ID = "svx_zc_same_program_scaled_adjacent_state_manual_v1"


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ActualSVZCNeuTraAdapter:
    """Batch-native posterior adapter bound to frozen T10 TT controls."""

    dtype = tf.float64
    parameter_dim = 2
    parameter_names = PARAMETER_NAMES
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    score_backend_id = SCORE_BACKEND_ID
    runtime_autodiff_for_hmc = False

    def __init__(
        self,
        *,
        contract: SSMTargetContract,
        program_tensors: Mapping[str, tf.Tensor],
        initial_core_hash: str,
        adjacent_core_hash: str,
    ) -> None:
        self.contract = contract
        self.target_scope = TARGET_SCOPE
        self.program_tensors = {
            str(name): tf.convert_to_tensor(value, tf.float64)
            for name, value in program_tensors.items()
        }
        self.initial_core_hash = str(initial_core_hash)
        self.adjacent_core_hash = str(adjacent_core_hash)
        payload = {
            "schema": "bayesfilter.testing.actual_sv_zc_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "target_scope": self.target_scope,
            "parameter_names": self.parameter_names,
            "initial_core_hash": self.initial_core_hash,
            "adjacent_core_hash": self.adjacent_core_hash,
        }
        self._adapter_signature = _semantic_hash(payload)

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_batched_fixed_adjacent_squared_tt_actual_sv_same_program_manual_score",
            evidence_path=(
                "docs/plans/bayesfilter_direct_factor_srukf_remaining_gaps_"
                "closure_execution_result_2026_08_17.md"
            ),
            target_scope=self.target_scope,
            nonclaims=NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score, _status = posterior_value_score_status(
            theta, **self.program_tensors
        )
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = posterior_value_score_status(
            theta, **self.program_tensors
        )
        return value, score

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score, status = (
            batched_fixed_tt_likelihood_analytic_score_status(
                theta, **self.program_tensors
            )
        )
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = batched_fixed_tt_likelihood_value_score_status(
            theta, **self.program_tensors
        )
        return status


class ActualSVZCLikelihoodRecomposer:
    """Independent likelihood-only component for posterior recomposition."""

    def __init__(self, adapter: ActualSVZCNeuTraAdapter) -> None:
        self.program_tensors = dict(adapter.program_tensors)

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = batched_fixed_tt_likelihood_value_score_status(
            theta, **self.program_tensors
        )
        return value, score


def posterior_value_score_status(
    theta: Any,
    **program_tensors: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Independently compose likelihood, prior, and full chart Jacobian."""

    likelihood_value, likelihood_score, status = (
        batched_fixed_tt_likelihood_value_score_status(theta, **program_tensors)
    )
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
        status,
    )


def make_actual_sv_zc_neutra_adapter() -> ActualSVZCNeuTraAdapter:
    """Build the T10 adapter with center-frozen repository UKF cores."""

    # Initializer construction is a one-time target-identity operation.  Keep
    # it on CPU so its issued core hashes do not depend on whether the caller
    # initialized a GPU before constructing the adapter.
    with tf.device("/CPU:0"):
        return _make_actual_sv_zc_neutra_adapter_on_cpu()


def _make_actual_sv_zc_neutra_adapter_on_cpu() -> ActualSVZCNeuTraAdapter:
    """Construct the frozen program tensors under the CPU device scope."""

    import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator

    model, center, transformed_observations = comparator._row_inputs("actual_sv", 10)
    raw = tf.convert_to_tensor(
        comparator._sv_dataset(81101)["observations"], tf.float64
    )[:10]
    initial, adjacent, initializer = comparator._ukf_initial_cores(
        model=model,
        theta=center,
        raw_observations=raw,
        degree=DEGREE,
        order=ORDER,
        rank=RANK,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
    )
    config = comparator._comparator_config(
        degree=DEGREE,
        order=ORDER,
        rank=RANK,
        seed="svx-zc-neutra-target-center-frozen-v1",
        transition_before_first_observation=False,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
        density_tau=0.0,
        initial_cores=initial,
        adjacent_initial_cores=adjacent,
        initialization_rule=str(initializer["initializer_rule"]),
    )
    nodes, weights = legendre_gauss_nodes_weights(ORDER)
    mesh = tf.meshgrid(nodes, nodes, indexing="ij")
    grid = tf.reshape(tf.stack(mesh, axis=-1), (-1, 2))
    weight_mesh = tf.meshgrid(0.5 * weights, 0.5 * weights, indexing="ij")
    grid_weights = tf.reshape(weight_mesh[0] * weight_mesh[1], (-1,))
    program_tensors = {
        "transformed_observations": transformed_observations[:, 0],
        "initial_core": initial[0].values,
        "adjacent_core0": adjacent[0].values,
        "adjacent_core1": adjacent[1].values,
        "reference_nodes": nodes,
        "reference_weights": 0.5 * weights,
        "reference_grid": grid,
        "reference_grid_weights": grid_weights,
        "basis_nodes": config.initial.product_basis.evaluate_axis(0, nodes),
        "basis_grid_axis0": config.adjacent.product_basis.evaluate_axis(0, grid[:, 0]),
        "basis_grid_axis1": config.adjacent.product_basis.evaluate_axis(1, grid[:, 1]),
    }
    data_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(transformed_observations).numpy())
    ).hexdigest()
    contract = make_actual_sv_zc_target_contract(
        data_hash=data_hash,
        initial_core_hash=str(initializer["initial_core_hash"]),
        adjacent_core_hash=str(initializer["adjacent_core_hash"]),
    )
    return ActualSVZCNeuTraAdapter(
        contract=contract,
        program_tensors=program_tensors,
        initial_core_hash=str(initializer["initial_core_hash"]),
        adjacent_core_hash=str(initializer["adjacent_core_hash"]),
    )


def make_actual_sv_zc_target_contract(
    *,
    data_hash: str,
    initial_core_hash: str,
    adjacent_core_hash: str,
) -> SSMTargetContract:
    """Issue the exact target identity for the fixed T10 finite program."""

    model_semantics = {
        "model_id": "zhao-cui-synthetic-sv-fixed-sigma-1",
        "fixed_sigma": 1.0,
        "observation_target": "log(y^2)-2log(beta)-x follows log-chi-square-1",
        "time_order": "initial-observation-then-transition",
    }
    problem = BayesianSSMProblem(
        problem_id="actual-sv-zc-fixed-adjacent-tt-t10",
        static_shape=SSMStaticShape(10, 1, 1, 1, 2),
        data_signature=SSMDataSignature(
            dataset_id="zhao_cui_sv_actual_nongaussian_seed81101_T10",
            observation_shape=(10, 1),
            data_hash=f"sha256:{data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": f"sha256:{_semantic_hash(model_semantics)}",
        },
    )
    chart_semantics = {
        "transform_id": "source-two-probit-uniform-box-chart",
        "gamma": "0.1+0.8*Phi(theta[0])",
        "beta": "0.1+0.8*Phi(theta[1])",
        "parameter_order": PARAMETER_NAMES,
    }
    chart = ParameterChart(
        parameter_names=PARAMETER_NAMES,
        unconstrained_dim=2,
        constrained_shape=(2,),
        transform_manifest={
            **chart_semantics,
            "transform_hash": f"sha256:{_semantic_hash(chart_semantics)}",
        },
        log_jacobian_convention="included_in_chart",
    )
    prior_semantics = {
        "prior_id": "zhao-cui-synthetic-sv-independent-uniform-box",
        "physical_support": ((0.1, 0.9), (0.1, 0.9)),
        "parameter_order": ("gamma", "beta"),
    }
    prior = ParameterPrior(
        prior_manifest={
            **prior_semantics,
            "prior_hash": f"sha256:{_semantic_hash(prior_semantics)}",
        },
        support_policy="enforced_by_transform",
        log_density_authority="graph_native",
    )
    filter_semantics = {
        "filter_id": BATCH_ROUTE_ID,
        "base_route_id": "zhao_cui_fixed_adjacent_state_squared_tt_v1",
        "degree": DEGREE,
        "rank": RANK,
        "quadrature_order": ORDER,
        "ridge": 1.0e-10,
        "max_sweeps": 2,
        "sweep_order": (0, 1, 1, 0),
        "coordinate_half_width": COORDINATE_HALF_WIDTH,
        "initializer_policy": (
            "UKF cores built once on CPU at validation center and frozen"
        ),
        "initial_core_hash": initial_core_hash,
        "adjacent_core_hash": adjacent_core_hash,
        "backend": "tensorflow_batch_native_static_time_and_sweep_loops",
    }
    filter_program = FilterProgram(
        filter_id=BATCH_ROUTE_ID,
        required_model_capabilities=(
            "scalar_stationary_sv",
            "exact_log_chi_square_observation_density",
            "batch_native_fixed_tt_als",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            **filter_semantics,
            "filter_hash": f"sha256:{_semantic_hash(filter_semantics)}",
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )


__all__ = [
    "ActualSVZCLikelihoodRecomposer",
    "ActualSVZCNeuTraAdapter",
    "PARAMETER_NAMES",
    "TARGET_SCOPE",
    "make_actual_sv_zc_neutra_adapter",
    "make_actual_sv_zc_target_contract",
    "posterior_value_score_status",
]
