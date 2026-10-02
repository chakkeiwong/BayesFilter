"""Simple nonlinear generic SSM target fixture for Phase 7.

This module exposes the BayesFilter-owned Model B nonlinear accumulation
fixture through the generic ``SSMTargetContract`` and batch-native posterior
adapter boundary.  It is an adapter/contract test fixture only: it does not
train NeuTra, run HMC, tune a sampler, or establish posterior correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
    tf_batched_svd_sigma_point_value_and_score,
)
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    GenericSSMPosteriorAdapter,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    build_ssm_posterior_adapter,
)
from bayesfilter.testing.nonlinear_models_tf import model_b_observations_tf


SIMPLE_NONLINEAR_GENERIC_TARGET_NONCLAIMS = (
    "simple nonlinear generic target adapter fixture only",
    "deterministic sigma-point approximation only",
    "no NeuTra training claim",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no production readiness claim",
    "no scientific validity claim",
)

MODEL_B_ALPHA = tf.constant(0.55, dtype=tf.float64)
MODEL_B_OBSERVATION_SIGMA = tf.constant(0.30, dtype=tf.float64)
MODEL_B_INITIAL_BATCH = tf.constant([[0.70, 0.25, 0.80]], dtype=tf.float64)
SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID = "model-b-svd-ukf-deterministic-loglikelihood"
SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID = (
    "model-b-svd-cubature-deterministic-loglikelihood"
)
SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID = "model-b-svd-cut4-deterministic-loglikelihood"
SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID = (
    "model-b-principal-sqrt-ukf-deterministic-loglikelihood"
)
SIMPLE_NONLINEAR_DEFAULT_FILTER_ID = SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID


@dataclass(frozen=True)
class SimpleNonlinearFilterRoute:
    """Phase 8 route inventory entry for the simple nonlinear target."""

    filter_id: str
    status: str
    sigma_point_backend: str
    filter_hash: str
    reason: str

    def manifest_payload(self) -> Mapping[str, str]:
        return {
            "filter_id": self.filter_id,
            "status": self.status,
            "sigma_point_backend": self.sigma_point_backend,
            "filter_hash": self.filter_hash,
            "reason": self.reason,
        }


_SIMPLE_NONLINEAR_FILTER_ROUTES = (
    SimpleNonlinearFilterRoute(
        filter_id=SIMPLE_NONLINEAR_SVD_UKF_FILTER_ID,
        status="admitted",
        sigma_point_backend="tf_svd_ukf",
        filter_hash="sha256:model-b-svd-ukf-deterministic-filter-v1",
        reason="Phase 7 admitted deterministic SVD-UKF value/score adapter route.",
    ),
    SimpleNonlinearFilterRoute(
        filter_id=SIMPLE_NONLINEAR_SVD_CUBATURE_FILTER_ID,
        status="admitted",
        sigma_point_backend="tf_svd_cubature",
        filter_hash="sha256:model-b-svd-cubature-deterministic-filter-v1",
        reason="Phase 8 admitted deterministic SVD cubature value/score adapter route.",
    ),
    SimpleNonlinearFilterRoute(
        filter_id=SIMPLE_NONLINEAR_SVD_CUT4_FILTER_ID,
        status="deferred",
        sigma_point_backend="tf_svd_cut4",
        filter_hash="sha256:model-b-svd-cut4-deterministic-filter-deferred-v1",
        reason=(
            "CUT4 is deferred until a dedicated generic-adapter finite-difference "
            "and branch-diagnostic gate is reviewed."
        ),
    ),
    SimpleNonlinearFilterRoute(
        filter_id=SIMPLE_NONLINEAR_PRINCIPAL_SQRT_UKF_FILTER_ID,
        status="deferred",
        sigma_point_backend="tf_principal_sqrt_ukf",
        filter_hash="sha256:model-b-principal-sqrt-ukf-deterministic-filter-deferred-v1",
        reason=(
            "Principal-square-root UKF is deferred until custom-op/principal-sqrt "
            "availability and provenance are gated for this target."
        ),
    ),
)
_SIMPLE_NONLINEAR_ROUTE_BY_ID = {
    route.filter_id: route for route in _SIMPLE_NONLINEAR_FILTER_ROUTES
}


@dataclass(frozen=True)
class SimpleNonlinearGenericTargetFixture:
    """Generic adapter fixture for Model B nonlinear accumulation."""

    contract: SSMTargetContract
    adapter: GenericSSMPosteriorAdapter
    initial_batch: tf.Tensor


def make_simple_nonlinear_generic_target_fixture(
    *,
    filter_id: str = SIMPLE_NONLINEAR_DEFAULT_FILTER_ID,
) -> SimpleNonlinearGenericTargetFixture:
    """Return a simple nonlinear non-DSGE target via the generic SSM adapter."""

    route = _require_admitted_filter_route(filter_id)
    contract = make_simple_nonlinear_generic_target_contract(filter_id=filter_id)
    target_scope = "model-b-nonlinear-accumulation-generic-target-fixture"
    if filter_id != SIMPLE_NONLINEAR_DEFAULT_FILTER_ID:
        target_scope = f"{target_scope}:{filter_id}"
    adapter = build_ssm_posterior_adapter(
        contract=contract,
        prior_log_prob_and_grad=simple_nonlinear_gaussian_prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=_filter_log_likelihood_fn(route),
        dtype=tf.float64,
        target_scope=target_scope,
        evidence_path="bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py",
        nonclaims=SIMPLE_NONLINEAR_GENERIC_TARGET_NONCLAIMS,
    )
    return SimpleNonlinearGenericTargetFixture(
        contract=contract,
        adapter=adapter,
        initial_batch=MODEL_B_INITIAL_BATCH,
    )


def make_simple_nonlinear_generic_target_contract(
    *,
    data_hash: str = "sha256:model-b-fixed-observations-v1",
    model_hash: str = "sha256:model-b-nonlinear-accumulation-v1",
    transform_hash: str = "sha256:model-b-identity-unconstrained-v1",
    prior_hash: str = "sha256:model-b-phase7-gaussian-prior-v1",
    filter_hash: str | None = None,
    filter_id: str = SIMPLE_NONLINEAR_DEFAULT_FILTER_ID,
) -> SSMTargetContract:
    """Build a stable generic target contract for Model B."""

    route = _require_admitted_filter_route(filter_id)
    selected_filter_hash = route.filter_hash if filter_hash is None else filter_hash
    observations = model_b_observations_tf()
    if observations.shape.rank != 2:
        raise ValueError("Model B observations must have shape [time, observation]")
    horizon = observations.shape[0]
    observation_dim = observations.shape[1]
    if horizon is None or observation_dim is None:
        raise ValueError("Model B observations must have static dimensions")
    parameter_dim = 3

    static_shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=2,
        observation_dim=int(observation_dim),
        innovation_dim=1,
        parameter_dim=parameter_dim,
    )
    problem = BayesianSSMProblem(
        problem_id="model-b-nonlinear-accumulation-generic-target-fixture",
        static_shape=static_shape,
        data_signature=SSMDataSignature(
            dataset_id="model-b-fixed-observations",
            observation_shape=tuple(observations.shape.as_list()),
            data_hash=data_hash,
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            "model_id": "model-b-nonlinear-accumulation",
            "model_hash": model_hash,
            "capabilities": (
                "smooth_nonlinear_transition",
                "additive_gaussian_observation",
                "deterministic_completion",
                "analytic_first_derivatives",
            ),
            "source_fixture": "bayesfilter.testing.nonlinear_models_tf",
            "fixed_alpha": float(MODEL_B_ALPHA.numpy()),
            "fixed_observation_sigma": float(MODEL_B_OBSERVATION_SIGMA.numpy()),
        },
    )
    chart = ParameterChart(
        parameter_names=("rho", "sigma", "beta"),
        unconstrained_dim=parameter_dim,
        constrained_shape=(parameter_dim,),
        transform_manifest={
            "transform_id": "identity-unconstrained-chart",
            "transform_hash": transform_hash,
            "coordinate_order": ("rho", "sigma", "beta"),
            "phase7_fixture_convention": (
                "sigma is an unconstrained innovation loading in this adapter fixture"
            ),
        },
        log_jacobian_convention="not_included",
    )
    prior = ParameterPrior(
        prior_manifest={
            "prior_id": "model-b-independent-gaussian-around-reference",
            "prior_hash": prior_hash,
            "parameter_order": chart.parameter_names,
            "prior_mean": (0.70, 0.25, 0.80),
            "prior_scale": (0.20, 0.10, 0.20),
        },
        support_policy="unbounded",
        log_density_authority="graph_native",
    )
    filter_program = FilterProgram(
        filter_id=route.filter_id,
        required_model_capabilities=(
            "smooth_nonlinear_transition",
            "additive_gaussian_observation",
            "analytic_first_derivatives",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            "filter_id": route.filter_id,
            "filter_hash": selected_filter_hash,
            "backend": "tensorflow",
            "value_score_source": (
                "bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf"
            ),
            "sigma_point_backend": route.sigma_point_backend,
            "innovation_floor": 1.0e-12,
            "spectral_gap_tolerance": 1.0e-8,
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )


def simple_nonlinear_filter_route_inventory() -> tuple[Mapping[str, str], ...]:
    """Return the Phase 8 filter-route inventory for Model B."""

    return tuple(route.manifest_payload() for route in _SIMPLE_NONLINEAR_FILTER_ROUTES)


def simple_nonlinear_admitted_filter_ids() -> tuple[str, ...]:
    """Return filter ids admitted through the generic adapter boundary."""

    return tuple(
        route.filter_id
        for route in _SIMPLE_NONLINEAR_FILTER_ROUTES
        if route.status == "admitted"
    )


def simple_nonlinear_gaussian_prior_log_prob_and_grad(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return rank-2 Gaussian prior value and score for Model B parameters."""

    theta_tensor = _rank2_theta(theta)
    mean = tf.constant([0.70, 0.25, 0.80], dtype=theta_tensor.dtype)
    scale = tf.constant([0.20, 0.10, 0.20], dtype=theta_tensor.dtype)
    centered = theta_tensor - mean[tf.newaxis, :]
    scaled = centered / scale[tf.newaxis, :]
    value = -0.5 * tf.reduce_sum(tf.square(scaled), axis=-1)
    score = -centered / tf.square(scale)[tf.newaxis, :]
    return value, score


def simple_nonlinear_svd_ukf_log_likelihood_and_grad(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return deterministic SVD-UKF likelihood value and first-order score."""

    return simple_nonlinear_sigma_point_log_likelihood_and_grad(
        theta,
        backend="tf_svd_ukf",
    )


def simple_nonlinear_svd_cubature_log_likelihood_and_grad(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return deterministic SVD cubature likelihood value and first-order score."""

    return simple_nonlinear_sigma_point_log_likelihood_and_grad(
        theta,
        backend="tf_svd_cubature",
    )


def simple_nonlinear_sigma_point_log_likelihood_and_grad(
    theta: Any,
    *,
    backend: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return deterministic sigma-point likelihood value and first-order score."""

    if backend not in {"tf_svd_ukf", "tf_svd_cubature"}:
        raise ValueError(f"simple nonlinear filter backend is not admitted: {backend}")
    theta_tensor = _rank2_theta(theta)
    model, derivatives = make_batched_model_b_svd_ukf_components(theta_tensor)
    value, score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        model_b_observations_tf(),
        model,
        derivatives,
        backend=backend,
        placement_floor=tf.constant(0.0, dtype=tf.float64),
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1.0e-8, dtype=tf.float64),
    )
    checks = [
        tf.debugging.assert_all_finite(value, "Model B SVD-UKF value must be finite"),
        tf.debugging.assert_all_finite(score, "Model B SVD-UKF score must be finite"),
        tf.debugging.assert_near(
            diagnostics["deterministic_residual"],
            tf.zeros_like(diagnostics["deterministic_residual"]),
            atol=tf.constant(1.0e-12, dtype=tf.float64),
        ),
    ]
    with tf.control_dependencies(checks):
        return tf.identity(value), tf.identity(score)


def simple_nonlinear_filter_diagnostics(
    theta: Any,
    *,
    backend: str = "tf_svd_ukf",
) -> Mapping[str, tf.Tensor]:
    """Return deterministic filter diagnostics for Phase 7 tests/results."""

    if backend not in {"tf_svd_ukf", "tf_svd_cubature"}:
        raise ValueError(f"simple nonlinear filter backend is not admitted: {backend}")
    theta_tensor = _rank2_theta(theta)
    model, derivatives = make_batched_model_b_svd_ukf_components(theta_tensor)
    _value, _score, diagnostics = tf_batched_svd_sigma_point_value_and_score(
        model_b_observations_tf(),
        model,
        derivatives,
        backend=backend,
        placement_floor=tf.constant(0.0, dtype=tf.float64),
        innovation_floor=tf.constant(1.0e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1.0e-8, dtype=tf.float64),
    )
    return diagnostics


def _require_admitted_filter_route(filter_id: str) -> SimpleNonlinearFilterRoute:
    key = str(filter_id)
    try:
        route = _SIMPLE_NONLINEAR_ROUTE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown simple nonlinear filter route: {key}") from exc
    if route.status != "admitted":
        raise ValueError(
            f"simple nonlinear filter route is not admitted: {key}; {route.reason}"
        )
    return route


def _filter_log_likelihood_fn(route: SimpleNonlinearFilterRoute):
    if route.sigma_point_backend == "tf_svd_ukf":
        return simple_nonlinear_svd_ukf_log_likelihood_and_grad
    if route.sigma_point_backend == "tf_svd_cubature":
        return simple_nonlinear_svd_cubature_log_likelihood_and_grad
    raise ValueError(
        "simple nonlinear filter route has no admitted likelihood function: "
        f"{route.filter_id}"
    )


def make_batched_model_b_svd_ukf_components(
    theta: Any,
) -> tuple[TFBatchedStructuralStateSpace, TFBatchedStructuralFirstDerivatives]:
    """Build batched Model B structural tensors and first derivatives."""

    theta_tensor = _rank2_theta(theta)
    rho = theta_tensor[:, 0]
    sigma = theta_tensor[:, 1]
    beta = theta_tensor[:, 2]
    batch_size = int(theta_tensor.shape[0])
    parameter_dim = 3
    state_dim = 2
    innovation_dim = 1
    observation_dim = 1

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m_next = rho[:, tf.newaxis] * previous[:, :, 0] + sigma[:, tf.newaxis] * eps
        k_next = (
            MODEL_B_ALPHA * previous[:, :, 1]
            + beta[:, tf.newaxis] * tf.math.tanh(m_next)
        )
        return tf.stack([m_next, k_next], axis=2)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return (states[:, :, 0] + states[:, :, 1])[:, :, tf.newaxis]

    def residual(
        previous: tf.Tensor,
        innovation: tf.Tensor,
        next_points: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        expected = MODEL_B_ALPHA * previous[:, :, 1] + beta[:, tf.newaxis] * tf.math.tanh(
            next_points[:, :, 0]
        )
        return (next_points[:, :, 1] - expected)[:, :, tf.newaxis]

    def transition_state_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m_next = rho[:, tf.newaxis] * previous[:, :, 0] + sigma[:, tf.newaxis] * eps
        sech2 = 1.0 - tf.square(tf.math.tanh(m_next))
        zeros = tf.zeros_like(m_next)
        row_m = tf.stack(
            [tf.broadcast_to(rho[:, tf.newaxis], tf.shape(m_next)), zeros],
            axis=2,
        )
        row_k = tf.stack(
            [
                beta[:, tf.newaxis] * sech2 * rho[:, tf.newaxis],
                tf.fill(tf.shape(m_next), MODEL_B_ALPHA),
            ],
            axis=2,
        )
        return tf.stack([row_m, row_k], axis=2)

    def transition_innovation_jacobian(
        previous: tf.Tensor,
        innovation: tf.Tensor,
    ) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m_next = rho[:, tf.newaxis] * previous[:, :, 0] + sigma[:, tf.newaxis] * eps
        sech2 = 1.0 - tf.square(tf.math.tanh(m_next))
        column = tf.stack(
            [
                tf.broadcast_to(sigma[:, tf.newaxis], tf.shape(m_next)),
                beta[:, tf.newaxis] * sech2 * sigma[:, tf.newaxis],
            ],
            axis=2,
        )
        return column[:, :, :, tf.newaxis]

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m_next = rho[:, tf.newaxis] * previous[:, :, 0] + sigma[:, tf.newaxis] * eps
        tanh_m = tf.math.tanh(m_next)
        sech2 = 1.0 - tf.square(tanh_m)
        zeros = tf.zeros_like(eps)
        d_rho = tf.stack(
            [previous[:, :, 0], beta[:, tf.newaxis] * sech2 * previous[:, :, 0]],
            axis=2,
        )
        d_sigma = tf.stack([eps, beta[:, tf.newaxis] * sech2 * eps], axis=2)
        d_beta = tf.stack([zeros, tanh_m], axis=2)
        return tf.stack([d_rho, d_sigma, d_beta], axis=1)

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        point_count = tf.shape(states)[1]
        return tf.broadcast_to(
            tf.constant([[[1.0, 1.0]]], dtype=tf.float64),
            [tf.shape(states)[0], point_count, observation_dim, state_dim],
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [tf.shape(states)[0], parameter_dim, tf.shape(states)[1], observation_dim],
            dtype=tf.float64,
        )

    model = TFBatchedStructuralStateSpace(
        initial_mean=tf.zeros([batch_size, state_dim], dtype=tf.float64),
        initial_covariance=tf.broadcast_to(
            tf.linalg.diag(tf.constant([0.25, 0.20], dtype=tf.float64)),
            [batch_size, state_dim, state_dim],
        ),
        innovation_covariance=tf.ones(
            [batch_size, innovation_dim, innovation_dim],
            dtype=tf.float64,
        ),
        observation_covariance=tf.fill(
            [batch_size, observation_dim, observation_dim],
            tf.square(MODEL_B_OBSERVATION_SIGMA),
        ),
        transition_fn=transition,
        observation_fn=observe,
        deterministic_residual_fn=residual,
        name="model_b_nonlinear_accumulation_phase7_batched",
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=tf.zeros([batch_size, parameter_dim, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros(
            [batch_size, parameter_dim, state_dim, state_dim],
            dtype=tf.float64,
        ),
        d_innovation_covariance=tf.zeros(
            [batch_size, parameter_dim, innovation_dim, innovation_dim],
            dtype=tf.float64,
        ),
        d_observation_covariance=tf.zeros(
            [batch_size, parameter_dim, observation_dim, observation_dim],
            dtype=tf.float64,
        ),
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="model_b_nonlinear_accumulation_phase7_first_derivatives",
    )
    return model, derivatives


def _rank2_theta(theta: Any) -> tf.Tensor:
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    if theta_tensor.shape.rank != 2:
        raise ValueError("simple nonlinear generic target requires rank 2 theta [B, D]")
    parameter_dim = theta_tensor.shape[-1]
    if parameter_dim is None:
        raise ValueError("simple nonlinear generic target requires static parameter dimension")
    if int(parameter_dim) != 3:
        raise ValueError("simple nonlinear generic target parameter dimension must be 3")
    batch_dim = theta_tensor.shape[0]
    if batch_dim is None:
        raise ValueError("simple nonlinear generic target requires static batch dimension")
    return theta_tensor
