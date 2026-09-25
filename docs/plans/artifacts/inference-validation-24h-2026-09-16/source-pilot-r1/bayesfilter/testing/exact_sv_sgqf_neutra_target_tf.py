"""Graph-native exact-transformed-SV SGQF posterior for the P2 admission ladder."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud
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


SVX_SGQF_DATASET_ID = "zhao_cui_sv_actual_nongaussian_T1000"
SVX_SGQF_DATASET_SEED = 81101
SVX_SGQF_HORIZON = 1000
SVX_SGQF_OBSERVATION_SHA256 = (
    "5e2423149e4f59eb588ccc7f16ec6d9ee984ccc4710a3ae07a3dbcf5c37db748"
)
SVX_SGQF_STATE_SHA256 = (
    "338b3ba4ce18fe6ef758c216a679f0537729d032e2ae5ff795ed8ecbe1fed453"
)
SVX_SGQF_SCOPE_PREFIX = "SVX-SGQF-source-chart-level"
SVX_SGQF_PARAMETER_NAMES = ("gamma_source_probit", "beta_source_probit")
SVX_SGQF_NONCLAIMS = (
    "exact-transformed observation target, not raw-return native likelihood",
    "fixed-SGQF deterministic approximation, not exact likelihood",
    "source-grounded prior/chart but BayesFilter graph-native filter extension",
    "no HMC convergence, NeuTra training, or scientific readiness claim",
)

_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
)
_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float64)
_LOG_POINT_EIGHT = tf.constant(math.log(0.8), tf.float64)
_MIN_VARIANCE = tf.constant(1.0e-14, tf.float64)


def generate_frozen_exact_sv_dataset_tf(
    *,
    seed: int = SVX_SGQF_DATASET_SEED,
    horizon: int = SVX_SGQF_HORIZON,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Replay the preserved dependent SV trajectory with a TensorFlow loop."""

    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    with tf.device("/CPU:0"):
        gamma = tf.constant(0.6, tf.float64)
        beta = tf.constant(0.4, tf.float64)
        sigma = tf.constant(1.0, tf.float64)
        generator = tf.random.Generator.from_seed(int(seed))
        states = tf.TensorArray(
            tf.float64, size=int(horizon), clear_after_read=False
        )
        observations = tf.TensorArray(
            tf.float64, size=int(horizon), clear_after_read=False
        )
        initial = sigma / tf.sqrt(1.0 - tf.square(gamma)) * generator.normal(
            (), dtype=tf.float64
        )
        initial_observation = beta * tf.exp(0.5 * initial) * generator.normal(
            (), dtype=tf.float64
        )
        states = states.write(0, initial)
        observations = observations.write(0, initial_observation)

    def condition(index, _state, _states, _observations):
        return index < tf.constant(int(horizon), tf.int32)

    def body(index, previous, state_array, observation_array):
        current = gamma * previous + sigma * generator.normal((), dtype=tf.float64)
        observed = beta * tf.exp(0.5 * current) * generator.normal(
            (), dtype=tf.float64
        )
        return (
            index + 1,
            current,
            state_array.write(index, current),
            observation_array.write(index, observed),
        )

    with tf.device("/CPU:0"):
        _, _, states, observations = tf.while_loop(
            condition,
            body,
            (tf.constant(1, tf.int32), initial, states, observations),
            parallel_iterations=1,
        )
        return (
            tf.reshape(states.stack(), (-1, 1)),
            tf.reshape(observations.stack(), (-1, 1)),
        )


def source_chart_physical_parameters(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    """Map source coordinates to physical gamma,beta in the paper prior box."""

    values = _rank2_theta(theta)
    probabilities = _NORMAL.cdf(values)
    physical = tf.constant(0.1, tf.float64) + tf.constant(0.8, tf.float64) * probabilities
    return physical[:, 0], physical[:, 1]


def source_uniform_prior_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate the physical Uniform([0.1,0.9]^2) prior in source coordinates."""

    values = _rank2_theta(theta)
    batch_shape = tf.shape(values)[:-1]
    value = tf.fill(batch_shape, -2.0 * _LOG_POINT_EIGHT)
    return value, tf.zeros_like(values)


def source_two_probit_jacobian_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate log|d(gamma,beta)/d(theta)| and its score."""

    values = _rank2_theta(theta)
    normal_log_density = -0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI
    value = tf.reduce_sum(_LOG_POINT_EIGHT + normal_log_density, axis=-1)
    return value, -values


def fixed_sgqf_likelihood_value_score(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate only the fixed-SGQF likelihood and manual source-chart score."""

    value, score, _status = _fixed_sgqf_value_score_status(
        theta,
        observations=observations,
        nodes=nodes,
        weights=weights,
    )
    return value, score


def _fixed_sgqf_value_score_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    values = _rank2_theta(theta)
    raw_observations = tf.convert_to_tensor(observations, tf.float64)
    if raw_observations.shape.rank != 2 or raw_observations.shape[1] != 1:
        raise ValueError("exact-SV observations must have shape [time, 1]")
    scalar_nodes = tf.reshape(tf.convert_to_tensor(nodes, tf.float64), (-1,))
    scalar_weights = tf.reshape(tf.convert_to_tensor(weights, tf.float64), (-1,))
    log_weights = tf.math.log(scalar_weights)
    transformed_observations = tf.math.log(tf.square(raw_observations[:, 0]))
    gamma, beta = source_chart_physical_parameters(values)
    density = _NORMAL.prob(values)
    dgamma = tf.constant(0.8, tf.float64) * density[:, 0]
    dbeta = tf.constant(0.8, tf.float64) * density[:, 1]
    dlog_beta = dbeta / beta
    one_minus_gamma_sq = 1.0 - tf.square(gamma)
    current_mean = tf.zeros_like(gamma)
    current_variance = tf.math.reciprocal(one_minus_gamma_sq)
    current_d_mean = tf.zeros((tf.shape(values)[0], 2), tf.float64)
    current_d_variance = tf.stack(
        (
            2.0 * gamma * dgamma / tf.square(one_minus_gamma_sq),
            tf.zeros_like(gamma),
        ),
        axis=1,
    )
    total_value = tf.zeros_like(gamma)
    total_score = tf.zeros_like(values)
    status_valid = tf.ones_like(gamma, tf.bool)
    minimum_variance = current_variance

    def condition(index, *_loop_values):
        return index < tf.shape(transformed_observations)[0]

    def body(
        index,
        mean,
        variance,
        d_mean,
        d_variance,
        value_total,
        score_total,
        valid,
        min_variance,
    ):
        is_positive_time = index > 0
        predicted_mean = tf.where(is_positive_time, gamma * mean, mean)
        predicted_variance = tf.where(
            is_positive_time,
            tf.square(gamma) * variance + 1.0,
            variance,
        )
        predicted_d_mean = tf.where(
            is_positive_time,
            tf.stack(
                (
                    dgamma * mean + gamma * d_mean[:, 0],
                    gamma * d_mean[:, 1],
                ),
                axis=1,
            ),
            d_mean,
        )
        predicted_d_variance = tf.where(
            is_positive_time,
            tf.stack(
                (
                    2.0 * gamma * dgamma * variance
                    + tf.square(gamma) * d_variance[:, 0],
                    tf.square(gamma) * d_variance[:, 1],
                ),
                axis=1,
            ),
            d_variance,
        )
        predicted_is_valid = tf.logical_and(
            tf.math.is_finite(predicted_variance), predicted_variance > 0.0
        )
        safe_variance = tf.maximum(predicted_variance, _MIN_VARIANCE)
        predicted_scale = tf.sqrt(safe_variance)
        predicted_d_scale = 0.5 * predicted_d_variance / predicted_scale[:, None]
        points = predicted_mean[:, None] + predicted_scale[:, None] * scalar_nodes[None, :]
        d_points = predicted_d_mean[:, :, None] + predicted_d_scale[:, :, None] * scalar_nodes[None, None, :]
        residual = (
            transformed_observations[index]
            - 2.0 * tf.math.log(beta)[:, None]
            - points
        )
        observation_log = 0.5 * residual - 0.5 * tf.exp(residual) - 0.5 * _LOG_TWO_PI
        log_normalizer = tf.reduce_logsumexp(log_weights[None, :] + observation_log, axis=1)
        normalized = tf.exp(log_weights[None, :] + observation_log - log_normalizer[:, None])
        direct_beta = tf.stack(
            (tf.zeros_like(dlog_beta), -2.0 * dlog_beta), axis=1
        )[:, :, None]
        d_residual = -d_points + direct_beta
        d_observation_log = 0.5 * (1.0 - tf.exp(residual))[:, None, :] * d_residual
        d_log_normalizer = tf.reduce_sum(normalized[:, None, :] * d_observation_log, axis=2)
        centered = d_observation_log - d_log_normalizer[:, :, None]
        filtered_mean = tf.reduce_sum(normalized * points, axis=1)
        filtered_second = tf.reduce_sum(normalized * tf.square(points), axis=1)
        filtered_variance = filtered_second - tf.square(filtered_mean)
        filtered_d_mean = tf.reduce_sum(
            normalized[:, None, :] * (d_points + centered * points[:, None, :]),
            axis=2,
        )
        filtered_d_second = tf.reduce_sum(
            normalized[:, None, :]
            * (2.0 * points[:, None, :] * d_points + centered * tf.square(points)[:, None, :]),
            axis=2,
        )
        filtered_d_variance = filtered_d_second - 2.0 * filtered_mean[:, None] * filtered_d_mean
        step_valid = tf.logical_and(
            predicted_is_valid,
            tf.logical_and(
                tf.math.is_finite(log_normalizer),
                tf.logical_and(
                    tf.math.is_finite(filtered_variance), filtered_variance > 0.0
                ),
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.reduce_all(tf.math.is_finite(d_log_normalizer), axis=1),
        )
        return (
            index + 1,
            filtered_mean,
            filtered_variance,
            filtered_d_mean,
            filtered_d_variance,
            value_total + log_normalizer,
            score_total + d_log_normalizer,
            tf.logical_and(valid, step_valid),
            tf.minimum(min_variance, tf.minimum(predicted_variance, filtered_variance)),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            current_mean,
            current_variance,
            current_d_mean,
            current_d_variance,
            total_value,
            total_score,
            status_valid,
            minimum_variance,
        ),
        parallel_iterations=1,
    )
    value = result[5]
    score = result[6]
    valid = result[7]
    minimum_variance = result[8]
    return value, score, {
        "status_code": tf.where(valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)),
        "valid_pre_regularized_score": valid,
        "floor_count_value": tf.zeros_like(value, tf.int32),
        "min_innovation_eigenvalue": minimum_variance,
        "innovation_condition_estimate": tf.ones_like(value),
    }


def _posterior_value_score(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    likelihood_value, likelihood_score = fixed_sgqf_likelihood_value_score(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
    )


class ExactSVSGQFNeuTraAdapter:
    """Batch-native fixed-SGQF posterior adapter for one frozen level/data pair."""

    dtype = tf.float64
    parameter_dim = 2
    parameter_names = SVX_SGQF_PARAMETER_NAMES

    def __init__(
        self,
        *,
        observations: tf.Tensor,
        sparse_level: int,
        contract: SSMTargetContract,
    ) -> None:
        cloud = tf_fixed_sgqf_cloud(dim=1, sparse_level=int(sparse_level))
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.nodes = tf.reshape(cloud.points, (-1,))
        self.weights = tf.convert_to_tensor(cloud.weights, tf.float64)
        self.sparse_level = int(sparse_level)
        self.contract = contract
        self.target_scope = f"{SVX_SGQF_SCOPE_PREFIX}-{self.sparse_level}"
        payload = {
            "schema": "bayesfilter.testing.exact_sv_sgqf_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "sparse_level": self.sparse_level,
            "point_count": int(cloud.point_count),
        }
        self._adapter_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_exact_sv_fixed_sgqf",
            evidence_path="bayesfilter/testing/exact_sv_sgqf_neutra_target_tf.py",
            target_scope=self.target_scope,
            nonclaims=SVX_SGQF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = _posterior_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return _posterior_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
        likelihood_value, likelihood_score, status = _fixed_sgqf_value_score_status(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = _fixed_sgqf_value_score_status(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status[
                "valid_pre_regularized_score"
            ],
        }


class ExactSVSGQFLikelihoodRecomposer:
    """Independent likelihood component callable for posterior recomposition."""

    def __init__(self, adapter: ExactSVSGQFNeuTraAdapter) -> None:
        self.observations = adapter.observations
        self.nodes = adapter.nodes
        self.weights = adapter.weights

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return fixed_sgqf_likelihood_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )


def make_exact_sv_sgqf_neutra_adapter(
    *,
    sparse_level: int,
    observations: tf.Tensor | None = None,
) -> ExactSVSGQFNeuTraAdapter:
    """Build one exact-SV SGQF adapter bound to the frozen dataset and level."""

    if observations is None:
        _states, observations = generate_frozen_exact_sv_dataset_tf()
    raw = tf.convert_to_tensor(observations, tf.float64)
    if raw.shape.rank != 2 or raw.shape[1] != 1:
        raise ValueError("observations must have shape [time, 1]")
    if not bool(tf.reduce_all(tf.math.is_finite(raw)).numpy()):
        raise ValueError("observations must be finite")
    if bool(tf.reduce_any(tf.equal(raw, 0.0)).numpy()):
        raise ValueError("exact log-square target forbids zero observations")
    data_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(raw).numpy())).hexdigest()
    cloud = tf_fixed_sgqf_cloud(dim=1, sparse_level=int(sparse_level))
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(cloud.points).numpy())
        + bytes(tf.io.serialize_tensor(cloud.weights).numpy())
    ).hexdigest()
    contract = make_exact_sv_sgqf_target_contract(
        horizon=int(raw.shape[0]),
        data_hash=data_hash,
        sparse_level=int(sparse_level),
        cloud_hash=cloud_hash,
        point_count=int(cloud.point_count),
    )
    return ExactSVSGQFNeuTraAdapter(
        observations=raw,
        sparse_level=int(sparse_level),
        contract=contract,
    )


def make_exact_sv_sgqf_target_contract(
    *,
    horizon: int,
    data_hash: str,
    sparse_level: int,
    cloud_hash: str,
    point_count: int,
) -> SSMTargetContract:
    shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=1,
        observation_dim=1,
        innovation_dim=1,
        parameter_dim=2,
    )
    model_semantics = {
        "model_id": "zhao-cui-synthetic-sv-fixed-sigma-1",
        "physical_truth": (0.6, 0.4, 1.0),
        "fixed_sigma": 1.0,
        "observation_target": "z=log(y^2)-2log(beta)-x ~ log(chi_square_1)",
        "transformed_data_measure": "z-space; raw-to-z Jacobian is data-only and excluded",
    }
    problem = BayesianSSMProblem(
        problem_id="exact-transformed-sv-fixed-sgqf-source-chart",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id=SVX_SGQF_DATASET_ID,
            observation_shape=(int(horizon), 1),
            data_hash=f"sha256:{data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": f"sha256:{_semantic_hash(model_semantics)}",
        },
    )
    transform_semantics = {
        "transform_id": "source-two-probit-uniform-box-chart",
        "gamma": "0.1+0.8*Phi(theta[0])",
        "beta": "0.1+0.8*Phi(theta[1])",
        "parameter_order": SVX_SGQF_PARAMETER_NAMES,
    }
    chart = ParameterChart(
        parameter_names=SVX_SGQF_PARAMETER_NAMES,
        unconstrained_dim=2,
        constrained_shape=(2,),
        transform_manifest={
            **transform_semantics,
            "transform_hash": f"sha256:{_semantic_hash(transform_semantics)}",
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
    filter_id = f"fixed-sgqf-exact-transformed-sv-level-{int(sparse_level)}"
    filter_semantics = {
        "filter_id": filter_id,
        "cloud_tensor_hash": cloud_hash,
        "sparse_level": int(sparse_level),
        "point_count": int(point_count),
        "backend": "tensorflow_xla_tf_while_loop",
        "score": "manual_forward_sensitivity_source_chart",
    }
    filter_program = FilterProgram(
        filter_id=filter_id,
        required_model_capabilities=(
            "scalar_stationary_sv",
            "exact_log_chi_square_observation_density",
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


def _rank2_theta(theta: Any) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 2:
        raise ValueError("exact-SV SGQF target requires theta shape [batch, 2]")
    return values


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
