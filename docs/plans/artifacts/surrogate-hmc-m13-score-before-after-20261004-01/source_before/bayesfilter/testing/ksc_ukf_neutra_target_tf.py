"""Graph-native scalar KSC principal-square-root-UKF posterior for P3."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture
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
from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    SVX_SGQF_OBSERVATION_SHA256,
    SVX_SGQF_STATE_SHA256,
    generate_frozen_exact_sv_dataset_tf,
    source_chart_physical_parameters,
    source_two_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)


KSC_UKF_DATASET_ID = "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"
KSC_UKF_DATASET_SEED = 81101
KSC_UKF_HORIZON = 1000
KSC_UKF_RAW_OBSERVATION_SHA256 = SVX_SGQF_OBSERVATION_SHA256
KSC_UKF_STATE_SHA256 = SVX_SGQF_STATE_SHA256
KSC_UKF_TRANSFORM_OFFSET = 1.0e-8
KSC_UKF_SCOPE = "KSC-UKF-source-chart-principal-sqrt-affine-v1"
KSC_UKF_PARAMETER_NAMES = (
    "gamma_source_probit",
    "beta_source_probit",
)
KSC_UKF_NONCLAIMS = (
    "KSC seven-component Gaussian-mixture transformed-SV target, not exact SV",
    "mixture moment-collapse UKF approximation, not exact latent-state filtering",
    "affine component update equivalent to scalar principal-square-root UKF",
    "no HMC convergence, NeuTra training, calibration, or readiness claim",
)

_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float64)
_MIN_VARIANCE = tf.constant(1.0e-14, tf.float64)
_WEIGHT_SUM_TOLERANCE = tf.constant(1.0e-10, tf.float64)


def transformed_ksc_observations(raw_observations: Any) -> tf.Tensor:
    """Apply the frozen KSC log-square-plus-offset transform."""

    raw = tf.convert_to_tensor(raw_observations, tf.float64)
    if raw.shape.rank != 2 or raw.shape[1] != 1:
        raise ValueError("KSC observations must have shape [time, 1]")
    return tf.math.log(
        tf.square(raw) + tf.constant(KSC_UKF_TRANSFORM_OFFSET, tf.float64)
    )


def ksc_ukf_likelihood_value_score(
    theta: Any,
    *,
    transformed_observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate the graph-native KSC-UKF likelihood and manual score."""

    value, score, _status = _ksc_ukf_likelihood_value_score_status(
        theta,
        transformed_observations=transformed_observations,
        mixture_weights=mixture_weights,
        mixture_means=mixture_means,
        mixture_variances=mixture_variances,
    )
    return value, score


def _ksc_ukf_likelihood_value_score_status(
    theta: Any,
    *,
    transformed_observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Run the batched scalar affine principal-square-root-UKF recurrence."""

    values = _rank2_theta(theta)
    observations = tf.reshape(
        tf.convert_to_tensor(transformed_observations, tf.float64), (-1,)
    )
    weights = tf.reshape(tf.convert_to_tensor(mixture_weights, tf.float64), (-1,))
    component_means = tf.reshape(
        tf.convert_to_tensor(mixture_means, tf.float64), (-1,)
    )
    component_variances = tf.reshape(
        tf.convert_to_tensor(mixture_variances, tf.float64), (-1,)
    )
    log_weights = tf.math.log(weights)

    gamma, beta = source_chart_physical_parameters(values)
    normal_density = tf.exp(-0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI)
    dgamma = tf.constant(0.8, tf.float64) * normal_density[:, 0]
    dbeta = tf.constant(0.8, tf.float64) * normal_density[:, 1]
    d_observation_offset = tf.stack(
        (tf.zeros_like(beta), 2.0 * dbeta / beta), axis=1
    )

    one_minus_gamma_sq = 1.0 - tf.square(gamma)
    mean = tf.zeros_like(gamma)
    variance = tf.math.reciprocal(one_minus_gamma_sq)
    d_mean = tf.zeros_like(values)
    d_variance = tf.stack(
        (
            2.0 * gamma * dgamma / tf.square(one_minus_gamma_sq),
            tf.zeros_like(gamma),
        ),
        axis=1,
    )
    value_total = tf.zeros_like(gamma)
    score_total = tf.zeros_like(values)
    valid = tf.ones_like(gamma, tf.bool)
    minimum_innovation_variance = tf.fill(
        tf.shape(gamma), tf.constant(float("inf"), tf.float64)
    )
    minimum_state_variance = variance
    maximum_weight_sum_error = tf.zeros_like(gamma)

    def condition(index, *_loop_values):
        return index < tf.shape(observations)[0]

    def body(
        index,
        current_mean,
        current_variance,
        current_d_mean,
        current_d_variance,
        current_value,
        current_score,
        current_valid,
        min_innovation_variance,
        min_state_variance,
        max_weight_sum_error,
    ):
        positive_time = index > 0
        predicted_mean = tf.where(
            positive_time, gamma * current_mean, current_mean
        )
        predicted_variance = tf.where(
            positive_time,
            tf.square(gamma) * current_variance + 1.0,
            current_variance,
        )
        predicted_d_mean = tf.where(
            positive_time,
            tf.stack(
                (
                    dgamma * current_mean + gamma * current_d_mean[:, 0],
                    gamma * current_d_mean[:, 1],
                ),
                axis=1,
            ),
            current_d_mean,
        )
        predicted_d_variance = tf.where(
            positive_time,
            tf.stack(
                (
                    2.0 * gamma * dgamma * current_variance
                    + tf.square(gamma) * current_d_variance[:, 0],
                    tf.square(gamma) * current_d_variance[:, 1],
                ),
                axis=1,
            ),
            current_d_variance,
        )

        innovation_variance = (
            predicted_variance[:, None] + component_variances[None, :]
        )
        safe_innovation_variance = tf.maximum(
            innovation_variance, _MIN_VARIANCE
        )
        observation_offset = (
            2.0 * tf.math.log(beta)[:, None] + component_means[None, :]
        )
        innovation = (
            observations[index] - predicted_mean[:, None] - observation_offset
        )
        log_component = (
            log_weights[None, :]
            - 0.5
            * (
                _LOG_TWO_PI
                + tf.math.log(safe_innovation_variance)
                + tf.square(innovation) / safe_innovation_variance
            )
        )
        d_innovation = (
            -predicted_d_mean[:, :, None] - d_observation_offset[:, :, None]
        )
        component_score = (
            -innovation[:, None, :] / safe_innovation_variance[:, None, :]
            * d_innovation
            + 0.5
            * (
                tf.square(innovation)[:, None, :]
                / tf.square(safe_innovation_variance)[:, None, :]
                - 1.0 / safe_innovation_variance[:, None, :]
            )
            * predicted_d_variance[:, :, None]
        )
        log_normalizer = tf.reduce_logsumexp(log_component, axis=1)
        normalized_weights = tf.exp(log_component - log_normalizer[:, None])
        log_normalizer_score = tf.reduce_sum(
            normalized_weights[:, None, :] * component_score, axis=2
        )

        gain = predicted_variance[:, None] / safe_innovation_variance
        d_gain = (
            component_variances[None, None, :]
            / tf.square(safe_innovation_variance)[:, None, :]
            * predicted_d_variance[:, :, None]
        )
        posterior_component_mean = predicted_mean[:, None] + gain * innovation
        d_posterior_component_mean = (
            predicted_d_mean[:, :, None]
            + d_gain * innovation[:, None, :]
            + gain[:, None, :] * d_innovation
        )
        posterior_component_variance = (
            predicted_variance[:, None]
            * component_variances[None, :]
            / safe_innovation_variance
        )
        d_posterior_component_variance = (
            tf.square(component_variances)[None, None, :]
            / tf.square(safe_innovation_variance)[:, None, :]
            * predicted_d_variance[:, :, None]
        )

        centered_score = component_score - log_normalizer_score[:, :, None]
        filtered_mean = tf.reduce_sum(
            normalized_weights * posterior_component_mean, axis=1
        )
        filtered_d_mean = tf.reduce_sum(
            normalized_weights[:, None, :]
            * (
                d_posterior_component_mean
                + centered_score * posterior_component_mean[:, None, :]
            ),
            axis=2,
        )
        component_second = (
            posterior_component_variance + tf.square(posterior_component_mean)
        )
        d_component_second = (
            d_posterior_component_variance
            + 2.0
            * posterior_component_mean[:, None, :]
            * d_posterior_component_mean
        )
        filtered_second = tf.reduce_sum(
            normalized_weights * component_second, axis=1
        )
        filtered_d_second = tf.reduce_sum(
            normalized_weights[:, None, :]
            * (d_component_second + centered_score * component_second[:, None, :]),
            axis=2,
        )
        filtered_variance = filtered_second - tf.square(filtered_mean)
        filtered_d_variance = (
            filtered_d_second - 2.0 * filtered_mean[:, None] * filtered_d_mean
        )

        weight_sum_error = tf.abs(
            tf.reduce_sum(normalized_weights, axis=1) - 1.0
        )
        step_valid = tf.logical_and(
            tf.math.is_finite(predicted_variance), predicted_variance > 0.0
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.reduce_all(
                tf.logical_and(
                    tf.math.is_finite(innovation_variance),
                    innovation_variance > 0.0,
                ),
                axis=1,
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.reduce_all(
                tf.logical_and(
                    tf.math.is_finite(posterior_component_variance),
                    posterior_component_variance > 0.0,
                ),
                axis=1,
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.logical_and(
                tf.math.is_finite(filtered_variance), filtered_variance > 0.0
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.logical_and(
                tf.math.is_finite(log_normalizer),
                tf.reduce_all(tf.math.is_finite(log_normalizer_score), axis=1),
            ),
        )
        step_valid = tf.logical_and(
            step_valid, weight_sum_error <= _WEIGHT_SUM_TOLERANCE
        )
        return (
            index + 1,
            filtered_mean,
            filtered_variance,
            filtered_d_mean,
            filtered_d_variance,
            current_value + log_normalizer,
            current_score + log_normalizer_score,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(
                min_innovation_variance,
                tf.reduce_min(innovation_variance, axis=1),
            ),
            tf.minimum(
                min_state_variance,
                tf.minimum(predicted_variance, filtered_variance),
            ),
            tf.maximum(max_weight_sum_error, weight_sum_error),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            mean,
            variance,
            d_mean,
            d_variance,
            value_total,
            score_total,
            valid,
            minimum_innovation_variance,
            minimum_state_variance,
            maximum_weight_sum_error,
        ),
        parallel_iterations=1,
    )
    value = result[5]
    score = result[6]
    valid = result[7]
    minimum_innovation_variance = result[8]
    minimum_state_variance = result[9]
    maximum_weight_sum_error = result[10]
    return value, score, {
        "status_code": tf.where(
            valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)
        ),
        "valid_pre_regularized_score": valid,
        "floor_count_value": tf.zeros_like(value, tf.int32),
        "min_innovation_eigenvalue": minimum_innovation_variance,
        "innovation_condition_estimate": tf.ones_like(value),
        "minimum_state_variance": minimum_state_variance,
        "maximum_mixture_weight_sum_error": maximum_weight_sum_error,
    }


def _posterior_value_score(
    theta: Any,
    *,
    transformed_observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    likelihood_value, likelihood_score = ksc_ukf_likelihood_value_score(
        theta,
        transformed_observations=transformed_observations,
        mixture_weights=mixture_weights,
        mixture_means=mixture_means,
        mixture_variances=mixture_variances,
    )
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
    )


class KSCUKFNeuTraAdapter:
    """Batch-native KSC principal-square-root-UKF posterior adapter."""

    dtype = tf.float64
    parameter_dim = 2
    parameter_names = KSC_UKF_PARAMETER_NAMES

    def __init__(
        self,
        *,
        raw_observations: tf.Tensor,
        transformed_observations: tf.Tensor,
        mixture_weights: tf.Tensor,
        mixture_means: tf.Tensor,
        mixture_variances: tf.Tensor,
        contract: SSMTargetContract,
    ) -> None:
        self.raw_observations = tf.convert_to_tensor(raw_observations, tf.float64)
        self.transformed_observations = tf.convert_to_tensor(
            transformed_observations, tf.float64
        )
        self.mixture_weights = tf.convert_to_tensor(mixture_weights, tf.float64)
        self.mixture_means = tf.convert_to_tensor(mixture_means, tf.float64)
        self.mixture_variances = tf.convert_to_tensor(
            mixture_variances, tf.float64
        )
        self.contract = contract
        self.target_scope = KSC_UKF_SCOPE
        payload = {
            "schema": "bayesfilter.testing.ksc_ukf_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "component_count": int(self.mixture_weights.shape[0]),
        }
        self._adapter_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_ksc_principal_sqrt_ukf_affine",
            evidence_path="bayesfilter/testing/ksc_ukf_neutra_target_tf.py",
            target_scope=self.target_scope,
            nonclaims=KSC_UKF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = _posterior_value_score(
            theta,
            transformed_observations=self.transformed_observations,
            mixture_weights=self.mixture_weights,
            mixture_means=self.mixture_means,
            mixture_variances=self.mixture_variances,
        )
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return _posterior_value_score(
            theta,
            transformed_observations=self.transformed_observations,
            mixture_weights=self.mixture_weights,
            mixture_means=self.mixture_means,
            mixture_variances=self.mixture_variances,
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
        likelihood_value, likelihood_score, status = (
            _ksc_ukf_likelihood_value_score_status(
                theta,
                transformed_observations=self.transformed_observations,
                mixture_weights=self.mixture_weights,
                mixture_means=self.mixture_means,
                mixture_variances=self.mixture_variances,
            )
        )
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = _ksc_ukf_likelihood_value_score_status(
            theta,
            transformed_observations=self.transformed_observations,
            mixture_weights=self.mixture_weights,
            mixture_means=self.mixture_means,
            mixture_variances=self.mixture_variances,
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status[
                "valid_pre_regularized_score"
            ],
        }


class KSCUKFLikelihoodRecomposer:
    """Independent likelihood component for posterior recomposition."""

    def __init__(self, adapter: KSCUKFNeuTraAdapter) -> None:
        self.transformed_observations = adapter.transformed_observations
        self.mixture_weights = adapter.mixture_weights
        self.mixture_means = adapter.mixture_means
        self.mixture_variances = adapter.mixture_variances

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return ksc_ukf_likelihood_value_score(
            theta,
            transformed_observations=self.transformed_observations,
            mixture_weights=self.mixture_weights,
            mixture_means=self.mixture_means,
            mixture_variances=self.mixture_variances,
        )


def make_ksc_ukf_neutra_adapter(
    *, raw_observations: tf.Tensor | None = None
) -> KSCUKFNeuTraAdapter:
    """Build the KSC-UKF adapter bound to one raw dataset."""

    if raw_observations is None:
        _states, raw_observations = generate_frozen_exact_sv_dataset_tf()
    raw = tf.convert_to_tensor(raw_observations, tf.float64)
    if raw.shape.rank != 2 or raw.shape[1] != 1:
        raise ValueError("raw observations must have shape [time, 1]")
    if not bool(tf.reduce_all(tf.math.is_finite(raw)).numpy()):
        raise ValueError("raw observations must be finite")
    transformed = transformed_ksc_observations(raw)
    mixture = ksc_1998_log_chi_square_mixture()
    raw_hash = _tensor_hash(raw)
    transformed_hash = _tensor_hash(transformed)
    mixture_hash = _tensor_bundle_hash(
        mixture.weights, mixture.means, mixture.variances
    )
    contract = make_ksc_ukf_target_contract(
        horizon=int(raw.shape[0]),
        raw_data_hash=raw_hash,
        transformed_data_hash=transformed_hash,
        mixture_hash=mixture_hash,
        mixture_source=mixture.source,
        component_count=mixture.component_count,
    )
    return KSCUKFNeuTraAdapter(
        raw_observations=raw,
        transformed_observations=transformed,
        mixture_weights=mixture.weights,
        mixture_means=mixture.means,
        mixture_variances=mixture.variances,
        contract=contract,
    )


def make_ksc_ukf_target_contract(
    *,
    horizon: int,
    raw_data_hash: str,
    transformed_data_hash: str,
    mixture_hash: str,
    mixture_source: str,
    component_count: int,
) -> SSMTargetContract:
    """Create the semantic KSC-UKF target contract."""

    shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=1,
        observation_dim=1,
        innovation_dim=1,
        parameter_dim=2,
    )
    model_semantics = {
        "model_id": "zhao-cui-synthetic-sv-fixed-sigma-1-ksc-mixture",
        "physical_truth": (0.6, 0.4, 1.0),
        "truth_role": "explanatory_only",
        "fixed_sigma": 1.0,
        "raw_observation_hash": f"sha256:{raw_data_hash}",
        "transformed_observation_hash": f"sha256:{transformed_data_hash}",
        "transform": "log(y^2+1e-8)",
        "transform_offset": KSC_UKF_TRANSFORM_OFFSET,
        "mixture_tensor_hash": f"sha256:{mixture_hash}",
        "mixture_source": str(mixture_source),
        "mixture_component_count": int(component_count),
    }
    problem = BayesianSSMProblem(
        problem_id="ksc-transformed-sv-principal-sqrt-ukf-source-chart",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id=KSC_UKF_DATASET_ID,
            observation_shape=(int(horizon), 1),
            data_hash=f"sha256:{raw_data_hash}",
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
        "parameter_order": KSC_UKF_PARAMETER_NAMES,
    }
    chart = ParameterChart(
        parameter_names=KSC_UKF_PARAMETER_NAMES,
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
        "filter_id": "ksc-principal-sqrt-ukf-affine-component-v1",
        "component_update": "scalar_affine_principal_sqrt_ukf_equals_gaussian_update",
        "mixture_collapse": "normalized_component_first_and_second_moments",
        "time_zero": "stationary_prior_then_observation_no_process_increment",
        "backend": "tensorflow_xla_tf_while_loop_tensorized_components",
        "score": "manual_forward_sensitivity_source_chart",
        "innovation_floor": None,
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "scalar_stationary_sv",
            "ksc_seven_component_observation_mixture",
            "principal_square_root_affine_component_update",
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
        raise ValueError("KSC-UKF target requires theta shape [batch, 2]")
    return values


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _tensor_bundle_hash(*values: tf.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
    return digest.hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
