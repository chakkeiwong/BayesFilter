"""Batch-native mass-preserving Gaussian-sum repair for the KSC target."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    generate_frozen_exact_sv_dataset_tf,
    source_chart_physical_parameters,
    source_two_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)
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
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.testing.ksc_gaussian_sum_ukf_scope import (
    KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP,
    KSC_GAUSSIAN_SUM_UKF_DATASET_ID,
    KSC_GAUSSIAN_SUM_UKF_HORIZON,
    KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES,
    KSC_GAUSSIAN_SUM_UKF_SCOPE,
    KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE,
)
KSC_GAUSSIAN_SUM_UKF_NONCLAIMS = (
    "KSC seven-component Gaussian-mixture transformed-SV target, not exact SV",
    "bounded mass-preserving Gaussian-sum UKF approximation, not exact latent-state filtering",
    "component pruning and moment merging are deterministic fixed target operations",
    "no HMC convergence, posterior correctness, superiority, or default-readiness claim",
)


_LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
_INVALID_LOG_WEIGHT = tf.constant(-1.0e100, tf.float64)
_MASS_FLOOR = tf.constant(1.0e-300, tf.float64)


def ksc_gaussian_sum_ukf_likelihood_value_score_status(
    theta: Any,
    *,
    transformed_observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
    component_cap: int,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate a deterministic reduced Gaussian-sum UKF approximation.

    Each affine KSC observation component uses the exact scalar Kalman/UKF
    update. Unlike the historical route, multiple posterior components are
    retained. Components outside the fixed cap are assigned to a retained
    top-weight center and moment-merged, preserving all normalized mass.
    """

    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 2:
        raise ValueError("theta must have shape [batch, 2]")
    if int(component_cap) < 7:
        raise ValueError("component_cap must be at least the KSC component count")
    observations = tf.reshape(
        tf.convert_to_tensor(transformed_observations, tf.float64), (-1,)
    )
    weights = tf.reshape(tf.convert_to_tensor(mixture_weights, tf.float64), (-1,))
    locations = tf.reshape(tf.convert_to_tensor(mixture_means, tf.float64), (-1,))
    variances = tf.reshape(
        tf.convert_to_tensor(mixture_variances, tf.float64), (-1,)
    )
    if int(weights.shape[0]) != 7:
        raise ValueError("the KSC repair requires exactly seven mixture components")

    with tf.GradientTape() as tape:
        tape.watch(values)
        value, diagnostics = _ksc_gaussian_sum_value(
            values,
            observations=observations,
            mixture_weights=weights,
            mixture_means=locations,
            mixture_variances=variances,
            component_cap=int(component_cap),
        )
    score = tape.gradient(value, values)
    if score is None:
        raise RuntimeError("TensorFlow did not produce the Gaussian-sum score")
    finite = tf.logical_and(
        tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1)
    )
    finite = tf.logical_and(finite, diagnostics["minimum_component_variance"] > 0.0)
    return value, score, {
        "status_code": tf.where(
            finite, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)
        ),
        "valid_pre_regularized_score": finite,
        "minimum_component_variance": diagnostics["minimum_component_variance"],
        "minimum_retained_mass_fraction": diagnostics[
            "minimum_retained_mass_fraction"
        ],
        "minimum_premerge_top_weight_mass_fraction": diagnostics[
            "minimum_premerge_top_weight_mass_fraction"
        ],
        "maximum_active_component_count": diagnostics[
            "maximum_active_component_count"
        ],
        "component_cap": tf.fill(tf.shape(value), tf.cast(component_cap, tf.int32)),
    }


def _ksc_gaussian_sum_value(
    theta: tf.Tensor,
    *,
    observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
    component_cap: int,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    batch_size = tf.shape(theta)[0]
    gamma, beta = source_chart_physical_parameters(theta)
    initial_variance = tf.math.reciprocal(1.0 - tf.square(gamma))
    means = tf.zeros([batch_size, component_cap], tf.float64)
    component_variances = tf.concat(
        (
            initial_variance[:, None],
            tf.ones([batch_size, component_cap - 1], tf.float64),
        ),
        axis=1,
    )
    log_weights = tf.concat(
        (
            tf.zeros([batch_size, 1], tf.float64),
            tf.fill([batch_size, component_cap - 1], _INVALID_LOG_WEIGHT),
        ),
        axis=1,
    )
    value = tf.zeros([batch_size], tf.float64)
    minimum_variance = initial_variance
    minimum_retained_mass = tf.ones([batch_size], tf.float64)
    minimum_premerge_mass = tf.ones([batch_size], tf.float64)
    maximum_active = tf.ones([batch_size], tf.int32)
    log_mixture_weights = tf.math.log(mixture_weights)

    def condition(index, *_):
        return index < tf.shape(observations)[0]

    def body(
        index,
        current_means,
        current_variances,
        current_log_weights,
        current_value,
        current_minimum_variance,
        current_minimum_retained_mass,
        current_minimum_premerge_mass,
        current_maximum_active,
    ):
        positive_time = index > 0
        predicted_means = tf.where(
            positive_time, gamma[:, None] * current_means, current_means
        )
        predicted_variances = tf.where(
            positive_time,
            tf.square(gamma)[:, None] * current_variances + 1.0,
            current_variances,
        )
        innovation_variances = (
            predicted_variances[:, :, None]
            + mixture_variances[None, None, :]
        )
        observation_offsets = (
            2.0 * tf.math.log(beta)[:, None, None]
            + mixture_means[None, None, :]
        )
        innovation = (
            observations[index]
            - predicted_means[:, :, None]
            - observation_offsets
        )
        expanded_log_weights = (
            current_log_weights[:, :, None]
            + log_mixture_weights[None, None, :]
            - 0.5
            * (
                _LOG_TWO_PI
                + tf.math.log(innovation_variances)
                + tf.square(innovation) / innovation_variances
            )
        )
        gains = predicted_variances[:, :, None] / innovation_variances
        posterior_means = predicted_means[:, :, None] + gains * innovation
        posterior_variances = (
            predicted_variances[:, :, None]
            * mixture_variances[None, None, :]
            / innovation_variances
        )
        flat_log_weights = tf.reshape(expanded_log_weights, [batch_size, -1])
        flat_means = tf.reshape(posterior_means, [batch_size, -1])
        flat_variances = tf.reshape(posterior_variances, [batch_size, -1])
        log_increment = tf.reduce_logsumexp(flat_log_weights, axis=1)
        normalized_flat = flat_log_weights - log_increment[:, None]
        center_log_weights, retained_indices = tf.math.top_k(
            normalized_flat, k=component_cap, sorted=True
        )
        center_means = tf.gather(
            flat_means, retained_indices, axis=1, batch_dims=1
        )
        center_variances = tf.gather(
            flat_variances, retained_indices, axis=1, batch_dims=1
        )
        premerge_mass = tf.exp(tf.reduce_logsumexp(center_log_weights, axis=1))

        # Assign every expanded component to a retained center, then preserve
        # its first two moments. The discrete assignment is deterministic; the
        # selected branch remains differentiable in its continuous quantities.
        sorted_center_indices = tf.argsort(
            center_means, axis=1, direction="ASCENDING", stable=True
        )
        sorted_center_means = tf.gather(
            center_means, sorted_center_indices, axis=1, batch_dims=1
        )
        insertion = tf.searchsorted(
            sorted_center_means, flat_means, side="left", out_type=tf.int32
        )
        left_position = tf.maximum(insertion - 1, 0)
        right_position = tf.minimum(insertion, component_cap - 1)
        left_mean = tf.gather(
            sorted_center_means, left_position, axis=1, batch_dims=1
        )
        right_mean = tf.gather(
            sorted_center_means, right_position, axis=1, batch_dims=1
        )
        selected_position = tf.where(
            tf.abs(flat_means - left_mean) <= tf.abs(flat_means - right_mean),
            left_position,
            right_position,
        )
        assignment = tf.gather(
            sorted_center_indices, selected_position, axis=1, batch_dims=1
        )
        flat_weights = tf.exp(normalized_flat)
        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, None] * component_cap
        global_assignment = tf.reshape(assignment + batch_offset, (-1,))
        segment_count = batch_size * component_cap
        retained_mass = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(flat_weights, (-1,)),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        )
        active = retained_mass > 0.0
        safe_mass = tf.where(active, retained_mass, tf.ones_like(retained_mass))
        retained_means = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(flat_weights * flat_means, (-1,)),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        ) / safe_mass
        retained_second = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(
                    flat_weights * (flat_variances + tf.square(flat_means)), (-1,)
                ),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        ) / safe_mass
        retained_variances = retained_second - tf.square(retained_means)
        retained_means = tf.where(active, retained_means, center_means)
        retained_variances = tf.where(
            active, retained_variances, center_variances
        )
        retained_log_weights = tf.where(
            active,
            tf.math.log(tf.maximum(retained_mass, _MASS_FLOOR)),
            _INVALID_LOG_WEIGHT,
        )
        total_retained_mass = tf.reduce_sum(retained_mass, axis=1)
        active_count = tf.reduce_sum(
            tf.cast(active, tf.int32), axis=1
        )
        return (
            index + 1,
            retained_means,
            retained_variances,
            retained_log_weights,
            current_value + log_increment,
            tf.minimum(
                current_minimum_variance,
                tf.reduce_min(retained_variances, axis=1),
            ),
            tf.minimum(current_minimum_retained_mass, total_retained_mass),
            tf.minimum(current_minimum_premerge_mass, premerge_mass),
            tf.maximum(current_maximum_active, active_count),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            means,
            component_variances,
            log_weights,
            value,
            minimum_variance,
            minimum_retained_mass,
            minimum_premerge_mass,
            maximum_active,
        ),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([None, component_cap]),
            tf.TensorShape([None, component_cap]),
            tf.TensorShape([None, component_cap]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        ),
        maximum_iterations=int(observations.shape[0]),
        parallel_iterations=1,
    )
    return result[4], {
        "minimum_component_variance": result[5],
        "minimum_retained_mass_fraction": result[6],
        "minimum_premerge_top_weight_mass_fraction": result[7],
        "maximum_active_component_count": result[8],
    }


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _tensor_bundle_hash(*values: tf.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
    return digest.hexdigest()


def make_ksc_gaussian_sum_ukf_target_contract(
    *,
    horizon: int,
    raw_data_hash: str,
    transformed_data_hash: str,
    mixture_hash: str,
    component_cap: int = KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP,
) -> SSMTargetContract:
    """Build the immutable scope contract for the repaired KSC route."""

    if int(component_cap) != KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP:
        raise ValueError("the admitted NeuTra scope is fixed to component cap 32")
    shape = SSMStaticShape(
        horizon=int(horizon), state_dim=1, observation_dim=1,
        innovation_dim=1, parameter_dim=2,
    )
    model_semantics = {
        "model_id": "zhao-cui-synthetic-sv-fixed-sigma-1-ksc-mixture",
        "fixed_sigma": 1.0,
        "raw_observation_hash": f"sha256:{raw_data_hash}",
        "transformed_observation_hash": f"sha256:{transformed_data_hash}",
        "transform": "log(y^2+1e-8)",
        "mixture_tensor_hash": f"sha256:{mixture_hash}",
        "mixture_component_count": 7,
        "component_cap": int(component_cap),
    }
    problem = BayesianSSMProblem(
        problem_id="ksc-transformed-sv-gaussian-sum-ukf-mass-preserving",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id=KSC_GAUSSIAN_SUM_UKF_DATASET_ID,
            observation_shape=(int(horizon), 1),
            data_hash=f"sha256:{raw_data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": "sha256:" + hashlib.sha256(
                json.dumps(model_semantics, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    chart_semantics = {
        "transform_id": "source-two-probit-uniform-box-chart",
        "gamma": "0.1+0.8*Phi(theta[0])",
        "beta": "0.1+0.8*Phi(theta[1])",
        "parameter_order": KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES,
    }
    chart = ParameterChart(
        parameter_names=KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES,
        unconstrained_dim=2, constrained_shape=(2,),
        transform_manifest={
            **chart_semantics,
            "transform_hash": "sha256:" + hashlib.sha256(
                json.dumps(chart_semantics, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
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
            "prior_hash": "sha256:" + hashlib.sha256(
                json.dumps(prior_semantics, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        support_policy="enforced_by_transform", log_density_authority="graph_native",
    )
    filter_semantics = {
        "filter_id": "ksc-gaussian-sum-mass-preserving-ukf-v1",
        "observation_components": 7,
        "component_cap": int(component_cap),
        "pruning": "top-weight-centers-with-nearest-center-moment-merge",
        "mass_policy": "all-normalized-component-mass-retained",
        "time_zero": "stationary-prior-then-observation-no-process-increment",
        "backend": "tensorflow_xla_batched_fixed_observation_while_loop",
        "score": "tensorflow_gradient_tape_same_program",
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "scalar_stationary_sv", "ksc_seven_component_observation_mixture",
            "mass_preserving_gaussian_sum_component_merge",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            **filter_semantics,
            "filter_hash": "sha256:" + hashlib.sha256(
                json.dumps(filter_semantics, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    return SSMTargetContract(problem=problem, chart=chart, prior=prior, filter_program=filter_program)


class KSCGaussianSumUKFNeuTraAdapter:
    """Batch-native posterior adapter for the admitted T20 cap-32 route."""

    dtype = tf.float64
    parameter_dim = 2
    parameter_names = KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES

    def __init__(self, *, raw_observations: tf.Tensor, transformed_observations: tf.Tensor,
                 mixture_weights: tf.Tensor, mixture_means: tf.Tensor,
                 mixture_variances: tf.Tensor, contract: SSMTargetContract) -> None:
        self.raw_observations = tf.convert_to_tensor(raw_observations, tf.float64)
        self.transformed_observations = tf.convert_to_tensor(transformed_observations, tf.float64)
        self.mixture_weights = tf.convert_to_tensor(mixture_weights, tf.float64)
        self.mixture_means = tf.convert_to_tensor(mixture_means, tf.float64)
        self.mixture_variances = tf.convert_to_tensor(mixture_variances, tf.float64)
        self.contract = contract
        self.target_scope = KSC_GAUSSIAN_SUM_UKF_SCOPE
        self.target_signature = stable_ssm_target_signature(contract)
        payload = {
            "schema": "bayesfilter.testing.ksc_gaussian_sum_ukf_neutra_adapter.v1",
            "target_signature": self.target_signature,
            "component_cap": KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP,
            "mixture_hash": _tensor_bundle_hash(self.mixture_weights, self.mixture_means, self.mixture_variances),
        }
        self._adapter_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_ksc_gaussian_sum_mass_preserving_ukf",
            evidence_path="bayesfilter/testing/ksc_gaussian_sum_ukf_neutra_target_tf.py",
            target_scope=self.target_scope, nonclaims=KSC_GAUSSIAN_SUM_UKF_NONCLAIMS,
        )

    def _posterior(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood, score, status = ksc_gaussian_sum_ukf_likelihood_value_score_status(
            theta, transformed_observations=self.transformed_observations,
            mixture_weights=self.mixture_weights, mixture_means=self.mixture_means,
            mixture_variances=self.mixture_variances,
            component_cap=KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP,
        )
        prior, prior_score = source_uniform_prior_value_score(theta)
        jacobian, jacobian_score = source_two_probit_jacobian_value_score(theta)
        return likelihood + prior + jacobian, score + prior_score + jacobian_score, status

    def log_prob(self, theta: Any) -> tf.Tensor:
        return self._posterior(theta)[0]

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self._posterior(theta)
        return value, score

    def neutra_batch_log_prob_and_grad_status(self, theta: Any):
        likelihood_value, likelihood_score, status = (
            ksc_gaussian_sum_ukf_likelihood_value_score_status(
                theta,
                transformed_observations=self.transformed_observations,
                mixture_weights=self.mixture_weights,
                mixture_means=self.mixture_means,
                mixture_variances=self.mixture_variances,
                component_cap=KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP,
            )
        )
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
        status = {
            **status,
            "floor_count_value": tf.zeros_like(status["status_code"]),
            "min_innovation_eigenvalue": status["minimum_component_variance"],
        }
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = self._posterior(theta)
        return {
            **status,
            "floor_count_value": tf.zeros_like(status["status_code"]),
            "min_innovation_eigenvalue": status["minimum_component_variance"],
            "innovation_condition_estimate": tf.ones_like(
                tf.cast(status["status_code"], tf.float64)
            ),
        }


def make_ksc_gaussian_sum_ukf_neutra_adapter(
    *, raw_observations: tf.Tensor | None = None,
) -> KSCGaussianSumUKFNeuTraAdapter:
    """Bind the repaired route to the frozen T20 KSC admission dataset."""

    if raw_observations is None:
        _states, raw_observations = generate_frozen_exact_sv_dataset_tf(
            horizon=KSC_GAUSSIAN_SUM_UKF_HORIZON
        )
    raw = tf.convert_to_tensor(raw_observations, tf.float64)
    if raw.shape != (KSC_GAUSSIAN_SUM_UKF_HORIZON, 1):
        raise ValueError("KSC Gaussian-sum NeuTra adapter requires raw shape [20, 1]")
    # Freeze the serialized scope input on CPU so its hash is independent of
    # whether the caller later initializes a GPU/XLA context.
    with tf.device("/CPU:0"):
        transformed = tf.math.log(
            tf.square(raw) + tf.constant(1.0e-8, tf.float64)
        )
    mixture = __import__(
        "bayesfilter.highdim.sv_mixture_cut4", fromlist=["ksc_1998_log_chi_square_mixture"]
    ).ksc_1998_log_chi_square_mixture()
    contract = make_ksc_gaussian_sum_ukf_target_contract(
        horizon=KSC_GAUSSIAN_SUM_UKF_HORIZON,
        raw_data_hash=_tensor_hash(raw), transformed_data_hash=_tensor_hash(transformed),
        mixture_hash=_tensor_bundle_hash(mixture.weights, mixture.means, mixture.variances),
    )
    return KSCGaussianSumUKFNeuTraAdapter(
        raw_observations=raw, transformed_observations=transformed,
        mixture_weights=mixture.weights, mixture_means=mixture.means,
        mixture_variances=mixture.variances, contract=contract,
    )


__all__ = [
    "KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP",
    "KSC_GAUSSIAN_SUM_UKF_HORIZON",
    "KSC_GAUSSIAN_SUM_UKF_SCOPE",
    "KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE",
    "KSCGaussianSumUKFNeuTraAdapter",
    "ksc_gaussian_sum_ukf_likelihood_value_score_status",
    "make_ksc_gaussian_sum_ukf_neutra_adapter",
    "make_ksc_gaussian_sum_ukf_target_contract",
]
