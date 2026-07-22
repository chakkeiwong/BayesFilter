"""Raw-observation SGQF route for the scalar generalized-SV source row."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.models import GeneralizedSVPriorMeanSSM
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
)


GENERALIZED_SV_SGQF_ROW_ID = (
    "zhao_cui_generalized_sv_synthetic_from_estimated_values"
)
GENERALIZED_SV_SGQF_ROUTE_ID = (
    "fixed_sgqf_generalized_sv_prior_mean_raw_y_level3_gaussian_projection_manual_score_v1"
)
GENERALIZED_SV_SGQF_TARGET_ID = (
    "zhao_cui_svmodels_prior_mean_tf_seed81105_transition_then_observe_raw_y_v1"
)
GENERALIZED_SV_SGQF_SEED = 81105
GENERALIZED_SV_SGQF_HORIZON = 1008
GENERALIZED_SV_SGQF_STATE_SHA256 = (
    "2a976493f58cc839667f8f4c892e92853dc232f5fe02fc617dad202b300ee358"
)
GENERALIZED_SV_SGQF_OBSERVATION_SHA256 = (
    "c990947b16d5ef108870fcb716836166a41b1fe31503b9cb5e0cd0459d4a53f2"
)

_DTYPE = tf.float64
_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), _DTYPE)
_MIN_VARIANCE = tf.constant(1.0e-12, _DTYPE)
_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, _DTYPE), scale=tf.constant(1.0, _DTYPE)
)


def _build_clouds():
    with tf.device("/CPU:0"):
        return {
            level: tf_fixed_sgqf_cloud(dim=1, sparse_level=level)
            for level in (2, 3, 5)
        }


_CLOUDS = _build_clouds()


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value, dtype=_DTYPE)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _theta_vector(theta: tf.Tensor) -> tf.Tensor:
    value = tf.convert_to_tensor(theta)
    if value.dtype != _DTYPE:
        raise ValueError("generalized-SV SGQF theta must use float64")
    if value.shape != (3,):
        raise ValueError("generalized-SV SGQF theta requires shape [3]")
    return value


def _observations(values: tf.Tensor) -> tf.Tensor:
    observations = tf.convert_to_tensor(values)
    if observations.dtype != _DTYPE:
        raise ValueError("generalized-SV SGQF observations must use float64")
    if observations.shape.rank != 2 or observations.shape[1] != 1:
        raise ValueError("generalized-SV SGQF observations require shape [T, 1]")
    return observations[:, 0]


def _cloud(level: int) -> tuple[tf.Tensor, tf.Tensor]:
    try:
        cloud = _CLOUDS[int(level)]
    except KeyError as exc:
        raise ValueError("generalized-SV SGQF level must be one of 2, 3, or 5") from exc
    weights = tf.convert_to_tensor(cloud.weights, _DTYPE)
    return tf.reshape(tf.convert_to_tensor(cloud.points, _DTYPE), [-1]), weights


def generalized_sv_sgqf_value_score_status(
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    sparse_level: int = 3,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the raw-y Gaussian-projection scalar and its manual score."""

    theta = _theta_vector(theta)
    y = _observations(observations)
    nodes, weights = _cloud(sparse_level)
    log_weights = tf.math.log(weights)

    gamma = _STD_NORMAL.cdf(theta[0])
    d_gamma = tf.stack([_STD_NORMAL.prob(theta[0]), 0.0, 0.0])
    tau = tf.exp(theta[1])
    d_tau = tf.stack([0.0, tau, 0.0])
    mu = theta[2] * tau
    d_mu = tf.stack([0.0, mu, tau])

    mean = mu
    variance = tf.math.reciprocal(1.0 - tf.square(gamma))
    d_mean = d_mu
    d_variance = 2.0 * gamma * d_gamma / tf.square(1.0 - tf.square(gamma))
    total_value = tf.constant(0.0, _DTYPE)
    total_score = tf.zeros([3], _DTYPE)
    valid = tf.constant(True)
    min_predictive = tf.constant(float("inf"), _DTYPE)
    min_filtered = tf.constant(float("inf"), _DTYPE)

    def body(
        time_index: tf.Tensor,
        current_mean: tf.Tensor,
        current_variance: tf.Tensor,
        current_d_mean: tf.Tensor,
        current_d_variance: tf.Tensor,
        value_total: tf.Tensor,
        score_total: tf.Tensor,
        current_valid: tf.Tensor,
        current_min_predictive: tf.Tensor,
        current_min_filtered: tf.Tensor,
    ):
        # The source program transitions before every observation, including y1.
        predicted_mean = mu + gamma * (current_mean - mu)
        d_predicted_mean = (
            d_mu
            + d_gamma * (current_mean - mu)
            + gamma * (current_d_mean - d_mu)
        )
        predicted_variance = tf.square(gamma) * current_variance + 1.0
        d_predicted_variance = (
            2.0 * gamma * d_gamma * current_variance
            + tf.square(gamma) * current_d_variance
        )
        predicted_scale = tf.sqrt(predicted_variance)
        d_predicted_scale = 0.5 * d_predicted_variance / predicted_scale
        points = predicted_mean + predicted_scale * nodes
        d_points = (
            d_predicted_mean[:, tf.newaxis]
            + d_predicted_scale[:, tf.newaxis] * nodes[tf.newaxis, :]
        )

        log_variance = tau * points
        d_log_variance = (
            d_tau[:, tf.newaxis] * points[tf.newaxis, :]
            + tau * d_points
        )
        standardized_square = tf.square(y[time_index]) * tf.exp(-log_variance)
        observation_log = -0.5 * (
            _LOG_TWO_PI + log_variance + standardized_square
        )
        d_observation_log = (
            0.5
            * (standardized_square - 1.0)[tf.newaxis, :]
            * d_log_variance
        )
        increment = tf.reduce_logsumexp(log_weights + observation_log)
        normalized_weights = tf.exp(log_weights + observation_log - increment)
        d_increment = tf.reduce_sum(
            normalized_weights[tf.newaxis, :] * d_observation_log, axis=1
        )
        centered_score = d_observation_log - d_increment[:, tf.newaxis]

        filtered_mean = tf.reduce_sum(normalized_weights * points)
        filtered_second = tf.reduce_sum(normalized_weights * tf.square(points))
        filtered_variance = filtered_second - tf.square(filtered_mean)
        d_filtered_mean = tf.reduce_sum(
            normalized_weights[tf.newaxis, :]
            * (d_points + centered_score * points[tf.newaxis, :]),
            axis=1,
        )
        d_filtered_second = tf.reduce_sum(
            normalized_weights[tf.newaxis, :]
            * (
                2.0 * points[tf.newaxis, :] * d_points
                + centered_score * tf.square(points)[tf.newaxis, :]
            ),
            axis=1,
        )
        d_filtered_variance = (
            d_filtered_second - 2.0 * filtered_mean * d_filtered_mean
        )
        step_valid = tf.logical_and(
            predicted_variance > _MIN_VARIANCE,
            filtered_variance > _MIN_VARIANCE,
        )
        step_valid = tf.logical_and(step_valid, tf.math.is_finite(increment))
        step_valid = tf.logical_and(
            step_valid, tf.reduce_all(tf.math.is_finite(d_increment))
        )

        return (
            time_index + 1,
            filtered_mean,
            filtered_variance,
            d_filtered_mean,
            d_filtered_variance,
            value_total + increment,
            score_total + d_increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predicted_variance),
            tf.minimum(current_min_filtered, filtered_variance),
        )

    result = tf.while_loop(
        lambda time_index, *_unused: time_index < tf.shape(y)[0],
        body,
        (
            tf.constant(0, tf.int32),
            mean,
            variance,
            d_mean,
            d_variance,
            total_value,
            total_score,
            valid,
            min_predictive,
            min_filtered,
        ),
        parallel_iterations=1,
    )

    return result[5], result[6], {
        "status_code": tf.where(result[7], tf.constant(0), tf.constant(1)),
        "valid_value_score": result[7],
        "transition_count": tf.shape(y)[0],
        "min_predictive_variance": result[8],
        "min_filtered_variance": result[9],
        "final_filtered_mean": result[1],
        "final_filtered_variance": result[2],
    }


def generalized_sv_sgqf_value_only_status(
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    sparse_level: int = 3,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the same scalar through an independent primal recurrence."""

    nodes, weights = _cloud(sparse_level)
    return _generalized_sv_value_only_with_rule(theta, observations, nodes, weights)


def generalized_sv_dense_value_reference_status(
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    order: int = 41,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Dense Gauss-Hermite reference for the same Gaussian-projection scalar."""

    if int(order) < 3:
        raise ValueError("dense generalized-SV reference order must be at least 3")
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(int(order))
    nodes = tf.constant(np.sqrt(2.0) * raw_nodes, _DTYPE)
    weights = tf.constant(raw_weights / np.sqrt(np.pi), _DTYPE)
    return _generalized_sv_value_only_with_rule(theta, observations, nodes, weights)


def _generalized_sv_value_only_with_rule(
    theta: tf.Tensor,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    theta = _theta_vector(theta)
    y = _observations(observations)
    nodes = tf.reshape(tf.convert_to_tensor(nodes, _DTYPE), [-1])
    weights = tf.reshape(tf.convert_to_tensor(weights, _DTYPE), [-1])
    if nodes.shape != weights.shape:
        raise ValueError("generalized-SV value rule requires aligned positive weights")
    log_weights = tf.math.log(weights)
    gamma = _STD_NORMAL.cdf(theta[0])
    tau = tf.exp(theta[1])
    mu = theta[2] * tau
    mean = mu
    variance = tf.math.reciprocal(1.0 - tf.square(gamma))
    total_value = tf.constant(0.0, _DTYPE)
    valid = tf.constant(True)
    min_predictive = tf.constant(float("inf"), _DTYPE)
    min_filtered = tf.constant(float("inf"), _DTYPE)

    def body(
        time_index: tf.Tensor,
        current_mean: tf.Tensor,
        current_variance: tf.Tensor,
        value_total: tf.Tensor,
        current_valid: tf.Tensor,
        current_min_predictive: tf.Tensor,
        current_min_filtered: tf.Tensor,
    ):
        predicted_mean = mu + gamma * (current_mean - mu)
        predicted_variance = tf.square(gamma) * current_variance + 1.0
        points = predicted_mean + tf.sqrt(predicted_variance) * nodes
        log_variance = tau * points
        observation_log = -0.5 * (
            _LOG_TWO_PI
            + log_variance
            + tf.square(y[time_index]) * tf.exp(-log_variance)
        )
        increment = tf.reduce_logsumexp(log_weights + observation_log)
        normalized_weights = tf.exp(log_weights + observation_log - increment)
        filtered_mean = tf.reduce_sum(normalized_weights * points)
        filtered_variance = (
            tf.reduce_sum(normalized_weights * tf.square(points))
            - tf.square(filtered_mean)
        )
        step_valid = tf.logical_and(
            predicted_variance > _MIN_VARIANCE,
            filtered_variance > _MIN_VARIANCE,
        )
        step_valid = tf.logical_and(step_valid, tf.math.is_finite(increment))
        return (
            time_index + 1,
            filtered_mean,
            filtered_variance,
            value_total + increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predicted_variance),
            tf.minimum(current_min_filtered, filtered_variance),
        )

    result = tf.while_loop(
        lambda time_index, *_unused: time_index < tf.shape(y)[0],
        body,
        (
            tf.constant(0, tf.int32),
            mean,
            variance,
            total_value,
            valid,
            min_predictive,
            min_filtered,
        ),
        parallel_iterations=1,
    )

    return result[3], {
        "status_code": tf.where(result[4], tf.constant(0), tf.constant(1)),
        "valid_value": result[4],
        "transition_count": tf.shape(y)[0],
        "min_predictive_variance": result[5],
        "min_filtered_variance": result[6],
        "final_filtered_mean": result[1],
        "final_filtered_variance": result[2],
    }


@dataclass(frozen=True)
class GeneralizedSVSGQFRoute:
    theta: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    sparse_level: int
    route_identity: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        theta = _theta_vector(self.theta)
        states = tf.convert_to_tensor(self.states, _DTYPE)
        observations = tf.convert_to_tensor(self.observations, _DTYPE)
        if states.shape != (GENERALIZED_SV_SGQF_HORIZON, 1):
            raise ValueError("canonical generalized-SV states require shape [1008, 1]")
        if observations.shape != (GENERALIZED_SV_SGQF_HORIZON, 1):
            raise ValueError("canonical generalized-SV observations require shape [1008, 1]")
        if _tensor_hash(states) != GENERALIZED_SV_SGQF_STATE_SHA256:
            raise ValueError("generalized-SV canonical state identity rejected")
        if _tensor_hash(observations) != GENERALIZED_SV_SGQF_OBSERVATION_SHA256:
            raise ValueError("generalized-SV canonical observation identity rejected")
        manifest = dict(self.manifest)
        if self.route_identity != _semantic_hash(manifest):
            raise ValueError("generalized-SV SGQF route identity rejected")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))

    def value_score_status(self) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return generalized_sv_sgqf_value_score_status(
            self.theta, self.observations, sparse_level=self.sparse_level
        )

    def value_only_status(self) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
        return generalized_sv_sgqf_value_only_status(
            self.theta, self.observations, sparse_level=self.sparse_level
        )


def make_generalized_sv_sgqf_route(
    *, sparse_level: int = 3
) -> GeneralizedSVSGQFRoute:
    """Issue the sealed full-horizon scalar raw-y SGQF route."""

    if int(sparse_level) != 3:
        raise ValueError("canonical generalized-SV route requires sparse level 3")
    # Fixture generation and identity hashing are CPU-pinned so GPU visibility
    # cannot change the sealed byte identity before the requested XLA run.
    with tf.device("/CPU:0"):
        data = _generalized_sv_prior_mean_dataset(GENERALIZED_SV_SGQF_SEED)
        theta = tf.convert_to_tensor(data["truth_theta"], _DTYPE)
        states = tf.convert_to_tensor(data["states"], _DTYPE)
        observations = tf.convert_to_tensor(data["observations"], _DTYPE)
    nodes, weights = _cloud(sparse_level)
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(nodes).numpy())
        + bytes(tf.io.serialize_tensor(weights).numpy())
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema": "bayesfilter.generalized_sv_source_sgqf_route.v1",
        "row_id": GENERALIZED_SV_SGQF_ROW_ID,
        "route_id": GENERALIZED_SV_SGQF_ROUTE_ID,
        "target_id": GENERALIZED_SV_SGQF_TARGET_ID,
        "result_kind": "value_score",
        "parameter_coordinate": "theta=(z_gamma,log_tau,mu_over_tau)",
        "seed": GENERALIZED_SV_SGQF_SEED,
        "horizon": GENERALIZED_SV_SGQF_HORIZON,
        "time_order": "stationary_x0_then_1008_transition_then_observe_steps",
        "state_sha256": GENERALIZED_SV_SGQF_STATE_SHA256,
        "observation_sha256": GENERALIZED_SV_SGQF_OBSERVATION_SHA256,
        "cloud_level": int(sparse_level),
        "cloud_point_count": int(nodes.shape[0]),
        "cloud_sha256": cloud_hash,
        "dtype": "float64",
        "model_family": GeneralizedSVPriorMeanSSM().manifest_payload()["family"],
        "observation_target": "raw_y_zero_mean_normal_variance_exp_tau_x",
        "approximation": "sequential_gaussian_projection_after_direct_likelihood_quadrature",
        "level_selection": (
            "level3 selected after level2/3/5 full-horizon ladder; level3-level5 "
            "value gap 6.0507077e-5 versus level2-level5 gap 4.7830381e-3"
        ),
        "score": "manual_forward_sensitivity_of_the_same_fixed_sgqf_scalar",
        "source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:34"
        ),
        "nonclaims": [
            "not exact nonlinear likelihood or exact posterior",
            "not NativeGeneralizedSVSSM and not a two-state target",
            "not actual-SV or KSC transformed-observation evidence",
            "not adaptive MATLAB TT-cross/SIRT reproduction",
            "not SGQF superiority or default-readiness evidence",
        ],
    }
    return GeneralizedSVSGQFRoute(
        theta=theta,
        states=states,
        observations=observations,
        sparse_level=int(sparse_level),
        route_identity=_semantic_hash(manifest),
        manifest=manifest,
    )


__all__ = [
    "GENERALIZED_SV_SGQF_HORIZON",
    "GENERALIZED_SV_SGQF_OBSERVATION_SHA256",
    "GENERALIZED_SV_SGQF_ROUTE_ID",
    "GENERALIZED_SV_SGQF_ROW_ID",
    "GENERALIZED_SV_SGQF_SEED",
    "GENERALIZED_SV_SGQF_STATE_SHA256",
    "GENERALIZED_SV_SGQF_TARGET_ID",
    "GeneralizedSVSGQFRoute",
    "generalized_sv_dense_value_reference_status",
    "generalized_sv_sgqf_value_only_status",
    "generalized_sv_sgqf_value_score_status",
    "make_generalized_sv_sgqf_route",
]
