"""Experimental precision policies for the selected SSL-LSTM UKF target.

This module is deliberately isolated from the production float64 contracts. It
mirrors the four-coordinate principal-square-root UKF score so mixed and
float32 precision can be tested without changing the admitted target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import ssl_lstm_parameter_slices


PrecisionPolicy = Literal["all_float64", "mixed_lstm32_filter64", "all_float32_tf32"]


class SSLPrecisionValueScore(NamedTuple):
    value: tf.Tensor
    score: tf.Tensor
    placement_floor_count: tf.Tensor
    innovation_floor_count: tf.Tensor
    max_factor_reconstruction_residual: tf.Tensor
    final_mean: tf.Tensor
    final_covariance: tf.Tensor


@dataclass(frozen=True)
class _Parameters:
    config: SSLLSTMStaticConfig
    lstm_input: tf.Tensor
    lstm_recurrent: tf.Tensor
    lstm_bias: tf.Tensor
    latent_weight: tf.Tensor
    latent_bias: tf.Tensor
    observation_weight: tf.Tensor
    observation_bias: tf.Tensor
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    innovation_covariance: tf.Tensor
    observation_covariance: tf.Tensor


def policy_dtypes(policy: PrecisionPolicy) -> tuple[tf.DType, tf.DType]:
    """Return ``(model_dtype, filter_dtype)`` for an experiment policy."""

    if policy == "all_float64":
        return tf.float64, tf.float64
    if policy == "mixed_lstm32_filter64":
        return tf.float32, tf.float64
    if policy == "all_float32_tf32":
        return tf.float32, tf.float32
    raise ValueError(f"unknown SSL-LSTM precision policy: {policy}")


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return (matrix + tf.linalg.matrix_transpose(matrix)) * tf.cast(0.5, matrix.dtype)


def _unpack(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    model_dtype: tf.DType,
    filter_dtype: tf.DType,
    std_floor: float = 1.0e-4,
) -> _Parameters:
    values = tf.cast(tf.convert_to_tensor(theta), filter_dtype)
    slices = ssl_lstm_parameter_slices(config)
    k = int(config.latent_dim)
    h = int(config.hidden_dim)
    d = int(config.observation_dim)
    n = int(config.augmented_state_dim)

    def take(start: int, size: int, dtype: tf.DType) -> tf.Tensor:
        return tf.cast(values[start : start + size], dtype)

    lstm_input = tf.reshape(take(slices.lstm_input_start, 4 * h * k, model_dtype), [4, h, k])
    lstm_recurrent = tf.reshape(
        take(slices.lstm_recurrent_start, 4 * h * h, model_dtype), [4, h, h]
    )
    lstm_bias = tf.reshape(take(slices.lstm_bias_start, 4 * h, model_dtype), [4, h])
    latent_weight = tf.reshape(take(slices.latent_weight_start, k * h, model_dtype), [k, h])
    latent_bias = take(slices.latent_bias_start, k, model_dtype)
    observation_weight = tf.reshape(
        take(slices.observation_weight_start, d * k, model_dtype), [d, k]
    )
    observation_bias = take(slices.observation_bias_start, d, model_dtype)
    initial_mean = take(slices.initial_mean_start, n, filter_dtype)
    floor = tf.cast(std_floor, filter_dtype)
    initial_std = tf.nn.softplus(take(slices.initial_std_start, n, filter_dtype)) + floor
    process_std = tf.nn.softplus(take(slices.process_std_start, k, filter_dtype)) + floor
    observation_std = (
        tf.nn.softplus(take(slices.observation_std_start, d, filter_dtype)) + floor
    )
    return _Parameters(
        config=config,
        lstm_input=lstm_input,
        lstm_recurrent=lstm_recurrent,
        lstm_bias=lstm_bias,
        latent_weight=latent_weight,
        latent_bias=latent_bias,
        observation_weight=observation_weight,
        observation_bias=observation_bias,
        initial_mean=initial_mean,
        initial_covariance=tf.linalg.diag(tf.square(initial_std)),
        innovation_covariance=tf.linalg.diag(tf.square(process_std)),
        observation_covariance=tf.linalg.diag(tf.square(observation_std)),
    )


def _gates(params: _Parameters, points: tf.Tensor) -> dict[str, tf.Tensor]:
    dtype = params.lstm_input.dtype
    values = tf.cast(points, dtype)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    z_prev = values[:, :k]
    a_prev = values[:, k : k + h]
    c_prev = values[:, k + h :]
    pre = (
        tf.einsum("ghk,rk->rgh", params.lstm_input, z_prev)
        + tf.einsum("ghj,rj->rgh", params.lstm_recurrent, a_prev)
        + params.lstm_bias[tf.newaxis, :, :]
    )
    input_gate = tf.math.sigmoid(pre[:, 0, :])
    forget_gate = tf.math.sigmoid(pre[:, 1, :])
    output_gate = tf.math.sigmoid(pre[:, 2, :])
    candidate = tf.math.tanh(pre[:, 3, :])
    cell = forget_gate * c_prev + input_gate * candidate
    return {
        "input": input_gate,
        "forget": forget_gate,
        "output": output_gate,
        "candidate": candidate,
        "cell": cell,
        "c_prev": c_prev,
    }


def _transition_model(params: _Parameters, points: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    gates = _gates(params, points)
    hidden = gates["output"] * tf.math.tanh(gates["cell"])
    latent = hidden @ tf.transpose(params.latent_weight) + params.latent_bias[tf.newaxis, :]
    return tf.concat([latent, hidden, gates["cell"]], axis=1), gates


def _transition_and_derivatives(
    params: _Parameters,
    previous_points: tf.Tensor,
    innovation_points: tf.Tensor,
    d_previous_points: tf.Tensor,
    d_innovation_points: tf.Tensor,
    *,
    filter_dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor]:
    model_dtype = params.lstm_input.dtype
    deterministic, gates = _transition_model(params, previous_points)
    tangents = tf.cast(d_previous_points, model_dtype)
    k = int(params.config.latent_dim)
    h = int(params.config.hidden_dim)
    n = int(params.config.augmented_state_dim)
    d_z_prev = tangents[:, :, :k]
    d_a_prev = tangents[:, :, k : k + h]
    d_c_prev = tangents[:, :, k + h :]
    d_preactivation = (
        tf.einsum("ghk,prk->prgh", params.lstm_input, d_z_prev)
        + tf.einsum("ghj,prj->prgh", params.lstm_recurrent, d_a_prev)
    )
    d_input = gates["input"][tf.newaxis, :, :] * (
        tf.cast(1.0, model_dtype) - gates["input"][tf.newaxis, :, :]
    ) * d_preactivation[:, :, 0, :]
    d_forget = gates["forget"][tf.newaxis, :, :] * (
        tf.cast(1.0, model_dtype) - gates["forget"][tf.newaxis, :, :]
    ) * d_preactivation[:, :, 1, :]
    d_output = gates["output"][tf.newaxis, :, :] * (
        tf.cast(1.0, model_dtype) - gates["output"][tf.newaxis, :, :]
    ) * d_preactivation[:, :, 2, :]
    d_candidate = (
        tf.cast(1.0, model_dtype) - tf.square(gates["candidate"])
    )[tf.newaxis, :, :] * d_preactivation[:, :, 3, :]
    d_cell = (
        d_forget * gates["c_prev"][tf.newaxis, :, :]
        + gates["forget"][tf.newaxis, :, :] * d_c_prev
        + d_input * gates["candidate"][tf.newaxis, :, :]
        + gates["input"][tf.newaxis, :, :] * d_candidate
    )
    tanh_cell = tf.math.tanh(gates["cell"])
    d_hidden = (
        d_output * tanh_cell[tf.newaxis, :, :]
        + gates["output"][tf.newaxis, :, :]
        * (tf.cast(1.0, model_dtype) - tf.square(tanh_cell))[tf.newaxis, :, :]
        * d_cell
    )
    d_latent = tf.einsum("kh,prh->prk", params.latent_weight, d_hidden)
    propagated = tf.cast(tf.concat([d_latent, d_hidden, d_cell], axis=2), filter_dtype)
    propagated += tf.concat(
        [d_innovation_points, tf.zeros([4, tf.shape(previous_points)[0], 2 * h], filter_dtype)],
        axis=2,
    )

    point_count = tf.shape(previous_points)[0]
    direct_model = tf.zeros([4, point_count, n], model_dtype)
    hidden = deterministic[:, k : k + h]
    latent_weight_row = tf.concat(
        [hidden[:, 0:1], tf.zeros([point_count, k - 1], model_dtype)], axis=1
    )
    latent_bias_row = tf.broadcast_to(
        tf.one_hot(0, k, dtype=model_dtype)[tf.newaxis, :], [point_count, k]
    )
    direct_model = tf.tensor_scatter_nd_update(
        direct_model,
        tf.constant([[0], [1]], tf.int32),
        tf.stack(
            [
                tf.concat([latent_weight_row, tf.zeros([point_count, 2 * h], model_dtype)], axis=1),
                tf.concat([latent_bias_row, tf.zeros([point_count, 2 * h], model_dtype)], axis=1),
            ],
            axis=0,
        ),
    )
    predicted = tf.cast(deterministic, filter_dtype)
    predicted = tf.concat(
        [predicted[:, :k] + innovation_points, predicted[:, k:]], axis=1
    )
    return predicted, propagated + tf.cast(direct_model, filter_dtype)


def _observation_and_derivatives(
    params: _Parameters,
    points: tf.Tensor,
    d_points: tf.Tensor,
    *,
    filter_dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor]:
    model_dtype = params.observation_weight.dtype
    values = tf.cast(points, model_dtype)
    tangents = tf.cast(d_points, model_dtype)
    k = int(params.config.latent_dim)
    z = values[:, :k]
    observed = z @ tf.transpose(params.observation_weight) + params.observation_bias[tf.newaxis, :]
    propagated = tf.einsum(
        "dk,prk->prd", params.observation_weight, tangents[:, :, :k]
    )
    point_count = tf.shape(points)[0]
    direct = tf.zeros([4, point_count, 1], model_dtype)
    direct = tf.tensor_scatter_nd_update(
        direct,
        tf.constant([[2], [3]], tf.int32),
        tf.stack([z[:, 0:1], tf.ones([point_count, 1], model_dtype)], axis=0),
    )
    return tf.cast(observed, filter_dtype), tf.cast(propagated + direct, filter_dtype)


def _principal_sqrt_first_derivatives(
    covariance: tf.Tensor,
    d_covariance: tf.Tensor,
    floor: tf.Tensor,
    *,
    require_no_floor: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    dtype = covariance.dtype
    covariance = _symmetrize(covariance)
    d_covariance = _symmetrize(d_covariance)
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    floored = tf.maximum(eigenvalues, floor)
    floor_count = tf.reduce_sum(tf.cast(eigenvalues <= floor, tf.int32))
    if require_no_floor:
        with tf.control_dependencies(
            [tf.debugging.assert_equal(floor_count, 0, message="precision experiment active floor")]
        ):
            floored = tf.identity(floored)
    sqrt_eigenvalues = tf.sqrt(floored)
    factor = eigenvectors @ tf.linalg.diag(sqrt_eigenvalues) @ tf.transpose(eigenvectors)
    projected = (
        tf.transpose(eigenvectors)[tf.newaxis, :, :]
        @ d_covariance
        @ eigenvectors[tf.newaxis, :, :]
    )
    denominator = sqrt_eigenvalues[:, tf.newaxis] + sqrt_eigenvalues[tf.newaxis, :]
    d_factor = _symmetrize(
        eigenvectors[tf.newaxis, :, :]
        @ (projected / denominator[tf.newaxis, :, :])
        @ tf.transpose(eigenvectors)[tf.newaxis, :, :]
    )
    implemented = _symmetrize(factor @ tf.transpose(factor))
    reconstructed = (
        tf.matmul(d_factor, factor[tf.newaxis, :, :], transpose_b=True)
        + tf.matmul(factor[tf.newaxis, :, :], d_factor, transpose_b=True)
    )
    residual = tf.reduce_max(tf.linalg.norm(reconstructed - d_covariance, axis=[-2, -1]))
    return factor, d_factor, implemented, tf.cast(residual, dtype), floor_count


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return _symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _weighted_covariance_derivative(
    centered: tf.Tensor, d_centered: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    return _symmetrize(
        tf.einsum("r,pri,rj->pij", weights, d_centered, centered)
        + tf.einsum("r,ri,prj->pij", weights, centered, d_centered)
    )


def _factor_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    rhs_matrix = rhs[:, tf.newaxis] if rhs.shape.rank == 1 else rhs
    first = tf.linalg.triangular_solve(factor, rhs_matrix, lower=True)
    second = tf.linalg.triangular_solve(tf.transpose(factor), first, lower=False)
    return second[:, 0] if rhs.shape.rank == 1 else second


def ssl_lstm_precision_value_and_score(
    free: tf.Tensor,
    fixture: tf.Tensor,
    observations: tf.Tensor,
    config: SSLLSTMStaticConfig,
    free_indices: tuple[int, ...],
    *,
    policy: PrecisionPolicy,
    prior_center: tf.Tensor,
    prior_standard_deviation: float,
) -> SSLPrecisionValueScore:
    """Evaluate the selected target under one explicit precision policy."""

    model_dtype, dtype = policy_dtypes(policy)
    free = tf.ensure_shape(tf.cast(free, dtype), [4])
    full = tf.tensor_scatter_nd_update(
        tf.cast(fixture, dtype),
        tf.constant([[index] for index in free_indices], tf.int32),
        free,
    )
    params = _unpack(full, config, model_dtype=model_dtype, filter_dtype=dtype)
    y = tf.cast(observations, dtype)
    p = 4
    k = int(config.latent_dim)
    n = int(config.augmented_state_dim)
    aug_dim = n + k
    point_count = 2 * aug_dim + 1
    eye = tf.eye(aug_dim, dtype=dtype)
    scale = tf.sqrt(tf.cast(aug_dim, dtype))
    offsets = tf.concat([tf.zeros([1, aug_dim], dtype), scale * eye, -scale * eye], axis=0)
    axis_weight = tf.cast(1.0 / (2.0 * aug_dim), dtype)
    mean_weights = tf.concat(
        [tf.zeros([1], dtype), tf.fill([2 * aug_dim], axis_weight)], axis=0
    )
    covariance_weights = tf.concat(
        [tf.fill([1], tf.cast(2.0, dtype)), tf.fill([2 * aug_dim], axis_weight)], axis=0
    )

    mean = params.initial_mean
    covariance = params.initial_covariance
    d_mean = tf.zeros([p, n], dtype)
    d_covariance = tf.zeros([p, n, n], dtype)
    innovation_covariance = params.innovation_covariance
    d_innovation_covariance = tf.zeros([p, k, k], dtype)
    observation_covariance = params.observation_covariance
    d_observation_covariance = tf.zeros([p, 1, 1], dtype)
    observation_identity = tf.eye(1, dtype=dtype)
    regularized_float32 = policy == "all_float32_tf32"
    # Pure FP32 loses strict positive definiteness on this recursion. Retaining
    # it as a timed diagnostic requires a visible changed-branch floor.
    placement_floor = tf.cast(1.0e-6 if regularized_float32 else 0.0, dtype)
    innovation_floor = tf.cast(1.0e-6 if regularized_float32 else 1.0e-12, dtype)
    log_likelihood = tf.cast(0.0, dtype)
    score = tf.zeros([p], dtype)
    max_placement_floor_count = tf.constant(0, tf.int32)
    max_innovation_floor_count = tf.constant(0, tf.int32)
    max_factor_residual = tf.cast(0.0, dtype)

    for t in range(int(config.horizon)):
        aug_mean = tf.concat([mean, tf.zeros([k], dtype)], axis=0)
        d_aug_mean = tf.concat([d_mean, tf.zeros([p, k], dtype)], axis=1)
        aug_covariance = tf.concat(
            [
                tf.concat([covariance, tf.zeros([n, k], dtype)], axis=1),
                tf.concat([tf.zeros([k, n], dtype), innovation_covariance], axis=1),
            ],
            axis=0,
        )
        d_aug_covariance = tf.concat(
            [
                tf.concat([d_covariance, tf.zeros([p, n, k], dtype)], axis=2),
                tf.concat(
                    [tf.zeros([p, k, n], dtype), d_innovation_covariance], axis=2
                ),
            ],
            axis=1,
        )
        placement_factor, d_placement_factor, _, placement_residual, placement_count = (
            _principal_sqrt_first_derivatives(
                aug_covariance,
                d_aug_covariance,
                placement_floor,
                require_no_floor=not regularized_float32,
            )
        )
        point_offsets = offsets @ tf.transpose(placement_factor)
        aug_points = aug_mean[tf.newaxis, :] + point_offsets
        d_aug_points = d_aug_mean[:, tf.newaxis, :] + tf.einsum(
            "rd,pad->pra", offsets, d_placement_factor
        )
        previous_points = aug_points[:, :n]
        innovation_points = aug_points[:, n:]
        predicted_points, d_predicted_points = _transition_and_derivatives(
            params,
            previous_points,
            innovation_points,
            d_aug_points[:, :, :n],
            d_aug_points[:, :, n:],
            filter_dtype=dtype,
        )
        predicted_mean = tf.linalg.matvec(tf.transpose(predicted_points), mean_weights)
        d_predicted_mean = tf.einsum("r,prn->pn", mean_weights, d_predicted_points)
        centered_x = predicted_points - predicted_mean[tf.newaxis, :]
        d_centered_x = d_predicted_points - d_predicted_mean[:, tf.newaxis, :]
        predicted_covariance = _weighted_covariance(centered_x, covariance_weights)
        d_predicted_covariance = _weighted_covariance_derivative(
            centered_x, d_centered_x, covariance_weights
        )

        observation_points, d_observation_points = _observation_and_derivatives(
            params, predicted_points, d_predicted_points, filter_dtype=dtype
        )
        observation_mean = tf.linalg.matvec(tf.transpose(observation_points), mean_weights)
        d_observation_mean = tf.einsum(
            "r,prm->pm", mean_weights, d_observation_points
        )
        centered_y = observation_points - observation_mean[tf.newaxis, :]
        d_centered_y = d_observation_points - d_observation_mean[:, tf.newaxis, :]
        raw_innovation_covariance = _symmetrize(
            _weighted_covariance(centered_y, covariance_weights) + observation_covariance
        )
        d_raw_innovation_covariance = _symmetrize(
            _weighted_covariance_derivative(centered_y, d_centered_y, covariance_weights)
            + d_observation_covariance
        )
        innovation_factor, _, implemented_innovation, innovation_residual, innovation_count = (
            _principal_sqrt_first_derivatives(
                raw_innovation_covariance,
                d_raw_innovation_covariance,
                innovation_floor,
                require_no_floor=not regularized_float32,
            )
        )
        cross_covariance = tf.transpose(centered_x) @ (
            centered_y * covariance_weights[:, tf.newaxis]
        )
        d_cross_covariance = (
            tf.einsum("r,prn,rm->pnm", covariance_weights, d_centered_x, centered_y)
            + tf.einsum("r,rn,prm->pnm", covariance_weights, centered_x, d_centered_y)
        )
        innovation = y[t] - observation_mean
        d_innovation = -d_observation_mean
        solve_innovation = _factor_solve(innovation_factor, innovation)
        innovation_precision = _factor_solve(innovation_factor, observation_identity)
        log_det = tf.cast(2.0, dtype) * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor))
        )
        mahalanobis = tf.reduce_sum(innovation * solve_innovation)
        log_likelihood += -tf.cast(0.5, dtype) * (
            tf.math.log(tf.cast(2.0 * math.pi, dtype)) + log_det + mahalanobis
        )
        trace_terms = tf.einsum(
            "ab,pba->p", innovation_precision, d_raw_innovation_covariance
        )
        innovation_derivative_terms = tf.einsum(
            "pm,m->p", d_innovation, solve_innovation
        )
        covariance_quadratic_terms = tf.einsum(
            "m,pmn,n->p",
            solve_innovation,
            d_raw_innovation_covariance,
            solve_innovation,
        )
        score += -tf.cast(0.5, dtype) * (
            trace_terms
            + tf.cast(2.0, dtype) * innovation_derivative_terms
            - covariance_quadratic_terms
        )
        kalman_gain = cross_covariance @ innovation_precision
        gain_transpose = tf.transpose(kalman_gain)
        d_kalman_gain = tf.stack(
            [
                tf.transpose(
                    _factor_solve(
                        innovation_factor,
                        tf.transpose(d_cross_covariance[i])
                        - d_raw_innovation_covariance[i] @ gain_transpose,
                    )
                )
                for i in range(p)
            ],
            axis=0,
        )
        mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
        d_mean = (
            d_predicted_mean
            + tf.einsum("pnm,m->pn", d_kalman_gain, innovation)
            + tf.einsum("nm,pm->pn", kalman_gain, d_innovation)
        )
        covariance = _symmetrize(
            predicted_covariance
            - kalman_gain @ implemented_innovation @ tf.transpose(kalman_gain)
        )
        d_covariance = _symmetrize(
            d_predicted_covariance
            - tf.einsum(
                "pnm,ml,kl->pnk", d_kalman_gain, implemented_innovation, kalman_gain
            )
            - tf.einsum(
                "nm,pml,kl->pnk", kalman_gain, d_raw_innovation_covariance, kalman_gain
            )
            - tf.einsum(
                "nm,ml,pkl->pnk", kalman_gain, implemented_innovation, d_kalman_gain
            )
        )
        max_factor_residual = tf.maximum(
            max_factor_residual, tf.maximum(placement_residual, innovation_residual)
        )
        max_placement_floor_count = tf.maximum(
            max_placement_floor_count, placement_count
        )
        max_innovation_floor_count = tf.maximum(
            max_innovation_floor_count, innovation_count
        )

    delta = free - tf.cast(prior_center, dtype)
    variance = tf.cast(prior_standard_deviation**2, dtype)
    value = log_likelihood - tf.cast(0.5, dtype) * tf.reduce_sum(tf.square(delta) / variance)
    score = score - delta / variance
    return SSLPrecisionValueScore(
        value=tf.ensure_shape(value, []),
        score=tf.ensure_shape(score, [4]),
        placement_floor_count=max_placement_floor_count,
        innovation_floor_count=max_innovation_floor_count,
        max_factor_reconstruction_residual=max_factor_residual,
        final_mean=mean,
        final_covariance=covariance,
    )


__all__ = [
    "PrecisionPolicy",
    "SSLPrecisionValueScore",
    "policy_dtypes",
    "ssl_lstm_precision_value_and_score",
]
