"""TensorFlow nonlinear model fixtures and tiny reference oracles.

These helpers are testing tools.  They keep benchmark model laws in one place
so value, score, branch, and benchmark tests do not copy equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import (
    TFSigmaPointRule,
    tf_svd_sigma_point_placement,
)
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace, make_affine_structural_tf


@dataclass(frozen=True)
class DenseProjectionStep:
    """One-step Gaussian moment-projection reference output."""

    log_likelihood: tf.Tensor
    predicted_mean: tf.Tensor
    predicted_covariance: tf.Tensor
    observation_mean: tf.Tensor
    observation_covariance: tf.Tensor
    cross_covariance: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_covariance: tf.Tensor
    deterministic_residual: tf.Tensor


def model_a_observations_tf() -> tf.Tensor:
    """Fixed observations for the affine Gaussian structural oracle."""

    return tf.constant([[0.2], [0.05], [-0.1], [0.15], [0.0]], dtype=tf.float64)


def model_b_observations_tf() -> tf.Tensor:
    """Fixed observations for the nonlinear accumulation fixture."""

    return tf.constant([[0.10], [0.04], [0.16]], dtype=tf.float64)


def model_c_observations_tf() -> tf.Tensor:
    """Fixed observations for the autonomous nonlinear growth fixture."""

    return tf.constant([[0.40], [1.20]], dtype=tf.float64)


def make_affine_gaussian_structural_oracle_tf() -> TFStructuralStateSpace:
    """Return Model A, an affine Gaussian structural oracle."""

    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.constant([[0.35, -0.10], [1.0, 0.0]], dtype=tf.float64),
        innovation_matrix=tf.constant([[0.25], [0.0]], dtype=tf.float64),
        innovation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
        name="model_a_affine_gaussian_structural_oracle",
    )


def make_nonlinear_accumulation_model_tf(
    *,
    rho: float = 0.70,
    sigma: float = 0.25,
    alpha: float = 0.55,
    beta: float = 0.80,
    observation_sigma: float = 0.30,
) -> TFStructuralStateSpace:
    """Return Model B, a smooth nonlinear structural accumulation fixture."""

    rho_t = tf.constant(rho, dtype=tf.float64)
    sigma_t = tf.constant(sigma, dtype=tf.float64)
    alpha_t = tf.constant(alpha, dtype=tf.float64)
    beta_t = tf.constant(beta, dtype=tf.float64)
    partition = StatePartition(
        state_names=("m", "k"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    config = StructuralFilterConfig(
        integration_space="innovation",
        deterministic_completion="required",
        approximation_label="nonlinear_accumulation_testing_fixture",
    )

    def transition_fn(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(previous_state).shape.rank == 1
        previous = _as_points(previous_state, name="previous_state")
        eps = _as_points(innovation, name="innovation")[:, 0]
        m_next = rho_t * previous[:, 0] + sigma_t * eps
        k_next = alpha_t * previous[:, 1] + beta_t * tf.math.tanh(m_next)
        next_points = tf.stack([m_next, k_next], axis=1)
        return next_points[0] if was_vector else next_points

    def observation_fn(state_points: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(state_points).shape.rank == 1
        states = _as_points(state_points, name="state_points")
        observations = (states[:, 0] + states[:, 1])[:, tf.newaxis]
        return observations[0] if was_vector else observations

    def residual_fn(
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        previous = _as_points(previous_state, name="previous_state")
        next_points = _as_points(next_state, name="next_state")
        expected = alpha_t * previous[:, 1] + beta_t * tf.math.tanh(next_points[:, 0])
        return (next_points[:, 1] - expected)[:, tf.newaxis]

    return TFStructuralStateSpace(
        partition=partition,
        config=config,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(tf.constant([0.25, 0.20], dtype=tf.float64)),
        innovation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[observation_sigma**2]], dtype=tf.float64),
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        deterministic_residual_fn=residual_fn,
        name="model_b_nonlinear_accumulation",
    )


def make_univariate_nonlinear_growth_model_tf(
    *,
    process_sigma: float = 1.0,
    observation_sigma: float = 1.0,
    initial_variance: float = 0.20,
) -> TFStructuralStateSpace:
    """Return Model C as an autonomous phase-state testing fixture."""

    process_sigma_t = tf.constant(process_sigma, dtype=tf.float64)
    partition = StatePartition(
        state_names=("x", "tau"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    config = StructuralFilterConfig(
        integration_space="innovation",
        deterministic_completion="required",
        approximation_label="autonomous_phase_nonlinear_growth_testing_fixture",
    )

    def transition_fn(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(previous_state).shape.rank == 1
        previous = _as_points(previous_state, name="previous_state")
        eps = _as_points(innovation, name="innovation")[:, 0]
        x_prev = previous[:, 0]
        tau_prev = previous[:, 1]
        x_next = (
            0.5 * x_prev
            + 25.0 * x_prev / (1.0 + tf.square(x_prev))
            + 8.0 * tf.math.cos(1.2 * tau_prev)
            + process_sigma_t * eps
        )
        tau_next = tau_prev + 1.0
        next_points = tf.stack([x_next, tau_next], axis=1)
        return next_points[0] if was_vector else next_points

    def observation_fn(state_points: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(state_points).shape.rank == 1
        states = _as_points(state_points, name="state_points")
        observations = (tf.square(states[:, 0]) / 20.0)[:, tf.newaxis]
        return observations[0] if was_vector else observations

    def residual_fn(
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        del innovation
        previous = _as_points(previous_state, name="previous_state")
        next_points = _as_points(next_state, name="next_state")
        expected_tau = previous[:, 1] + 1.0
        return (next_points[:, 1] - expected_tau)[:, tf.newaxis]

    return TFStructuralStateSpace(
        partition=partition,
        config=config,
        initial_mean=tf.constant([0.0, 1.0], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(
            tf.constant([initial_variance, 0.0], dtype=tf.float64)
        ),
        innovation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[observation_sigma**2]], dtype=tf.float64),
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        deterministic_residual_fn=residual_fn,
        name="model_c_autonomous_nonlinear_growth",
    )


def dense_gaussian_projection_step(
    *,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    innovation_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation: tf.Tensor,
    transition_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    deterministic_residual_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    nodes_per_dim: int = 9,
) -> DenseProjectionStep:
    """Dense tensor-product Gaussian moment projection for one filter step.

    This oracle approximates the Gaussian projection used by sigma-point
    filters.  It is intentionally small-dimensional and testing-only.
    """

    mean = tf.convert_to_tensor(mean, dtype=tf.float64)
    covariance = _symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    innovation_covariance = _symmetrize(
        tf.convert_to_tensor(innovation_covariance, dtype=tf.float64)
    )
    observation_covariance = _symmetrize(
        tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    )
    observation = tf.convert_to_tensor(observation, dtype=tf.float64)
    if observation.shape.rank == 0:
        observation = observation[tf.newaxis]
    if observation.shape.rank != 1:
        raise ValueError("observation must be scalar or one-dimensional")

    aug_mean = tf.concat(
        [mean, tf.zeros([tf.shape(innovation_covariance)[0]], dtype=tf.float64)],
        axis=0,
    )
    upper = tf.concat(
        [covariance, tf.zeros([tf.shape(mean)[0], tf.shape(innovation_covariance)[0]], dtype=tf.float64)],
        axis=1,
    )
    lower = tf.concat(
        [
            tf.zeros([tf.shape(innovation_covariance)[0], tf.shape(mean)[0]], dtype=tf.float64),
            innovation_covariance,
        ],
        axis=1,
    )
    aug_covariance = tf.concat([upper, lower], axis=0)
    points, weights = _dense_gaussian_quadrature_points(
        aug_mean,
        aug_covariance,
        nodes_per_dim=nodes_per_dim,
    )
    state_dim = int(mean.shape[0])
    previous_points = points[:, :state_dim]
    innovation_points = points[:, state_dim:]
    predicted_points = transition_fn(previous_points, innovation_points)
    residuals = deterministic_residual_fn(
        previous_points,
        innovation_points,
        predicted_points,
    )
    if residuals.shape[-1] == 0:
        deterministic_residual = tf.constant(0.0, dtype=tf.float64)
    else:
        deterministic_residual = tf.reduce_max(tf.abs(residuals))
    predicted_mean = tf.linalg.matvec(tf.transpose(predicted_points), weights)
    centered_x = predicted_points - predicted_mean[tf.newaxis, :]
    predicted_covariance = _weighted_covariance(centered_x, weights)
    observation_points = observation_fn(predicted_points)
    observation_mean = tf.linalg.matvec(tf.transpose(observation_points), weights)
    centered_y = observation_points - observation_mean[tf.newaxis, :]
    innovation_cov = _symmetrize(_weighted_covariance(centered_y, weights) + observation_covariance)
    cross_covariance = tf.transpose(centered_x) @ (centered_y * weights[:, tf.newaxis])
    innovation = observation - observation_mean
    chol = tf.linalg.cholesky(innovation_cov)
    precision_innovation = tf.linalg.cholesky_solve(chol, innovation[:, tf.newaxis])[:, 0]
    innovation_precision = tf.linalg.cholesky_solve(
        chol,
        tf.eye(int(observation.shape[0]), dtype=tf.float64),
    )
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    mahalanobis = tf.reduce_sum(innovation * precision_innovation)
    obs_dim = tf.cast(tf.shape(observation)[0], tf.float64)
    log_likelihood = -0.5 * (
        obs_dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=tf.float64))
        + log_det
        + mahalanobis
    )
    gain = cross_covariance @ innovation_precision
    filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
    filtered_covariance = _symmetrize(
        predicted_covariance - gain @ innovation_cov @ tf.transpose(gain)
    )
    return DenseProjectionStep(
        log_likelihood=log_likelihood,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        observation_mean=observation_mean,
        observation_covariance=innovation_cov,
        cross_covariance=cross_covariance,
        filtered_mean=filtered_mean,
        filtered_covariance=filtered_covariance,
        deterministic_residual=deterministic_residual,
    )


def dense_projection_first_step(
    model: TFStructuralStateSpace,
    observation: tf.Tensor,
    *,
    nodes_per_dim: int = 9,
) -> DenseProjectionStep:
    """Dense Gaussian projection reference for the first step of a model."""

    return dense_gaussian_projection_step(
        mean=model.initial_mean,
        covariance=model.initial_covariance,
        innovation_covariance=model.innovation_covariance,
        observation_covariance=model.observation_covariance,
        observation=observation,
        transition_fn=model.transition,
        observation_fn=model.observe,
        deterministic_residual_fn=model.deterministic_residual,
        nodes_per_dim=nodes_per_dim,
    )


def sigma_point_projection_first_step(
    model: TFStructuralStateSpace,
    observation: tf.Tensor,
    *,
    sigma_rule: TFSigmaPointRule,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
) -> DenseProjectionStep:
    """One-step Gaussian projection induced by a fixed sigma-point rule."""

    observation = tf.convert_to_tensor(observation, dtype=tf.float64)
    if observation.shape.rank == 0:
        observation = observation[tf.newaxis]
    if observation.shape.rank != 1:
        raise ValueError("observation must be scalar or one-dimensional")

    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = _symmetrize(model.initial_covariance)
    innovation_covariance = _symmetrize(model.innovation_covariance)
    state_dim = int(mean.shape[0])
    innovation_dim = int(innovation_covariance.shape[0])
    aug_mean = tf.concat(
        [mean, tf.zeros([innovation_dim], dtype=tf.float64)],
        axis=0,
    )
    upper = tf.concat(
        [covariance, tf.zeros([state_dim, innovation_dim], dtype=tf.float64)],
        axis=1,
    )
    lower = tf.concat(
        [tf.zeros([innovation_dim, state_dim], dtype=tf.float64), innovation_covariance],
        axis=1,
    )
    aug_covariance = tf.concat([upper, lower], axis=0)
    aug_points, _diagnostics = tf_svd_sigma_point_placement(
        aug_mean,
        aug_covariance,
        sigma_rule,
        singular_floor=placement_floor,
    )
    previous_points = aug_points[:, :state_dim]
    innovation_points = aug_points[:, state_dim:]
    predicted_points = model.transition(previous_points, innovation_points)
    residuals = model.deterministic_residual(
        previous_points,
        innovation_points,
        predicted_points,
    )
    if residuals.shape[-1] == 0:
        deterministic_residual = tf.constant(0.0, dtype=tf.float64)
    else:
        deterministic_residual = tf.reduce_max(tf.abs(residuals))
    predicted_mean = tf.linalg.matvec(
        tf.transpose(predicted_points),
        sigma_rule.mean_weights,
    )
    centered_x = predicted_points - predicted_mean[tf.newaxis, :]
    predicted_covariance = _weighted_covariance(centered_x, sigma_rule.covariance_weights)
    observation_points = model.observe(predicted_points)
    observation_mean = tf.linalg.matvec(
        tf.transpose(observation_points),
        sigma_rule.mean_weights,
    )
    centered_y = observation_points - observation_mean[tf.newaxis, :]
    raw_observation_covariance = _symmetrize(
        _weighted_covariance(centered_y, sigma_rule.covariance_weights)
        + _symmetrize(model.observation_covariance)
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(raw_observation_covariance)
    floored = tf.maximum(eigenvalues, tf.cast(innovation_floor, tf.float64))
    implemented_observation_covariance = _symmetrize(
        eigenvectors @ tf.linalg.diag(floored) @ tf.transpose(eigenvectors)
    )
    cross_covariance = tf.transpose(centered_x) @ (
        centered_y * sigma_rule.covariance_weights[:, tf.newaxis]
    )
    innovation = observation - observation_mean
    chol = tf.linalg.cholesky(implemented_observation_covariance)
    precision_innovation = tf.linalg.cholesky_solve(chol, innovation[:, tf.newaxis])[:, 0]
    observation_precision = tf.linalg.cholesky_solve(
        chol,
        tf.eye(int(observation.shape[0]), dtype=tf.float64),
    )
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    mahalanobis = tf.reduce_sum(innovation * precision_innovation)
    obs_dim = tf.cast(tf.shape(observation)[0], tf.float64)
    log_likelihood = -0.5 * (
        obs_dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=tf.float64))
        + log_det
        + mahalanobis
    )
    gain = cross_covariance @ observation_precision
    filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
    filtered_covariance = _symmetrize(
        predicted_covariance
        - gain @ implemented_observation_covariance @ tf.transpose(gain)
    )
    return DenseProjectionStep(
        log_likelihood=log_likelihood,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        observation_mean=observation_mean,
        observation_covariance=implemented_observation_covariance,
        cross_covariance=cross_covariance,
        filtered_mean=filtered_mean,
        filtered_covariance=filtered_covariance,
        deterministic_residual=deterministic_residual,
    )


def _as_points(values: tf.Tensor, *, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        return tensor[tf.newaxis, :]
    if tensor.shape.rank == 2:
        return tensor
    raise ValueError(f"{name} must be one- or two-dimensional")


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return _symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _dense_gaussian_quadrature_points(
    mean: tf.Tensor,
    covariance: tf.Tensor,
    *,
    nodes_per_dim: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, raw_weights = tf.numpy_function(
        _hermgauss,
        [tf.constant(nodes_per_dim, dtype=tf.int32)],
        [tf.float64, tf.float64],
    )
    nodes.set_shape([nodes_per_dim])
    raw_weights.set_shape([nodes_per_dim])
    standard_nodes = tf.sqrt(tf.constant(2.0, dtype=tf.float64)) * nodes
    standard_weights = raw_weights / tf.sqrt(tf.constant(3.141592653589793, dtype=tf.float64))
    dim = int(mean.shape[0])
    grids = tf.meshgrid(*([standard_nodes] * dim), indexing="ij")
    weight_grids = tf.meshgrid(*([standard_weights] * dim), indexing="ij")
    offsets = tf.stack([tf.reshape(grid, [-1]) for grid in grids], axis=1)
    weights = tf.ones([tf.shape(offsets)[0]], dtype=tf.float64)
    for grid in weight_grids:
        weights = weights * tf.reshape(grid, [-1])
    eigenvalues, eigenvectors = tf.linalg.eigh(_symmetrize(covariance))
    floored = tf.maximum(eigenvalues, tf.constant(0.0, dtype=tf.float64))
    factor = eigenvectors @ tf.linalg.diag(tf.sqrt(floored))
    points = mean[tf.newaxis, :] + offsets @ tf.transpose(factor)
    return points, weights


def _hermgauss(nodes_per_dim: tf.Tensor) -> tuple[object, object]:
    import numpy as np

    return np.polynomial.hermite.hermgauss(int(nodes_per_dim))
