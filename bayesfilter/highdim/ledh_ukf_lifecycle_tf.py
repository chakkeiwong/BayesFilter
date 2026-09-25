"""Per-particle UKF covariance lifecycle for the canonical LEDH lane (P1).

Implements the prediction and update arrows of Li(2017) Algorithm 1's
covariance lifecycle (ch19c eq. bf-pfpf-alg1-covariance-lifecycle) and the
triple-resampling rule (eq. bf-pfpf-alg1-resampling-triple), vectorized
over particles with one covariance per particle.

Semantic authority: the reviewed single-particle implementation in
``experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py``
(additive-noise unscented form, June 2026 Alg1-UKF campaign). This module
vectorizes that construction; the unscented weights and sigma-point
placement follow the same rule (alpha=1, beta=2, kappa=0 defaults ->
mean weight lambda/(d+lambda), covariance weight adds (1-alpha^2+beta)).
Cholesky sigma placement with symmetrization and jitter replaces the SVD
placement for the vectorized path; on SPD inputs the two agree, and the
conformance gates (C-1 linear-Kalman equality) bind the result.

Contract step ids: ukf_predict, ukf_update, triple_resampling
(``bayesfilter.highdim.ledh_alg1_contract``).
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

Tensor = tf.Tensor

DEFAULT_JITTER = 1.0e-12


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _unscented_weights(
    dim: int, dtype: tf.dtypes.DType, *, alpha: float, beta: float, kappa: float
) -> tuple[Tensor, Tensor, float]:
    lam = alpha * alpha * (dim + kappa) - dim
    count = 2 * dim + 1
    mean_w0 = lam / (dim + lam)
    cov_w0 = mean_w0 + (1.0 - alpha * alpha + beta)
    rest = 1.0 / (2.0 * (dim + lam))
    mean_weights = tf.concat(
        [
            tf.constant([mean_w0], dtype=dtype),
            tf.constant([rest] * (count - 1), dtype=dtype),
        ],
        axis=0,
    )
    cov_weights = tf.concat(
        [
            tf.constant([cov_w0], dtype=dtype),
            tf.constant([rest] * (count - 1), dtype=dtype),
        ],
        axis=0,
    )
    return mean_weights, cov_weights, dim + lam


def _sigma_points(
    means: Tensor, covariances: Tensor, scale: float, jitter: float
) -> Tensor:
    """[N, 2d+1, d] sigma points per particle via jittered Cholesky."""

    dim = tf.shape(means)[1]
    stabilized = _sym(covariances) + tf.cast(jitter, means.dtype) * tf.eye(
        dim, dtype=means.dtype
    )
    scaled_chol = tf.linalg.cholesky(
        tf.cast(scale, means.dtype) * stabilized
    )
    offsets = tf.linalg.matrix_transpose(scaled_chol)  # rows are sqrt cols
    plus = means[:, None, :] + offsets
    minus = means[:, None, :] - offsets
    return tf.concat([means[:, None, :], plus, minus], axis=1)


def ukf_predict_per_particle(
    states: Tensor,
    covariances: Tensor,
    transition_mean_fn: Callable[[Tensor], Tensor],
    process_noise_covariance: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    jitter: float = DEFAULT_JITTER,
) -> tuple[Tensor, Tensor]:
    """Additive-noise unscented prediction, one covariance per particle.

    ``states``: [N, d]; ``covariances``: [N, d, d];
    ``transition_mean_fn``: maps [M, d] -> [M, d] (deterministic dynamics);
    returns predicted means [N, d] and predicted covariances [N, d, d].
    """

    states = tf.convert_to_tensor(states)
    covariances = tf.convert_to_tensor(covariances, states.dtype)
    dim = int(states.shape[1])
    count = tf.shape(states)[0]
    mean_w, cov_w, scale = _unscented_weights(
        dim, states.dtype, alpha=alpha, beta=beta, kappa=kappa
    )
    points = _sigma_points(states, covariances, scale, jitter)
    flat = tf.reshape(points, [-1, dim])
    pushed = tf.reshape(
        transition_mean_fn(flat), [count, 2 * dim + 1, dim]
    )
    predicted_means = tf.einsum("s,nsd->nd", mean_w, pushed)
    centered = pushed - predicted_means[:, None, :]
    predicted_covariances = _sym(
        tf.einsum("s,nsi,nsj->nij", cov_w, centered, centered)
        + tf.cast(process_noise_covariance, states.dtype)[None]
    )
    return predicted_means, predicted_covariances


def ukf_update_per_particle(
    predicted_means: Tensor,
    predicted_covariances: Tensor,
    observation_mean_fn: Callable[[Tensor], Tensor],
    observation_covariance: Tensor,
    observation: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    jitter: float = DEFAULT_JITTER,
) -> tuple[Tensor, Tensor]:
    """Additive-noise unscented update, one covariance per particle.

    Returns posterior means [N, d] and posterior covariances [N, d, d]
    (the P_k^i of the Algorithm 1 lifecycle).
    """

    predicted_means = tf.convert_to_tensor(predicted_means)
    dtype = predicted_means.dtype
    predicted_covariances = tf.convert_to_tensor(predicted_covariances, dtype)
    observation = tf.convert_to_tensor(observation, dtype)
    dim = int(predicted_means.shape[1])
    obs_dim = int(observation.shape[-1])
    count = tf.shape(predicted_means)[0]
    mean_w, cov_w, scale = _unscented_weights(
        dim, dtype, alpha=alpha, beta=beta, kappa=kappa
    )
    points = _sigma_points(predicted_means, predicted_covariances, scale, jitter)
    flat = tf.reshape(points, [-1, dim])
    observed = tf.reshape(
        observation_mean_fn(flat), [count, 2 * dim + 1, obs_dim]
    )
    observed_means = tf.einsum("s,nso->no", mean_w, observed)
    centered_x = points - predicted_means[:, None, :]
    centered_y = observed - observed_means[:, None, :]
    innovation_cov = _sym(
        tf.einsum("s,nsi,nsj->nij", cov_w, centered_y, centered_y)
        + tf.cast(observation_covariance, dtype)[None]
    )
    cross_cov = tf.einsum("s,nsi,nsj->nij", cov_w, centered_x, centered_y)
    chol = tf.linalg.cholesky(
        innovation_cov
        + tf.cast(jitter, dtype) * tf.eye(obs_dim, dtype=dtype)
    )
    gain = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(cross_cov))
    )
    innovation = observation[None, :] - observed_means
    posterior_means = predicted_means + tf.einsum(
        "nij,nj->ni", gain, innovation
    )
    posterior_covariances = _sym(
        predicted_covariances
        - tf.einsum("nij,njk,nlk->nil", gain, innovation_cov, gain)
    )
    return posterior_means, posterior_covariances


def triple_gather(
    states: Tensor,
    covariances: Tensor,
    weights: Tensor,
    ancestry_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resample the Algorithm 1 triple {x, P, w} together (C-7 contract)."""

    return (
        tf.gather(states, ancestry_indices),
        tf.gather(covariances, ancestry_indices),
        tf.gather(weights, ancestry_indices),
    )


__all__ = [
    "ukf_predict_per_particle",
    "ukf_update_per_particle",
    "triple_gather",
]
