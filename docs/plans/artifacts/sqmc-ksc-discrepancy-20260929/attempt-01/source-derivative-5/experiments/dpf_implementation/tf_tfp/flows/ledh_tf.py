"""TensorFlow local EDH/LEDH affine flow for experimental PF-PF diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf


DTYPE = tf.float64


@dataclass(frozen=True)
class LedhFlowBatchResult:
    pre_flow_particles: tf.Tensor
    post_flow_particles: tf.Tensor
    pre_flow_log_density: tf.Tensor
    forward_log_det: tf.Tensor
    local_posterior_means: tf.Tensor
    local_posterior_covariances: tf.Tensor
    diagnostics: dict[str, Any]


def ledh_flow_batch_tf(
    *,
    pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    jitter: float = 1e-9,
) -> LedhFlowBatchResult:
    """Map bootstrap transition proposals through local Gaussian LEDH transports.

    The returned `forward_log_det` uses the frozen local-affine LEDH map
    convention: the observation Jacobian is evaluated at each proposal particle
    and then held fixed for the per-particle affine transport determinant.
    """

    x0_batch = tf.cast(pre_flow_particles, DTYPE)
    ancestors = tf.cast(ancestors, DTYPE)
    observation = tf.cast(observation, DTYPE)
    transition_matrix = tf.cast(transition_matrix, DTYPE)
    transition_covariance = _stabilize_covariance(transition_covariance, jitter)
    observation_covariance = _stabilize_covariance(observation_covariance, jitter)
    prior_means = tf.linalg.matmul(ancestors, transition_matrix, transpose_b=True)
    pre_flow_log_density = gaussian_logpdf_tf(x0_batch - prior_means, transition_covariance)

    mapped = []
    logdets = []
    posterior_means = []
    posterior_covariances = []
    min_singular_values = []
    max_singular_values = []
    transform_signs = []

    for x0, prior_mean in zip(tf.unstack(x0_batch, axis=0), tf.unstack(prior_means, axis=0)):
        x1, post_mean, post_cov, affine_transform, logabsdet = _local_ledh_map_tf(
            x0=x0,
            prior_mean=prior_mean,
            prior_covariance=transition_covariance,
            observation=observation,
            observation_covariance=observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            jitter=jitter,
        )
        sign = tf.linalg.det(affine_transform)
        singular_values = tf.linalg.svd(affine_transform, compute_uv=False)
        mapped.append(x1)
        logdets.append(logabsdet)
        posterior_means.append(post_mean)
        posterior_covariances.append(post_cov)
        min_singular_values.append(tf.reduce_min(singular_values))
        max_singular_values.append(tf.reduce_max(singular_values))
        transform_signs.append(sign)

    post_flow_particles = tf.stack(mapped, axis=0)
    forward_log_det = tf.stack(logdets, axis=0)
    local_posterior_means = tf.stack(posterior_means, axis=0)
    local_posterior_covariances = tf.stack(posterior_covariances, axis=0)
    min_singular_tensor = tf.stack(min_singular_values, axis=0)
    max_singular_tensor = tf.stack(max_singular_values, axis=0)
    transform_sign_tensor = tf.stack(transform_signs, axis=0)
    diagnostics = {
        "component_id": "tf_tfp_ledh_local_affine_flow",
        "map_convention": "x1 = local_posterior_mean + L_post L_prior^{-1}(x0 - prior_mean)",
        "forward_log_det": "frozen_local_affine_log_abs_det",
        "forward_log_det_scope": "local_observation_jacobian_held_fixed_per_particle",
        "pre_flow_density": "transition_prior_q0",
        "local_linearization": "per_particle_observation_jacobian",
        "finite_pre_flow": _finite_bool(x0_batch),
        "finite_post_flow": _finite_bool(post_flow_particles),
        "finite_forward_log_det": _finite_bool(forward_log_det),
        "finite_pre_flow_log_density": _finite_bool(pre_flow_log_density),
        "min_forward_log_det": _float(tf.reduce_min(forward_log_det)),
        "max_forward_log_det": _float(tf.reduce_max(forward_log_det)),
        "max_abs_forward_log_det": _float(tf.reduce_max(tf.abs(forward_log_det))),
        "min_jacobian_singular_value": _float(tf.reduce_min(min_singular_tensor)),
        "max_jacobian_singular_value": _float(tf.reduce_max(max_singular_tensor)),
        "min_affine_transform_det": _float(tf.reduce_min(transform_sign_tensor)),
        "max_affine_transform_det": _float(tf.reduce_max(transform_sign_tensor)),
        "backend": "tensorflow",
    }
    if not diagnostics["finite_post_flow"] or not diagnostics["finite_forward_log_det"]:
        raise FloatingPointError("LEDH flow emitted non-finite map or log-det values")
    if diagnostics["min_jacobian_singular_value"] <= 1e-12:
        raise FloatingPointError("LEDH flow Jacobian is numerically singular")
    return LedhFlowBatchResult(
        pre_flow_particles=x0_batch,
        post_flow_particles=post_flow_particles,
        pre_flow_log_density=pre_flow_log_density,
        forward_log_det=forward_log_det,
        local_posterior_means=local_posterior_means,
        local_posterior_covariances=local_posterior_covariances,
        diagnostics=diagnostics,
    )


def gaussian_logpdf_tf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.cast(residuals, DTYPE)
    if len(residuals.shape) == 1:
        residuals = residuals[None, :]
    covariance = _stabilize_covariance(covariance)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE)) + logdet + quad)


def _local_ledh_map_tf(
    *,
    x0: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_covariance: tf.Tensor,
    observation: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    jitter: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    prior_covariance = _stabilize_covariance(prior_covariance, jitter)
    observation_covariance = _stabilize_covariance(observation_covariance, jitter)
    prior_chol = tf.linalg.cholesky(prior_covariance)
    prior_precision = tf.linalg.cholesky_solve(
        prior_chol,
        tf.eye(int(prior_covariance.shape[0]), dtype=DTYPE),
    )
    obs_chol = tf.linalg.cholesky(observation_covariance)
    obs_precision = tf.linalg.cholesky_solve(
        obs_chol,
        tf.eye(int(observation_covariance.shape[0]), dtype=DTYPE),
    )
    h_ref = tf.cast(observation_fn(x0), DTYPE)
    h_jac = tf.cast(observation_jacobian_fn(x0), DTYPE)
    residual = tf.reshape(observation_residual_fn(h_ref, observation), [-1])
    pseudo_observation = tf.linalg.matvec(h_jac, x0) + residual
    post_precision = prior_precision + tf.transpose(h_jac) @ obs_precision @ h_jac
    post_covariance = tf.linalg.inv(_stabilize_covariance(post_precision, jitter))
    post_covariance = _stabilize_covariance(post_covariance, jitter)
    info = (
        tf.linalg.matvec(prior_precision, prior_mean)
        + tf.linalg.matvec(tf.transpose(h_jac) @ obs_precision, pseudo_observation)
    )
    post_mean = tf.linalg.matvec(post_covariance, info)
    post_chol = tf.linalg.cholesky(post_covariance)
    prior_inv = tf.linalg.triangular_solve(
        prior_chol,
        tf.eye(int(prior_chol.shape[0]), dtype=DTYPE),
    )
    affine_transform = post_chol @ prior_inv
    x1 = post_mean + tf.linalg.matvec(affine_transform, x0 - prior_mean)
    affine_logdet = (
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(post_chol)))
        - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)))
    )
    return x1, post_mean, post_covariance, affine_transform, affine_logdet


def _stabilize_covariance(covariance: tf.Tensor, jitter: float = 1e-9) -> tf.Tensor:
    covariance = tf.cast(covariance, DTYPE)
    sym = 0.5 * (covariance + tf.transpose(covariance))
    eigvals = tf.linalg.eigvalsh(sym)
    min_eig = tf.reduce_min(eigvals)
    needed = tf.maximum(tf.constant(jitter, dtype=DTYPE) - min_eig, 0.0)
    return sym + needed * tf.eye(int(sym.shape[0]), dtype=DTYPE)


def _finite_bool(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
