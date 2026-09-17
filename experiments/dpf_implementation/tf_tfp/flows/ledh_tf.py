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

    state_dim = x0_batch.shape[-1]
    if state_dim is None:
        raise ValueError("LEDH flow requires a static state dimension")
    # These repeated stabilizations were part of each local map. Keep them,
    # but evaluate the particle-invariant factors outside the particle loop.
    # This also avoids Grappler lifting eigensolver state out of nested loops.
    prior_chol = tf.linalg.cholesky(_stabilize_covariance(transition_covariance, jitter))
    prior_precision = tf.linalg.cholesky_solve(prior_chol, tf.eye(state_dim, dtype=DTYPE))
    obs_chol = tf.linalg.cholesky(_stabilize_covariance(observation_covariance, jitter))
    obs_precision = tf.linalg.cholesky_solve(obs_chol,
        tf.eye(int(observation_covariance.shape[0]), dtype=DTYPE))

    def map_one(inputs):
        x0, prior_mean = inputs
        x1, post_mean, post_cov, affine_transform, logabsdet = _local_ledh_map_tf(
            x0=x0,
            prior_mean=prior_mean,
            prior_chol=prior_chol,
            prior_precision=prior_precision,
            observation=observation,
            obs_precision=obs_precision,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            jitter=jitter,
        )
        singular_values = tf.linalg.svd(affine_transform, compute_uv=False)
        return (
            x1,
            post_mean,
            post_cov,
            logabsdet,
            tf.stop_gradient(tf.reduce_min(singular_values)),
            tf.stop_gradient(tf.reduce_max(singular_values)),
            # L_post @ inverse(L_prior) is lower triangular.
            tf.stop_gradient(tf.reduce_prod(tf.linalg.diag_part(affine_transform))),
        )

    count = x0_batch.shape[0]
    if count is None:
        raise ValueError("LEDH flow requires a fixed particle count")
    # Dense fixed-shape buffers avoid a map_fn input TensorList whose gradient
    # length can become a dynamic capture inside an enclosing date recurrence.
    buffers = (tf.zeros([count, state_dim], DTYPE), tf.zeros([count, state_dim], DTYPE),
               tf.zeros([count, state_dim, state_dim], DTYPE),
               *(tf.zeros([count], DTYPE) for _ in range(4)))
    def particle_step(index, buffers):
        values = map_one((x0_batch[index], prior_means[index]))
        buffers = tf.nest.map_structure(lambda buffer, value: tf.tensor_scatter_nd_update(
            buffer, tf.reshape(index, [1, 1]), value[None]), buffers, values)
        return index+1, buffers
    _, buffers = tf.while_loop(lambda index, _: index < count, particle_step, (tf.constant(0), buffers),
                               maximum_iterations=count, parallel_iterations=1)
    (
        post_flow_particles,
        local_posterior_means,
        local_posterior_covariances,
        forward_log_det,
        min_singular_tensor,
        max_singular_tensor,
        transform_sign_tensor,
    ) = buffers
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
    diagnostics["valid_flow"] = (tf.reduce_all(tf.math.is_finite(post_flow_particles))
        & tf.reduce_all(tf.math.is_finite(forward_log_det))
        & (tf.reduce_min(min_singular_tensor) > tf.constant(1e-12, DTYPE)))
    if tf.executing_eagerly() and (not diagnostics["finite_post_flow"] or not diagnostics["finite_forward_log_det"]):
        raise FloatingPointError("LEDH flow emitted non-finite map or log-det values")
    if tf.executing_eagerly() and diagnostics["min_jacobian_singular_value"] <= 1e-12:
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
    prior_chol: tf.Tensor,
    prior_precision: tf.Tensor,
    observation: tf.Tensor,
    obs_precision: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    jitter: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    result = tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE)))
    return bool(result.numpy()) if tf.executing_eagerly() else result


def _float(value: tf.Tensor) -> float:
    result = tf.cast(value, DTYPE)
    return float(result.numpy()) if tf.executing_eagerly() else result
