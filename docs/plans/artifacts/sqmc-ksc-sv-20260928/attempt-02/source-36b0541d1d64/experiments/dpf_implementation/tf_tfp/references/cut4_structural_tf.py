"""Differentiable TF/TFP CUT4 comparator for structural AR(1) completion."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.cubature.cut4_tf import (
    cut4_standard_normal_nodes_weights_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    DTYPE,
    StructuralAR1QuadraticTFFixture,
    complete_k_tf,
    structural_observation_mean_tf,
)


@dataclass(frozen=True)
class Cut4StructuralResult:
    scalar: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    log_normalizer: tf.Tensor
    max_deterministic_residual: tf.Tensor
    finite: bool
    comparator_id: str = "cut4_structural_tf_differentiable_comparator_not_ground_truth"


def run_cut4_structural_filter_tf(
    fixture: StructuralAR1QuadraticTFFixture,
    *,
    b: tf.Tensor,
) -> Cut4StructuralResult:
    b = tf.cast(b, DTYPE)
    nodes, node_weights = cut4_standard_normal_nodes_weights_tf()
    mean = tf.stack([fixture.m0_mean, fixture.k0], axis=0)
    covariance = tf.linalg.diag(tf.stack([fixture.m0_variance, tf.constant(1e-8, DTYPE)]))
    filtered_means = []
    filtered_variances = []
    residuals = []
    log_normalizer = tf.constant(0.0, dtype=DTYPE)

    for observation in tf.unstack(tf.cast(fixture.observations, DTYPE), axis=0):
        prev_m = mean[0]
        prev_k = mean[1]
        m_nodes = fixture.rho * prev_m + fixture.sigma * nodes
        k_nodes = complete_k_tf(
            previous_k=prev_k,
            previous_m=prev_m,
            current_m=m_nodes,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        state_nodes = tf.stack([m_nodes, k_nodes], axis=1)
        obs_mean = structural_observation_mean_tf(state_nodes, fixture.lam)
        log_lik = _normal_logpdf(observation - obs_mean, fixture.observation_scale)
        log_terms = tf.math.log(node_weights) + log_lik
        increment = tf.reduce_logsumexp(log_terms)
        posterior_weights = tf.exp(log_terms - increment)
        mean = tf.reduce_sum(posterior_weights[:, None] * state_nodes, axis=0)
        centered = state_nodes - mean
        variance = tf.reduce_sum(posterior_weights[:, None] * centered * centered, axis=0)
        covariance = tf.linalg.diag(tf.maximum(variance, tf.constant(1e-10, DTYPE)))
        deterministic_check = complete_k_tf(
            previous_k=prev_k,
            previous_m=prev_m,
            current_m=m_nodes,
            a=fixture.a,
            b=b,
            c=fixture.c,
            d=fixture.d,
        )
        residuals.append(tf.reduce_max(tf.abs(k_nodes - deterministic_check)))
        filtered_means.append(mean)
        filtered_variances.append(variance)
        log_normalizer = log_normalizer + increment

    del covariance
    means = tf.stack(filtered_means, axis=0)
    variances = tf.stack(filtered_variances, axis=0)
    max_residual = tf.reduce_max(tf.stack(residuals, axis=0))
    scalar = -log_normalizer
    finite = bool(
        tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(variances)).numpy()
        and tf.math.is_finite(scalar).numpy()
        and tf.math.is_finite(max_residual).numpy()
    )
    return Cut4StructuralResult(
        scalar=scalar,
        filtered_means=means,
        filtered_variances=variances,
        log_normalizer=log_normalizer,
        max_deterministic_residual=max_residual,
        finite=finite,
    )


def _normal_logpdf(residual: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(residual, DTYPE)
    scale = tf.cast(scale, DTYPE)
    variance = scale * scale
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE) * variance)
        + residual * residual / variance
    )
