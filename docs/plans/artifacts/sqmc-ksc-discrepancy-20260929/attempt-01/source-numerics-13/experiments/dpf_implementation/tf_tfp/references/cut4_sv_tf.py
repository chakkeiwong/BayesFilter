"""Differentiable TF/TFP CUT4 comparator for one-dimensional SV filtering."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.cubature.cut4_tf import (
    cut4_standard_normal_nodes_weights_tf,
)
from experiments.dpf_implementation.tf_tfp.fixtures.stochastic_volatility_tf import (
    DTYPE,
    StochasticVolatilityTFFixture,
    sv_observation_log_density_tf,
    sv_transition_mean_tf,
)


@dataclass(frozen=True)
class Cut4SVResult:
    scalar: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    log_normalizer: tf.Tensor
    finite: bool
    comparator_id: str = "cut4_sv_tf_differentiable_comparator_not_ground_truth"


def run_cut4_sv_filter_tf(
    fixture: StochasticVolatilityTFFixture,
    *,
    mu: tf.Tensor,
    phi: tf.Tensor | None = None,
    sigma: tf.Tensor | None = None,
) -> Cut4SVResult:
    phi = fixture.phi if phi is None else tf.cast(phi, DTYPE)
    sigma = fixture.sigma if sigma is None else tf.cast(sigma, DTYPE)
    mu = tf.cast(mu, DTYPE)
    mean = tf.cast(fixture.h0_mean, DTYPE)
    variance = tf.cast(fixture.h0_variance, DTYPE)
    nodes, node_weights = cut4_standard_normal_nodes_weights_tf()
    filtered_means = []
    filtered_variances = []
    log_normalizer = tf.constant(0.0, dtype=DTYPE)

    for observation in tf.unstack(tf.cast(fixture.observations, DTYPE), axis=0):
        predictive_mean = sv_transition_mean_tf(mean, mu=mu, phi=phi)
        predictive_variance = phi * phi * variance + sigma * sigma
        predictive_scale = tf.sqrt(tf.maximum(predictive_variance, tf.constant(1e-12, DTYPE)))
        h_nodes = predictive_mean + predictive_scale * nodes
        log_lik = sv_observation_log_density_tf(h_nodes, observation)
        log_terms = tf.math.log(node_weights) + log_lik
        increment = tf.reduce_logsumexp(log_terms)
        posterior_weights = tf.exp(log_terms - increment)
        mean = tf.reduce_sum(posterior_weights * h_nodes)
        centered = h_nodes - mean
        variance = tf.reduce_sum(posterior_weights * centered * centered)
        variance = tf.maximum(variance, tf.constant(1e-10, DTYPE))
        filtered_means.append(mean)
        filtered_variances.append(variance)
        log_normalizer = log_normalizer + increment

    means = tf.stack(filtered_means, axis=0)
    variances = tf.stack(filtered_variances, axis=0)
    scalar = -log_normalizer
    finite = bool(
        tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(variances)).numpy()
        and tf.math.is_finite(scalar).numpy()
    )
    return Cut4SVResult(
        scalar=scalar,
        filtered_means=means,
        filtered_variances=variances,
        log_normalizer=log_normalizer,
        finite=finite,
    )
