"""Independent FP64 LG1 posterior quadrature for surrogate-HMC comparisons.

This reference does not select DCTR controls or certify a sampler. Its refinement
allowance is empirical. All numerical work uses TensorFlow, with explicit
bounded grid sizes and stable compiled kernels.
"""
from __future__ import annotations

import math
import tensorflow as tf

from .surrogate_hmc_lg1_tf import kalman_physical, kalman_unconstrained


@tf.function(input_signature=[tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([10, 1], tf.float64)], jit_compile=True,
             autograph=False)
def evaluate_kalman(phi, observations):
    # Four declared grid lengths compile at most four XLA executables per data shape.
    return kalman_physical(phi, observations)


@tf.function(input_signature=[tf.TensorSpec([9], tf.float64),
                             tf.TensorSpec([10, 1], tf.float64)], jit_compile=True,
             autograph=False)
def joint_gaussian_log_likelihood(phi, observations):
    """Independent stationary observation covariance, without filtering updates."""
    lag = tf.abs(tf.range(10)[:, None] - tf.range(10)[None, :])
    # Integer-valued powers retain the sign of negative AR coefficients.
    covariance = tf.pow(phi[:, None, None], tf.cast(lag, tf.float64)[None, ...])
    covariance /= (1 - tf.square(phi))[:, None, None]
    covariance += .5 * tf.eye(10, dtype=tf.float64)[None, ...]
    factor = tf.linalg.cholesky(covariance)
    residual = tf.broadcast_to(observations[None, ...], [9, 10, 1])
    solved = tf.linalg.triangular_solve(factor, residual)
    return -.5 * (10 * tf.constant(math.log(2 * math.pi), tf.float64)
                  + 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)), axis=1)
                  + tf.reduce_sum(tf.square(solved), axis=[1, 2]))


def _trapz(phi, values):
    return tf.reduce_sum((phi[1:] - phi[:-1]) * (values[1:] + values[:-1]) * .5)


def _interp(x, knots, values):
    indices = tf.clip_by_value(tf.searchsorted(knots, x, side="right") - 1,
                               0, tf.size(knots) - 2)
    left, right = tf.gather(knots, indices), tf.gather(knots, indices + 1)
    fraction = (x - left) / (right - left)
    return tf.gather(values, indices) + fraction * (tf.gather(values, indices + 1) - tf.gather(values, indices))


@tf.function(input_signature=[tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([5], tf.float64)], autograph=False,
             jit_compile=True)
def grid_functions(phi, density, predictive, summary):
    center, scale = summary[0], summary[1]
    standardized = (phi - center) / scale
    interval_mass = (phi[1:] - phi[:-1]) * (density[1:] + density[:-1]) * .5
    cdf = tf.concat([tf.zeros([1], tf.float64), tf.cumsum(interval_mass)], 0)
    return tf.concat([
        tf.stack([_trapz(phi, density * standardized), _trapz(phi, density * tf.square(standardized))]),
        _interp(summary[2:], phi, cdf),
        tf.reshape(_trapz(phi, density * predictive), [1])], 0)


def prepare_reference(observations):
    """Return qualified grid tensors and JSON-compatible numerical diagnostics."""
    obs = tf.cast(tf.convert_to_tensor(observations, tf.float32), tf.float64)
    tf.debugging.assert_equal(tf.shape(obs), [10, 1])
    tf.debugging.assert_all_finite(obs, "observations")
    points = tf.constant([-.95, -.8, -.5, -.2, 0., .2, .5, .8, .95], tf.float64)
    value, score, _ = evaluate_kalman(points, obs)
    independent = joint_gaussian_log_likelihood(points, obs)
    value_error = tf.reduce_max(tf.abs(value - independent))
    h = tf.constant(1e-6, tf.float64)
    finite_difference = (evaluate_kalman(points + h, obs)[0] - evaluate_kalman(points - h, obs)[0]) / (2 * h)
    score_error = tf.reduce_max(tf.abs(score - finite_difference) / tf.maximum(1., tf.abs(score)))
    z = tf.constant([[-7.], [-3.], [-1.], [0.], [1.], [3.], [7.]], tf.float64)
    s = tf.math.sigmoid(z[:, 0])
    chart_logp, chart_score, _ = kalman_unconstrained(z, obs)
    raw_value, raw_score, _ = evaluate_kalman(-.95 + 1.9 * s, obs)
    chart_error = tf.reduce_max(tf.abs(chart_logp - (raw_value + tf.math.log(s) + tf.math.log1p(-s))))
    chart_score_error = tf.reduce_max(tf.abs(chart_score[:, 0] - (raw_score * 1.9 * s * (1 - s) + 1 - 2 * s)))

    grids = []
    for intervals in (1024, 2048, 4096, 8192):
        phi = tf.linspace(tf.constant(-.95, tf.float64), tf.constant(.95, tf.float64), intervals + 1)
        log_likelihood, _, predictive = evaluate_kalman(phi, obs)
        weights = tf.exp(log_likelihood - tf.reduce_max(log_likelihood))
        normalizer = _trapz(phi, weights)
        tf.debugging.assert_positive(normalizer)
        density = weights / normalizer
        tf.debugging.assert_all_finite(density, "posterior density")
        mean = _trapz(phi, density * phi)
        variance = _trapz(phi, density * tf.square(phi - mean))
        tf.debugging.assert_positive(variance)
        cdf = tf.concat([tf.zeros([1], tf.float64), tf.cumsum((phi[1:] - phi[:-1]) * (density[1:] + density[:-1]) * .5)], 0)
        grids.append((phi, density, predictive, mean, variance, cdf))
    phi, density, predictive, mean, variance, cdf = grids[-1]
    quantiles = _interp(tf.constant([.1, .5, .9], tf.float64), cdf, phi)
    summary = tf.concat([tf.stack([mean, tf.sqrt(variance)]), quantiles], 0)
    functions = tf.stack([grid_functions(p, d, pred, summary) for p, d, pred, *_ in grids])
    differences = tf.abs(functions[1:] - functions[:-1])
    allowance = 2 * tf.reduce_max(differences[-2:], axis=0)
    passed = bool(value_error <= 1e-10 and score_error <= 1e-5
                  and chart_error <= 1e-10 and chart_score_error <= 1e-10
                  and tf.reduce_max(differences[-1]) <= 1e-5)
    report = dict(schema="bayesfilter.lg1_model_posterior_reference.v1",
        qualified=passed, target="LG1 T10, stationary x0, Q=1, R=.5, phi uniform[-.95,.95]",
        observation_representation="float32 data promoted to float64, identical to DCTR consumer",
        cpu_reference_exception=True, jit_compile=True,
        interval_counts=[1024, 2048, 4096, 8192],
        independent_log_likelihood_error=float(value_error), relative_score_fd_error=float(score_error),
        chart_value_error=float(chart_error), chart_score_error=float(chart_score_error),
        posterior_mean=float(mean), posterior_sd=float(tf.sqrt(variance)),
        cdf_thresholds=quantiles.numpy().tolist(),
        function_names=["standardized_phi", "centered_second_moment", "cdf_0.1", "cdf_0.5", "cdf_0.9", "Pr(y11>0|y)"],
        functions_by_grid=functions.numpy().tolist(), refinement_differences=differences.numpy().tolist(),
        reference_expectations=functions[-1].numpy().tolist(),
        empirical_refinement_allowance=allowance.numpy().tolist(), rigorous_error_bound=False,
        margins=[.1, .1, .02, .02, .02, .03], sampler_accuracy_established=False)
    return report, dict(phi=phi, density=density, predictive=predictive, summary=summary)
