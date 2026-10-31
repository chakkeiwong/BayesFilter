"""Independent actual-SV posterior quadrature; no sampler or DCTR admission.

Reuses model-independent quadrature arithmetic from the linear diagnostic
reference, while all likelihood and prediction evaluations use actual SV.
"""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_sv_reference import make_sv_reference, LOG_TWO_PI
from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import (
    trapezoid_weights, summarize_grid, quantile_graph, expectation_graph)

FUNCTION_NAMES = (
    "standardized_gamma", "standardized_beta", "centered_gamma_second_moment",
    "centered_beta_second_moment", "standardized_cross_moment",
    "gamma_cdf_q10", "gamma_cdf_q50", "gamma_cdf_q90",
    "beta_cdf_q10", "beta_cdf_q50", "beta_cdf_q90",
    "predictive_abs_y21_le_0.25", "predictive_abs_y21_le_1.0",
)
MARGINS = (.1,)*5 + (.02,)*6 + (.03,)*2
PARAMETER_INTERVALS = (16, 32, 64, 128)
LATENT_GRIDS = ((512, 16.), (1024, 16.), (1280, 20.))
BATCH_SIZE = 32


def make_statistics(*, intervals=512, half_width=16., batch_size=BATCH_SIZE, horizon=20):
    reference = make_sv_reference(intervals=intervals, half_width=half_width,
                                  batch_size=batch_size, horizon=horizon)
    grid, weights = reference.grid, reference.weights

    @tf.function(input_signature=[tf.TensorSpec([batch_size, 2], tf.float64),
                                  tf.TensorSpec([horizon], tf.float64)], jit_compile=True, autograph=False)
    def statistics(theta, observations):
        result = reference.physical_value_score(theta, observations)
        residual = grid[None, None, :]-theta[:, 0, None, None]*grid[None, :, None]
        transition = tf.exp(-.5*(residual*residual+LOG_TWO_PI))*weights[None, None, :]
        predicted = tf.linalg.matvec(transition, result["final_mass"], transpose_a=True)
        denominator = tf.sqrt(tf.constant(2., tf.float64))*theta[:, 1, None, None]*tf.exp(.5*grid[None, :, None])
        conditional = tf.math.erf(tf.constant([.25, 1.], tf.float64)[None, None, :]/denominator)
        predictive = tf.reduce_sum(predicted[:, :, None]*conditional, axis=1)
        loss = tf.maximum(result["maximum_predicted_domain_loss"], 1-tf.reduce_sum(predicted, axis=1))
        return result["log_likelihood"], predictive, loss

    return statistics


@tf.function(input_signature=[tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64),
                              tf.TensorSpec([None, None], tf.float64)], jit_compile=True, autograph=False)
def normalize_likelihood(first, second, likelihood):
    shift = tf.reduce_max(likelihood)
    raw = tf.exp(likelihood-shift)
    normalizer = tf.reduce_sum(raw*trapezoid_weights(first)[:, None]*trapezoid_weights(second)[None, :])
    # Physical prior is constant 1/.8**2 on the rectangle.
    return raw/normalizer, shift+tf.math.log(normalizer)-2*tf.math.log(tf.constant(.8, tf.float64))


def assess_refinement(grids):
    """Six grids: four parameter refinements, then two latent refinements."""
    if len(grids) != 6:
        raise ValueError("four parameter and two latent refinements required")
    first, second, density, _ = grids[-1]
    means, scales = summarize_grid(first, second, density)
    tf.debugging.assert_all_finite(means, "posterior means")
    tf.debugging.assert_positive(scales, "posterior scales")
    tf.debugging.assert_all_finite(scales, "posterior scales")
    cuts = tf.stack([quantile_graph(first, tf.linalg.matvec(density, trapezoid_weights(second))),
                     quantile_graph(second, tf.linalg.matvec(density, trapezoid_weights(first), transpose_a=True))])
    expectations = tf.stack([expectation_graph(*row, means, scales, cuts) for row in grids])
    tf.debugging.assert_all_finite(expectations, "posterior expectations")
    changes = tf.abs(expectations[1:]-expectations[:-1])
    parameter_change = tf.reduce_max(changes[1:3], axis=0)
    latent_change = tf.reduce_max(changes[3:], axis=0)
    allowance = tf.maximum(2*(parameter_change+latent_change), tf.constant(1e-12, tf.float64))
    contraction = changes[2] <= .5*changes[1]+1e-5
    precision = allowance <= tf.constant(MARGINS, tf.float64)/10
    return dict(means=means, scales=scales, cuts=cuts, expectations=expectations, changes=changes,
                allowance=allowance, contraction=contraction, precision=precision,
                qualified=tf.reduce_all(contraction & precision))
