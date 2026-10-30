"""Independent physical-parameter posterior integration for the exact KSC T2 law."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ksc_reference import (
    FUNCTION_NAMES, MARGINS, make_ksc_reference,
)
from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import (
    trapezoid_weights, summarize_grid, quantile_graph, expectation_graph,
)

PARAMETER_INTERVALS = (32, 64, 128, 256)
BATCH_SIZE = 256


def prepare_ksc_posterior(dataset, *, resource_check=lambda: None):
    """Qualify all 13 functionals on the full proper prior, without a chain."""
    reference = make_ksc_reference(dataset, batch_size=BATCH_SIZE)
    grids, log_normalizers = [], []
    for intervals in PARAMETER_INTERVALS:
        resource_check()
        knots = tf.linspace(tf.constant(.1, tf.float64), tf.constant(.9, tf.float64), intervals + 1)
        points = tf.reshape(tf.stack(tf.meshgrid(knots, knots, indexing="ij"), -1), [-1, 2])
        values, predictions = [], []
        for offset in range(0, int(points.shape[0]), BATCH_SIZE):
            resource_check()
            count = min(BATCH_SIZE, int(points.shape[0]) - offset)
            batch = points[offset:offset + count]
            if count < BATCH_SIZE:
                batch = tf.concat([batch, tf.repeat(batch[:1], BATCH_SIZE - count, axis=0)], 0)
            likelihood, predictive = reference.statistics(batch)
            tf.debugging.assert_all_finite(likelihood, "KSC parameter-grid likelihood")
            tf.debugging.assert_all_finite(predictive, "KSC parameter-grid prediction")
            tf.debugging.assert_non_negative(predictive)
            tf.debugging.assert_less_equal(predictive, tf.ones_like(predictive))
            values.append(likelihood[:count])
            predictions.append(predictive[:count])
        likelihood = tf.reshape(tf.concat(values, 0), [intervals + 1, intervals + 1])
        predictive = tf.reshape(tf.concat(predictions, 0), [intervals + 1, intervals + 1, 2])
        shift = tf.reduce_max(likelihood)
        raw = tf.exp(likelihood - shift)
        weights = trapezoid_weights(knots)
        normalizer = tf.reduce_sum(raw * weights[:, None] * weights[None, :])
        tf.debugging.assert_positive(normalizer)
        log_normalizers.append(shift + tf.math.log(normalizer) - 2 * tf.math.log(tf.constant(.8, tf.float64)))
        grids.append((knots, knots, raw / normalizer, predictive))
    first, second, density, _ = grids[-1]
    means, scales = summarize_grid(first, second, density)
    tf.debugging.assert_all_finite(means, "KSC posterior means")
    tf.debugging.assert_all_finite(scales, "KSC posterior scales")
    tf.debugging.assert_positive(scales)
    cuts = tf.stack([quantile_graph(first, tf.linalg.matvec(density, trapezoid_weights(second))),
                     quantile_graph(second, tf.linalg.matvec(density, trapezoid_weights(first), transpose_a=True))])
    expectations = tf.stack([expectation_graph(*grid, means, scales, cuts) for grid in grids])
    tf.debugging.assert_all_finite(expectations, "KSC posterior expectations")
    changes = tf.abs(expectations[1:] - expectations[:-1])
    allowance = tf.maximum(2 * tf.reduce_max(changes[-2:], axis=0), tf.constant(1e-12, tf.float64))
    contraction = changes[-1] <= .5 * changes[-2] + 1e-5
    precision = allowance <= tf.constant(MARGINS, tf.float64) / 10
    qualified = bool(tf.reduce_all(contraction & precision))
    return dict(model_id="M03-KSC-T2", qualified=qualified,
                status="QUALIFIED_MODEL_POSTERIOR_REFERENCE" if qualified else "REFERENCE_PRECISION_FAILED",
                means=means, scales=scales, cuts=cuts, expectations=expectations, changes=changes,
                allowance=allowance, margins=MARGINS, contraction=contraction, precision=precision,
                parameter_intervals=PARAMETER_INTERVALS, function_names=FUNCTION_NAMES,
                log_normalizers=tf.stack(log_normalizers),
                numerical_error_status="empirical grid refinement allowance; no rigorous global quadrature bound",
                full_prior_integrated=True, surrogate_hmc_tested=False), grids
