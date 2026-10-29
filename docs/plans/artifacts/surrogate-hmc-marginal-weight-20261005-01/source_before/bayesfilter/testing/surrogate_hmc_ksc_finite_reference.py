"""Independent full-prior integration of the named KSC finite DCTR density."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import trapezoid_weights, expectation_graph
from bayesfilter.testing.surrogate_hmc_lg1_finite_reference import compare_references

PARAMETER_INTERVALS = (16, 32, 64, 128)


def summarize_finite_grids(knots, log_density, predictive, reference):
    """Use fixed R-defined functionals for both D and R; no chains are involved."""
    tf.debugging.assert_shapes([(knots, (129,)), (log_density, (129, 129)),
                               (predictive, (129, 129, 2))])
    tf.debugging.assert_all_finite(log_density, "finite KSC log density")
    tf.debugging.assert_all_finite(predictive, "KSC prediction")
    tf.debugging.assert_greater_equal(predictive, tf.zeros_like(predictive))
    tf.debugging.assert_less_equal(predictive, tf.ones_like(predictive))
    means, scales, cuts = (tf.constant(reference[k], tf.float64) for k in ("means", "scales", "cuts"))
    grids, expectations, log_normalizers = [], [], []
    for intervals in PARAMETER_INTERVALS:
        stride = 128 // intervals
        axis, value, prediction = knots[::stride], log_density[::stride, ::stride], predictive[::stride, ::stride]
        shift = tf.reduce_max(value)
        raw = tf.exp(value-shift)
        weights = trapezoid_weights(axis)
        normalizer = tf.reduce_sum(raw * weights[:, None] * weights[None, :])
        tf.debugging.assert_positive(normalizer)
        density = raw / normalizer
        grids.append((axis, axis, density, prediction))
        expectations.append(expectation_graph(axis, axis, density, prediction, means, scales, cuts))
        log_normalizers.append(shift + tf.math.log(normalizer))
    expectations = tf.stack(expectations)
    tf.debugging.assert_all_finite(expectations, "finite posterior expectations")
    changes = tf.abs(expectations[1:]-expectations[:-1])
    allowance = tf.maximum(2*tf.reduce_max(changes[-2:], axis=0), tf.constant(1e-12, tf.float64))
    margins = tf.constant(reference["margins"], tf.float64)
    contraction = changes[-1] <= .5*changes[-2]+1e-5
    precision = allowance <= margins/10
    qualified = bool(tf.reduce_all(contraction & precision))
    comparison = compare_references(expectations[-1], allowance,
        tf.constant(reference["expectations"][-1], tf.float64), tf.constant(reference["allowance"], tf.float64),
        margins, qualified)
    return dict(qualified=qualified,
        status="QUALIFIED_FINITE_POSTERIOR_REFERENCE" if qualified else "FINITE_REFERENCE_PRECISION_FAILED",
        parameter_intervals=PARAMETER_INTERVALS, means=means, scales=scales, cuts=cuts,
        expectations=expectations, changes=changes, allowance=allowance, margins=margins,
        contraction=contraction, precision=precision, log_normalizers=tf.stack(log_normalizers),
        function_names=reference["function_names"], D_vs_R=comparison, full_prior_integrated=True,
        predictive_definition="exact-model conditional predictive averaged over D parameter posterior",
        numerical_error_status="empirical grid refinement allowance; no rigorous global error bound",
        surrogate_hmc_tested=False,
        posterior_comparisons={"B/R": "NOT_RUN", "C/D": "NOT_RUN", "D/R": comparison["status"], "C/R": "NOT_RUN"}), grids
