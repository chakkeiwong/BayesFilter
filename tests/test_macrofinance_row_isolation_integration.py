"""Optional real-consumer row-isolation regression; CPU reference/debug only.

Set MACROFINANCE_ROOT to the sibling repository. No fixture asserts posterior
correctness or makes a GPU readiness claim.
"""
import os
from pathlib import Path
import sys

import pytest

ROOT = os.environ.get("MACROFINANCE_ROOT")
if not ROOT or not (Path(ROOT) / "one_country_affine_macro_phase9_hmc.py").is_file():
    pytest.skip("set MACROFINANCE_ROOT for the real-consumer integration", allow_module_level=True)
sys.path.insert(0, ROOT)
import one_country_affine_macro_phase9_hmc as consumer

tf = pytest.importorskip("tensorflow")
from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_sqrt_kalman_score_factors_batched_static as kernel,
)


def evaluator(mode, count):
    observations = tf.constant([[0.1, -0.1], [0.2, 0.05]], tf.float64)
    def evaluate(theta):
        payload = consumer._tf_chart_batch_with_derivatives(theta, tf)
        value, score = consumer._analytic_likelihood_value_score(
            theta, observations, tf, kernel)
        return value, score, payload["support_valid"]
    if mode == "eager":
        return evaluate
    return tf.function(evaluate, input_signature=[tf.TensorSpec([count, 9], tf.float64)],
                       autograph=False, jit_compile=mode == "xla")


@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
def test_healthy_rows_survive_overflow_unitroot_and_nonfinite_companions(mode):
    healthy = tf.constant(consumer.default_truth_parameters(), tf.float64)
    bad_overflow = tf.tensor_scatter_nd_update(healthy, [[0]], [400.0])
    bad_unitroot = tf.constant([-20., -20., 0., 0., 0., 0., 0., 0., 0.], tf.float64)
    bad_nonfinite = tf.tensor_scatter_nd_update(healthy, [[0]], [float("nan")])
    batch = tf.stack([healthy, bad_overflow, bad_unitroot, bad_nonfinite])
    multi, single = evaluator(mode, 4), evaluator(mode, 1)
    value, score, valid = multi(batch)
    assert valid.numpy().tolist() == [True, False, False, False]
    assert bool(tf.reduce_all(tf.math.is_nan(value[1:])))
    assert bool(tf.reduce_all(tf.math.is_nan(score[1:])))
    singleton = single(healthy[None, :])
    duplicate = multi(tf.repeat(healthy[None, :], 4, axis=0))
    permuted = multi(tf.reverse(batch, [0]))
    for other, index in ((singleton, 0), (duplicate, 0), (permuted, 3)):
        tf.debugging.assert_near(value[0], other[0][index], atol=1e-8, rtol=1e-10)
        tf.debugging.assert_near(score[0], other[1][index], atol=1e-7, rtol=1e-9)


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("failure", ["overflow", "unitroot"])
def test_invalid_singleton_keeps_strict_exception(mode, failure):
    if failure == "overflow":
        point = tf.tensor_scatter_nd_update(
            tf.constant(consumer.default_truth_parameters(), tf.float64), [[0]], [400.0])
        message = "Phase 9 K must be finite before expm"
    else:
        point = tf.constant([-20., -20., 0., 0., 0., 0., 0., 0., 0.], tf.float64)
        message = "Phase 9 F must be stable"
    with pytest.raises(tf.errors.InvalidArgumentError, match=message):
        evaluator(mode, 1)(point[None, :])


def test_unchecked_likelihood_route_retains_final_finite_assertion():
    theta = tf.repeat(tf.constant(consumer.default_truth_parameters()[None, :], tf.float64), 2, 0)
    def nonfinite_kernel(**kwargs):
        return tf.fill([2], tf.constant(float("nan"), tf.float64)), tf.zeros([2, 9], tf.float64)
    with pytest.raises(tf.errors.InvalidArgumentError, match="Phase 9 analytic value must be finite"):
        consumer._analytic_likelihood_value_score(theta, tf.zeros([2, 2], tf.float64),
                                                  tf, nonfinite_kernel, include_support_checks=False)


@pytest.mark.parametrize("mode", ["graph", "xla"])
def test_diagnostic_target_value_and_score_match_independent_covariance_reference(mode):
    # Direct target boundary only: public source admission is separately tested.
    # NumPy is confined to the independent external covariance-form reference.
    observations = [[0.1, -0.1], [0.2, 0.05]]
    point = consumer.default_truth_parameters()
    value, score, valid = evaluator(mode, 1)(tf.constant(point[None, :], tf.float64))
    assert bool(valid[0])
    reference = consumer.numpy_log_likelihood(point, observations)
    assert abs(float(value[0]) - reference) <= 1e-8
    # Inherited external reference-test step/tolerances; a second step checks
    # that cancellation or truncation at one convenience step is not decisive.
    for step in (1e-5, 5e-6):
        derivative = []
        for coordinate in range(9):
            plus, minus = point.copy(), point.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            derivative.append((consumer.numpy_log_likelihood(plus, observations)
                               - consumer.numpy_log_likelihood(minus, observations)) / (2 * step))
        tf.debugging.assert_near(score[0], tf.constant(derivative, tf.float64), atol=1e-4, rtol=1e-5)
