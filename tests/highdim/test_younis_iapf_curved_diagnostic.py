"""Independent checks for the diagnostic reference, not filter admission."""
import tensorflow as tf
import pytest
from docs.benchmarks import diagnose_younis_iapf_curved as diagnostic
from bayesfilter.score_study.nonlinear_tf import make_grid_reference

diagnostic.tf = tf


@pytest.mark.parametrize("curves", [(0., 0.), (.35, .12)])
def test_backward_recursion_reproduces_forward_likelihood(curves):
    theta = tf.constant(diagnostic.THETA, tf.float64)
    observations = tf.constant([[.3], [-.4]], tf.float64)
    backward = diagnostic.backward_kernel(201, 9., *curves)(theta, observations)
    forward = make_grid_reference(2, 201, 9., *curves)(theta, observations)
    assert abs(float(backward[3]-forward[0])) < 1e-11
    assert float(tf.reduce_max(tf.abs(tf.reduce_sum(backward[2], 1)-1))) < 1e-13
    assert bool(tf.reduce_all(tf.math.is_finite(backward[1])))


def test_independent_box_competitor_recovers_known_gaussian():
    x = tf.linspace(tf.constant(-3., tf.float64), tf.constant(3., tf.float64), 257)[:, None]
    target = -.5*((x[:, 0]-.4)/.7)**2
    center, variance, _, residual, _ = diagnostic.box_kernel(257, 4., .2, 4.)(x, target)
    assert abs(float(center[0])-.4) < 1e-4
    assert abs(float(variance[0, 0])-.49) < 1e-4
    assert float(residual) < 1e-8


def test_shape_comparison_distinguishes_floor_from_gaussian_fit():
    x = tf.linspace(tf.constant(-3., tf.float64), tf.constant(3., tf.float64), 201)
    target = tf.stack([-.5*x*x, -.5*x*x])
    weights = tf.ones([2, 201], tf.float64)/201
    gaussian, floored, fractions = diagnostic.shape_kernel(201)(x, target, weights,
        tf.zeros([2, 1], tf.float64), tf.ones([2, 1, 1], tf.float64),
        tf.constant([-2., -2.], tf.float64))
    assert float(tf.reduce_max(gaussian)) < 1e-12
    assert float(tf.reduce_min(floored)) > .01
    assert bool(tf.reduce_all((fractions > 0) & (fractions < 1)))


def test_diagnostic_calls_actual_nonlinear_iapf_consumer():
    observations = tf.constant([[.3], [-.4]], tf.float64)
    back = diagnostic.backward_kernel(401, 9., .12, .04)(
        tf.constant(diagnostic.THETA, tf.float64), observations)
    record = diagnostic.consumer_fit(1390, "weak", observations, back,
                                     "shape_2000", diagnostic.Budget(120))
    assert record["status"] == "valid"
    assert record["details"]["fit_model"] == {
        "transition_curve": .12, "observation_curve": .04}
    assert record["details"]["fit_parameter_derivative"] == "frozen_coefficients_at_declared_nominal_theta"


def test_diagnostic_downstream_executes_all_heuristic_consumers():
    observations = tf.constant([[.3], [-.4]], tf.float64)
    oracle = make_grid_reference(2, 201, 9., .12, .04)(
        tf.constant(diagnostic.THETA, tf.float64), observations)
    reference = {"value": float(oracle[0]), "score": oracle[1].numpy().tolist()}
    result = diagnostic.downstream(1391, "weak", observations, {}, [16], reference,
                                   diagnostic.Budget(120))
    assert set(result["deterministic"]) == {"ekf", "ukf"}
    assert set(result["sizes"]["16"]["summary"]) == {
        "bootstrap", "local_linear", "no_resampling"}
    assert all(record["replicates"] == 32 for record in result["sizes"]["16"]["summary"].values())
