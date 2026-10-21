"""Finite-likelihood probes cannot promote a posterior or hide invalid rows."""
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import TargetPotentialEvaluation
from bayesfilter.testing.surrogate_hmc_ar2_finite import (
    independence_checks, probe_design, scan_batches,
)


def _consumer(coupled=False):
    def evaluate(points):
        valid = points[:, 0] < 3.
        value = -tf.reduce_sum(points**2, -1)
        if coupled:
            value += points[0, 0]
        value = tf.where(valid, value, -float("inf"))
        status = dict(value_valid=valid, force_valid=valid,
                      numerical_failure=~valid, outside_declared_support=tf.zeros_like(valid))
        return value, -2*points, status

    def endpoint(points):
        value, _, status = evaluate(points)
        return TargetPotentialEvaluation(-value, status)

    return SimpleNamespace(evaluate_force=evaluate, evaluate_reference=lambda p: (-tf.reduce_sum(p**2, -1), -2*p),
                           finite_endpoint=SimpleNamespace(function=endpoint))


def test_probe_design_covers_prior_tails_and_reference_bulk_separately():
    points, labels = probe_design(dict(means=tf.constant([.3, -.1], tf.float64),
                                      scales=tf.constant([.3, .6], tf.float64)))
    assert points.shape == (58, 2)
    assert labels.count("prior_center") == 9
    assert labels.count("prior_tail") == 40
    assert labels.count("posterior_bulk") == 9
    tf.debugging.assert_near(points[53], tf.constant([.3, -.1], tf.float64))


def test_padded_scan_preserves_real_failure_and_actual_endpoint_wiring():
    points = tf.constant([[0., 0.], [1., 0.], [2., 1.], [3., 0.], [.5, -.5]], tf.float64)
    values, exact, status, timing, wiring = scan_batches(_consumer(), points)
    assert values.shape == exact.shape == (5,)
    assert status["value_valid"].numpy().tolist() == [True, True, True, False, True]
    assert bool(tf.math.is_inf(values[3]))
    assert bool(tf.reduce_all(tf.math.is_finite(exact)))
    assert wiring and len(timing) == 2
    assert timing[0]["compilation_included"] and not timing[1]["compilation_included"]


def test_mechanics_detects_cross_row_coupling():
    points = tf.constant([[0., 0.], [1., .1], [2., -.2], [.5, .4]], tf.float64)
    assert all(independence_checks(_consumer(), points).values())
    bad = independence_checks(_consumer(coupled=True), points)
    assert bad["reproducible"]
    assert not bad["row_permutation_invariant"]
    assert not bad["row_independent"]
