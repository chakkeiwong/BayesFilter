"""Finite-likelihood probes cannot promote a posterior or hide invalid rows."""
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import TargetPotentialEvaluation
from bayesfilter.testing.surrogate_hmc_ar2_finite import (
    classify_precision, finite_endpoint_validity, independence_checks, probe_design, scan_batches,
)


def _consumer(coupled=False):
    def evaluate(points):
        valid = points[:, 0] < 3.
        value = -tf.reduce_sum(points**2, -1)
        if coupled:
            value += points[0, 0]
        value = tf.where(valid, value, -float("inf"))
        status = dict(value_valid=valid, force_valid=valid,
                      outside_declared_support=tf.zeros_like(valid))
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


def test_raw_canonical_status_needs_no_adapter_derived_failure_field():
    value = tf.constant([0., -float("inf"), -3., -2.], tf.float64)
    status = dict(value_valid=tf.constant([True, True, False, True]),
                  force_valid=tf.constant([False, True, True, True]),
                  outside_declared_support=tf.constant([False, False, False, True]))
    assert finite_endpoint_validity(value, status).numpy().tolist() == [True, False, False, False]


def test_precision_localization_classifies_only_failed_prefix():
    assert classify_precision([False, False, True], [True, True, True], 2) == "FP32_PRECISION_SENSITIVITY"
    assert classify_precision([False, False], [False, True], 2) == "BOTH_PRECISIONS_CORRECTION_FAILURE"
    assert classify_precision([True, True], [True, True], 2) == "REPLAY_NO_LONGER_FAILS"
    assert classify_precision([False, True], [True, False], 2) == "MIXED_PRECISION_RESULT"


def test_localization_keeps_support_separate_from_numeric_validity():
    assert finite_endpoint_validity(
        tf.constant([-1., -float("inf")], tf.float64),
        dict(value_valid=tf.constant([True, False]),
             force_valid=tf.constant([True, False]),
             outside_declared_support=tf.constant([False, False]))
    ).numpy().tolist() == [True, False]

def test_instrumented_target_casts_at_precision_boundary_and_trims_padding():
    from bayesfilter.testing.surrogate_hmc_ar2_finite import _target_diagnostics
    calls = []
    def compile_target(batch_size, *, filter_moments):
        assert batch_size == 4 and filter_moments
        @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float32)],
                     autograph=False, jit_compile=False)
        def evaluate(points):
            return (tf.reduce_sum(points, -1), tf.zeros_like(points),
                    {"value_valid": tf.ones([4], tf.bool)})
        return evaluate
    target = SimpleNamespace(inputs={"initial_states": tf.zeros([2, 2], tf.float32)},
                             compile=compile_target)
    values, status = _target_diagnostics(target, tf.ones([5, 2], tf.float64))
    assert values.dtype == tf.float32 and values.shape == (5,)
    assert status["value_valid"].shape == (5,)
    assert values.numpy().tolist() == [2.] * 5
