"""Pinned public P72 gate decisions, edge cases and complete XLA callbacks."""

import math

import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_gate_runtime_tf as native
from tests.test_filter_repair_remaining_routes import _graph, _original
from tests.test_filter_repair_squared_density import _density

D = tf.float64


def _same_record(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            if math.isnan(value):
                assert math.isnan(actual[key]), key
            elif math.isinf(value):
                assert actual[key] == value, key
            else:
                assert actual[key] == pytest.approx(value, abs=1e-10, rel=1e-10), key
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("case", ["ordinary", "single", "empty", "empty_fit", "missing",
    "missing_nonfinite", "nonfinite", "nonfinite_fit", "clipped", "clip_nan", "far", "boundary"])
def test_support_records_preserve_baseline_and_invalid_clouds(case):
    fit = tf.constant([[-1., 0., 1.], [0., 0., 0.]], D)
    points = tf.constant([[-.8, -.1, .6, 1.], [.1, .2, -.2, .4]], D)
    extra = {}
    if case == "single":
        fit = fit[:, :1]
    elif case == "empty":
        points = tf.zeros([2, 0], D)
    elif case == "empty_fit":
        fit = tf.zeros([2, 0], D)
    elif case in ("missing", "missing_nonfinite"):
        points = None
        if case == "missing_nonfinite":
            fit = tf.fill([2, 3], tf.constant(float("nan"), D))
    elif case == "nonfinite":
        points = tf.constant([[float("nan"), float("inf")], [0., 0.]], D)
    elif case == "nonfinite_fit":
        fit = tf.constant([[float("nan")], [0.]], D)
    elif case == "clipped":
        extra = {"clip_fraction": 1., "local_max_abs_before_clip": 2.}
    elif case == "clip_nan":
        extra = {"clip_fraction": float("nan"), "local_max_abs_before_clip": float("inf")}
    elif case == "far":
        points = tf.constant([[0.], [10.01]], D)
    elif case == "boundary":
        points = tf.constant([[0.], [10.]], D)
    arguments = dict(role="guard", fit_points=fit, points=points, **extra)
    _same_record(candidate.p72_support_clipping_coverage(**arguments),
                 _original("source_route").p72_support_clipping_coverage(**arguments))
    if points is not None:
        assert "HloModule" in native.support_statistics.experimental_get_compiler_ir(points, fit)(stage="hlo")


class _Predictions:
    def __init__(self, values):
        self.values = tf.constant(values, D)

    def evaluate(self, points):
        return self.values + tf.zeros_like(points[:, 0])


@pytest.mark.parametrize("mode", ["none", "maximum", "indices"])
@pytest.mark.parametrize("case", ["ordinary", "rms_boundary", "absolute_boundary", "veto",
                                  "nan", "inf", "start_nan", "start_inf", "empty"])
def test_line_records_preserve_thresholds_nonfinite_values_and_order(mode, case):
    predictions = [1., 2., -3.]
    targets = [.7, 2.2, -2.8]
    if case == "rms_boundary":
        predictions, targets = [10., 10., 10.], [0., 0., 0.]
    elif case == "absolute_boundary":
        predictions, targets = [1000., 1000., 1000.], [0., 0., 0.]
    elif case == "veto":
        predictions, targets = [1001., 1., -1001.], [0., 0., 0.]
    elif case == "nan":
        predictions = [float("nan"), 1., 2.]
    elif case == "inf":
        predictions = [float("inf"), 1., 2.]
    elif case == "empty":
        predictions, targets = [], []
    fitted = _Predictions(predictions)
    points = tf.zeros([2, len(predictions)], D)
    starts = None if mode == "none" else tf.constant([1., 2.], D)
    if mode != "none" and case in ("start_nan", "start_inf"):
        starts = tf.constant([float("nan" if case == "start_nan" else "inf"), 2.], D)
    indices = None if mode != "indices" else [0, 1, 0][:len(predictions)]
    arguments = {"fitted_tt": fitted, "line_points": points, "line_target_values": tf.constant(targets, D),
                 "start_prediction_values": starts, "line_start_indices": indices, "target_scale": 1.}
    if case in ("nan", "inf"):
        # The original API rejects nonfinite arrays during provenance hashing;
        # preserve that behavior instead of inventing a report on this branch.
        for module in (_original("source_route"), candidate):
            with pytest.raises(ValueError, match="branch manifest tensors must be finite"):
                module.p72_line_probe_diagnostics(**arguments)
        return
    _same_record(candidate.p72_line_probe_diagnostics(**arguments),
                 _original("source_route").p72_line_probe_diagnostics(**arguments))


@pytest.mark.parametrize("indices", [[-1, 0], [0, 2], [0]])
def test_line_indices_remain_rejected_including_xla_clamping(indices):
    for module in (_original("source_route"), candidate):
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            module.p72_line_probe_diagnostics(fitted_tt=_Predictions([1., 2.]),
                line_points=tf.zeros([2, 2], D), line_target_values=tf.ones([2], D),
                start_prediction_values=tf.ones([2], D), line_start_indices=indices, target_scale=1.)


def test_owned_fitted_tt_prediction_and_reductions_compile_together_without_retracing():
    fitted = _density(3, lebesgue=True).sqrt_tt
    points = tf.reshape(tf.linspace(tf.constant(-.4, D), tf.constant(.6, D), 9), [3, 3])
    inputs = (points, tf.constant([.9, 1.1, .8], D), tf.constant([1., .7], D),
              tf.constant([0, 1, 0]), tf.constant(1., D))
    program = native.line_probe_program(fitted, (3, 3), (2,), (3,), "indices")
    arguments = {"fitted_tt": fitted, "line_points": inputs[0], "line_target_values": inputs[1],
        "start_prediction_values": inputs[2], "line_start_indices": inputs[3], "target_scale": 1.}
    actual = candidate.p72_line_probe_diagnostics(**arguments)
    expected = _original("source_route").p72_line_probe_diagnostics(**arguments)
    # Prediction hashes bind the realized candidate values, not rounded parity.
    actual_hash, expected_hash = actual.pop("line_prediction_hash"), expected.pop("line_prediction_hash")
    assert actual_hash == candidate._p69_hash_tensor("p72_line_probe_predictions_hash.v1", program(*inputs)[0])
    assert expected_hash
    _same_record(actual, expected)
    assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(program)
    program(points * .9, *inputs[1:])
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("records", [
    [],
    [{"condition_number": 1e10, "scaled_augmented_singular_values": [2., 2e-12, 2.1e-12]}],
    [{"condition_number": 1., "scaled_augmented_singular_values": []}],
    [{"condition_number": 2., "scaled_augmented_singular_values": [float("nan"), 1.]},
     {"condition_number": 3., "effective_rank": 2.}],
    [{"condition_number": 1., "scaled_augmented_singular_values": [2., 1.]},
     {"scaled_augmented_condition_number": float("inf"), "condition_number": 2.,
      "scaled_augmented_singular_values": [[1., 0.], [1e-14, 0.]]},
     {"condition_number": 4., "effective_rank": 0.}],
    [{"condition_number": float("nan"), "effective_rank": float("nan")},
     {"condition_number": 1e11, "scaled_augmented_singular_values": [float("inf")]}],
])
def test_condition_records_preserve_heterogeneous_spectra_and_unavailable_reasons(records):
    _same_record(candidate.p72_condition_effective_rank_gate(records),
                 _original("source_route").p72_condition_effective_rank_gate(records))


@pytest.mark.parametrize("jit", [False, True])
def test_spectrum_reduction_native_loop_preserves_strict_rank_threshold(jit):
    inputs = (tf.constant([2., 2e-12, 2.1e-12], D), tf.constant([], D),
              tf.constant([[2., 1.], [0., 0.]], D), tf.constant([float("nan")], D))
    program = native.spectrum_rank_program(tuple(tuple(value.shape) for value in inputs), jit_compile=jit)
    ranks, valid = program(inputs, tf.constant(1e-12, D))
    tf.debugging.assert_equal(valid, [True, False, True, False])
    tf.debugging.assert_equal(tf.gather(ranks, [0, 2]), tf.constant([2., 2.], D))
    assert any("While" in node.op for node in _graph(program))
    if jit:
        assert "HloModule" in program.experimental_get_compiler_ir(inputs, tf.constant(1e-12, D))(stage="hlo")
    program(inputs, tf.constant(1e-10, D))
    assert program.experimental_get_tracing_count() == 1
