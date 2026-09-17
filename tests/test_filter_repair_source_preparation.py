"""Pinned parity and enclosing XLA checks for source-route preparation.

The independent baseline is an execution-refactor oracle, not a historical
LEDH result or a source-faithfulness claim for repository initializers.
"""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_preparation_tf as native
from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.highdim.tt import TTCore
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("count,dimension", [(1, 2), (5, 3), (9, 6)])
@pytest.mark.parametrize("unit", [False, True])
def test_reference_design_preserves_baseline_and_has_hlo(count, dimension, unit):
    name = "_p59_author_sir_unit_reference_points" if unit else "_p59_author_sir_reference_points"
    expected = getattr(_original("source_route"), name)(count, dimension)
    actual = getattr(candidate, name)(count, dimension)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    call = native.reference_points_program(count, dimension, unit)
    assert "HloModule" in call.experimental_get_compiler_ir()(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("ranks,width", [((1, 1), 1), ((1, 3, 1), 1),
                                       ((1, 3, 2, 4, 1), 4)])
@pytest.mark.parametrize("seeded", [False, True])
def test_initial_cores_preserve_all_channels_and_compilation(ranks, width, seeded):
    name = "_source_route_seeded_channel_initial_cores" if seeded else "_source_route_constant_path_initial_cores"
    arguments = {"ranks": ranks, "basis_dim": width, "constant_value": tf.constant(1.73, D)}
    expected = getattr(_original("source_route"), name)(**arguments)
    actual = getattr(candidate, name)(**arguments)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.values, right.values)
    call = native.initial_cores_program(ranks, width, seeded)
    assert "HloModule" in call.experimental_get_compiler_ir(tf.constant(1.73, D),
                       tf.constant(1e-3, D))(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("ranks,widths,fit_rank", [((1, 1), (2,), 1),
    ((1, 3, 1), (2, 4), 4), ((1, 3, 2, 4, 1), (2, 3, 1, 4), 4)])
@pytest.mark.parametrize("zero", [False, True])
def test_channel_activity_preserves_decisions_and_heterogeneous_shapes(ranks, widths, fit_rank, zero):
    cores = tuple(TTCore(tf.reshape(tf.sin(tf.cast(tf.range(left * width * right), D)) *
        (0. if zero else .03), [left, width, right]))
        for left, width, right in zip(ranks[:-1], widths, ranks[1:], strict=True))
    arguments = {"cores": cores, "target_dim": len(cores), "fit_rank": fit_rank}
    expected = _original("source_route")._p70_channel_activity_diagnostics(**arguments)
    actual = candidate._p70_channel_activity_diagnostics(**arguments)
    for name in ("status", "minimum_active_bonds", "extra_channel_active_bond_counts", "inactive_extra_channels"):
        assert actual[name] == expected[name]
    for actual_row, expected_row in zip(actual["score_by_bond"], expected["score_by_bond"], strict=True):
        np.testing.assert_allclose(actual_row, expected_row, atol=1e-10, rtol=1e-10)
    for name in ("reference_channel_score", "activity_threshold"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("use_quantile,minimum_ess", [(False, 1000.), (True, 0.), (True, 1000.)])
@pytest.mark.parametrize("masked", [False, True])
def test_recenter_preserves_pinned_frame_and_jit_with_discarded_samples(use_quantile, minimum_ess, masked):
    values = tf.cast(tf.range(32), D)
    samples = tf.stack([tf.sin(.37 * values), tf.cos(.53 * values), .1 * values], axis=0)
    weights = tf.math.log(.5 + tf.square(tf.cos(.23 * values)))
    if masked:
        samples = tf.tensor_scatter_nd_update(samples, [[0, 2], [2, 7]],
                                              tf.constant([float("nan"), float("inf")], D))
        weights = tf.tensor_scatter_nd_update(weights, [[5]], tf.constant([float("nan")], D))
    arguments = {"samples": samples, "log_weights": weights, "expansion_factor": 1.3,
        "covariance_jitter": 1e-5, "use_quantile_scale": use_quantile,
        "min_ess_for_quantile_scale": minimum_ess, "quantile_fraction": .1}
    expected = _original("source_route").source_route_recenter(**arguments)
    actual = candidate.source_route_recenter(**arguments)
    np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-10, rtol=1e-10)
    inputs = (samples, weights, tf.constant(1.3, D), tf.constant(1e-5, D),
              tf.constant(.1, D), tf.constant(minimum_ess, D))
    call = tf.function(lambda *args: native.recenter.python_function(*args,
        use_quantile_scale=use_quantile), input_signature=tuple(tf.TensorSpec(x.shape, x.dtype)
        for x in inputs), jit_compile=True, autograph=False)
    assert bool(call(*inputs)[2])
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(call)


def test_recenter_rejects_empty_finite_cloud():
    with pytest.raises(ValueError, match="NONFINITE_VALUE"):
        candidate.source_route_recenter(samples=tf.constant([[float("nan")]], D),
            log_weights=tf.zeros([1], D), expansion_factor=1.)


def test_coordinate_time_recurrence_matches_pinned_and_reuses_signature():
    model = zhao_cui_sir_austria_model()
    call = native.coordinate_frame_program(model)
    for time in (1, 4, 9):
        expected = _original("source_route")._p59_author_sir_36d_coordinate_frame_for_time(model, time_index=time)
        actual = candidate._p59_author_sir_36d_coordinate_frame_for_time(model, time_index=time)
        np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(actual.matrix, expected.matrix)
    assert call.experimental_get_tracing_count() == 1
    assert "HloModule" in call.experimental_get_compiler_ir(tf.constant(4), model.initial_mean,
        model.process_covariance, model.initial_covariance)(stage="hlo")
    _graph(call)
    changed = replace(model, kappa=model.kappa * .9)
    alternate = candidate._p59_author_sir_36d_coordinate_frame_for_time(changed, time_index=4)
    assert np.max(np.abs(alternate.mu - actual.mu)) > 1e-10
    assert native.coordinate_frame_program(changed) is not call


def test_channel_decisions_execute_in_xla_at_the_inclusive_threshold():
    cores = tf.ones([2, 1, 1, 1], D)
    inputs = (cores, tf.constant([1]), tf.constant(1), tf.constant(1., D), tf.constant(0., D))
    call = tf.function(native.channel_activity.python_function,
        input_signature=tuple(tf.TensorSpec(x.shape, x.dtype) for x in inputs),
        jit_compile=True, autograph=False)
    scores, _, threshold, counts, _, valid = call(*inputs)
    np.testing.assert_array_equal(scores, [[1.]])
    np.testing.assert_array_equal(counts, [1])
    assert bool(valid) and float(threshold) == 1.
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(call)


def test_design_and_channel_graphs_do_not_grow_with_dimension():
    for create in (lambda dimension: native.reference_points_program(8, dimension, True),
                   lambda dimension: native.initial_cores_program((1, *([3] * (dimension - 1)), 1), 4, True)):
        assert len(_graph(create(3))) == len(_graph(create(8)))
