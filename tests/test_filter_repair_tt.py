"""Independent small reference checks for native heterogeneous TT recurrences."""

import sys
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.squared_tt_engine_v0_tf import _product_basis
from bayesfilter.highdim.squared_tt_engine_xla_tf import (
    _fit_als_graph,
    _solve_scaled_qr,
)
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import core_matrices, row_environments


def _inputs():
    basis = _product_basis(3, 2)
    shapes = ((1, 3, 2), (2, 3, 2), (2, 3, 1))
    cores = tuple(TTCore(tf.random.stateless_normal(shape, [1729, i], dtype=tf.float64))
                  for i, shape in enumerate(shapes))
    rows = tf.random.stateless_uniform([32, 3], [1729, 3], minval=-1., maxval=1., dtype=tf.float64)
    return basis, shapes, cores, rows


def test_heterogeneous_environments_equal_independent_contractions():
    basis, _, cores, rows = _inputs()
    reference = tuple(tf.einsum("nl,alb->nab", basis.evaluate_axis(i, rows[:, i]), core.values)
                      for i, core in enumerate(cores))
    actual = core_matrices(basis, rows, cores)
    for a, b in zip(actual, reference):
        tf.debugging.assert_near(a, b, atol=1e-12, rtol=1e-12)
    left = row_environments(actual)
    right = row_environments(actual, reverse=True)
    state = tf.ones([32, 1], tf.float64)
    for i in range(3):
        tf.debugging.assert_near(left[i], state, atol=1e-12, rtol=1e-12)
        state = tf.einsum("na,nab->nb", state, reference[i])
    state = tf.ones([32, 1], tf.float64)
    for i in reversed(range(3)):
        tf.debugging.assert_near(right[i], state, atol=1e-12, rtol=1e-12)
        state = tf.einsum("nab,nb->na", reference[i], state)


def test_native_als_preserves_schedule_and_has_bounded_graph():
    basis, shapes, cores, rows = _inputs()
    weights = tf.fill([32], tf.constant(1. / 32, tf.float64))
    target = tf.exp(-tf.reduce_sum(tf.square(rows), axis=1))
    ridge = tf.constant(1e-5, tf.float64)
    fitter = FixedTTFitter()
    current = list(cores)
    for _ in range(2):
        for axis in range(3):
            design = fitter._build_design_matrix(basis, rows, current, axis)
            solution, _ = _solve_scaled_qr(design, weights, target, ridge)
            current[axis] = TTCore(tf.reshape(solution, shapes[axis]))

    def compile_for(sweeps):
        def evaluate(target):
            fitted, worst, rms = _fit_als_graph(fitter, basis, rows, target, weights,
                tuple(core.values for core in cores), shapes, sweeps, ridge)
            return tuple(core.values for core in fitted), worst, rms
        return tf.function(evaluate, input_signature=[tf.TensorSpec([32], tf.float64)],
                           jit_compile=True, autograph=False)

    compiled = compile_for(2)
    fitted, worst, rms = compiled(target)
    for actual, expected in zip(fitted, current):
        tf.debugging.assert_near(actual, expected.values, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_all_finite([worst, rms], "native ALS diagnostics")
    def nodes(function):
        graph = function.get_concrete_function().graph.as_graph_def()
        return len(graph.node) + sum(len(f.node_def) for f in graph.library.function)
    assert nodes(compile_for(8)) == nodes(compiled)


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_actual_sv_manual_parameter_batch_matches_ad_and_xla(horizon):
    from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
        batched_fixed_tt_likelihood_value_score_status,
    )
    # Both benchmark arms use this one frozen input builder.
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        from filter_repair_benchmark_worker import fixture
        evaluate, inputs, _ = fixture(tf, 'tt_actual', 1, True)
    finally:
        sys.path.remove(scripts)
    inputs = (inputs[0], inputs[1][:horizon], *inputs[2:])
    compiled = tf.function(evaluate, input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs],
                           jit_compile=True, autograph=False)
    eager = evaluate(*inputs)
    actual = compiled(*inputs)
    names = ('transformed_observations', 'initial_core', 'adjacent_core0', 'adjacent_core1',
             'reference_nodes', 'reference_weights', 'reference_grid', 'reference_grid_weights',
             'basis_nodes', 'basis_grid_axis0', 'basis_grid_axis1')
    reference = batched_fixed_tt_likelihood_value_score_status(inputs[0], **dict(zip(names, inputs[1:])))
    tf.debugging.assert_near(actual[0], reference[0], atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(actual[1], reference[1], atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(actual[1], eager[1], atol=1e-12, rtol=1e-12)
    tf.debugging.assert_equal(actual[2]['status_code'], reference[2]['status_code'])
    assert bool(tf.reduce_all(actual[2]['valid_pre_regularized_score']))
