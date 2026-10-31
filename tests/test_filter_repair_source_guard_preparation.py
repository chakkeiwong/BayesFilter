"""Pinned P72 design and fit/guard preparation; no training-quality claim."""

import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_preparation_runtime_tf as native
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("case", ["ordinary", "duplicate", "constant", "ties", "custom",
                                 "nonzero_constant"])
def test_guard_lines_preserve_selection_order_duplicate_keys_and_frozen_design(case):
    fit = tf.constant([[-.8, .1, .9], [-.3, .2, .5]], D)
    guard = tf.constant([[-.7, .7, .7], [-.2, .4, .4]], D)
    fractions = (.0, .25, .5, .75, 1.)
    if case == "ordinary":
        guard = tf.constant([[-.6, .3, .9, .2], [-.1, .8, .2, .4]], D)
    elif case == "constant":
        fit, guard = tf.zeros([2, 3], D), tf.zeros([2, 1], D)
    elif case == "ties":
        fit = tf.zeros([2, 3], D)
        guard = tf.constant([[1., -1., 0., 0.], [0., 0., 1., -1.]], D)
    elif case == "custom":
        fractions = (0., .31, .31, 1., .75, 0.)
    elif case == "nonzero_constant":
        fit, guard = tf.fill([2, 4], tf.constant(.37, D)), tf.fill([2, 2], tf.constant(.37, D))
        fractions = (0., .31, .31, 1., .75, .44)
    arguments = {"fit_points": fit, "guard_points": guard, "line_fractions": fractions}
    expected, expected_manifest = _original("source_route").p72_guard_line_points(**arguments)
    with tf.GradientTape() as tape:
        tape.watch((fit, guard))
        actual, manifest = candidate.p72_guard_line_points(**arguments)
        loss = tf.reduce_sum(actual)
    assert tape.gradient(loss, (fit, guard)) == (None, None)
    if actual.shape != expected.shape:
        program = native.guard_line_program(fit.shape[0], fit.shape[1], guard.shape[1], len(fractions))
        raw, selected = program(fit, guard, tf.constant(fractions, D))
        order, count = native.guard_line_unique_program(fit.shape[0], raw.shape[1])(raw)
        print("candidate", actual.numpy().tolist(), manifest)
        print("baseline", expected.numpy().tolist(), expected_manifest)
        print("raw_bits", tf.bitcast(raw, tf.uint64).numpy().tolist(),
              "selected", selected.numpy().tolist(), "order", order.numpy().tolist(), int(count))
    tf.debugging.assert_near(actual, expected, atol=1e-12, rtol=1e-12)
    for field in ("line_fractions", "selected_guard_indices", "line_start_indices",
                  "line_point_count", "raw_line_point_count", "duplicate_removal"):
        assert manifest[field] == expected_manifest[field]
    if case in ("constant", "ties"):
        assert manifest["line_hash"] == expected_manifest["line_hash"]
    else:
        assert manifest["line_hash"] == candidate._p69_hash_tensor("p72_guard_line_points_hash.v1", actual)
    call = native.guard_line_program(fit.shape[0], fit.shape[1], guard.shape[1], len(fractions))
    inputs = fit, guard, tf.constant(fractions, D)
    raw, _selected = call(*inputs)
    unique = native.guard_line_unique_program(fit.shape[0], raw.shape[1])
    order, count = unique(raw)
    select = native.guard_line_selection_program(fit.shape[0], raw.shape[1], int(count))
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert "HloModule" in unique.experimental_get_compiler_ir(raw)(stage="hlo")
    assert "HloModule" in select.experimental_get_compiler_ir(raw, order)(stage="hlo")
    _graph(call)
    _graph(unique)
    _graph(select)
    call(fit * .9, guard * .9, inputs[-1])
    assert call.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("invalid", ["shape", "empty", "nonfinite", "fractions"])
def test_guard_lines_retain_input_rejection(invalid):
    fit, guard = tf.zeros([2, 3], D), tf.zeros([2, 2], D)
    fractions = (0., .5, 1.)
    if invalid == "shape":
        guard = tf.zeros([3, 2], D)
    elif invalid == "empty":
        guard = tf.zeros([2, 0], D)
    elif invalid == "nonfinite":
        fit = tf.fill([2, 3], tf.constant(float("nan"), D))
    else:
        fractions = ()
    for module in (candidate, _original("source_route")):
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            module.p72_guard_line_points(fit_points=fit, guard_points=guard, line_fractions=fractions)


def test_fit_guard_batch_preserves_full_values_and_input_gradients():
    sources = (tf.constant([[-.4, .2, .7], [-.2, .1, .6]], D),
               tf.constant([[-.3, .8], [.1, .4]], D),
               tf.constant([.8, 1.4, 1.1], D), tf.constant([1.3, .7], D),
               tf.constant([.3, .2, 1.1], D), tf.constant([.1, .9], D))
    names = ("fit_points", "guard_points", "fit_target_values", "guard_target_values",
             "fit_weights", "guard_weights")
    outcomes = []
    for module in (_original("source_route"), candidate):
        with tf.GradientTape() as tape:
            tape.watch(sources)
            batch, manifest = module.p72_training_batch_from_fit_and_guard(
                **dict(zip(names, sources, strict=True)), alpha_guard=.7)
            loss = (tf.reduce_sum(batch.points ** 2)
                    + tf.reduce_sum(batch.weights * batch.target_values ** 2))
        outcomes.append(((batch.points, batch.target_values, batch.weights), tape.gradient(loss, sources)))
        assert manifest["audit_point_count_used_for_training"] == 0
        assert manifest["fit_point_count"] == 3 and manifest["guard_point_count"] == 2
        assert manifest["fit_weight_mass"] == pytest.approx(1.)
        assert manifest["guard_weight_mass_after_alpha"] == pytest.approx(.7)
    for actual, expected in zip(tf.nest.flatten(outcomes[1]), tf.nest.flatten(outcomes[0]), strict=True):
        assert actual is not None and expected is not None
        tf.debugging.assert_near(actual, expected, atol=1e-12, rtol=1e-12)

    @tf.function(input_signature=tuple(tf.TensorSpec(x.shape, D) for x in sources),
                 jit_compile=True, autograph=False)
    def compiled(*values):
        with tf.GradientTape() as tape:
            tape.watch(values)
            fit_weights, fit_valid = native.normalized_set_weights.python_function(values[4])
            guard_weights, guard_valid = native.normalized_set_weights.python_function(values[5])
            arrays = native.fit_guard_arrays.python_function(*values[:4], fit_weights,
                                                              guard_weights, tf.constant(.7, D))
            loss = tf.reduce_sum(arrays[0] ** 2) + tf.reduce_sum(arrays[2] * arrays[1] ** 2)
        return arrays, tape.gradient(loss, values), fit_valid & guard_valid

    result = compiled(*sources)
    assert bool(result[-1])
    for actual, expected in zip(tf.nest.flatten(result[:2]), tf.nest.flatten(outcomes[0]), strict=True):
        tf.debugging.assert_near(actual, expected, atol=1e-12, rtol=1e-12)
    assert "HloModule" in compiled.experimental_get_compiler_ir(*sources)(stage="hlo")
    _graph(compiled)


@pytest.mark.parametrize("weights", [[], [0., 0.], [-1., 2.], [float("nan"), 1.], [[1.]]])
def test_set_weight_normalization_preserves_invalid_inputs(weights):
    for module in (candidate, _original("source_route")):
        with pytest.raises(ValueError):
            module._p72_normalize_set_weights(tf.constant(weights, D))
