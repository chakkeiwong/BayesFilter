"""Pinned target preparation and exact minibatch draw-stream checks."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from bayesfilter.ops.stateless_random_tf import philox_shuffle_indices
from tests.test_filter_repair_centered_tt import _fresh_parent
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("size", [1, 2, 7, 32, 65])
@pytest.mark.parametrize("seed", [(1729, 3), (-17, 0)])
def test_compiled_shuffle_preserves_tensorflow_original_stream(size, seed):
    inputs = tf.constant(seed, tf.int32)
    call = tf.function(lambda seed: philox_shuffle_indices(size, seed),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    expected = tf.random.experimental.stateless_shuffle(tf.range(size), inputs)
    np.testing.assert_array_equal(call(inputs), expected)
    np.testing.assert_array_equal(tf.sort(call(inputs)), tf.range(size))
    assert "HloModule" in call.experimental_get_compiler_ir(inputs)(stage="hlo")
    _graph(call)


def test_rotating_batches_preserve_three_complete_seeded_epochs():
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    for update in range(12):
        arguments = {"pool_size": 32, "batch_size": 8, "update": update, "seed": 8401}
        np.testing.assert_array_equal(candidate.rotating_prefix_minibatch_indices(**arguments),
                                      before.rotating_prefix_minibatch_indices(**arguments))
    program = candidate._prefix_minibatch_program(32, 8)
    assert program.experimental_get_tracing_count() == 1
    graph = tf.function(program.python_function, input_signature=program.input_signature,
                        jit_compile=False, autograph=False)
    np.testing.assert_array_equal(graph(tf.constant([8401, 2]), tf.constant(3)),
        candidate.rotating_prefix_minibatch_indices(pool_size=32, batch_size=8, update=11, seed=8401))
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in graph.get_concrete_function().graph.as_graph_def().library.function)


@pytest.mark.parametrize("count", [4, 8])
def test_complete_batch_and_ratio_targets_match_pinned_program(count):
    parent = _fresh_parent()
    from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame

    diagonal = tf.linspace(tf.constant(.7, D), 1.3, 36)
    matrix = tf.linalg.diag(diagonal) + .01 * tf.linalg.band_part(tf.ones([36, 36], D), -1, 0)
    parent = SimpleNamespace(**{**vars(parent), "frame": SourceRouteCoordinateFrame(
        mu=tf.linspace(tf.constant(-.2, D), .2, 36), matrix=matrix, expansion_factor=1.)})
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    theta = tf.constant([[0., 0., 0.], [.02, -.01, .03]], D)
    initial = .05 * tf.reshape(tf.sin(tf.cast(tf.range(count * 18), D)), [count, 18])
    noise = .03 * tf.cos(initial)
    arguments = {"parent": parent, "theta": theta, "initial_noise": initial,
                 "transition_noise": noise, "role": "frozen_execution_reference"}
    actual = candidate.build_t1_parameter_density_batch(**arguments)
    expected = before.build_t1_parameter_density_batch(**arguments)
    names = tuple(field.name for field in fields(actual) if field.name != "role")
    ratio_names = tuple(field.name for field in fields(candidate.RatioScoreEstimate))
    for name in names:
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-10, rtol=1e-10)
    for index in range(2):
        result = candidate.estimate_t1_ratio_score(actual, theta_index=index)
        reference = before.estimate_t1_ratio_score(expected, theta_index=index)
        for name in ratio_names:
            np.testing.assert_allclose(getattr(result, name), getattr(reference, name), atol=1e-10, rtol=1e-10)

    def evaluate(parameters, initial, noise):
        batch = candidate.build_t1_parameter_density_batch(parent=parent, theta=parameters,
            initial_noise=initial, transition_noise=noise, role="enclosing_execution_reference")
        ratio = candidate.estimate_t1_ratio_score(batch, theta_index=1)
        return (tuple(getattr(batch, name) for name in names),
                tuple(getattr(ratio, name) for name in ratio_names))

    specs = [tf.TensorSpec(value.shape, D) for value in (theta, initial, noise)]
    graph = tf.function(evaluate, input_signature=specs, jit_compile=False, autograph=False)
    compiled = tf.function(evaluate, input_signature=specs, jit_compile=True, autograph=False)
    for value, authority in zip(tf.nest.flatten(compiled(theta, initial, noise)),
                                tf.nest.flatten(graph(theta, initial, noise)), strict=True):
        np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    _graph(graph)
    assert "HloModule" in compiled.experimental_get_compiler_ir(theta, initial, noise)(stage="hlo")


@pytest.mark.parametrize("name", [
    "test_parameter_batch_binds_physical_chart_and_absolute_weight_identity",
    "test_independent_ratio_and_prefix_estimators_report_finite_uncertainty",
])
def test_existing_preparation_consumers_with_fresh_parent(name, monkeypatch):
    from tests.highdim import (
        test_zhao_cui_austria_sir_parameter_density_training_tf as consumers,
    )

    monkeypatch.setattr(consumers, "_parent", _fresh_parent)
    getattr(consumers, name)()
