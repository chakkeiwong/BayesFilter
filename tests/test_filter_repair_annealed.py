"""September execution parity against the original active-subset calculation."""

import numpy as np
import pytest
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    _transport_active,
    annealed_transport_resample_tf,
    make_annealed_transport_resample_tf,
)


def _inputs():
    x = tf.reshape(tf.sin(tf.cast(tf.range(24), tf.float64) * .71), [3, 4, 2])
    logw = tf.nn.log_softmax(tf.reshape(tf.cos(tf.cast(tf.range(12), tf.float64)), [3, 4]))
    return x, logw


@pytest.mark.parametrize("mask", [[True, False, True], [False, True, False], [False, False, False]])
@pytest.mark.parametrize("jit", [False, True])
def test_active_subset_stopping_values_and_gradients(mask, jit):
    x, logw = _inputs()
    active = tf.constant(mask)
    settings = dict(epsilon=.7, scaling=.8, convergence_threshold=1e-3,
                    max_iterations=8, transport_gradient_mode="filterflow_clipped")
    with tf.GradientTape() as tape:
        tape.watch((x, logw))
        if any(mask):
            points, transport, diag = _transport_active(
                tf.boolean_mask(x, active), tf.boolean_mask(logw, active), **settings)
            expected = tf.tensor_scatter_nd_update(x, tf.where(active), points)
            expected_transport = tf.tensor_scatter_nd_update(
                tf.zeros([3, 4, 4], tf.float64), tf.where(active), transport)
            iterations = diag["max_iterations_used"]
        else:
            expected, expected_transport, iterations = x, tf.zeros([3, 4, 4], tf.float64), 0.
        objective = tf.reduce_sum(tf.square(expected)) + .13 * tf.reduce_sum(tf.square(expected_transport))
    expected_grad = tape.gradient(objective, (x, logw), unconnected_gradients=tf.UnconnectedGradients.ZERO)

    with tf.GradientTape() as tape:
        tape.watch((x, logw))
        actual = annealed_transport_resample_tf(x, logw, ess_mask=active,
            jit_compile=jit, **settings)
        objective = tf.reduce_sum(tf.square(actual.particles)) + .13 * tf.reduce_sum(tf.square(actual.transport_matrix))
    actual_grad = tape.gradient(objective, (x, logw), unconnected_gradients=tf.UnconnectedGradients.ZERO)
    np.testing.assert_allclose(actual.particles, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual.transport_matrix, expected_transport, atol=1e-10, rtol=1e-10)
    assert actual.diagnostics["max_iterations_used"] == float(iterations)
    for got, want in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)


def test_factory_defaults_to_xla_with_bounded_graph_and_cache():
    spec = tf.TensorSpec([3, 4, 2], tf.float64)
    call = make_annealed_transport_resample_tf(spec, max_iterations=8)
    assert call is make_annealed_transport_resample_tf(spec, max_iterations=8)
    graph = call.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
    assert call.get_concrete_function().function_def.attr["_XlaMustCompile"].b
    x, logw = _inputs()
    first = call(x, logw, tf.constant([True, False, True]))
    second = call(x, logw, tf.constant([False, True, False]))
    assert bool(first[10]) and bool(second[11])
    assert call.experimental_get_tracing_count() == 1
