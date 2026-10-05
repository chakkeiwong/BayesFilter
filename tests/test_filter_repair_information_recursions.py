"""Diagnostic parity for Kalman and backward-information execution repairs."""

import ast
import inspect

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as information
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _evaluate(module, theta, observations, points):
    kalman = getattr(module.exact_kalman_value, "python_function", module.exact_kalman_value)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = kalman(theta, observations)
    return (
        value, tape.gradient(value, theta),
        module._conditional_future_log_likelihood(theta, points, observations),
        *module._backward_information_parameters(theta, observations),
        *module._finite_lookahead_information_parameters(theta, observations, 2),
    )


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_information_recursions_preserve_values_and_derivatives(horizon, jit):
    theta = tf.constant([0.2, 0.3, 0.4, 0.5, 0.8], D)
    observations = tf.reshape(tf.linspace(tf.constant(-0.1, D), 0.2, horizon * 3), [horizon, 3])
    points = tf.constant([[0.1, 0.2, -0.1], [-0.3, 0.1, 0.4]], D)
    baseline = _original("ledh_contract_e_tp_lgssm_tf")
    expected = _evaluate(baseline, theta, observations, points)
    call = tf.function(lambda parameters: _evaluate(information, parameters, observations, points),
                       input_signature=[tf.TensorSpec([5], D)], jit_compile=jit, autograph=False)
    actual = call(theta)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
        assert bool(tf.reduce_all(tf.math.is_finite(result)))
    delta = 1e-5
    for axis in range(5):
        direction = tf.one_hot(axis, 5, dtype=D) * delta
        plus, minus = call(theta + direction)[0], call(theta - direction)[0]
        np.testing.assert_allclose(actual[1][axis], (plus - minus) / (2 * delta), rtol=1e-7, atol=1e-8)
    assert call.experimental_get_tracing_count() == 1
    _graph(call)
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(theta)(stage="hlo")


def test_kalman_default_is_compiled_and_horizon_does_not_unroll():
    assert information.exact_kalman_value._jit_compile
    theta = tf.constant([0.2, 0.3, 0.4, 0.5, 0.8], D)
    counts = []
    for horizon in (2, 8):
        observations = tf.zeros([horizon, 3], D)
        program = information.exact_kalman_value.get_concrete_function(theta, observations)
        graph = program.graph.as_graph_def()
        counts.append(len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function))
        assert "HloModule" in information.exact_kalman_value.experimental_get_compiler_ir(
            theta, observations)(stage="hlo")
    assert counts[0] == counts[1]
    for function in (
        information.exact_kalman_value.python_function,
        information._conditional_future_log_likelihood,
        information._backward_information_parameters,
        information._finite_lookahead_information_parameters,
    ):
        assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
                       for node in ast.walk(ast.parse(inspect.getsource(function))))
