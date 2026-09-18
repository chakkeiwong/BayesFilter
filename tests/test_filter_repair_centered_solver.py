"""Pinned finite-solver parity and independent quadratic solution checks."""

from dataclasses import fields

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.centered_training_native_tf import quadratic_cg_program
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_training_tf import (
    solve_quadratic_value_gradient_with_conjugate_gradient as solve,
)
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("case", ["positive", "already_solved", "budget", "negative", "zero_curvature"])
def test_quadratic_solver_preserves_solution_status_and_trace(case):
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    diagonal = tf.constant([1., 2., 4., 8.], D)
    if case == "negative":
        diagonal = -diagonal
    if case == "zero_curvature":
        diagonal = tf.zeros_like(diagonal)
    rhs = tf.constant([.2, -.3, .5, -.7], D)

    def callback(position):
        return .5 * tf.reduce_sum(diagonal * position**2) - tf.reduce_sum(rhs * position), diagonal * position - rhs

    initial = rhs / diagonal if case == "already_solved" else tf.zeros([4], D)
    maximum = 2 if case == "budget" else 12
    options = {"initial_position": initial, "tolerance": 1e-10, "max_iterations": maximum, "trace_interval": 3}
    actual = solve(callback, **options)
    expected = before.solve_quadratic_value_gradient_with_conjugate_gradient(callback, **options)
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "trace":
            assert tuple(row[0] for row in left) == tuple(row[0] for row in right)
        for value, authority in zip(tf.nest.flatten(left), tf.nest.flatten(right), strict=True):
            if isinstance(value, (int, bool)):
                assert value == authority
            else:
                np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    if case in ("positive", "already_solved"):
        assert actual.converged and not actual.failed
        np.testing.assert_allclose(actual.position, rhs / diagonal, atol=1e-10, rtol=1e-10)
    else:
        assert not actual.converged
        assert actual.failed == (case != "budget")
    program = quadratic_cg_program(callback, tf.TensorSpec([4], D), maximum, 3)
    assert "HloModule" in program.experimental_get_compiler_ir(initial, tf.constant(1e-10, D))(stage="hlo")
    assert any(node.op in ("While", "StatelessWhile") for node in _graph(program))
    assert program.experimental_get_tracing_count() == 1


def test_quadratic_solver_preserves_nonfinite_action_veto():
    def callback(position):
        return tf.zeros([], D), tf.fill(position.shape, tf.constant(float("nan"), D))

    with pytest.raises(tf.errors.InvalidArgumentError, match="quadratic Hessian action"):
        solve(callback, initial_position=tf.zeros([4], D), tolerance=1e-10, max_iterations=4)
