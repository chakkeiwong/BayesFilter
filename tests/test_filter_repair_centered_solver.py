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


@pytest.mark.parametrize("width", [3, 5])
def test_additive_and_pair_encodings_preserve_cores_and_linear_pullback(width):
    from bayesfilter.highdim import centered_training_native_tf as native
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as candidate,
    )

    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    count = 36 * width + 18 * width * width
    coefficients = tf.sin(tf.cast(tf.range(count), D))
    for name, values, kwargs in (
        ("_additive_rank_two_component", tf.reshape(coefficients[:36 * width], [36, width]), {"basis_dims": (width,) * 36}),
        ("_additive_pair_rank_seven_component", coefficients, {"basis_dim": width}),
    ):
        expected = getattr(before, name)(values, **kwargs)
        actual = getattr(candidate, name)(values, **kwargs)
        for value, authority in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, authority)
        gradients = []
        for module in (candidate, before):
            with tf.GradientTape() as tape:
                tape.watch(values)
                result = getattr(module, name)(values, **kwargs)
                objective = sum(tf.reduce_sum(core**2) for core in result)
            gradients.append(tape.gradient(objective, values))
        np.testing.assert_array_equal(*gradients)
    additive = tf.reshape(coefficients[:36 * width], [1, 36, width])
    pair = tf.reshape(coefficients[36 * width:], [1, 18, width, width])
    component_scales = tf.constant([.2, -.3, .5], D)
    banks = native.additive_pair_core_banks(component_scales[:, None, None] * additive,
        component_scales[:, None, None, None] * pair)
    for index, packed in enumerate(tf.unstack(banks)):
        expected = before._additive_pair_rank_seven_component(component_scales[index] * coefficients, basis_dim=width)
        for actual, reference in zip(native.unpack_additive_bank(packed), expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
    assert "HloModule" in native.additive_pair_core_banks.experimental_get_compiler_ir(additive, pair)(stage="hlo")
    assert "HloModule" in native.additive_core_banks.experimental_get_compiler_ir(additive)(stage="hlo")
