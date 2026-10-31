"""Independent diagnostic derivatives; no runtime/canonical score admission."""

import gc
import hashlib
import subprocess
import weakref
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as current
from tests.filter_repair_highest_dot_candidate import (
    HighestPrecisionDotFamily,
    HighestPrecisionDotProgram,
)
from tests.test_filter_repair_genut_layout import _highest_dot_candidate
from tests.test_filter_repair_genut_transitive import _fixture, _owner, _write


def _values(dtype, left_axis, right_axis):
    # Unequal extents and nonsymmetric entries expose orientation errors.
    left = (np.arange(15, dtype=np.float64).reshape(3, 5) - 6.1) / 11
    right = np.cos(np.arange(20, dtype=np.float64).reshape(5, 4) * .17)
    cotangent = np.sin(np.arange(12, dtype=np.float64).reshape(3, 4) + .3)
    first = np.sin(left + .2) / 7
    second = np.cos(right - .3) / 5
    if left_axis == 0:
        left, first = left.T, first.T
    if right_axis == 1:
        right, second = right.T, second.T
    return tuple(tf.constant(x, dtype) for x in (left, right, cotangent, first, second))


def _matrix_reference(left, right, cotangent, left_axis, right_axis):
    oriented_left = left.T if left_axis == 0 else left
    oriented_right = right.T if right_axis == 1 else right
    grad_left = cotangent @ oriented_right.T
    grad_right = oriented_left.T @ cotangent
    return (oriented_left @ oriented_right,
            grad_left.T if left_axis == 0 else grad_left,
            grad_right.T if right_axis == 1 else grad_right)


def _differentiated(program, specs):
    def evaluate(left, right, cotangent, first, second):
        with tf.GradientTape() as outer:
            outer.watch((left, right))
            with tf.GradientTape() as inner:
                inner.watch((left, right))
                value = program(left, right)
                loss = tf.reduce_sum(value * cotangent)
            gradients = inner.gradient(loss, (left, right))
            projected = tf.reduce_sum(gradients[0] * first) + tf.reduce_sum(gradients[1] * second)
        mixed = outer.gradient(projected, (left, right))
        return value, gradients, mixed
    return tf.function(evaluate, input_signature=specs, jit_compile=True, autograph=False)


def _assert_close(actual, expected, tolerance):
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("left_axis,right_axis", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_dot_value_pullback_and_mixed_derivative(dtype, left_axis, right_axis, request):
    operands = _values(dtype, left_axis, right_axis)
    left, right, cotangent, first, second = operands
    owner = HighestPrecisionDotProgram(left.shape, right.shape, left_axis, right_axis, dtype)
    derivative = _differentiated(owner,
        [tf.TensorSpec(x.shape, x.dtype) for x in operands])
    graph_parent = tf.function(lambda a, b, program=owner: program(a, b),
        input_signature=owner.function.input_signature, jit_compile=False, autograph=False)
    tolerance = 2e-5 if dtype == tf.float32 else 1e-10
    report = {"role": "uninstalled_dot_derivative_diagnostic", "dtype": dtype.name,
              "axes": [left_axis, right_axis], "graph_parent_contains_xla_dot": True,
              "cases": []}
    try:
        for change in (0., .13):
            args = (left + change, right - change / 2, cotangent, first, second)
            actual = derivative(*args)
            arrays = [x.numpy().astype(np.float64) for x in args]
            ref = _matrix_reference(*arrays[:3], left_axis, right_axis)
            mixed_left = _matrix_reference(arrays[0], arrays[4], arrays[2], left_axis, right_axis)[1]
            mixed_right = _matrix_reference(arrays[3], arrays[1], arrays[2], left_axis, right_axis)[2]
            reference = (ref[0], (ref[1], ref[2]), (mixed_left, mixed_right))
            step = 1e-5
            upper = _matrix_reference(arrays[0] + step * arrays[3], arrays[1] + step * arrays[4], arrays[2], left_axis, right_axis)[0]
            lower = _matrix_reference(arrays[0] - step * arrays[3], arrays[1] - step * arrays[4], arrays[2], left_axis, right_axis)[0]
            finite_difference = np.sum((upper - lower) * arrays[2]) / (2 * step)
            projected = np.sum(ref[1] * arrays[3]) + np.sum(ref[2] * arrays[4])
            report["cases"].append({"change": change, "actual": actual, "reference": reference,
                                    "fd": finite_difference, "reference_projection": projected})
            _assert_close(finite_difference, projected, 1e-6)
            for observed, expected in zip(tf.nest.flatten(actual), tf.nest.flatten(reference)):
                _assert_close(observed.numpy(), expected, tolerance)
            _assert_close(graph_parent(*args[:2]).numpy(), ref[0], tolerance)
            replay = derivative(*args)
            assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(replay)))
        hlo = derivative.experimental_get_compiler_ir(*operands)(stage="optimized_hlo")
        assert hlo == derivative.experimental_get_compiler_ir(*operands)(stage="optimized_hlo")
        report["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
        report["traces"] = [f.experimental_get_tracing_count() for f in (owner.function, derivative, graph_parent)]
        assert report["traces"] == [1, 1, 1]
        references = tuple(weakref.ref(x) for x in (owner, owner.function, derivative, graph_parent))
        del owner, derivative, graph_parent
        gc.collect()
        report["collected"] = [ref() is None for ref in references]
        assert all(report["collected"])
    finally:
        _write(request, f"dot-pullback-{dtype.name}-{left_axis}-{right_axis}.json", report)


def test_dot_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="float32 or float64"):
        HighestPrecisionDotProgram((3, 5), (5, 4), 1, 0, tf.int32)
    with pytest.raises(ValueError, match="zero or one"):
        HighestPrecisionDotProgram((3, 5), (5, 4), 2, 0, tf.float32)
    with pytest.raises(ValueError, match="fixed rank-two"):
        HighestPrecisionDotProgram((None, 5), (5, 4), 1, 0, tf.float32)
    with pytest.raises(ValueError, match="dimensions must agree"):
        HighestPrecisionDotProgram((3, 5), (6, 4), 1, 0, tf.float32)


def _program_gradient(implementation, inputs, *, jit=True, steps=2, coefficients=None):
    def evaluate(*args):
        with tf.GradientTape() as tape:
            tape.watch(args)
            outputs = implementation(*args, diagonal_steps=steps, pairwise_steps=steps)
            particles = outputs["particles"]
            multiplier = tf.reshape(tf.cast(tf.range(tf.size(particles)), particles.dtype), particles.shape)
            multiplier = tf.cos(multiplier * .11) if coefficients is None else tf.cast(coefficients, particles.dtype)
            loss = tf.reduce_sum(tf.sin(particles) * multiplier)
        return outputs, loss, tape.gradient(loss, args)
    return _owner(evaluate, inputs, jit)


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
def test_current_genut_bounded_derivative(dtype, request):
    source = subprocess.check_output(["git", "show",
        "28cbdb536:bayesfilter/highdim/dual_cap_genut_primal_tf.py"], text=True)
    original = ModuleType("_genut_pre_dot_pullback_repair_reference")
    exec(compile(source, "<genut-pre-dot-pullback-repair>", "exec"), original.__dict__)  # noqa: S102
    inputs = _fixture(dtype)
    report = {"role": "current_primal_execution_and_independent_derivative_reference",
              "reference_commit": "28cbdb536", "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "dtype": dtype.name, "cases": []}
    try:
        for steps in (2, 0, 4):
            reference = _program_gradient(original.dual_cap_genut_primal, inputs, jit=False, steps=steps)
            current_graph = _program_gradient(current.dual_cap_genut_primal, inputs, jit=False, steps=steps)
            actual_owner = _program_gradient(current.dual_cap_genut_primal, inputs, steps=steps)
            old_value = _owner(original.dual_cap_genut_primal, inputs,
                diagonal_steps=steps, pairwise_steps=steps)
            expected = reference(*inputs)
            row = {"steps": steps, "reference": expected}
            report["cases"].append(row)
            actual = actual_owner(*inputs)
            graph = current_graph(*inputs)
            row.update(actual=actual, current_graph=graph, old_xla_value=old_value(*inputs))
            tolerance = 2e-5 if dtype == tf.float32 else 2e-10
            # Thresholded report disagreement across modes is independently
            # open. Compare ALL values against the old same-mode program.
            for a, b in zip(tf.nest.flatten(actual[0]), tf.nest.flatten(row["old_xla_value"])):
                _assert_close(a.numpy(), b.numpy(), tolerance)
            for a, b in zip(tf.nest.flatten(graph), tf.nest.flatten(expected)):
                _assert_close(a.numpy(), b.numpy(), tolerance)
            for a, b in zip(tf.nest.flatten(actual[1:]), tf.nest.flatten(expected[1:])):
                _assert_close(a.numpy(), b.numpy(), tolerance)
            assert bool(actual[0]["valid"])
            changed = (inputs[0] + .1, inputs[1], inputs[2] - .03)
            changed_actual, changed_reference = actual_owner(*changed), reference(*changed)
            row.update(changed_actual=changed_actual, changed_reference=changed_reference)
            for a, b in zip(tf.nest.flatten(changed_actual[1:]), tf.nest.flatten(changed_reference[1:])):
                _assert_close(a.numpy(), b.numpy(), tolerance)
            assert actual_owner.experimental_get_tracing_count() == 1
            if dtype == tf.float64:
                directions = tuple(tf.reshape(tf.sin(tf.cast(tf.range(tf.size(x)), dtype) + .31), x.shape) / tf.cast(tf.size(x), dtype) for x in inputs)
                step = tf.constant(1e-4, dtype)
                high = reference(*(x + step * v for x, v in zip(inputs, directions)))[1]
                low = reference(*(x - step * v for x, v in zip(inputs, directions)))[1]
                finite_difference = (high - low) / (2 * step)
                projection = tf.add_n([tf.reduce_sum(g * v) for g, v in zip(actual[2], directions)])
                row.update(finite_difference=finite_difference, projection=projection)
                _assert_close(projection.numpy(), finite_difference.numpy(), 1e-6)
    finally:
        _write(request, f"genut-bounded-derivative-{dtype.name}.json", report)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["f32", "f64"])
def test_complete_genut_value_and_derivative(dtype, request):
    family = HighestPrecisionDotFamily()
    candidate, source, _ = _highest_dot_candidate()
    candidate.highest_dot = family
    inputs = _fixture(dtype)
    actual_owner = _program_gradient(candidate.dual_cap_genut_primal, inputs)
    reference_owner = _program_gradient(current.dual_cap_genut_primal, inputs)
    report = {"role": "uninstalled_full_genut_derivative_diagnostic", "dtype": dtype.name,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    try:
        actual, expected = actual_owner(*inputs), reference_owner(*inputs)
        report.update(actual=actual, reference=expected)
        tolerance = 2e-5 if dtype == tf.float32 else 1e-10
        for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
            _assert_close(a.numpy(), b.numpy(), tolerance)
        replay = actual_owner(*inputs)
        assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(replay)))
        assert actual_owner.experimental_get_tracing_count() == reference_owner.experimental_get_tracing_count() == 1
        report["dot_program_count"] = len(family.programs)
        report["dot_traces"] = [program.function.experimental_get_tracing_count() for program in family.programs.values()]
        assert all(count == 1 for count in report["dot_traces"])
        if dtype == tf.float64:
            # Centered finite difference of the current reduction value program.
            directions = tuple(tf.reshape(tf.sin(tf.cast(tf.range(tf.size(x)), dtype) + .31), x.shape) / tf.cast(tf.size(x), dtype) for x in inputs)
            step = tf.constant(1e-4, dtype)
            high = reference_owner(*(x + step * direction for x, direction in zip(inputs, directions)))[1]
            low = reference_owner(*(x - step * direction for x, direction in zip(inputs, directions)))[1]
            finite_difference = (high - low) / (2 * step)
            projected = tf.add_n([tf.reduce_sum(g * direction) for g, direction in zip(actual[2], directions)])
            report.update(finite_difference=finite_difference, projected=projected)
            _assert_close(projected.numpy(), finite_difference.numpy(), 1e-6)
    finally:
        _write(request, f"genut-dot-derivative-{dtype.name}.json", report)
