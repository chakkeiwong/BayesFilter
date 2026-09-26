"""Independent FP64/TF32 diagnostics for reduced-primal reverse precision."""

import hashlib
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops import linalg_grad

from bayesfilter.highdim import dual_cap_genut_primal_tf as current
from tests import filter_repair_genut_reverse_candidate as reverse_candidate
from tests.test_filter_repair_genut_dot_pullback import _program_gradient
from tests.test_filter_repair_genut_transitive import _fixture, _write


def _coefficients(shape):
    with tf.device("/CPU:0"):
        indices = tf.reshape(tf.cast(tf.range(shape.num_elements()), tf.float32), shape)
        return tf.cos(indices * .11)


def _fields(result):
    values, loss, gradients = result
    return {**values, "scalar_loss": loss, "source_gradient": gradients[0],
            "weight_gradient": gradients[1], "reset_gradient": gradients[2]}


def _compare(actual, expected):
    return {key: {"passed": bool(np.allclose(np.asarray(value, dtype=np.float64),
                      np.asarray(expected[key], dtype=np.float64), rtol=2e-5, atol=2e-5)),
                  "max_abs_error": float(np.max(np.abs(np.asarray(value, dtype=np.float64) -
                      np.asarray(expected[key], dtype=np.float64))))}
            for key, value in actual.items()}


def _independent_reference(inputs, coefficients):
    rounded = tuple(tf.cast(value, tf.float64) for value in inputs)
    owner = _program_gradient(current.dual_cap_genut_primal, rounded, jit=False,
        coefficients=coefficients)
    reference = owner(*rounded)
    directions = tuple(tf.reshape(tf.sin(tf.cast(tf.range(tf.size(x)), tf.float64) + .31), x.shape)
                       / tf.cast(tf.size(x), tf.float64) for x in rounded)
    step = tf.constant(1e-4, tf.float64)
    upper = owner(*(x + step * v for x, v in zip(rounded, directions)))[1]
    lower = owner(*(x - step * v for x, v in zip(rounded, directions)))[1]
    finite_difference = (upper - lower) / (2 * step)
    projection = tf.add_n([tf.reduce_sum(g * v) for g, v in zip(reference[2], directions)])
    np.testing.assert_allclose(projection.numpy(), finite_difference.numpy(), rtol=1e-6, atol=1e-6)
    return _fields(reference), {"projection": projection, "finite_difference": finite_difference}


def test_reverse_precision_toggle(request):
    inputs = _fixture(tf.float32)
    coefficients = _coefficients(inputs[0].shape)
    reference, fd = _independent_reference(inputs, coefficients)
    source = Path(inspect.getfile(linalg_grad)).read_bytes()
    points, weights = (x.numpy().astype(np.float64) for x in inputs[:2])
    centered = points - weights @ points
    prior = tf.config.experimental.tensor_float_32_execution_enabled()
    report = {"role": "reverse_precision_localization_not_runtime_qualification",
              "inputs": inputs, "coefficients": coefficients, "reference": reference,
              "reference_fd": fd, "tensorflow_gradient_source_sha256": hashlib.sha256(source).hexdigest(),
              "covariance_condition": float(np.linalg.cond((centered.T * weights) @ centered)),
              "arms": []}
    try:
        for tf32 in (True, False):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            for jit in (False, True):
                owner = _program_gradient(current.dual_cap_genut_primal, inputs, jit=jit,
                    coefficients=coefficients)
                result = _fields(owner(*inputs))
                report["arms"].append({"tf32": tf32, "jit_compile": jit, "result": result,
                    "comparison": _compare(result, reference)})
                assert owner.experimental_get_tracing_count() == 1
                assert all(np.all(np.isfinite(x.numpy())) for x in result.values())
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(prior)
        _write(request, "genut-reverse-precision-toggle.json", report)


def _substitution(kind):
    source = inspect.getsource(current)
    if kind in ("cholesky", "both", "all"):
        assert source.count("tf.linalg.cholesky(") == 2
        source = source.replace("tf.linalg.cholesky(", "native_cholesky(")
    if kind in ("solve", "both", "solve_matrix", "all", "solve_diagonal_primal"):
        assert source.count("tf.linalg.triangular_solve(") == 1
        source = source.replace("tf.linalg.triangular_solve(", "native_triangular_solve(")
    if kind in ("matrix_solve", "solve_matrix", "all"):
        assert source.count("tf.linalg.solve(") == 1
        source = source.replace("tf.linalg.solve(", "native_matrix_solve(")
    if kind == "all":
        before = "tf.linalg.matmul(jacobian, jacobian, transpose_a=True)"
        assert source.count(before) == 1
        source = source.replace(before, "native_gram(jacobian)")
        before = "tf.linalg.matvec(jacobian, residual, transpose_a=True)"
        assert source.count(before) == 1
        source = source.replace(before, "native_transposed_matvec(jacobian, residual)")
    if kind in ("diagonal_primal", "solve_diagonal_primal"):
        before = "tf.linalg.matmul(jacobian, jacobian, transpose_a=True)"
        assert source.count(before) == 1
        source = source.replace(before, "precise_gram(jacobian)")
        before = "tf.linalg.matvec(jacobian, residual, transpose_a=True)"
        assert source.count(before) == 1
        source = source.replace(before, "precise_transposed_matvec(jacobian, residual)")
    module = ModuleType("_genut_reverse_precision_" + kind)
    module.native_cholesky = reverse_candidate.native_cholesky
    module.native_triangular_solve = reverse_candidate.native_triangular_solve
    module.native_matrix_solve = reverse_candidate.native_matrix_solve
    module.native_gram = reverse_candidate.native_gram
    module.native_transposed_matvec = reverse_candidate.native_transposed_matvec
    module.precise_gram = reverse_candidate.precise_gram
    module.precise_transposed_matvec = reverse_candidate.precise_transposed_matvec
    exec(compile(source, "<genut-reverse-precision-diagnostic>", "exec"), module.__dict__)  # noqa: S102
    return module, source


def test_reverse_precision_substitutions(request):
    _site_trial(("cholesky", "solve", "both"), "both", "genut-reverse-substitutions.json", request)


def test_reverse_precision_additional_sites(request):
    _site_trial(("matrix_solve", "solve_matrix", "all"), "all", "genut-reverse-additional-sites.json", request)


def test_reverse_precision_primal_products(request):
    _site_trial(("diagonal_primal", "solve_diagonal_primal"), "solve_diagonal_primal",
        "genut-reverse-primal-products.json", request, require_bitwise=False)


def _site_trial(kinds, required, filename, request, *, require_bitwise=True):
    assert tf.config.experimental.tensor_float_32_execution_enabled()
    inputs = _fixture(tf.float32)
    coefficients = _coefficients(inputs[0].shape)
    reference, fd = _independent_reference(inputs, coefficients)
    output = Path(request.config.getoption("xmlpath")).parent
    helper = inspect.getsource(reverse_candidate)
    (output / "genut-reverse-helper.py").write_text(helper)
    report = {"role": "reverse_precision_site_substitution_no_runtime_change",
              "reference": reference, "reference_fd": fd, "inputs": inputs,
              "coefficients": coefficients, "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(),
              "arms": []}
    try:
        for jit in (False, True):
            baseline = _fields(_program_gradient(current.dual_cap_genut_primal, inputs,
                jit=jit, coefficients=coefficients)(*inputs))
            for kind in kinds:
                module, source = _substitution(kind)
                (output / f"genut-reverse-{kind}.py").write_text(source)
                owner = _program_gradient(module.dual_cap_genut_primal, inputs,
                    jit=jit, coefficients=coefficients)
                result = _fields(owner(*inputs))
                row = {"kind": kind, "jit_compile": jit, "result": result,
                       "comparison": _compare(result, reference),
                       "current_comparison": _compare(result, baseline),
                       "forward_bitwise_same": all(np.array_equal(result[key].numpy(), value.numpy())
                           for key, value in baseline.items() if not key.endswith("_gradient")),
                       "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
                report["arms"].append(row)
                if require_bitwise:
                    assert row["forward_bitwise_same"]
                assert owner.experimental_get_tracing_count() == 1
                assert all(np.all(np.isfinite(x.numpy())) for x in result.values())
    finally:
        _write(request, filename, report)
    assert all(field["passed"] for arm in report["arms"] if arm["kind"] == required
               for key, field in arm["comparison"].items() if key != "fraction_coordinatewise_cap_active")


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["cholesky", "solve"])
@pytest.mark.parametrize("dimension", [3, 18])
def test_reverse_primitives(kind, dtype, dimension, request):
    rng = np.random.default_rng(719 + dimension)
    lower = np.tril(.1 * rng.normal(size=(dimension, dimension))) + np.eye(dimension) * 2
    right = rng.normal(size=(dimension, 7))
    np_dtype = np.float32 if dtype == tf.float32 else np.float64
    matrix = np.asarray(lower @ lower.T if kind == "cholesky" else lower, dtype=np_dtype)
    right = np.asarray(right, dtype=np_dtype)
    cotangent = np.asarray(rng.normal(size=matrix.shape if kind == "cholesky" else right.shape), dtype=np_dtype)
    direction = rng.normal(size=matrix.shape) / dimension
    direction = .5 * (direction + direction.T) if kind == "cholesky" else np.tril(direction)
    other_direction = rng.normal(size=right.shape) / dimension
    m64, b64, g64 = (x.astype(np.float64) for x in (matrix, right, cotangent))
    if kind == "cholesky":
        expected_value = np.linalg.cholesky(m64)
        inverse = np.linalg.inv(expected_value)
        middle = expected_value.T @ g64
        middle[np.diag_indices(dimension)] *= .5
        expected_grad = inverse.T @ np.tril(middle) @ inverse
        expected_grad = .5 * (expected_grad + expected_grad.T)
        expected_gradient = (expected_grad,)
        def reference_value(m, _b):
            return np.linalg.cholesky(m)
    else:
        expected_value = np.linalg.solve(m64, b64)
        grad_right = np.linalg.solve(m64.T, g64)
        expected_gradient = (np.tril(-grad_right @ expected_value.T), grad_right)
        def reference_value(m, b):
            return np.linalg.solve(m, b)

    def calculate(a, b, g):
        operands = (a,) if kind == "cholesky" else (a, b)
        with tf.GradientTape() as tape:
            tape.watch(operands)
            value = reverse_candidate.native_cholesky(a) if kind == "cholesky" else reverse_candidate.native_triangular_solve(a, b)
            loss = tf.reduce_sum(value * g)
        return value, tape.gradient(loss, operands)

    tensors = tuple(tf.constant(x, dtype) for x in (matrix, right, cotangent))
    owner = tf.function(calculate, input_signature=[tf.TensorSpec(x.shape, dtype) for x in tensors],
        autograph=False, jit_compile=True)
    actual = owner(*tensors)
    step = 1e-5
    fd = np.sum((reference_value(m64 + step * direction, b64 + step * other_direction)
                 - reference_value(m64 - step * direction, b64 - step * other_direction)) * g64) / (2 * step)
    projected = np.sum(actual[1][0].numpy() * direction)
    if kind == "solve":
        projected += np.sum(actual[1][1].numpy() * other_direction)
    _write(request, f"genut-reverse-primitive-{kind}-{dimension}-{dtype.name}.json",
        {"role": "independent_pullback_reference", "actual": actual,
         "expected": (expected_value, expected_gradient), "fd": fd, "projected": projected})
    tolerance = 2e-5 if dtype == tf.float32 else 2e-10
    for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten((expected_value, expected_gradient))):
        np.testing.assert_allclose(a.numpy(), b, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(projected, fd, rtol=2e-5 if dtype == tf.float32 else 1e-6,
        atol=2e-5 if dtype == tf.float32 else 1e-6)
    assert owner.experimental_get_tracing_count() == 1
