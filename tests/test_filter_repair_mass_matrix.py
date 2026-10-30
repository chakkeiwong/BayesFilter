"""Pinned public records, independent identities and XLA mass derivatives."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import mass_matrix as candidate
from bayesfilter.inference import mass_matrix_tf as native

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/mass_matrix.py"], text=True)
    spec = importlib.util.spec_from_loader("mass_matrix_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - independent pinned reference
    return module


def _compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key])
    elif (isinstance(expected, (str, bool)) or expected is None
          or isinstance(expected, (list, tuple)) and expected and isinstance(expected[0], dict)):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("matrix,options", [
    ([[4., 1.], [1., 3.]], {"jitter": 0.}),
    ([[2., .5], [0., 3.]], {"jitter": 1e-9}),
    ([[2., 0.], [0., -.5]], {"jitter": 0., "eigenvalue_floor": .25}),
    ([[2., 0.], [0., .5]], {"jitter": 0., "eigenvalue_floor": .5}),
    ([[2., 0.], [0., .1]], {"jitter": 0., "max_condition_number": 4.}),
    ([[-2., 0.], [0., -1.]], {"jitter": 0., "eigenvalue_floor": .5}),
])
def test_public_mass_records_preserve_projection_and_floor_decisions(baseline, dense, matrix, options):
    expected = baseline.covariance_from_precision(matrix, source="pinned", dense=dense, **options)
    actual = candidate.covariance_from_precision(matrix, source="pinned", dense=dense, **options)
    _compare(vars(actual), vars(expected))
    if dense:
        np.testing.assert_allclose(actual.regularized_precision @ actual.covariance, np.eye(2), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("matrix,options,message", [
    ([[1., np.nan], [0., 2.]], {}, "precision must be finite"),
    ([[1., np.inf], [0., 2.]], {}, "precision must be finite"),
    ([[0., 0.], [0., 0.]], {"jitter": 0.}, "positive eigenvalue"),
    ([[-2., 0.], [0., -1.]], {"jitter": 0.}, "positive eigenvalue"),
    ([[1., 0.], [0., 1.]], {"jitter": -1.}, "jitter must be"),
    ([[1., 0.], [0., 1.]], {"eigenvalue_floor": np.nan}, "eigenvalue_floor must be"),
    ([[1., 0.], [0., 1.]], {"max_condition_number": 1.}, "max_condition_number must be"),
    ([[1., 0.]], {}, "square matrix"),
])
def test_public_mass_keeps_fail_closed_validation(baseline, matrix, options, message):
    for module in (baseline, candidate):
        with pytest.raises(ValueError, match=message):
            module.regularize_precision(matrix, **options)


@pytest.mark.parametrize("diagonal,widths", [(True, (1, 1, 1, 1)), (False, (2, 1, 3)), (False, (2, 2, 2))])
@pytest.mark.parametrize("shrinkage", [0., .1, 1.])
def test_structural_projection_preserves_complete_public_records(baseline, diagonal, widths, shrinkage):
    dimension = sum(widths)
    matrix = np.arange(dimension ** 2, dtype=float).reshape(dimension, dimension) / (dimension ** 2)
    matrix += np.diag(np.linspace(-1., 3., dimension))
    stops = np.cumsum((0, *widths))
    blocks = tuple({"name": str(i), "start": int(stops[i]), "stop": int(stops[i + 1])} for i in range(len(widths)))
    options = {"diagonal": diagonal, "blocks": None if diagonal else blocks,
        "shrinkage": shrinkage, "eigenvalue_floor": .2, "max_condition_number": 8.}
    expected = baseline.structured_covariance_from_empirical(matrix, **options)
    actual = candidate.structured_covariance_from_empirical(matrix, **options)
    _compare(vars(actual), vars(expected))
    for row in range(len(widths)):
        for column in range(len(widths)):
            if row != column:
                assert not np.any(actual.covariance[stops[row]:stops[row + 1], stops[column]:stops[column + 1]])


@pytest.mark.parametrize("kind", ["regularized", "dense", "diagonal", "structural", "whitening"])
def test_matrix_pullback_retains_original_frozen_floor_boundary(baseline, kind):
    # A condition-based active floor depends on the matrix numerically, but the
    # prior host boundary intentionally froze that dependence in its pullback.
    matrix = tf.constant([[3., .2, .1], [.2, .4, .05], [.1, .05, 2.]], D)
    weights = tf.reshape(tf.linspace(tf.constant(.3, D), 1.1, 9), [3, 3])

    def result(module, point):
        options = {"jitter": 0., "max_condition_number": 5.}
        if kind == "regularized":
            return module.regularize_precision(point, **options)[0]
        if kind in ("dense", "diagonal"):
            return module.covariance_from_precision(point, source="gradient", dense=kind == "dense", **options).covariance
        if kind == "whitening":
            return module.whitening_from_covariance(point, jitter=1e-9)
        return module.structured_covariance_from_empirical(point, blocks=(
            {"name": "a", "start": 0, "stop": 2}, {"name": "b", "start": 2, "stop": 3}),
            shrinkage=.1, eigenvalue_floor=.2, max_condition_number=5.).covariance

    gradients = []
    values = []
    for module in (baseline, candidate):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            value = result(module, matrix)
            objective = tf.reduce_sum(value * weights)
        gradients.append(tape.gradient(objective, matrix))
        values.append(value)
    np.testing.assert_allclose(values[1], values[0], atol=1e-10, rtol=1e-10)
    assert all(gradient is not None for gradient in gradients)
    np.testing.assert_allclose(gradients[1], gradients[0], atol=1e-10, rtol=1e-10)


def test_compiled_precision_gradient_matches_inverse_identity_and_finite_difference():
    matrix = tf.constant([[3., .2], [.2, 1.]], D)
    direction = tf.constant([[.3, -.2], [-.2, .1]], D)
    program = native.precision_program(2, dense=True)

    @tf.function(input_signature=[tf.TensorSpec([2, 2], D)], jit_compile=True, autograph=False)
    def value_and_gradient(point):
        with tf.GradientTape() as tape:
            tape.watch(point)
            covariance = program(point, tf.constant(0., D), tf.constant(.01, D), tf.constant(0., D))[1]
            value = tf.linalg.trace(covariance)
        return value, tape.gradient(value, point)

    _, gradient = value_and_gradient(matrix)
    inverse = np.linalg.inv(matrix)
    np.testing.assert_allclose(gradient, -inverse @ inverse, atol=1e-10, rtol=1e-10)
    step = 1e-4
    plus = value_and_gradient(matrix + step * direction)[0]
    minus = value_and_gradient(matrix - step * direction)[0]
    np.testing.assert_allclose(tf.reduce_sum(gradient * direction), (plus - minus) / (2 * step), atol=1e-8, rtol=1e-7)
    assert "HloModule" in value_and_gradient.experimental_get_compiler_ir(matrix)(stage="hlo")


def test_structural_graph_reuses_block_body_and_keeps_vjp_inside_xla():
    counts = []
    for block_count in (2, 8):
        dimension = 2 * block_count
        partition = tuple((2 * i, 2 * i + 2) for i in range(block_count))
        program = native.structured_program(dimension, partition)
        inputs = (tf.eye(dimension, dtype=D), tf.constant(.1, D), tf.constant(.2, D), tf.constant(100., D))
        program(*inputs)
        program(*inputs)
        assert program.experimental_get_tracing_count() == 1
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)
        assert any(node.op in ("While", "StatelessWhile") for node in nodes)
        counts.append(len(nodes))
    assert counts[1] <= counts[0] + 10


def test_covariance_jitter_preserves_original_cast_and_whitening(baseline):
    matrix = tf.constant([[2., .2], [.2, 1.]], D)
    for jitter in (0., 1e-9, .123456789):
        expected = baseline.regularize_covariance(matrix, jitter=jitter)
        actual = candidate.regularize_covariance(matrix, jitter=jitter)
        np.testing.assert_array_equal(actual, expected)
        factor = candidate.whitening_from_covariance(matrix, jitter=jitter)
        np.testing.assert_allclose(factor @ tf.transpose(factor), expected, atol=1e-10, rtol=1e-10)


def test_summary_fields_for_empty_and_nonpositive_matrices(baseline):
    for matrix in (np.empty((0, 0)), np.diag([-1., 2.]), np.zeros((2, 2))):
        _compare(candidate._eigen_summary(matrix), baseline._eigen_summary(matrix))


def test_eigensolver_residual_localization(baseline):
    from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

    matrix = tf.constant([[3., .2, .1], [.2, .4, .05], [.1, .05, 2.]], D)
    expected, _ = baseline.regularize_precision(matrix, jitter=0., max_condition_number=5.)
    def build(epsilon):
        @tf.function(input_signature=[tf.TensorSpec([3, 3], D)], jit_compile=True, autograph=False)
        def evaluate(point):
            values, vectors = xla_self_adjoint_eig(point, lower=True, max_iter=100, epsilon=epsilon)
            floor = tf.reduce_max(values) / 5.
            reconstructed = tf.matmul(vectors * tf.maximum(values, floor)[None, :], vectors, transpose_b=True)
            return reconstructed, point @ vectors - vectors * values[None, :]
        return evaluate

    for epsilon in (np.finfo(float).eps, np.finfo(float).eps ** 2):
        projected, residual = build(float(epsilon))(matrix)
        print("mass_eigensolver_residual", {"epsilon": float(epsilon),
            "residual_max": float(tf.reduce_max(tf.abs(residual))),
            "projection_error": float(tf.reduce_max(tf.abs(projected - expected)))})
    repaired, _ = candidate.regularize_precision(matrix, jitter=0., max_condition_number=5.)
    np.testing.assert_allclose(repaired, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dimension", [1, 2, 3, 6, 12])
@pytest.mark.parametrize("spectrum,scale", [("distinct", 1.), ("repeated", 1.), ("clustered", 1.),
    ("indefinite", 1e-12), ("indefinite", 1e12)])
def test_refined_eigenpairs_preserve_residuals_and_orthogonality(dimension, spectrum, scale):
    generator = np.random.default_rng(731 + dimension)
    frame, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    if spectrum == "repeated":
        diagonal = np.ones(dimension)
    elif spectrum == "clustered":
        diagonal = 1. + np.arange(dimension) * 1e-12
    else:
        diagonal = np.linspace(-2. if spectrum == "indefinite" else .2, 3., dimension)
    matrix = tf.constant((frame * diagonal) @ frame.T * scale, D)
    program = tf.function(native._eigh, input_signature=[tf.TensorSpec([dimension, dimension], D)],
        jit_compile=True, autograph=False)
    values, vectors = program(matrix)
    np.testing.assert_allclose(values / scale, np.linalg.eigvalsh(matrix / scale), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tf.transpose(vectors) @ vectors, np.eye(dimension), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose((matrix @ vectors - vectors * values[None, :]) / scale,
        np.zeros((dimension, dimension)), atol=1e-12, rtol=1e-12)
