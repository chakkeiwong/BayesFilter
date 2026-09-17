"""Independent numerical/order checks for the execution-policy migration."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.sqmc_tf import hilbert_integer_keys, hilbert_transpose_words
from bayesfilter.ops.quadrature_tf import gauss_legendre, gauss_legendre_product


def reference_hilbert(row, bits):
    """Scalar Skilling transpose authority, using exact Python integer bits."""
    axes = list(row)
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for axis in range(len(axes)):
            if axes[axis] & q:
                axes[0] ^= p
            else:
                exchange = (axes[0] ^ axes[axis]) & p
                axes[0] ^= exchange
                axes[axis] ^= exchange
        q >>= 1
    for axis in range(1, len(axes)):
        axes[axis] ^= axes[axis - 1]
    correction = 0
    q = 1 << (bits - 1)
    while q > 1:
        if axes[-1] & q:
            correction ^= q - 1
        q >>= 1
    axes = [value ^ correction for value in axes]
    key = 0
    for bit in range(bits - 1, -1, -1):
        for axis in axes:
            key = (key << 1) | ((axis >> bit) & 1)
    return key


@pytest.mark.parametrize("dimension,bits", [(1, 1), (1, 20), (2, 3), (3, 10), (4, 8), (18, 20)])
def test_hilbert_compiled_matches_integer_authority(dimension, bits):
    rng = np.random.default_rng(421)
    coordinates = rng.integers(0, 1 << bits, size=(41, dimension))
    points = (coordinates + 0.25) / (1 << bits)
    target = tf.function(lambda x: hilbert_transpose_words(x, bits=bits), input_signature=[tf.TensorSpec([41, dimension], tf.float64)], jit_compile=True, autograph=False)
    words = target(tf.constant(points)).numpy().tolist()
    width = dimension * bits
    for row, observed in zip(coordinates.tolist(), words, strict=True):
        expected = reference_hilbert(row, bits)
        reconstructed = 0
        for index, word in enumerate(observed):
            reconstructed = (reconstructed << min(30, width - 30 * index)) | word
        assert reconstructed == expected
    if dimension in (2, 3):
        packed = hilbert_integer_keys(tf.constant(points), bits=bits).numpy()
        assert packed.tolist() == [reference_hilbert(row, bits) for row in coordinates.tolist()]


@pytest.mark.parametrize("order", [1, 2, 8, 40])
def test_gauss_legendre_nodes_weights_and_exact_moments(order):
    nodes, weights = gauss_legendre(order)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(order)
    np.testing.assert_allclose(nodes, ref_nodes, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(weights, ref_weights, rtol=1e-11, atol=1e-14)
    for degree in range(2 * order):
        value = tf.reduce_sum(weights * nodes ** degree)
        expected = 0. if degree % 2 else 2. / (degree + 1)
        np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-13)


def test_quadrature_product_order_and_mass():
    nodes, weights = gauss_legendre_product(3, 4)
    one, w = np.polynomial.legendre.leggauss(4)
    expected = np.stack(np.meshgrid(one, one, one, indexing="ij"), axis=-1).reshape(-1, 3)
    expected_w = np.prod(np.stack(np.meshgrid(w / 2, w / 2, w / 2, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    np.testing.assert_allclose(nodes, expected, atol=1e-14)
    np.testing.assert_allclose(weights, expected_w, atol=1e-14)


@pytest.mark.parametrize("jit", [False, True])
def test_fixed_quadrature_prepares_once_and_remains_valid_across_graphs(monkeypatch, jit):
    from bayesfilter.ops import quadrature_tf as quadrature

    quadrature._legendre_constants.cache_clear()
    prepare = quadrature._legendre_program
    preparations = []

    def tracked_program(*args):
        preparations.append(args)
        return prepare(*args)

    monkeypatch.setattr(quadrature, "_legendre_program", tracked_program)

    def moment(scale):
        nodes, weights = gauss_legendre(7, jit_compile=jit)
        with tf.GradientTape() as tape:
            tape.watch(scale)
            value = tf.reduce_sum(weights * tf.square(scale * nodes))
        return value, tape.gradient(value, scale)

    for _ in range(2):
        call = tf.function(moment, input_signature=[tf.TensorSpec([], tf.float64)],
                           jit_compile=jit, autograph=False)
        for scale in (1.0, 2.0):
            value, gradient = call(tf.constant(scale, tf.float64))
            np.testing.assert_allclose(value, 2.0 * scale**2 / 3.0, atol=1e-13)
            np.testing.assert_allclose(gradient, 4.0 * scale / 3.0, atol=1e-13)
        graph = call.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        assert not {node.op for node in nodes} & {"SelfAdjointEigV2", "XlaSelfAdjointEig"}
        assert call.experimental_get_tracing_count() == 1
    assert len(preparations) == 1
    if jit:
        assert "HloModule" in prepare(7, tf.float64, True).experimental_get_compiler_ir()(stage="hlo")


@pytest.mark.parametrize("family", ["dns", "retained_moments"])
def test_graph_reference_disables_nested_quadrature_compilation(family):
    from types import SimpleNamespace

    from bayesfilter.hardbound.dns_curve_tf import yield_curve
    from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
    from bayesfilter.highdim.squared_tt_engine_v0_tf import _initial_tt_cores, _product_basis
    from bayesfilter.highdim.tt import TTCore

    basis = _product_basis(2, 2)
    cores = _initial_tt_cores(2, 3, 2)
    inputs = (tf.constant([[.01, -.02, .03]], tf.float64) if family == "dns"
              else cores[0].values)

    def evaluate(value):
        if family == "dns":
            return yield_curve(value, tf.constant([.25, 1., 5.], tf.float64),
                .5, 0., .01, "softplus", jit_compile=False)
        retained = SimpleNamespace(prefix_cores=(TTCore(value), cores[1]),
            prefix_basis=basis, suffix_gram=tf.ones([1, 1], tf.float64),
            tau=tf.constant(1e-6, tf.float64), z_complete_ref=tf.constant(1.000001, tf.float64))
        return retained_reference_moments(retained, jit_compile=False)

    call = tf.function(evaluate, input_signature=[tf.TensorSpec(inputs.shape, inputs.dtype)],
        jit_compile=False, autograph=False)
    graph = call.get_concrete_function().graph.as_graph_def()
    assert not any(fn.attr.get("_XlaMustCompile", None) and fn.attr["_XlaMustCompile"].b
                   for fn in graph.library.function)
    assert not any(node.op.startswith("Xla") for fn in graph.library.function for node in fn.node_def)
    for expected, actual in zip(tf.nest.flatten(evaluate(inputs)), tf.nest.flatten(call(inputs))):
        np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=1e-12)


def test_analytical_predator_prey_rk4_jacobians():
    from bayesfilter.highdim.models import p30_predator_prey_fixture_model
    from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (
        tf_predator_prey_to_fixed_sgqf_model,
    )
    model = p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    points = tf.constant([[50., 5.], [63., 4.], [41., 6.]], tf.float64)
    adapter = tf_predator_prey_to_fixed_sgqf_model(model, theta, with_derivatives=True)
    value, state_jac = model.transition_mean_state_jacobian(theta, points)
    np.testing.assert_allclose(value, model.transition_mean(theta, points), atol=1e-12)
    for axis in range(2):
        perturbation = tf.one_hot(axis, 2, dtype=tf.float64) * 1e-5
        fd = (model.transition_mean(theta, points + perturbation) - model.transition_mean(theta, points - perturbation)) / 2e-5
        np.testing.assert_allclose(state_jac[:, :, axis], fd, rtol=1e-7, atol=1e-8)
    parameter_jac = adapter.derivatives.d_transition_fn(points)
    for axis in range(6):
        perturbation = tf.one_hot(axis, 6, dtype=tf.float64) * 1e-5
        fd = (model.transition_mean(theta + perturbation, points) - model.transition_mean(theta - perturbation, points)) / 2e-5
        np.testing.assert_allclose(parameter_jac[axis], fd, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.uint8, np.bool_])
def test_identity_tensor_bytes_preserve_historical_numeric_encoding(dtype):
    import hashlib

    from bayesfilter.highdim.ledh_contract_e_identity import _tensor_record
    for array in (np.array(1, dtype=dtype), np.array([[1, 0], [0, 1]], dtype=dtype)):
        record, tensor = _tensor_record("fixture", array)
        assert record["shape"] == list(array.shape)
        assert record["dtype_encoding"] == array.dtype.str
        assert record["value_sha256"] == hashlib.sha256(array.tobytes(order="C")).hexdigest()
        np.testing.assert_equal(tensor, array)
