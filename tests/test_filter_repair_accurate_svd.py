"""Independent reconstruction, rank and pullback checks for scaled XLA SVD."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.accurate_svd_tf import accurate_svd
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


@pytest.mark.parametrize('rows,columns', [(1, 1), (3, 3), (3, 4), (4, 3)])
def test_svd_scale_reconstruction_and_rank(rows, columns, request):
    @tf.function(input_signature=[tf.TensorSpec([2, rows, columns], D)], jit_compile=True, autograph=False)
    def decompose(matrix):
        return accurate_svd(matrix)

    rng = np.random.default_rng(293)
    base = rng.normal(size=(2, rows, columns))
    records, hlos = [], []
    for scale in (0., 1e-140, 1e-12, 1., 1e140):
        matrix = base * scale
        singular, left, right = decompose(tf.constant(matrix, D))
        expected = np.linalg.svd(matrix, compute_uv=False)
        np.testing.assert_allclose(singular, expected, rtol=1e-12, atol=0.)
        reconstructed = (left.numpy() * (singular.numpy() / (scale or 1.))[:, None, :]) @ np.swapaxes(right.numpy(), -1, -2)
        reference = base if scale else np.zeros_like(base)
        np.testing.assert_allclose(reconstructed, reference, rtol=1e-12, atol=1e-12)
        extent = min(rows, columns)
        np.testing.assert_allclose(np.swapaxes(left, -1, -2) @ left, np.broadcast_to(np.eye(extent), (2, extent, extent)), atol=1e-12)
        np.testing.assert_allclose(np.swapaxes(right, -1, -2) @ right, np.broadcast_to(np.eye(extent), (2, extent, extent)), atol=1e-12)
        records.append({'scale': scale, 'singular': clean(singular), 'reference': clean(expected),
            'normalized_reconstruction_error': float(np.max(np.abs(reconstructed-reference)))})
        hlos.append(stable_hlo(decompose.experimental_get_compiler_ir(tf.constant(matrix, D))(stage='hlo')))
    if rows == columns == 3:
        tied = np.broadcast_to(np.array([[1., 1e-7, 0.], [1e-7, 1., 0.], [0., 0., .5]]), (2, 3, 3))
        tied_singular = decompose(tf.constant(tied, D))[0]
        np.testing.assert_allclose(tied_singular, np.broadcast_to([1. + 1e-7, 1. - 1e-7, .5], (2, 3)), rtol=1e-14, atol=0.)
        repeated = np.broadcast_to(np.eye(3), (2, 3, 3))
        np.testing.assert_allclose(decompose(tf.constant(repeated, D))[0], np.ones((2, 3)), atol=0., rtol=1e-14)
    deficient = np.ones_like(base) * 1e-12
    singular, _, _ = decompose(tf.constant(deficient, D))
    rank = np.sum(singular.numpy() > singular.numpy()[:, :1] * 1e-12, axis=-1)
    np.testing.assert_array_equal(rank, [1, 1])
    save(request, f'accurate-svd-{rows}-{columns}.json', {'records': records,
        'deficient_rank': clean(rank), 'trace_count': decompose.experimental_get_tracing_count(),
        'hlo_unchanged': len(set(hlos)) == 1})
    assert len(set(hlos)) == 1 and decompose.experimental_get_tracing_count() == 1


@pytest.mark.parametrize('rows,columns', [(3, 3), (3, 4), (4, 3)])
def test_svd_pullback_preserves_tensorflow_convention(rows, columns, request):
    rng = np.random.default_rng(852)
    matrix = tf.constant(rng.normal(size=(2, rows, columns)), D)
    direction = tf.constant(rng.normal(size=matrix.shape), D)
    left_weight = tf.constant(rng.normal(size=(rows, rows)), D)
    right_weight = tf.constant(rng.normal(size=(columns, columns)), D)

    def objective(matrix, corrected):
        singular, left, right = (accurate_svd(matrix) if corrected else
            tf.linalg.svd(matrix, full_matrices=False))
        weights = tf.cast(tf.range(min(rows, columns)), D) + 1.
        left_projection = tf.matmul(left * weights[None, None, :], left, transpose_b=True)
        right_projection = tf.matmul(right * weights[None, None, :], right, transpose_b=True)
        return (tf.reduce_sum(tf.math.sin(singular)) +
            tf.reduce_sum(left_projection * left_weight) + tf.reduce_sum(right_projection * right_weight))

    @tf.function(input_signature=[tf.TensorSpec(matrix.shape, D)], jit_compile=True, autograph=False)
    def evaluated(matrix):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            value = objective(matrix, True)
        return value, tape.gradient(value, matrix)

    actual_value, actual_gradient = evaluated(matrix)
    with tf.GradientTape() as tape:
        tape.watch(matrix)
        original_value = objective(matrix, False)
    original_gradient = tape.gradient(original_value, matrix)
    np.testing.assert_allclose(actual_value, original_value, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_gradient, original_gradient, atol=1e-10, rtol=1e-10)
    eps = 1e-6
    finite_difference = (evaluated(matrix + eps * direction)[0] - evaluated(matrix - eps * direction)[0]) / (2. * eps)
    directional_gradient = tf.reduce_sum(actual_gradient * direction)
    np.testing.assert_allclose(directional_gradient, finite_difference, atol=1e-7, rtol=1e-7)
    save(request, f'accurate-svd-gradient-{rows}-{columns}.json', clean({
        'value': actual_value, 'reference_value': original_value,
        'gradient': actual_gradient, 'reference_gradient': original_gradient,
        'finite_difference': finite_difference, 'directional_gradient': directional_gradient}))
