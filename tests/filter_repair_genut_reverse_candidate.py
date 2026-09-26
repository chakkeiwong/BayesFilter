"""Diagnostic-only FP32 products in existing TensorFlow first pullbacks.

Equations follow TensorFlow linalg_grad.py, Cholesky and MatrixTriangularSolve.
No runtime imports this candidate and no canonical LEDH score is defined here.
"""

import tensorflow as tf


def _product(left, right):
    return tf.reduce_sum(left[..., :, :, None] * right[..., None, :, :], axis=-2)


@tf.custom_gradient
def native_cholesky(matrix):
    lower = tf.linalg.cholesky(matrix)

    def pullback(gradient):
        inverse = tf.linalg.triangular_solve(lower,
            tf.eye(tf.shape(lower)[0], dtype=lower.dtype))
        middle = _product(tf.transpose(lower), gradient)
        middle = tf.linalg.set_diag(middle, .5 * tf.linalg.diag_part(middle))
        middle = tf.linalg.band_part(middle, -1, 0)
        result = _product(_product(tf.transpose(inverse), middle), inverse)
        return .5 * (result + tf.transpose(result))

    return lower, pullback


@tf.custom_gradient
def native_triangular_solve(lower, right):
    solution = tf.linalg.triangular_solve(lower, right)

    def pullback(gradient):
        grad_right = tf.linalg.triangular_solve(lower, gradient, adjoint=True)
        grad_lower = -_product(grad_right, tf.transpose(solution))
        return tf.linalg.band_part(grad_lower, -1, 0), grad_right

    return solution, pullback


@tf.custom_gradient
def native_matrix_solve(matrix, right):
    solution = tf.linalg.solve(matrix, right)

    def pullback(gradient):
        grad_right = tf.linalg.solve(matrix, gradient, adjoint=True)
        return -_product(grad_right, tf.linalg.matrix_transpose(solution)), grad_right

    return solution, pullback


@tf.custom_gradient
def native_gram(matrix):
    result = tf.linalg.matmul(matrix, matrix, transpose_a=True)

    def pullback(gradient):
        return _product(matrix, gradient + tf.linalg.matrix_transpose(gradient))

    return result, pullback


@tf.custom_gradient
def native_transposed_matvec(matrix, vector):
    result = tf.linalg.matvec(matrix, vector, transpose_a=True)

    def pullback(gradient):
        return (vector[..., :, None] * gradient[..., None, :],
                tf.reduce_sum(matrix * gradient[..., None, :], axis=-1))

    return result, pullback


def precise_gram(matrix):
    return _product(tf.linalg.matrix_transpose(matrix), matrix)


def precise_transposed_matvec(matrix, vector):
    return tf.reduce_sum(matrix * vector[..., :, None], axis=-2)
