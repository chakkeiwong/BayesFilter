"""TensorFlow Gauss–Legendre preparation using the Jacobi eigenproblem."""

from __future__ import annotations

from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig


def _legendre_rule(order, dtype, jit_compile):
    """Return ascending nodes and weights on [-1, 1] (Golub–Welsch)."""
    order = int(order)
    if order < 1:
        raise ValueError("Gauss-Legendre order must be positive")
    if order == 1:
        return tf.zeros([1], dtype), tf.fill([1], tf.constant(2.0, dtype))
    index = tf.cast(tf.range(1, order), dtype)
    off_diagonal = index / tf.sqrt(4.0 * tf.square(index) - 1.0)
    jacobi = tf.linalg.diag(off_diagonal, k=1) + tf.linalg.diag(off_diagonal, k=-1)
    if jit_compile:
        nodes, _ = xla_self_adjoint_eig(jacobi, lower=True, max_iter=100,
            epsilon=2.220446049250313e-16 if dtype == tf.float64 else 1.1920928955078125e-7)
        nodes.set_shape([order])
    else:
        nodes = tf.linalg.eigvalsh(jacobi)

    def step(index, previous, current):
        k = tf.cast(index, dtype)
        following = ((2.0*k-1.0)*nodes*current-(k-1.0)*previous)/k
        return index+1, current, following

    _, previous, current = tf.while_loop(lambda index, *_: index <= order, step,
        (tf.constant(2), tf.ones_like(nodes), nodes), maximum_iterations=order-1, parallel_iterations=1)
    derivative = tf.cast(order, dtype)*(previous-nodes*current)/(1.0-tf.square(nodes))
    return nodes, 2.0/((1.0-tf.square(nodes))*tf.square(derivative))


@lru_cache(maxsize=32)
def _legendre_program(order, dtype, jit_compile):
    return tf.function(lambda: _legendre_rule(order, dtype, jit_compile),
        input_signature=[], jit_compile=jit_compile, autograph=False)


@lru_cache(maxsize=32)
def _legendre_constants(order, dtype, jit_compile):
    # The rule has no numerical inputs. Lift its compiled preparation out of
    # consumer graphs so fixed-order eigensolves are not repeated on each call.
    with tf.init_scope():
        return _legendre_program(order, dtype, jit_compile)()


def gauss_legendre(order: int, dtype: tf.DType = tf.float64, *, jit_compile=True) -> tuple[tf.Tensor, tf.Tensor]:
    """Ascending nodes/weights, prepared once per static rule with XLA by default."""
    return _legendre_constants(int(order), tf.as_dtype(dtype), bool(jit_compile))


@lru_cache(maxsize=32)
def _product_program(dimension, order, jit_compile):
    @tf.function(input_signature=[], jit_compile=jit_compile, autograph=False)
    def generate():
        nodes, weights = _legendre_rule(order, tf.float64, jit_compile)
        powers = tf.pow(tf.cast(order, tf.int64), tf.range(dimension - 1, -1, -1, dtype=tf.int64))
        digits = (tf.range(order ** dimension, dtype=tf.int64)[:, None] // powers[None, :]) % order
        return tf.gather(nodes, digits), tf.reduce_prod(tf.gather(weights * 0.5, digits), axis=1)
    return generate


def gauss_legendre_product(dimension: int, order: int, *, jit_compile=True) -> tuple[tf.Tensor, tf.Tensor]:
    """Compiled lexicographic tensor product under uniform probability measure."""
    if int(dimension) < 1:
        raise ValueError("dimension must be positive")
    return _product_program(int(dimension), int(order), bool(jit_compile))()
