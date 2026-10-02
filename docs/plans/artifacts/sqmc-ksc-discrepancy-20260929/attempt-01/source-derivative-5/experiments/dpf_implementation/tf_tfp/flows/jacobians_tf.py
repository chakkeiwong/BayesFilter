"""TensorFlow observation Jacobians for experimental LEDH diagnostics."""

from __future__ import annotations

import tensorflow as tf


DTYPE = tf.float64


def linear_observation_jacobian_tf(matrix: tf.Tensor):
    """Return a constant linear-observation Jacobian function."""

    matrix = tf.cast(matrix, DTYPE)

    def jacobian(_state: tf.Tensor) -> tf.Tensor:
        return matrix

    return jacobian


def range_bearing_jacobian_tf(state: tf.Tensor, *, eps: float = 1e-9) -> tf.Tensor:
    """Analytic Jacobian of [range, bearing] with respect to [px, py, vx, vy]."""

    state = tf.cast(state, DTYPE)
    px = state[0]
    py = state[1]
    radius_sq = tf.maximum(px * px + py * py, tf.constant(eps, dtype=DTYPE))
    radius = tf.sqrt(radius_sq)
    return tf.stack(
        [
            tf.stack([px / radius, py / radius, 0.0, 0.0]),
            tf.stack([-py / radius_sq, px / radius_sq, 0.0, 0.0]),
        ],
        axis=0,
    )
