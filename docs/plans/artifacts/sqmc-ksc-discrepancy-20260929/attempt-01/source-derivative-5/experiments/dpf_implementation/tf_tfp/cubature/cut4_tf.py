"""TensorFlow one-dimensional CUT4-style Gaussian cubature rule."""

from __future__ import annotations

import tensorflow as tf


DTYPE = tf.float64


def cut4_standard_normal_nodes_weights_tf() -> tuple[tf.Tensor, tf.Tensor]:
    """Return symmetric nodes/weights exact for N(0, 1) moments through degree 5.

    In one dimension this is the three-point symmetric rule with nodes
    `[-sqrt(3), 0, sqrt(3)]` and weights `[1/6, 2/3, 1/6]`.  It is used here as
    a CUT4-style deterministic differentiable comparator, not as ground truth.
    """

    sqrt_three = tf.sqrt(tf.constant(3.0, dtype=DTYPE))
    nodes = tf.stack([-sqrt_three, tf.constant(0.0, dtype=DTYPE), sqrt_three])
    weights = tf.constant([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=DTYPE)
    return nodes, weights
