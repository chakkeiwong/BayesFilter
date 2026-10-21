"""Versioned TensorFlow/XLA probe clouds for host-owned geometry preparation.

This stream intentionally differs from the historical NumPy PCG64 stream.
Reproducibility binds the stream ID, seed, call order/shapes and TF version.
Every draw runs on CPU/XLA so GPU placement cannot silently change the design.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import tensorflow as tf

STREAM_ID = "geometry_tf_philox_cpu_xla_v1"


@lru_cache(maxsize=64)
def _draw_kernel(kind, shape):
    @tf.function(
        input_signature=[tf.TensorSpec([2], tf.int32)],
        jit_compile=True,
        autograph=False,
    )
    def draw(seed):
        if kind == "normal":
            return tf.random.stateless_normal(
                shape, seed, dtype=tf.float64, alg="philox"
            )
        count = shape[0]
        if count < 2:
            return tf.range(count)

        def swap(i, values):
            bound = tf.cast(i + 1, tf.uint64)
            space = tf.constant(1 << 32, tf.uint64)
            limit = space - space % bound
            step_seed = tf.random.experimental.stateless_fold_in(seed, i, alg="philox")

            def word(attempt):
                return tf.cast(
                    tf.random.stateless_uniform(
                        [],
                        tf.random.experimental.stateless_fold_in(
                            step_seed, attempt, alg="philox"
                        ),
                        minval=None,
                        maxval=None,
                        dtype=tf.uint32,
                        alg="philox",
                    ),
                    tf.uint64,
                )

            _, selected = tf.while_loop(
                lambda attempt, value: value >= limit,
                lambda attempt, value: (attempt + 1, word(attempt)),
                (tf.constant(1), word(tf.constant(0))),
            )
            j = tf.cast(selected % bound, tf.int32)
            values = tf.tensor_scatter_nd_update(
                values, [[i], [j]], [values[j], values[i]]
            )
            return i - 1, values

        return tf.while_loop(
            lambda i, values: i > 0,
            swap,
            (tf.constant(count - 1), tf.range(count)),
            maximum_iterations=count - 1,
        )[1]

    return draw


@lru_cache(maxsize=64)
def _ball_kernel(rows, dimension):
    @tf.function(
        input_signature=[
            tf.TensorSpec([2], tf.int32),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.float64),
        ],
        jit_compile=True,
        autograph=False,
    )
    def draw(seed, radius, minimum_uniform):
        def directions(index):
            return tf.random.stateless_normal(
                [rows, dimension],
                tf.random.experimental.stateless_fold_in(seed, index, alg="philox"),
                dtype=tf.float64,
                alg="philox",
            )

        def redraw(index, values):
            zero = tf.reduce_sum(tf.square(values), axis=1, keepdims=True) == 0.0
            return index + 1, tf.where(zero, directions(index), values)

        _, values = tf.while_loop(
            lambda index, values: tf.reduce_any(
                tf.reduce_sum(tf.square(values), axis=1) == 0.0
            ),
            redraw,
            (tf.constant(2), directions(tf.constant(0))),
        )
        uniform = tf.random.stateless_uniform(
            [rows, 1],
            tf.random.experimental.stateless_fold_in(seed, 1, alg="philox"),
            minval=minimum_uniform,
            maxval=tf.constant(1.0, tf.float64),
            dtype=tf.float64,
            alg="philox",
        )
        radial = radius * uniform ** (1.0 / dimension)
        return values / tf.linalg.norm(values, axis=1, keepdims=True) * radial

    return draw


class GeometryTensorStream:
    """Host call counter over explicitly seeded, compiled TensorFlow draws."""

    def __init__(self, seed):
        self.seed = tuple(int(value) for value in seed)
        if not self.seed or any(value < 0 for value in self.seed):
            raise ValueError("seed must be a nonempty sequence of nonnegative integers")
        self.call_index = 0

    def _next_seed(self, kind):
        identity = json.dumps(
            [STREAM_ID, self.seed, self.call_index, kind], separators=(",", ":")
        )
        digest = hashlib.sha256(identity.encode("ascii")).digest()
        self.call_index += 1
        return tf.constant(
            [
                int.from_bytes(digest[:4], "little") & 0x7FFFFFFF,
                int.from_bytes(digest[4:8], "little") & 0x7FFFFFFF,
            ],
            tf.int32,
        )

    def normal(self, *, size):
        shape = tuple(int(value) for value in size)
        if any(value < 0 for value in shape):
            raise ValueError("draw shape must be nonnegative")
        with tf.device("/CPU:0"):
            return _draw_kernel("normal", shape)(self._next_seed("normal"))

    def permutation(self, count):
        count = int(count)
        if count < 0:
            raise ValueError("permutation count must be nonnegative")
        with tf.device("/CPU:0"):
            return _draw_kernel("permutation", (count,))(self._next_seed("permutation"))

    def ball(self, rows, dimension, *, radius, minimum_uniform=0.0):
        rows, dimension = int(rows), int(dimension)
        if rows < 0 or dimension < 1:
            raise ValueError(
                "ball shape must have nonnegative rows and positive dimension"
            )
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not 0.0 <= minimum_uniform < 1.0
        ):
            raise ValueError("invalid ball radius or minimum_uniform")
        with tf.device("/CPU:0"):
            return _ball_kernel(rows, dimension)(
                self._next_seed("ball"),
                tf.constant(radius, tf.float64),
                tf.constant(minimum_uniform, tf.float64),
            )
