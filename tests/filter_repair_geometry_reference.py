"""Independent diagnostic legacy probe stream for matched numerical fixtures."""

import copy
import json

import numpy as np
import tensorflow as tf


class FrozenLegacyGeometryStream:
    """Replay the original seeded clouds without altering any fit arithmetic."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed[0]) ^ (int(seed[1]) << 16))

    def normal(self, *, size):
        return self.rng.normal(size=size)

    def permutation(self, count):
        return self.rng.permutation(count)

    def ball(self, rows, dimension, *, radius, minimum_uniform=0.0):
        directions = self.rng.normal(size=(rows, dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        while np.any(norms == 0.0):
            zero = np.flatnonzero(norms[:, 0] == 0.0)
            directions[zero] = self.rng.normal(size=(zero.size, dimension))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
        radial = radius * self.rng.uniform(minimum_uniform, 1.0, size=(rows, 1)) ** (
            1.0 / dimension
        )
        return directions / norms * radial


def install_legacy_geometry_inputs(monkeypatch):
    """Freeze original clouds and every possible finite-count permutation.

    Test-only preparation replaces random inputs, never fit/selection arithmetic.
    The numerical program still decides the finite count and partition in XLA.
    """
    from bayesfilter.inference import quadratic_geometry_full_tf as native
    from bayesfilter.inference.quadratic_geometry_prepare_tf import (
        make_geometry_partition_program,
    )

    saved = {}
    last_program = None

    def prepare(dimension, config):
        rank, _, samples, directions = native.geometry_extents(dimension, config)
        stream = FrozenLegacyGeometryStream(config.seed)
        raw = stream.normal(size=(directions, dimension)) if rank else np.zeros((0, dimension))
        offsets = stream.ball(samples, dimension, radius=config.trust_radius)
        saved['state'] = copy.deepcopy(stream.rng.bit_generator.state)
        return tf.constant(raw, tf.float64), tf.constant(offsets, tf.float64), tf.constant([0, 0])

    def partition_factory(dimension, capacity, required, fraction, *, jit_compile=True):
        table = np.zeros((capacity + 1, capacity), dtype=np.int32)
        for count in range(capacity + 1):
            stream = np.random.default_rng(0)
            stream.bit_generator.state = copy.deepcopy(saved['state'])
            table[count, :count] = stream.permutation(count)
        table = tf.constant(table)
        partition = make_geometry_partition_program(dimension, capacity, required, fraction,
            jit_compile=jit_compile)

        @tf.function(input_signature=[tf.TensorSpec([capacity, dimension], tf.float64),
            tf.TensorSpec([capacity], tf.float64), tf.TensorSpec([capacity, dimension], tf.float64),
            tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([2], tf.int32)],
            autograph=False, jit_compile=jit_compile)
        def prepared(offsets, values, scores, scale, seed):
            del seed
            count = tf.math.count_nonzero(tf.math.is_finite(values) &
                tf.reduce_all(tf.math.is_finite(scores), axis=1), dtype=tf.int32)
            return partition(offsets, values, scores, scale, tf.gather(table, count))

        return prepared

    def program_factory(callback, dimension, config, *, batched_callback=None, jit_compile=True):
        nonlocal last_program
        # Each frozen permutation table is determined by the exact PCG state
        # after cloud preparation. Reuse only with that state and the complete
        # runtime signature unchanged, just as ordinary seeded fits reuse their
        # program. Changing centers/scales remains a change of tensor operands.
        key = (dimension, config, jit_compile, json.dumps(saved['state'], sort_keys=True))
        previous = last_program
        if (previous is not None and previous[0] is callback
                and previous[1] is batched_callback and previous[2] == key):
            return previous[3]
        last_program = None
        del previous
        program = native.make_geometry_program(callback, dimension, config,
            batched_callback=batched_callback, jit_compile=jit_compile)
        last_program = (callback, batched_callback, key, program)
        return program

    monkeypatch.setattr(native, 'prepare_geometry_inputs', prepare)
    monkeypatch.setattr(native, 'make_geometry_seeded_partition_program', partition_factory)
    monkeypatch.setattr(native, 'geometry_program', program_factory)
