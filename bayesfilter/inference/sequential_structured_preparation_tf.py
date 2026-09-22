"""Native fresh/reused-row preparation for sequential factor fitting.

Fresh clouds, exact target calls, stable reuse packing and weights execute in
one XLA program. The enclosing refinement controller is a separate boundary.
This is preparation, not NeuTra parameter training.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference._exact_incumbent import _incumbent_selection
from bayesfilter.inference.program_cache_scope import (
    independent_trace_scope,
    scoped_program_cache,
)
from bayesfilter.inference.sequential_preparation_tf import (
    cloud_program,
    evaluation_program,
)

D = tf.float64


@tf.custom_gradient
def _fresh_positions(fresh, center, scale):
    """Preserve separately rounded multiplication/addition and their pullback."""
    def step(index, previous):
        value = tf.cond(index == 0, lambda: fresh * scale[None, :],
            lambda: center[None, :] + previous)
        return index + 1, value

    _, positions = tf.while_loop(lambda index, _: index < 2, step,
        (tf.constant(0), tf.zeros_like(fresh)), maximum_iterations=2, parallel_iterations=1)

    def pullback(upstream):
        return (upstream * scale[None, :], tf.reduce_sum(upstream, axis=0),
            tf.reduce_sum(upstream * fresh, axis=0))

    return positions, pullback


@lru_cache(maxsize=64)
def _position_program(row_count, dimension, jit_compile):
    # TensorFlow's custom-gradient registry retains its traced closure. Trace
    # this resource-free affine map outside any consuming FuncGraph, so it
    # cannot retain the consumer's covariance guard or optimizer graph.
    with independent_trace_scope():
        @tf.function(input_signature=[tf.TensorSpec([row_count, dimension], D),
            tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)],
            jit_compile=jit_compile, autograph=False)
        def position(fresh, center, scale):
            return _fresh_positions(fresh, center, scale)

        position.get_concrete_function()
    return position


@scoped_program_cache(maxsize=32)
def structured_data_program(scalar, batched, dimension, fresh_count, search_count,
                            reuse_search_scores, *, jit_compile=True):
    if fresh_count < 4 * dimension or fresh_count % 2:
        raise ValueError('structured fresh sample count must be even and at least 4N')
    train_count = fresh_count // 2
    position = _position_program(fresh_count, dimension, jit_compile)
    generate = cloud_program(train_count, dimension, True).python_function
    evaluate = evaluation_program(scalar, batched, fresh_count, dimension).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], D), tf.TensorSpec([2], tf.int32),
        tf.TensorSpec([search_count, dimension], D), tf.TensorSpec([search_count, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def prepare(center, center_score, scale, radius, seed, search_positions, search_scores):
        train = generate(radius, seed)
        holdout = generate(radius, seed + tf.constant([0, 104729], tf.int32))
        fresh = tf.concat([train, holdout], axis=0)
        positions = position(fresh, center, scale)
        values, scores = evaluate(positions)
        if reuse_search_scores:
            translated = (search_positions - center[None, :]) / scale[None, :]
            finite = tf.reduce_all(tf.math.is_finite(translated) & tf.math.is_finite(search_scores), axis=1)
            norms = tf.linalg.norm(translated, axis=1)
            eligible = finite & (norms <= radius * (1. + 1.e-12)) & (norms > 1.e-12)
            reused_count = tf.math.count_nonzero(eligible, dtype=tf.int32)
            # Unique integer keys preserve original eligible-row order.
            keys = tf.range(search_count) + tf.cast(~eligible, tf.int32) * search_count
            order = tf.argsort(keys, stable=True)
            selected = tf.range(search_count) < reused_count
            reused = tf.where(selected[:, None], tf.gather(translated, order), tf.zeros_like(translated))
            reused_scores = tf.where(selected[:, None], tf.gather(search_scores, order), tf.zeros_like(search_scores))
            reused_weight = tf.where(selected, .5 / tf.cast(tf.maximum(reused_count, 1), D), 0.)
            training = tf.concat([train, reused], axis=0)
            training_scores = tf.concat([scores[:train_count], reused_scores], axis=0)
        else:
            reused_count = tf.constant(0)
            reused_weight = tf.zeros([0], D)
            training = train
            training_scores = scores[:train_count]
        fresh_weight = tf.where(reused_count > 0,
            tf.constant(.5 / train_count, D), tf.constant(1. / train_count, D))
        weights = tf.concat([tf.fill([train_count], fresh_weight), reused_weight], axis=0)
        _, winner = _incumbent_selection.python_function(tf.reshape(positions, [-1]),
            tf.reshape(scores, [-1]), values, tf.ones([fresh_count], tf.bool),
            tf.range(1, fresh_count + 1) * dimension)
        safe = tf.maximum(winner, 0)
        return {'center_score_z': scale * center_score, 'training_offsets_z': training,
            'training_scores_z': training_scores * scale[None, :], 'holdout_offsets_z': holdout,
            'holdout_scores_z': scores[train_count:] * scale[None, :], 'training_weights': weights,
            'active_training_rows': train_count + reused_count, 'reused_training_count': reused_count,
            'fresh_training_count': tf.constant(train_count), 'fresh_holdout_count': tf.constant(train_count),
            'unique_fresh_evaluations': tf.constant(fresh_count), 'best_index': winner,
            'best_value': tf.stop_gradient(tf.gather(values, safe)),
            'best_position': tf.stop_gradient(tf.gather(positions, safe)),
            'best_score': tf.stop_gradient(tf.gather(scores, safe))}

    return prepare
