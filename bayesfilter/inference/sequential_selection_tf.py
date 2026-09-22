"""Native exact replay and search selection for the sequential locator.

Eligibility follows the inherited value/score rule. Position finiteness is not
an additional selection gate. Optimizer and refinement lifecycles are separate.
"""


import tensorflow as tf

from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_preparation_tf import (
    cloud_program,
    evaluation_program,
)

D = tf.float64


def selection_numerics(positions, values, scores, *, keep_first=False):
    if positions.shape[0] == 0:
        return {'index': tf.constant(-1), 'finite_count': tf.constant(0),
            'value': tf.constant(0., D), 'position': tf.zeros([positions.shape[1]], D),
            'score': tf.zeros([positions.shape[1]], D)}
    finite = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
    if keep_first:
        finite = tf.concat([tf.ones([1], tf.bool), finite[1:]], axis=0)
    count = tf.math.count_nonzero(finite, dtype=tf.int32)
    index = tf.argmax(tf.where(finite, values, tf.constant(-float('inf'), D)), output_type=tf.int32)
    if keep_first:
        # Python stable sorting retains a leading NaN incumbent. Normally the
        # enclosing locator supplies a finite incumbent; preserve its boundary.
        index = tf.where(tf.math.is_nan(values[0]), 0, index)
    return {'index': tf.where(count > 0, index, -1), 'finite_count': count,
        # Scalar StridedSlice indices force winner dependencies to compile-time
        # constants in TF/XLA. Gather keeps these data-dependent rows dynamic.
        'value': tf.gather(values, index), 'position': tf.gather(positions, index),
        'score': tf.gather(scores, index)}


@scoped_program_cache(maxsize=64)
def replay_program(scalar, rows, dimension, *, jit_compile=True):
    evaluate = evaluation_program(scalar, None, rows, dimension).python_function

    @tf.function(input_signature=[tf.TensorSpec([rows, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def replay(positions):
        if rows == 0:
            return selection_numerics(positions, tf.zeros([0], D), tf.zeros([0, dimension], D))
        values, scores = evaluate(positions)
        return selection_numerics(positions, values, scores)

    return replay


@scoped_program_cache(maxsize=64)
def search_program(scalar, batched, rows, dimension, orthogonal, *, jit_compile=True):
    generate = cloud_program(rows, dimension, orthogonal).python_function
    evaluate = evaluation_program(scalar, batched, rows, dimension).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([2], tf.int32)], jit_compile=jit_compile, autograph=False)
    def search(center, center_value, center_score, scale, radius, seed):
        cloud = generate(radius, seed)
        positions = center[None, :] + scale[None, :] * cloud
        values, scores = evaluate(positions)
        result = selection_numerics(tf.concat([center[None, :], positions], axis=0),
            tf.concat([center_value[None], values], axis=0),
            tf.concat([center_score[None, :], scores], axis=0), keep_first=True)
        return {**result, 'search_positions': positions, 'search_values': values,
            'search_scores': scores, 'recentered': result['value'] > center_value}

    return search
