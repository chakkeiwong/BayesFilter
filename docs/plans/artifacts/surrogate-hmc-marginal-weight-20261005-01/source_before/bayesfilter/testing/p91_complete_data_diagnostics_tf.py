"""Non-pfor derivative reference for P91 complete-data diagnostic runners.

This is autodiff of the shared complete-data density, not an analytical
filtering score or a training target. Callers own the enclosing fixed-signature
XLA function. Keep the density batch-native. Each score direction owns its
forward tape inside the loop body so nested RK4 derivative graphs stay local.
"""

import tensorflow as tf

from bayesfilter.highdim.models import (
    zhao_cui_sir_austria_batched_local_complete_data_log_density_xla,
)


def complete_data_values_and_scores_tf(theta, states, observations):
    values = zhao_cui_sir_austria_batched_local_complete_data_log_density_xla(
        theta, states, observations)
    count = tf.shape(values)[0]
    scores = tf.zeros([count, tf.shape(theta)[0]], theta.dtype)

    def body(index, output):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            direction_values = zhao_cui_sir_austria_batched_local_complete_data_log_density_xla(
                theta, states, observations)
        score = tape.gradient(direction_values, theta,
            output_gradients=tf.one_hot(index, count, dtype=direction_values.dtype))
        if score is None:
            score = tf.fill(tf.shape(theta), tf.constant(float('nan'), theta.dtype))
        return index + 1, tf.tensor_scatter_nd_update(output, [[index]], [score])

    _, scores = tf.while_loop(lambda index, _: index < count, body,
        (tf.constant(0), scores), parallel_iterations=1, maximum_iterations=count)
    return values, scores
