"""Native bounded iAPF decision arithmetic; endpoint migration is separate.

Action codes are -1 invalid history, 0 fit, 1 final and 2 capacity veto.
Absent padded history is never included in the coefficient of variation.
"""

from functools import lru_cache

import tensorflow as tf


@lru_cache(maxsize=12)
def make_iapf_iteration_decision(max_history, k, max_particles, jit_compile=True):
    """Preserve strict iteration/CV tests and sequential FP64 accumulation."""
    if type(k) is not int or k < 1:
        raise ValueError("positive integer k required")
    if type(max_history) is not int or max_history < k + 2:
        raise ValueError("history capacity must allow the first stopping check")
    if type(max_particles) is not int or not 2 <= max_particles <= (2**63 - 1)//2:
        raise ValueError("particle cap must permit exact int64 doubling")

    @tf.function(input_signature=[tf.TensorSpec([max_history], tf.float64),
        tf.TensorSpec([max_history], tf.int64), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.float64)], jit_compile=jit_compile, autograph=False)
    def decide(log_values, particle_counts, length, tau):
        active = tf.range(max_history) < length
        valid = (length >= 1) & (length <= max_history) & tf.math.is_finite(tau) & (tau > 0)
        valid &= tf.reduce_all(~active | tf.math.is_finite(log_values))
        valid &= tf.reduce_all(~active | ((particle_counts >= 2) & (particle_counts <= max_particles)))
        index = tf.clip_by_value(length - 1, 0, max_history - 1)
        particles = particle_counts[index]
        complete = valid & (length > k)

        def complete_window():
            indices = tf.range(k + 1) + length - k - 1
            window = tf.gather(log_values, indices)
            counts = tf.gather(particle_counts, indices)
            values = tf.exp(window - tf.reduce_max(window))

            def accumulate(i, total):
                return i + 1, total + values[i]

            total = tf.while_loop(lambda i, _: i < k + 1, accumulate,
                (tf.constant(0), tf.constant(0., tf.float64)),
                maximum_iterations=k + 1, parallel_iterations=1)[1]
            mean = total / tf.cast(k + 1, tf.float64)

            def squared_error(i, total):
                return i + 1, total + tf.square(values[i] - mean)

            squares = tf.while_loop(lambda i, _: i < k + 1, squared_error,
                (tf.constant(0), tf.constant(0., tf.float64)),
                maximum_iterations=k + 1, parallel_iterations=1)[1]
            cv = tf.sqrt(squares / tf.cast(k, tf.float64)) / mean
            stop = (length > k + 1) & (cv < tau)
            grow = tf.reduce_all(counts == particles) & ~tf.reduce_all(window[:-1] <= window[1:])
            next_particles = tf.where(~stop & grow, 2 * particles, particles)
            action = tf.where(stop, 1, tf.where(next_particles > max_particles, 2, 0))
            return action, next_particles, cv

        action, next_particles, cv = tf.cond(complete, complete_window,
            lambda: (tf.constant(0), particles, tf.constant(float("nan"), tf.float64)))
        return {"valid": valid, "action": tf.where(valid, action, -1),
                "next_particles": next_particles, "cv": cv, "complete_window": complete}

    return decide
