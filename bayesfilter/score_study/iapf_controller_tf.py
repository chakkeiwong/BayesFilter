"""Native bounded iAPF decision arithmetic; endpoint migration is separate.

Action codes are -1 invalid history, 0 fit, 1 final and 2 capacity veto.
Absent padded history is never included in the coefficient of variation.
"""

from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier


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
        def divide(numerator, denominator):
            # Constant reciprocal multiplication and nested-quotient rewriting
            # can change a strict CV stopping decision by one FP64 ULP.
            if jit_compile:
                numerator, denominator = xla_optimization_barrier(input=[numerator, denominator])
            return numerator / denominator

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
            # Complete, valid windows have exactly k+1 valid count entries.
            # Compute that cardinality from runtime data: a constant divisor
            # can be rewritten to a reciprocal before the XLA barrier exists.
            sample_count = tf.reduce_sum(tf.cast(counts >= 2, tf.float64))
            mean = divide(total, sample_count)

            def squared_error(i, total):
                return i + 1, total + tf.square(values[i] - mean)

            squares = tf.while_loop(lambda i, _: i < k + 1, squared_error,
                (tf.constant(0), tf.constant(0., tf.float64)),
                maximum_iterations=k + 1, parallel_iterations=1)[1]
            cv = divide(tf.sqrt(divide(squares, sample_count - 1.)), mean)
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


@lru_cache(maxsize=12)
def make_checked_iapf_iteration_decision(max_history, k, max_particles, jit_compile=True):
    """Retain raw decisions but reject numerically unresolved CV comparisons.

    ``action == -2`` means the stopping threshold lies in the propagated
    rounding interval. This diagnostic is not a statistical confidence bound.
    Its exp assumption is at most one ULP plus one outward rounding unit.
    Endpoint callers must treat -2 as an error, never as a fit/final action.
    """
    raw = make_iapf_iteration_decision(max_history, k, max_particles, jit_compile)

    @tf.function(input_signature=raw.input_signature, jit_compile=jit_compile, autograph=False)
    def checked(log_values, particle_counts, length, tau):
        result = raw(log_values, particle_counts, length, tau)
        zero = tf.constant(0., tf.float64)
        negative = tf.constant(float("-inf"), tf.float64)
        positive = tf.constant(float("inf"), tf.float64)

        def down(value):
            return tf.math.nextafter(value, negative)

        def up(value):
            return tf.math.nextafter(value, positive)

        def divide(a, b):
            if jit_compile:
                a, b = xla_optimization_barrier(input=[a, b])
            return a/b

        def interval():
            indices = tf.range(k + 1) + length - k - 1
            window = tf.gather(log_values, indices)
            counts = tf.gather(particle_counts, indices)
            maximum = tf.reduce_max(window)
            shifted = window - maximum
            lower = tf.maximum(zero, down(down(tf.exp(down(shifted)))))
            upper = up(up(tf.exp(up(shifted))))
            lower = tf.where(window == maximum, tf.ones_like(lower), lower)
            upper = tf.where(window == maximum, tf.ones_like(upper), upper)

            def add(i, lo, hi):
                return i + 1, down(lo + lower[i]), up(hi + upper[i])

            total = tf.while_loop(lambda i, *_: i < k + 1, add,
                (tf.constant(0), tf.constant(0., tf.float64), tf.constant(0., tf.float64)),
                maximum_iterations=k + 1, parallel_iterations=1)
            n = tf.reduce_sum(tf.cast(counts >= 2, tf.float64))
            mean_lo, mean_hi = down(divide(total[1], n)), up(divide(total[2], n))
            delta_lo, delta_hi = down(lower - mean_hi), up(upper - mean_lo)
            square_lo = tf.where((delta_lo <= 0.) & (delta_hi >= 0.), zero,
                tf.maximum(zero, down(tf.minimum(tf.square(delta_lo), tf.square(delta_hi)))))
            square_hi = up(tf.maximum(tf.square(delta_lo), tf.square(delta_hi)))

            def add_square(i, lo, hi):
                return i + 1, tf.maximum(zero, down(lo + square_lo[i])), up(hi + square_hi[i])

            squares = tf.while_loop(lambda i, *_: i < k + 1, add_square,
                (tf.constant(0), tf.constant(0., tf.float64), tf.constant(0., tf.float64)),
                maximum_iterations=k + 1, parallel_iterations=1)
            variance_lo = tf.maximum(zero, down(divide(squares[1], n - 1.)))
            variance_hi = up(divide(squares[2], n - 1.))
            root_lo = tf.maximum(zero, down(tf.sqrt(variance_lo)))
            root_hi = up(tf.sqrt(variance_hi))
            cv_lo = tf.maximum(zero, down(divide(root_lo, mean_hi)))
            cv_hi = up(divide(root_hi, mean_lo))
            equal = tf.reduce_all(window == maximum)
            return tf.where(equal, zero, cv_lo), tf.where(equal, zero, cv_hi)

        lower, upper = tf.cond(result["complete_window"], interval,
            lambda: (tf.constant(float("nan"), tf.float64), tf.constant(float("nan"), tf.float64)))
        interval_valid = (tf.math.is_finite(lower) & tf.math.is_finite(upper) &
                          (lower <= result["cv"]) & (result["cv"] <= upper))
        eligible = result["valid"] & (length > k + 1)
        unresolved = eligible & (~interval_valid | ((lower <= tau) & (tau <= upper)))
        return {**result, "raw_action": result["action"],
                "action": tf.where(unresolved, -2, result["action"]),
                "decision_resolved": result["valid"] & ~unresolved,
                "cv_lower": lower, "cv_upper": upper,
                "cv_interval_valid": interval_valid}

    return checked
