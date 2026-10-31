"""Optional, branch-conditioned safety checks for finite moment fitting.

All accepted values and tangents come from the same analytical finite program.
The acceptance decisions are locally constant; no derivative at a switching
boundary is claimed. This module never changes the empirical moment teacher.
"""
from __future__ import annotations

import tensorflow as tf


def roundoff_tolerance(value):
    epsilon = 2.0 ** (-23 if value.dtype == tf.float32 else -52)
    return tf.cast(32.0 * epsilon, value.dtype) * (1.0 + tf.abs(value))


def moment_loss(points, skew, kurtosis, co_skew, co_kurtosis, mask3, mask4):
    """Dimensionless squared error of the configured whitened moments."""
    square = tf.square(points)
    n = tf.cast(tf.shape(points)[0], points.dtype)
    observed = (tf.reduce_mean(square * points, axis=0),
                tf.reduce_mean(tf.square(square), axis=0),
                tf.linalg.matmul(square, points, transpose_a=True) / n,
                tf.linalg.matmul(square, square, transpose_a=True) / n)
    targets = (skew, kurtosis, co_skew, co_kurtosis)
    masks = (tf.ones_like(skew), tf.ones_like(kurtosis), mask3, mask4)
    terms = []
    for actual, target, mask in zip(observed, targets, masks):
        scale = tf.maximum(tf.ones_like(target), tf.abs(target))
        residual = mask * (actual / scale - target / scale)
        terms.append(tf.reduce_sum(tf.square(residual)))
    return tf.add_n(terms)


def empty_safety_stats(dtype):
    # trials, rejected steps, minimum accepted alpha, input validity, max loss rise
    return tf.constant([0., 0., 1., 1., 0.], dtype)


def combine_safety_stats(previous, current):
    return tf.stack((previous[0] + current[0], previous[1] + current[1],
                     tf.minimum(previous[2], current[2]),
                     tf.minimum(previous[3], current[3]),
                     tf.maximum(previous[4], current[4])))


def guarded_moment_step(points, tangent, displacement, displacement_tangent,
                        *, normalize, loss, trials=8):
    """Backtrack before whitening, accepting finite non-increasing moment loss.

    The covariance Loewner bounds 1/4 C <= C_trial <= 9/4 C limit the
    relative whitening gain to two. A failed trial is never Cholesky factored.
    Rejection returns the input value AND its tangent, with an explicit flag.
    """
    dtype = points.dtype
    n = tf.cast(tf.shape(points)[0], dtype)
    centered = points - tf.reduce_mean(points, axis=0)
    covariance = tf.linalg.matmul(centered, centered, transpose_a=True) / n
    old_loss = loss(points)
    finite_input = (tf.reduce_all(tf.math.is_finite(points))
                    & tf.reduce_all(tf.math.is_finite(tangent))
                    & tf.reduce_all(tf.math.is_finite(covariance))
                    & tf.math.is_finite(old_loss))
    scale = tf.linalg.trace(covariance) / tf.cast(tf.shape(points)[1], dtype)
    epsilon = tf.cast(2.0 ** (-23 if dtype == tf.float32 else -52), dtype)
    margin = 32.0 * epsilon * scale
    input_valid = tf.cond(
        finite_input,
        lambda: (scale > 0.) & (tf.reduce_min(tf.linalg.eigvalsh(covariance)) > margin),
        lambda: tf.constant(False))

    def body(index, accepted, chosen, chosen_tangent, chosen_alpha, chosen_loss):
        alpha = tf.pow(tf.cast(.5, dtype), tf.cast(index, dtype))
        candidate = points + alpha * displacement
        candidate_tangent = tangent + alpha * displacement_tangent
        c = candidate - tf.reduce_mean(candidate, axis=0)
        candidate_cov = tf.linalg.matmul(c, c, transpose_a=True) / n
        finite = (tf.reduce_all(tf.math.is_finite(candidate))
                  & tf.reduce_all(tf.math.is_finite(candidate_tangent))
                  & tf.reduce_all(tf.math.is_finite(candidate_cov)))

        def covariance_ok():
            lower = tf.reduce_min(tf.linalg.eigvalsh(candidate_cov - .25 * covariance))
            upper = tf.reduce_min(tf.linalg.eigvalsh(2.25 * covariance - candidate_cov))
            # The Loewner tolerance alone can admit a singular trial when
            # the input's smallest eigenvalue is near the roundoff margin.
            trial_scale = tf.linalg.trace(candidate_cov) / tf.cast(tf.shape(points)[1], dtype)
            positive = (tf.reduce_min(tf.linalg.eigvalsh(candidate_cov))
                        > 32.0 * epsilon * trial_scale)
            return (lower >= -margin) & (upper >= -margin) & positive

        admissible = tf.cond(finite, covariance_ok, lambda: tf.constant(False))

        def evaluate():
            out, dout = normalize(candidate, candidate_tangent)
            new_loss = loss(out)
            ok = (tf.reduce_all(tf.math.is_finite(out))
                  & tf.reduce_all(tf.math.is_finite(dout))
                  & tf.math.is_finite(new_loss)
                  & (new_loss <= old_loss + roundoff_tolerance(old_loss)))
            return ok, out, dout, new_loss

        ok, out, dout, new_loss = tf.cond(
            admissible, evaluate,
            lambda: (tf.constant(False), points, tangent, old_loss))
        return (index + 1, ok, tf.where(ok, out, chosen),
                tf.where(ok, dout, chosen_tangent),
                tf.where(ok, alpha, chosen_alpha),
                tf.where(ok, new_loss, chosen_loss))

    count, accepted, output, output_tangent, alpha, final_loss = tf.while_loop(
        lambda index, accepted, *_: input_valid & ~accepted & (index < trials),
        body, (tf.constant(0), tf.constant(False), points, tangent,
               tf.zeros([], dtype), old_loss), parallel_iterations=1)
    stats = tf.stack((tf.cast(count, dtype), tf.cast(~accepted, dtype), alpha,
                      tf.cast(input_valid, dtype),
                      tf.maximum(final_loss - old_loss, 0.)))
    return output, output_tangent, stats


SAFETY_KEYS = (
    "moment_safety_enabled", "moment_safety_trials", "moment_safety_rejected_steps",
    "moment_safety_minimum_step", "moment_safety_input_valid",
    "moment_safety_maximum_loss_increase", "moment_safety_baseline_loss",
    "moment_safety_final_loss", "moment_safety_final_rejected",
)


def safety_diagnostics(stats, enabled):
    zero = tf.zeros([], stats.dtype)
    return dict(zip(SAFETY_KEYS, (tf.cast(enabled, stats.dtype), *tf.unstack(stats),
                                 zero, zero, zero)))
