"""TensorFlow trace reductions for ordinary HMC acceptance and hard vetoes.

For log MH ratio a, acceptance probability is exp(min(a,0)) (monograph
eq:hmc_accept). The finite-subset mean is explanatory only when any entry is
nonfinite; health independently fails. Empty traces never establish health.
No sampler, tuning choice, state clipping or target modification occurs here.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec([None], tf.float64)], autograph=False, jit_compile=False)
def finite_trace_statistics(values: tf.Tensor) -> dict[str, tf.Tensor]:
    """Reduce a flat trace with explicit finite counts and nonempty evidence.

    Masking is only for reporting extrema/acceptance. A masked NaN/Inf remains
    visible in nonfinite_count and all_finite=False, never a healthy proposal.
    Stable input rank/signature bounds tracing across scalar and batched traces.
    """

    finite = tf.math.is_finite(values)
    total_count = tf.size(values, out_type=tf.int64)
    finite_count = tf.reduce_sum(tf.cast(finite, tf.int64))
    has_finite = finite_count > 0
    positive_infinity = tf.constant(float("inf"), tf.float64)
    minimum = tf.reduce_min(tf.where(finite, values, positive_infinity))
    maximum = tf.reduce_max(tf.where(finite, values, -positive_infinity))
    max_abs = tf.reduce_max(tf.where(finite, tf.abs(values), tf.zeros_like(values)))
    acceptance = tf.where(finite, tf.exp(tf.minimum(tf.where(finite, values, 0.0), 0.0)), 0.0)
    return {
        "finite_count": finite_count, "nonfinite_count": total_count - finite_count,
        "total_count": total_count, "all_finite": tf.logical_and(total_count > 0, finite_count == total_count),
        "min_finite": tf.where(has_finite, minimum, 0.0),
        "max_finite": tf.where(has_finite, maximum, 0.0),
        "max_abs_finite": tf.where(has_finite, max_abs, 0.0),
        "mean_acceptance_probability": tf.math.divide_no_nan(
            tf.reduce_sum(acceptance), tf.cast(finite_count, tf.float64),
        ),
    }


def trace_health_payload(values: Any) -> dict[str, Any]:
    """Materialize only reduced scalars at the host-side decision boundary."""

    tensor = tf.cast(tf.convert_to_tensor(values, dtype_hint=tf.float64), tf.float64)
    statistics = finite_trace_statistics(tf.reshape(tensor, [-1]))
    result = {name: value.numpy().item() for name, value in statistics.items()}
    if result["finite_count"] == 0:
        for name in ("min_finite", "max_finite", "max_abs_finite", "mean_acceptance_probability"):
            result[name] = None
    return result
