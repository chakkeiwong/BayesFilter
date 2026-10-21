"""Named full-DCTR KSC density with FP32 physical evaluation and FP64 probit chart.

This module binds the existing canonical evaluator; it does not implement a
second filter or admit the analytical proposal field as the endpoint gradient.
"""
import hashlib
import math
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import FrozenTargetPotential, TargetPotentialEvaluation
from bayesfilter.testing.surrogate_hmc_lg1_finite_endpoint import _coordinate_status

ENDPOINT_VERSION = "ksc_t2_dctr_physical_fp32_probit_fp64_v1"


def make_finite_endpoint(consumer):
    batch = consumer.manifest["batch_size"]
    fixture = consumer.fixture
    lower, upper = tf.constant(.1, tf.float64), tf.constant(.9, tf.float64)
    lo32, hi32 = tf.cast(lower, tf.float32), tf.cast(upper, tf.float32)
    safe_lower = tf.math.nextafter(lo32, hi32)
    safe_upper = tf.math.nextafter(hi32, lo32)
    tf.debugging.assert_equal(fixture.lower, [lo32, lo32])
    tf.debugging.assert_equal(fixture.upper, [hi32, hi32])

    def evaluate(theta):
        rounded = tf.cast(theta, tf.float32)
        adjusted = tf.clip_by_value(rounded, safe_lower, safe_upper)
        value, _, status = fixture.physical(adjusted)
        return tf.cast(value, tf.float64), status, tf.reduce_any(rounded != adjusted, axis=-1)

    @tf.function(input_signature=[tf.TensorSpec([batch, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def physical(theta):
        value, original, rounding = evaluate(theta)
        finite = tf.reduce_all(tf.math.is_finite(theta), axis=-1)
        support = finite & tf.reduce_all((theta > lower) & (theta < upper), axis=-1)
        status = _coordinate_status(original, support & tf.math.is_finite(value), finite & ~support)
        status["coordinate_boundary_rounding"] = rounding
        return value, status

    @tf.function(input_signature=[tf.TensorSpec([batch, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def chart(q):
        theta = lower + (upper - lower) * .5 * tf.math.erfc(-q / tf.sqrt(tf.constant(2., tf.float64)))
        value, original, rounding = evaluate(theta)
        jacobian = tf.reduce_sum(tf.math.log(upper - lower) - .5 * q**2
                                 - .5 * tf.math.log(tf.constant(2 * math.pi, tf.float64)), axis=-1)
        density = value + jacobian
        valid = tf.reduce_all(tf.math.is_finite(q), axis=-1) & tf.math.is_finite(density)
        status = _coordinate_status(original, valid, tf.zeros_like(valid))
        status["coordinate_boundary_rounding"] = rounding
        return density, status

    def potential(q):
        value, status = chart(q)
        return TargetPotentialEvaluation(-value, status)

    # Both the coordinate adapter and its shared status helper are part of the
    # identity. The fixture already binds the full canonical source closure.
    status_source = Path(__file__).with_name("surrogate_hmc_lg1_finite_endpoint.py")
    digest = hashlib.sha256(Path(__file__).read_bytes() + status_source.read_bytes()).hexdigest()
    identity = ENDPOINT_VERSION + ":" + fixture.identity + ":" + digest
    return FrozenTargetPotential(potential, identity=identity), physical, chart
