"""Named full-DCTR finite endpoint candidate with explicit coordinate rounding.

The physical density, including its constant prior, uses the complete shared
FP32 DCTR evaluator. The bounded coordinate map and Jacobian use FP64. This is
a different finite program from the old FP32 unconstrained evaluator.
"""
import hashlib
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import FrozenTargetPotential, TargetPotentialEvaluation
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode, FailureStage

ENDPOINT_VERSION = "lg1_dctr_physical_fp32_chart_fp64_v1"


def _coordinate_status(original, coordinate_valid, outside):
    status = dict(original)
    status["value_valid"] &= coordinate_valid
    status["force_valid"] &= coordinate_valid
    status["outside_declared_support"] = outside
    code = tf.where(outside, int(FailureCode.OUTSIDE_SUPPORT), int(FailureCode.NONFINITE_INPUT))
    for name in ("value_failure_code", "force_failure_code"):
        status[name] = tf.where(coordinate_valid,status[name],code)
    for name in ("first_failure_time","first_failure_particle"):
        status[name] = tf.where(coordinate_valid,status[name],tf.constant(-1,tf.int32))
    status["first_failure_stage"] = tf.where(coordinate_valid,status["first_failure_stage"],int(FailureStage.PRIOR))
    if "program_valid" in status:
        status["program_valid"] = status["value_valid"] & status["force_valid"]
    for name in ("direction_value_valid","direction_force_valid"):
        if name in status:
            status[name] &= coordinate_valid[:,None]
    return status


def make_finite_endpoint(consumer):
    """Return the frozen potential plus graph diagnostics; never alter the force."""
    batch = consumer.manifest["batch_size"]
    fixture = consumer.fixture
    lower = tf.constant(-.95, tf.float64)
    upper = tf.constant(.95, tf.float64)
    lo32, hi32 = tf.cast(lower, tf.float32), tf.cast(upper, tf.float32)
    safe_lower = tf.math.nextafter(lo32, hi32)
    safe_upper = tf.math.nextafter(hi32, lo32)
    tf.debugging.assert_equal(fixture.lower, [lo32])
    tf.debugging.assert_equal(fixture.upper, [hi32])

    @tf.function(input_signature=[tf.TensorSpec([batch, 1], tf.float64)],
                 jit_compile=True, autograph=False)
    def physical(phi):
        rounded = tf.cast(phi, tf.float32)
        adjusted = tf.clip_by_value(rounded, safe_lower, safe_upper)
        value, _, original = fixture.physical(adjusted)
        finite = tf.reduce_all(tf.math.is_finite(phi), axis=-1)
        support = finite & tf.reduce_all((phi > lower) & (phi < upper), axis=-1)
        status = _coordinate_status(original, support & tf.math.is_finite(value), finite & ~support)
        status["coordinate_boundary_rounding"] = tf.reduce_any(rounded != adjusted, axis=-1)
        return tf.cast(value, tf.float64), status

    @tf.function(input_signature=[tf.TensorSpec([batch, 1], tf.float64)],
                 jit_compile=True, autograph=False)
    def chart(z):
        # For extreme finite logits even FP64 sigmoid can round to a boundary.
        # This is coordinate rounding, not a mathematical support exclusion.
        phi = lower + (upper-lower)*tf.sigmoid(z)
        rounded = tf.cast(phi, tf.float32)
        adjusted = tf.clip_by_value(rounded, safe_lower, safe_upper)
        value, _, original = fixture.physical(adjusted)
        jacobian = tf.reduce_sum(tf.math.log(upper-lower)-tf.nn.softplus(-z)-tf.nn.softplus(z),axis=-1)
        log_density = tf.cast(value, tf.float64) + jacobian
        valid = (tf.reduce_all(tf.math.is_finite(z),axis=-1)
                 & tf.math.is_finite(log_density))
        status = _coordinate_status(original,valid,tf.zeros_like(valid))
        status["coordinate_boundary_rounding"] = tf.reduce_any(rounded != adjusted,axis=-1)
        return log_density, status

    def potential(z):
        value, status = chart(z)
        return TargetPotentialEvaluation(-value, status)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = ENDPOINT_VERSION + ":" + fixture.identity + ":" + source_hash
    return FrozenTargetPotential(potential, identity=identity), physical, chart
