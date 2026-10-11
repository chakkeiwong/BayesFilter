"""Compiled native-trace validity reductions; host code owns reason labels."""
from functools import lru_cache


@lru_cache(maxsize=32)
def target_status_program(status_dtype, floor_dtype, conditioning_dtypes, rank):
    """Reduce status bits without changing integer ranges or optional dtypes."""
    import tensorflow as tf

    shape = [None] * rank
    signature = [tf.TensorSpec(shape, tf.as_dtype(status_dtype)),
                 tf.TensorSpec(shape, tf.bool),
                 tf.TensorSpec(shape, tf.as_dtype(floor_dtype))]
    signature.extend(tf.TensorSpec(shape, tf.as_dtype(dtype)) for dtype in conditioning_dtypes)

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def validate(status, valid, floors, *conditioning):
        failed = (status != 0) | (~valid)
        flags = [tf.reduce_any(failed), tf.reduce_any((~failed) & (floors < 0))]
        for value in conditioning:
            if value.dtype.is_complex:
                finite = tf.math.is_finite(tf.math.real(value)) & tf.math.is_finite(tf.math.imag(value))
            elif value.dtype.is_integer:
                finite = tf.ones_like(value, tf.bool)
            else:
                finite = tf.math.is_finite(value)
            flags.append(tf.reduce_any((~failed) & (~finite)))
        return tf.stack(flags)

    return validate


@lru_cache(maxsize=1)
def retained_finite_program():
    """One exact Boolean transfer for the retained health validity checks."""
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float64),
        tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None, None], tf.float64)],
        autograph=False, jit_compile=True)
    def validate(samples, log_accept, target):
        return tf.stack([tf.reduce_all(tf.math.is_finite(value))
                         for value in (samples, log_accept, target)])

    return validate


@lru_cache(maxsize=2)
def native_health_program(jit_compile=True):
    """Use one stable graph for finite values, score bits and the MH state.

    Chain count and parameter dimension are fixed by each execution binding.
    Draw count varies over its declared chunk/evidence lengths. The caller
    transfers the Boolean vector once at its validation boundary.
    """
    import tensorflow as tf

    state = tf.TensorSpec([None, None, None], tf.float64)
    scalar = tf.TensorSpec([None, None], tf.float64)
    bits = tf.TensorSpec([None, None], tf.bool)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float64),
        state, state, state, state, scalar, scalar, scalar, scalar, bits, bits],
        autograph=False, jit_compile=jit_compile)
    def validate(initial, samples, proposal, initial_momentum, final_momentum,
                 log_accept, target, proposed_target, correction, accepted, score_finite):
        previous = tf.concat([initial[None], samples[:-1]], axis=0)
        values = (samples, initial, proposal, proposal - previous,
                  log_accept, target, proposed_target, initial_momentum,
                  final_momentum, correction)
        finite = [tf.reduce_all(tf.math.is_finite(value)) for value in values]
        expected = tf.where(accepted[..., None], proposal, previous)
        return tf.stack([*finite, tf.reduce_all(score_finite),
                         tf.reduce_all(tf.equal(expected, samples))])

    return validate
