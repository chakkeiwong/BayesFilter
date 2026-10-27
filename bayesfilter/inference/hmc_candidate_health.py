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
        state, state, state, state, scalar, scalar, scalar, scalar, bits, bits, bits],
        autograph=False, jit_compile=jit_compile)
    def validate(initial, samples, proposal, initial_momentum, final_momentum,
                 log_accept, target, proposed_target, correction, accepted, score_finite,
                 accepted_score_finite):
        previous = tf.concat([initial[None], samples[:-1]], axis=0)
        # TFP MH safe_sum records invalid proposed arithmetic as -inf and
        # rejects it. This is alpha=0, not missing acceptance evidence.
        rejected = (~accepted) & (log_accept == -float('inf')) & accepted_score_finite
        def proposed_finite(value):
            mask = rejected[..., None] if value.shape.rank == 3 else rejected
            return tf.reduce_all(mask | tf.math.is_finite(value))
        values = (samples, initial, proposal, proposal - previous,
                  log_accept, target, proposed_target, initial_momentum,
                  final_momentum, correction)
        proposal_indices = {2, 3, 4, 6, 8, 9}
        finite = [proposed_finite(value) if index in proposal_indices
                  else tf.reduce_all(tf.math.is_finite(value))
                  for index, value in enumerate(values)]
        expected = tf.where(accepted[..., None], proposal, previous)
        return tf.stack([*finite, tf.reduce_all((rejected | score_finite) & accepted_score_finite),
                         tf.reduce_all(tf.equal(expected, samples))])

    return validate


@lru_cache(maxsize=2)
def modern_selection_program(jit_compile=True):
    """Validate actual per-transition values; modern kernels carry no cache.

    The raw ratio and first failure remain observable. Only a numerical first
    failure with a valid current target can be represented by zero acceptance.
    This does not declare a finite invalid target status harmless.
    """
    import tensorflow as tf

    scalar = tf.TensorSpec([None, None], tf.float64)
    state = tf.TensorSpec([None, None, None], tf.float64)
    bits = tf.TensorSpec([None, None], tf.bool)
    integers = tf.TensorSpec([None, None], tf.int32)

    @tf.function(input_signature=[bits, scalar, scalar, bits, bits, scalar,
        scalar, scalar, state, state, state, bits, bits, integers, integers,
        integers, bits, bits, bits], autograph=False, jit_compile=jit_compile)
    def validate(accepted, ratio, raw_ratio, rejected, previous_valid,
                 previous_value, proposed_value, selected_value, previous_score,
                 proposed_score, selected_score, selected_score_finite,
                 proposal_good, stage, substep, steps, position_finite,
                 value_finite, score_finite):
        numeric = ((stage == 2) | (stage == 3) | (stage == 5) | (stage == 6)
            | ((stage == 4) & (~position_finite | ~value_finite | ~score_finite)))
        expected_rejection = previous_valid & numeric & ~accepted
        first_failure_ok = ((stage >= 0) & (stage <= 6)
            & tf.where(stage == 0, substep == -1,
                tf.where(stage == 1, substep == 0,
                    tf.where(stage == 6, substep == steps + 1,
                             (substep >= 1) & (substep <= steps)))))
        expected_ratio = tf.where(rejected, tf.constant(-float('inf'), tf.float64), raw_ratio)
        value = tf.where(accepted, proposed_value, previous_value)
        score = tf.where(accepted[..., None], proposed_score, previous_score)
        return tf.stack([
            tf.reduce_all(previous_valid & tf.math.is_finite(previous_value)
                & tf.reduce_all(tf.math.is_finite(previous_score), -1)),
            tf.reduce_all(first_failure_ok & (rejected == expected_rejection)),
            tf.reduce_all(tf.equal(ratio, expected_ratio)),
            tf.reduce_all((~rejected | (~accepted & (ratio == -float('inf'))))
                & ((stage == 0) | (~accepted & rejected))),
            tf.reduce_all(tf.equal(selected_value, value)),
            tf.reduce_all(tf.equal(selected_score, score)),
            tf.reduce_all(selected_score_finite
                & tf.reduce_all(tf.math.is_finite(selected_score), -1)),
            tf.reduce_all(proposal_good == (stage == 0)),
        ])

    return validate
