"""Enclose analytical parameter directions without changing their authority.

This maps parameter directions, never training samples. The caller owns the
returned function; configuration-specific factories may retain it in their
existing bounded caches. No numerical evaluation is used to infer the schema.
"""

import tensorflow as tf


def make_direction_kernel(directional_kernel, direction_count=6, *,
                          jit_compile=True, validity_output_index=None):
    """Compile a fixed set of directions and return explicit host-check flags.

    The directional function must already declare a stable input signature:
    theta, one direction, then the remaining dynamic operands. Its first two
    results must be a scalar value and analytical directional score. Nested
    auxiliary results are retained for every direction. Tracing that declared
    signature supplies their shapes without executing a dummy numerical call.
    """
    signature = directional_kernel.input_signature
    if signature is None or len(signature) < 2 or direction_count < 1:
        raise ValueError("a stable directional signature and positive count are required")
    direction_spec = signature[1]
    if direction_spec.shape.rank != 1 or not direction_spec.shape.is_fully_defined():
        raise ValueError("parameter direction shape must be fixed")
    dtype = direction_spec.dtype
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("analytical direction assembly requires float32 or float64")
    concrete = directional_kernel.get_concrete_function()

    def output_spec(tensor):
        if not tensor.shape.is_fully_defined():
            raise ValueError("directional auxiliary output shapes must be declared")
        return tf.TensorSpec(tensor.shape, tensor.dtype)

    outputs_spec = tf.nest.map_structure(output_spec, concrete.structured_outputs)
    if outputs_spec[0] != tf.TensorSpec([], dtype) or outputs_spec[1] != tf.TensorSpec([], dtype):
        raise ValueError("directional value and score must be scalar and match direction dtype")
    if validity_output_index is not None and outputs_spec[validity_output_index] != tf.TensorSpec([], tf.bool):
        raise ValueError("directional validity output must be a boolean scalar")
    # Match the existing assert_near default (10 machine eps for both terms).
    epsilon = 2**-23 if dtype == tf.float32 else 2**-52

    @tf.function(input_signature=[signature[0],
        tf.TensorSpec([direction_count, direction_spec.shape[0]], dtype),
        *signature[2:]], jit_compile=jit_compile, autograph=False)
    def owner(theta, directions, *operands):
        outputs = tf.map_fn(lambda direction: concrete(theta, direction, *operands),
            directions, fn_output_signature=outputs_spec, parallel_iterations=1)
        values, scores = outputs[:2]
        finite = tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
        invariant = tf.reduce_all(tf.abs(values-values[0]) <
            tf.constant(10*epsilon, dtype)*(1+tf.abs(values[0])))
        valid = finite
        if validity_output_index is not None:
            valid &= tf.reduce_all(outputs[validity_output_index])
        return {"value": values[0], "score": scores, "outputs": outputs,
                "first_auxiliary": tf.nest.map_structure(lambda tensor: tensor[0], outputs[2:]),
                "value_invariant": invariant, "valid": valid}

    return owner
