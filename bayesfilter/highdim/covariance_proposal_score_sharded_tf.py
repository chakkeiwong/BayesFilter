"""Bounded-memory evaluation of a logical batch using native batched shards.

This is a value/analytical-score adapter, not an optimizer or NeuTra training
step. All rows retain the same fixed data/design. The shard loop does not
evaluate a scalar target; every call preserves a native batch greater than one.
"""
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter


def make_sharded_score_filter(spec, n, horizon, controls, dtype=tf.float64,
                              jit_compile=True, batch_size=64, shard_size=8,
                              record_history=False, precision_policy="uniform",
                              fixed_draw_design=None):
    if not 1 < shard_size <= batch_size:
        raise ValueError("require 1 < shard_size <= batch_size")
    native = make_batched_score_filter(spec, n, horizon, controls, dtype,
        jit_compile=jit_compile, batch_size=shard_size,
        record_history=record_history, precision_policy=precision_policy,
        fixed_draw_design=fixed_draw_design)
    signature = list(native.input_signature)
    signature[0] = tf.TensorSpec([batch_size, spec.parameter_count], dtype)
    outputs = native.get_concrete_function().structured_outputs
    keys = tuple(key for key, tensor in outputs.items() if tensor.shape.rank)
    scalars = tuple(key for key, tensor in outputs.items() if not tensor.shape.rank)
    if scalars != ("covariance_tolerance",):
        raise ValueError("unrecognized scalar output; define its shard semantics")
    groups = (batch_size + shard_size - 1) // shard_size

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def program(theta, *data):
        arrays = tuple(tf.TensorArray(outputs[key].dtype, groups,
            element_shape=outputs[key].shape) for key in keys)

        def body(index, *arrays):
            rows = tf.minimum(index*shard_size + tf.range(shard_size), batch_size-1)
            result = native(tf.gather(theta, rows), *data)
            return (index+1, *(array.write(index, result[key])
                              for key, array in zip(keys, arrays)))

        result = tf.while_loop(lambda index, *_: index < groups, body,
                              (0, *arrays), parallel_iterations=1)
        combined = {key: tf.reshape(array.stack(),
            [groups*shard_size, *outputs[key].shape[1:]])[:batch_size]
            for key, array in zip(keys, result[1:])}
        from bayesfilter.highdim.covariance_proposal_tf import covariance_restoration_tolerance
        combined["covariance_tolerance"] = covariance_restoration_tolerance(n, spec.dimension, dtype)
        return combined

    return program
