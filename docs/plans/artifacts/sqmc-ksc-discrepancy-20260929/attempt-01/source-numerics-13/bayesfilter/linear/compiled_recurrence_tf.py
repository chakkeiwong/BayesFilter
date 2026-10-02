"""Compile a fixed-shape numerical recurrence for model/callback APIs.

Callers should construct their parameter-to-result tf.function once with an
explicit input_signature. This helper is then traced once inside that function;
no parameter-dependent tensor is cached across target evaluations. Eager calls
are convenience calls and construct a fresh compiled recurrence.
"""

from __future__ import annotations

import tensorflow as tf


def compiled_tensor_recurrence(body, state, steps, *, jit_compile=True):
    """Run sequential steps in XLA with tensor-only input/output state.

    Signature construction iterates over the fixed state schema, never dates,
    particles, parameters or numerical array coordinates.
    """
    if not jit_compile:
        # Explicit independent-reference/debug exception; never a fallback.
        return tf.while_loop(
            lambda t, *_: t < steps,
            body,
            (tf.constant(0), *state),
            parallel_iterations=1,
            maximum_iterations=steps,
        )
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in state)

    @tf.function(input_signature=signature, jit_compile=True)
    def run(*initial):
        return tf.while_loop(
            lambda t, *_: t < steps,
            body,
            (tf.constant(0), *initial),
            parallel_iterations=1,
            maximum_iterations=steps,
        )

    return run(*state)
