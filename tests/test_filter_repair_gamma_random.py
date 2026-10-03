"""Independent CPU stream check before changing gamma preparation execution."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.stateless_gamma_tf import philox_gamma_float64


@pytest.mark.parametrize("alpha", [.05, .5, 1., 1.01, 2.5, 20.])
@pytest.mark.parametrize("seed,shape", [((713, 33), (8, 3)), ((-1729, 13), (256, 1))])
def test_stateless_gamma_xla_preserves_existing_cpu_stream(alpha, seed, shape):
    inputs = (tf.constant(seed, tf.int32), tf.constant(alpha, tf.float64))

    def draw(seed, concentration):
        return tf.random.stateless_gamma(shape, seed, alpha=concentration,
                                          beta=tf.constant(.5, tf.float64), dtype=tf.float64)

    call = tf.function(lambda seed, concentration: philox_gamma_float64(
        shape, seed, concentration, tf.constant(.5, tf.float64)),
        input_signature=[tf.TensorSpec([2], tf.int32),
        tf.TensorSpec([], tf.float64)], jit_compile=True, autograph=False)
    expected, actual = draw(*inputs), call(*inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
