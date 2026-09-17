"""Independent checks of legacy TensorFlow random-stream compatibility."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import philox_uniform_float64


@pytest.mark.parametrize("count", [7, 16, 33])
def test_uniform_words_preserve_non_xla_double_stream(count):
    seed = tf.constant([71, 7000], tf.int32)
    with tf.device('/CPU:0'):
        expected = tf.random.stateless_uniform([count], seed, dtype=tf.float64, alg='philox')
    call = tf.function(lambda seed: philox_uniform_float64([count], seed),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    np.testing.assert_array_equal(call(seed).numpy(), expected.numpy())


def test_gpu_categorical_engine_diagnostic():
    if not tf.config.list_physical_devices('GPU'):
        pytest.skip('GPU categorical engine comparison requires the GPU group')
    count = 16
    seed = tf.constant([71, 7000], tf.int32)
    logits = -tf.square(tf.linspace(tf.constant(-0.9, tf.float64), 0.9, count)) / 0.06
    with tf.device('/GPU:0'):
        expected = tf.random.stateless_categorical(logits[None, :], count, seed, dtype=tf.int32)[0]
        doubles = philox_uniform_float64([count, count], seed)
        singles = tf.cast(tf.random.stateless_uniform([count, count], seed, dtype=tf.float32, alg='philox'), tf.float64)
    candidates = {}
    for name, uniforms in [('double', doubles), ('float', singles)]:
        for orientation, values in [('samples_first', uniforms), ('classes_first', tf.transpose(uniforms))]:
            candidate = tf.argmax(logits[None, :] - tf.math.log(-tf.math.log(values)), axis=1, output_type=tf.int32)
            candidates[name + '_' + orientation] = candidate.numpy().tolist()
    print('GPU_CATEGORICAL', expected.numpy().tolist(), candidates)
    assert expected.numpy().tolist() in candidates.values()
