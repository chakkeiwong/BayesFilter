"""Independent log-domain DSF boundary checks; CPU-only numerical reference."""
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_transport_core import sigmoid_mixture


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_finite_tiny_weights_keep_identical_sigmoids_affine(dtype):
    # For equal slopes/offsets the normalized mixture is exactly one sigmoid,
    # so applying logit gives 2*x+.3 regardless of the normalized weights.
    @tf.function(input_signature=[tf.TensorSpec([4], dtype)], jit_compile=True, autograph=False)
    def evaluate(x):
        slopes = tf.math.log(tf.fill([4, 2], tf.constant(2., dtype)))
        offsets = tf.fill([4, 2], tf.constant(.3, dtype))
        logits = tf.broadcast_to(tf.constant([0., -1000.], dtype), [4, 2])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y, ld, score = sigmoid_mixture(x, slopes, offsets, logits)
        return y, ld, score, tape.gradient(y, x)
    x = tf.constant([-100., -.7, .2, 100.], dtype)
    y, ld, score, derivative = evaluate(x)
    tolerance = 2.e-5 if dtype == tf.float32 else 1.e-11
    tf.debugging.assert_near(y, 2.*x+.3, atol=tolerance, rtol=tolerance)
    tf.debugging.assert_near(ld, tf.fill([4], tf.math.log(tf.constant(2., dtype))), atol=tolerance, rtol=tolerance)
    tf.debugging.assert_near(score, tf.zeros([4], dtype), atol=tolerance, rtol=tolerance)
    tf.debugging.assert_near(derivative, tf.fill([4], tf.constant(2., dtype)), atol=tolerance, rtol=tolerance)


def test_tiny_weight_can_still_determine_a_tail_in_log_space():
    # Component 2 has tiny prior weight but a much larger sigmoid probability
    # at x=-200. Its contribution must survive the log-domain mixture.
    x = tf.constant([-200.], tf.float32)
    logs = tf.math.log(tf.constant([[1., .001]], tf.float32))
    offsets = tf.constant([[0., 0.]], tf.float32)
    logits = tf.constant([[0., -119.]], tf.float32)
    @tf.function(input_signature=[tf.TensorSpec([1], tf.float32)], jit_compile=True, autograph=False)
    def evaluate(x):
        return sigmoid_mixture(x, logs, offsets, logits)
    actual = evaluate(x)
    # Independent FP64 probability-space equations remain representable here.
    xd = tf.cast(x[:, None], tf.float64)
    slopes = tf.cast(tf.exp(logs), tf.float64)
    p = tf.math.sigmoid(slopes*xd)
    w = tf.nn.softmax(tf.cast(logits, tf.float64))
    mixture = tf.reduce_sum(w*p, axis=-1)
    derivative = tf.reduce_sum(w*slopes*p*(1.-p), axis=-1)
    y = tf.math.log(mixture)-tf.math.log1p(-mixture)
    ld = tf.math.log(derivative)-tf.math.log(mixture)-tf.math.log1p(-mixture)
    tf.debugging.assert_near(tf.cast(actual[0], tf.float64), y, atol=3.e-5, rtol=1.e-6)
    tf.debugging.assert_near(tf.cast(actual[1], tf.float64), ld, atol=3.e-5, rtol=1.e-6)
    assert float(actual[0][0]) > -130.  # Dropping the tiny component gives -200.


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_log_weights_still_fail(bad):
    result = sigmoid_mixture(tf.constant([.2]), tf.constant([[0., 0.]]),
        tf.constant([[0., 0.]]), tf.constant([[0., bad]]))
    assert all(not bool(tf.reduce_all(tf.math.is_finite(value))) for value in result)
