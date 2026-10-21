"""Independent arithmetic checks of combined transport reductions."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_transport_reduction_tf import logsumexp_and_weights

@pytest.mark.parametrize("dtype",[tf.float32,tf.float64])
@pytest.mark.parametrize("offset",[-10000.,0.,10000.])
def test_combined_reduction_matches_independent_primitives(dtype,offset):
    x=tf.random.stateless_normal([2,31,37],[62,17],dtype=dtype)*7+tf.cast(offset,dtype)
    @tf.function(input_signature=[tf.TensorSpec(x.shape,dtype)],jit_compile=True,autograph=False)
    def actual(x): return logsumexp_and_weights(x,2)
    value,weights=actual(x)
    tolerance=3e-6 if dtype==tf.float32 else 1e-12
    tf.debugging.assert_near(value,tf.reduce_logsumexp(x,2),atol=tolerance,rtol=tolerance)
    tf.debugging.assert_near(weights,tf.nn.softmax(x,2),atol=tolerance,rtol=tolerance)


def test_nonfinite_logits_preserve_invalidity():
    x=tf.constant([[float("-inf"),float("-inf")],[float("inf"),1.],[float("nan"),1.]],tf.float32)
    value,weights=logsumexp_and_weights(x)
    tf.debugging.assert_equal(tf.math.is_finite(value),tf.math.is_finite(tf.reduce_logsumexp(x,-1)))
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_finite(weights),-1),tf.reduce_all(tf.math.is_finite(tf.nn.softmax(x,-1)),-1))
