"""CPU-only independent arithmetic and derivative diagnostic checks."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_transport_cache_diagnostic import cached_normalization, combined_normalization


def test_fp64_all_parameter_derivatives_and_primal():
    dtype=tf.float64
    c=-tf.square(tf.random.stateless_normal([2,7,7],[1,7],dtype=dtype))
    a=tf.random.stateless_normal([2,7],[2,7],dtype=dtype)
    dc=tf.random.stateless_normal([3,2,7,7],[3,7],dtype=dtype)
    da=tf.random.stateless_normal([3,2,7],[4,7],dtype=dtype)
    value,weights,fallback=cached_normalization(c,tf.exp(c),a)
    assert not bool(fallback)
    tf.debugging.assert_near(value,tf.reduce_logsumexp(c+a[:,None],-1),atol=2e-14,rtol=2e-14)
    score=tf.reduce_sum(weights[None]*(dc+da[:,:,None]),-1)
    step=tf.constant(1e-5,dtype)
    expected=(tf.reduce_logsumexp(c[None]+step*dc+(a[None]+step*da)[:,:,None],-1)-tf.reduce_logsumexp(c[None]-step*dc+(a[None]-step*da)[:,:,None],-1))/(2*step)
    tf.debugging.assert_near(score,expected,atol=2e-9,rtol=2e-9)


def test_underflow_cancellation_requires_exact_log_domain_fallback():
    c=tf.constant([[[-100.,0.],[0.,-100.]]])
    a=tf.constant([[100.,0.]])
    v,w,fallback=cached_normalization(c,tf.exp(c),a)
    assert bool(fallback)
    logits=c+a[:,None]
    tf.debugging.assert_equal(v,tf.reduce_logsumexp(logits,-1))
    tf.debugging.assert_equal(w,tf.nn.softmax(logits,-1))


def test_common_shift_and_distant_negligible_pairs():
    c=tf.constant([[[0.,-1000.],[-1000.,0.]]])
    a=tf.constant([[.2,-.3]])
    v,w,fallback=cached_normalization(c,tf.exp(c),a)
    shifted,shiftw,shiftfallback=cached_normalization(c,tf.exp(c),a+100.)
    assert not bool(fallback) and not bool(shiftfallback)
    tf.debugging.assert_near(shifted,v+100.,atol=1e-5)
    tf.debugging.assert_equal(shiftw,w)


def test_combined_reduction_value_and_all_directions():
    dtype=tf.float64
    c=-tf.square(tf.random.stateless_normal([2,7,7],[5,7],dtype=dtype))
    a=tf.random.stateless_normal([2,7],[6,7],dtype=dtype)
    dc=tf.random.stateless_normal([3,2,7,7],[7,7],dtype=dtype)
    da=tf.random.stateless_normal([3,2,7],[8,7],dtype=dtype)
    value,weights,fallback=combined_normalization(c,a)
    assert not bool(fallback)
    tf.debugging.assert_near(value,tf.reduce_logsumexp(c+a[:,None],-1),atol=2e-14,rtol=2e-14)
    tf.debugging.assert_near(weights,tf.nn.softmax(c+a[:,None],-1),atol=2e-14,rtol=2e-14)
    score=tf.reduce_sum(weights[None]*(dc+da[:,:,None]),-1)
    h=tf.constant(1e-5,dtype)
    positive=tf.reduce_logsumexp(c[None]+h*dc+(a[None]+h*da)[:,:,None],-1)
    negative=tf.reduce_logsumexp(c[None]-h*dc+(a[None]-h*da)[:,:,None],-1)
    tf.debugging.assert_near(score,(positive-negative)/(2*h),atol=2e-9,rtol=2e-9)


def test_combined_reduction_extreme_and_nonfinite_fallback():
    c=tf.constant([[[-1000.,0.],[0.,-1000.]]]);a=tf.constant([[1000.,-1000.]])
    value,weights,fallback=combined_normalization(c,a)
    assert not bool(fallback)
    logits=c+a[:,None,:]
    tf.debugging.assert_near(value,tf.reduce_logsumexp(logits,-1))
    tf.debugging.assert_near(weights,tf.nn.softmax(logits,-1))
    _,_,fallback=combined_normalization(tf.constant([[[-float("inf")]]]),tf.zeros([1,1]))
    assert bool(fallback)
