"""Actual SGQF-to-LEDH call-chain, nonlinear moments and signed-rule vetoes."""
import tensorflow as tf

from bayesfilter.nonlinear.sgqf_covariance_provider_tf import SGQFCovarianceProvider
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFNonlinearModel, TFFixedSGQFBranchConfig, tf_fixed_sgqf_filter,
)
from bayesfilter.score_study.covariance_adapter_tf import make_covariance_kernel
from bayesfilter.score_study.canonical_adapter_tf import make_canonical_kernel
from bayesfilter.score_study.gaussian_tf import parameterized_model
from tests.highdim.test_younis_score_master_canonical_tf import CONTROLS


def test_batched_nonlinear_provider_matches_standalone_sgqf_and_tangents():
    provider = SGQFCovarianceProvider(2, 3)
    dtype = tf.float64
    cov = tf.constant([[[.4, .1], [.1, .3]], [[.2, .02], [.02, .5]]], dtype)
    states = tf.constant([[.2, -.4], [.8, .1]], dtype)
    q, r = tf.eye(2, dtype=dtype)*.1, tf.constant([[.4]], dtype)
    y = tf.constant([.5], dtype)
    def transition(x): return .7*x + .1*tf.sin(x)
    def tangent(x, dx): return (.7+.1*tf.cos(x))*dx
    def observe(x): return x[:, :1]+.2*x[:, 1:]**2
    def dobs(x, dx): return dx[:, :1]+.4*x[:, 1:]*dx[:, 1:]
    @tf.function(input_signature=[tf.TensorSpec([2, 2], dtype)], jit_compile=True)
    def moments(x):
        pred = provider.predict(x, cov, tf.ones_like(x), tf.zeros_like(cov), transition, tangent, q)
        return provider.update(*pred[:2], *pred[2:], observe, dobs, r, y)
    result = moments(states)
    for i in range(2):
        model = TFFixedSGQFNonlinearModel(initial_mean=states[i], initial_covariance=cov[i],
            process_covariance=q, observation_covariance=r, transition_fn=transition, observation_fn=observe)
        reference = tf_fixed_sgqf_filter(y[None, :], model, cloud=provider.cloud,
            branch_config=TFFixedSGQFBranchConfig(predictive_epsilon=0., innovation_epsilon=0.), return_filtered=True)
        assert reference.failure is None
        tf.debugging.assert_near(result[0][i], reference.filtered_means[0], atol=1e-11)
        tf.debugging.assert_near(result[1][i], reference.filtered_covariances[0], atol=1e-11)
    h = tf.constant(1e-5, dtype)
    plus, minus = moments(states+h), moments(states-h)
    tf.debugging.assert_near(result[2], (plus[0]-minus[0])/(2*h), atol=1e-9)
    tf.debugging.assert_near(result[3], (plus[1]-minus[1])/(2*h), atol=1e-9)
    assert moments.experimental_get_tracing_count() == 1


def test_signed_rule_is_not_a_sampling_law_and_invalid_covariance_fails_closed():
    p = SGQFCovarianceProvider(4, 2)
    assert p.metadata["negative_weight_count"] == 1
    assert not p.metadata["signed_weights_used_for_sampling"]
    x, P = tf.zeros([2, 4], tf.float64), tf.eye(4, batch_shape=[2], dtype=tf.float64)
    @tf.function(input_signature=[], jit_compile=True)
    def kernel():
        healthy = p.predict(x, P, x, tf.zeros_like(P), lambda z:z, lambda z,dz:dz,
                            tf.eye(4, dtype=tf.float64)*.1)
        # At this level E[(sum X_i^2)^2]=12 and (E sum X_i^2)^2=16.
        broken = p.predict(x, P, x, tf.zeros_like(P),
            lambda z:tf.repeat(tf.reduce_sum(z*z, axis=1, keepdims=True),4,axis=1),
            lambda z,dz:tf.repeat(tf.reduce_sum(2*z*dz,axis=1,keepdims=True),4,axis=1),
            tf.eye(4, dtype=tf.float64)*.1)
        return healthy, broken
    healthy, broken = kernel()
    tf.debugging.assert_near(healthy[1], P*1.1, atol=1e-12)
    assert bool(tf.reduce_all(tf.math.is_nan(broken[1])))


def test_sgqf_full_ledh_consumer_all_six_directions_and_reset_carry():
    dtype = tf.float64
    theta = tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype)
    args = (tf.constant([[.5],[-.2]],dtype),
            tf.random.stateless_normal([8,2],[4,8],dtype=dtype),
            tf.random.stateless_normal([2,8,2],[5,8],dtype=dtype),
            tf.random.stateless_normal([8,2],[6,8],dtype=dtype))
    settings = (2,1,8,2,tuple(sorted(CONTROLS.items())))
    kernel, metadata = make_covariance_kernel(*settings,3)
    canonical = make_canonical_kernel(*settings)
    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
        value, score, trace = kernel(theta,direction,*args)
        reference_value, reference_score = canonical(theta,direction,*args)
        tf.debugging.assert_near(value,reference_value,atol=2e-9)
        tf.debugging.assert_near(score,reference_score,atol=2e-8)
        h = tf.constant(2e-5,dtype)
        fd = (kernel(theta+h*direction,direction,*args)[0]-kernel(theta-h*direction,direction,*args)[0])/(2*h)
        tf.debugging.assert_near(score,fd,atol=3e-5,rtol=3e-4)
        assert bool(tf.reduce_all(tf.math.is_finite(trace[-1]["d_covariances_after_reset"])))
        A,dA,_,_,_,_,_,_,Q,dQ,_,_=parameterized_model(theta,2,1)
        expected=A@trace[0]["covariances_after_reset"]@tf.transpose(A)+Q
        tf.debugging.assert_near(trace[1]["predicted_covariances"],expected,atol=1e-12)
    assert metadata["point_count"] > 5
    assert kernel.experimental_get_tracing_count() == 1
