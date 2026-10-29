"""Independent density/FD references and actual shared-core wiring checks."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from dataclasses import replace
import math
from unittest.mock import patch
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_marginal_weights_tf import (
    gaussian_mixture_log_density_tangent, marginal_prior_ratio_tangent)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel, canonical_value_and_analytical_score)
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, canonical_batch_fused_value_score,
    canonical_batch_fused_value_score_whileloop)
D = tf.float64


@pytest.mark.parametrize("policy", ["ancestor", "marginal_mixture"])
def test_density_nonuniform_weights_total_derivative(policy):
    rng = np.random.default_rng(109)
    n, d = 8, 3
    x, dx, m, dm = [rng.normal(size=(n,d)) for _ in range(4)]
    a = rng.normal(size=(n,d,d)); cov = a@a.transpose(0,2,1)+np.eye(d)[None]
    dcov = rng.normal(size=(n,d,d)); dcov = .05*(dcov+dcov.transpose(0,2,1))
    logits, dl = rng.normal(size=(2,n))
    def ref(h):
        w = np.exp(logits+h*dl); w /= w.sum()
        c = cov+h*dcov
        residual = (x+h*dx)[:,None]-(m+h*dm)[None]
        terms = np.log(w)[None]-.5*(d*math.log(2*math.pi)+np.linalg.slogdet(c)[1][None]
            +np.einsum('nmi,mij,nmj->nm',residual,np.linalg.inv(c),residual))
        return np.diag(terms) if policy == "ancestor" else np.logaddexp.reduce(terms,axis=1)
    w = np.exp(logits); w /= w.sum()
    args = [tf.constant(v,D) for v in (x,dx,m,dm,cov,dcov,np.log(w),dl-w@dl)]
    @tf.function(input_signature=[tf.TensorSpec(t.shape,D) for t in args],jit_compile=True,autograph=False)
    def run(*t): return gaussian_mixture_log_density_tangent(*t,component_policy=policy)
    value, tangent, valid, force = run(*args)
    assert bool(valid & force)
    np.testing.assert_allclose(value,ref(0),atol=1e-12,rtol=1e-12)
    h=1e-5
    np.testing.assert_allclose(tangent,(ref(h)-ref(-h))/(2*h),atol=2e-8,rtol=2e-8)


@pytest.mark.parametrize("n", [2, 8])
def test_identical_components_collapse_without_losing_outer_weights(n):
    points=tf.reshape(tf.linspace(tf.constant(-1.,D),tf.constant(1.,D),n),[n,1])
    anchors=tf.zeros_like(points); means=tf.ones_like(points)*.2
    q=tf.ones([1,1],D)*.7; b=tf.ones([n,1,1],D)*.6
    lw=tf.nn.log_softmax(tf.cast(tf.range(n),D))
    dl=tf.cast(tf.range(n),D);dl-=tf.reduce_sum(tf.exp(lw)*dl)
    args=(points,tf.ones_like(points)*.1,anchors,anchors,q,q*.1,means,means*.2,b,b*.1,lw,dl)
    own=marginal_prior_ratio_tangent(*args,component_policy="ancestor")
    full=marginal_prior_ratio_tangent(*args,component_policy="marginal_mixture")
    assert all(bool(v) for v in (*own[2:],*full[2:]))
    np.testing.assert_allclose(own[0],full[0],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(own[1],full[1],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(tf.nn.softmax(lw+own[0]),tf.nn.softmax(lw+full[0]),atol=1e-12)


def single_model():
    return NonlinearScoreModel(
        transition_mean_fn=lambda t,x:t[0]*x,
        transition_mean_tangent_fn=lambda t,x,dx:x+t[0]*dx,
        observation_fn=lambda x:x, observation_tangent_fn=lambda x,dx:dx,
        observation_jacobian_fn=lambda x:tf.ones([x.shape[0],1,1],D),
        process_covariance=tf.constant([[.4]],D),observation_covariance=tf.constant([[.6]],D))


def batch_model():
    return PerPointScoreModel(
        transition_mean_fn=lambda t,x:t*x,
        transition_mean_tangent_fn=lambda t,x,dx,dt:dt*x+t*dx,
        observation_fn=lambda x:x, observation_tangent_fn=lambda x,dx:dx,
        observation_jacobian_fn=lambda x:tf.ones([x.shape[0],1,1],D),
        process_covariance=tf.constant([[.4]],D),observation_covariance=tf.constant([[.6]],D))


def inputs():
    x=tf.reshape(tf.linspace(tf.constant(-1.,D),tf.constant(1.,D),8),[8,1])
    p=tf.ones([8,1,1],D)
    noise=tf.random.stateless_normal([3,8,1],[23,10],dtype=D)
    obs=tf.constant([[.2],[-.1],[.3]],D)
    reset=x/tf.sqrt(tf.reduce_mean(x*x))
    return x,p,noise,obs,reset


CONTROLS=dict(reset_policy="contract_e",reset_sinkhorn_steps=8,reset_balance_steps=8,
              reset_epsilon=2.,reset_ridge=1e-5,correction_steps=1,pairwise_steps=1,
              correction_strength=.2,correction_lm_damping=.01,correction_lm_scale_floor=1e-4,
              pairwise_strength=.02,
              correction_trust_radius=.1,pairwise_rms_cap=2.,coordinate_cap=16.,coordinate_cap_power=8)


@pytest.mark.parametrize("endpoint",[canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop])
@pytest.mark.parametrize("policy",["ancestor","marginal_mixture"])
def test_batch_endpoints_reach_shared_correction_and_match_scalar(endpoint,policy):
    x,p,z,y,reset=inputs();theta=tf.constant([[.7],[.8]],D)
    options=dict(CONTROLS,reset_design=reset,importance_weight_policy=policy)
    with patch('bayesfilter.highdim.ledh_canonical_score_tf.marginal_prior_ratio_tangent',
               wraps=marginal_prior_ratio_tangent) as call:
        @tf.function(input_signature=[tf.TensorSpec([2,1],D)],jit_compile=True,autograph=False)
        def run(t):return endpoint(batch_model(),t,tf.ones([2,1],D),x,p,z,y,substeps=4,**options)
        values,scores,status=run(theta)
        assert call.called
        assert all(c.kwargs['component_policy']==policy for c in call.call_args_list)
    expected=[canonical_value_and_analytical_score(single_model(),t,x,p,z,y,
        flow_substeps=4,with_score=True,**options) for t in theta]
    assert bool(tf.reduce_all(status['program_valid']))
    np.testing.assert_allclose(values,[v for v,s in expected],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(scores,[s[0] for v,s in expected],atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize("policy",["ancestor","marginal_mixture"])
def test_recursive_score_and_trace_total_finite_difference(policy):
    x,p,z,y,reset=inputs()
    @tf.function(input_signature=[tf.TensorSpec([1],D)],jit_compile=True,autograph=False)
    def run(t):return canonical_value_and_analytical_score(single_model(),t,x,p,z,y,
        with_score=True,return_trace=True,flow_substeps=4,reset_design=reset,
        importance_weight_policy=policy,**CONTROLS)
    t=tf.constant([.7],D);value,score,trace=run(t)
    assert len(trace)==3 and bool(tf.math.is_finite(value))
    h=tf.constant(1e-5,D);fd=(run(t+h)[0]-run(t-h)[0])/(2*h)
    np.testing.assert_allclose(score[0],fd,atol=1e-7,rtol=1e-7)


def test_default_is_diagonal_and_unsupported_mixture_fails():
    x,p,z,y,_=inputs();t=tf.constant([.7],D);kwargs=dict(flow_substeps=4,with_score=True)
    default=canonical_value_and_analytical_score(single_model(),t,x,p,z,y,**kwargs)
    own=canonical_value_and_analytical_score(single_model(),t,x,p,z,y,importance_weight_policy="ancestor",**kwargs)
    for a,b in zip(default,own):np.testing.assert_array_equal(a,b)
    for extra in [dict(importance_weight_policy="unknown"),
                  dict(importance_weight_policy="marginal_mixture",annealed_stages=2)]:
        with pytest.raises(ValueError):
            canonical_value_and_analytical_score(single_model(),t,x,p,z,y,**kwargs,**extra)
    custom=replace(single_model(),transition_log_density_fn=lambda *args:None)
    with pytest.raises(ValueError,match="Gaussian"):
        canonical_value_and_analytical_score(custom,t,x,p,z,y,importance_weight_policy="marginal_mixture",**kwargs)


def test_multiple_exact_tiles_match_closed_form():
    # N=4096 requires K=2048: exercise row and column accumulation, including
    # nonuniform weights whose total derivative sums to zero.
    n=4096
    x=tf.reshape(tf.linspace(tf.constant(-2.,D),tf.constant(2.,D),n),[n,1])
    mean=tf.ones_like(x)*.3;dx=tf.ones_like(x)*.2;dm=tf.ones_like(x)*.1
    cov=tf.ones([n,1,1],D)*.7;dcov=tf.ones_like(cov)*.05
    lw=tf.nn.log_softmax(tf.linspace(tf.constant(-1.,D),tf.constant(1.,D),n))
    dl=tf.linspace(tf.constant(-.2,D),tf.constant(.4,D),n)
    dl-=tf.reduce_sum(tf.exp(lw)*dl)
    args=(x,dx,mean,dm,cov,dcov,lw,dl)
    @tf.function(input_signature=[tf.TensorSpec(t.shape,D) for t in args],jit_compile=True,autograph=False)
    def run(*t):return gaussian_mixture_log_density_tangent(*t)
    value,tangent,valid,force=run(*args)
    residual=(x-mean)[:,0]
    expected=-.5*(math.log(2*math.pi*.7)+residual**2/.7)
    derivative=-residual*.1/.7+.5*(residual**2/.7-1)*.05/.7
    assert bool(valid & force)
    np.testing.assert_allclose(value,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(tangent,derivative,atol=1e-12,rtol=1e-12)
