"""Normalized proposal law, fit target and total finite-score tests."""
import tensorflow as tf
from bayesfilter.score_study.fitted_twist_tf import (normalizer,gaussian_floor_log,
    fit_log_quadratic,make_recursive_fit_kernel,make_fitted_twist_kernel)
from bayesfilter.score_study.gaussian_tf import gaussian_log_density_and_tangent,parameterized_model


def logn(x,m,C):
    d=x.shape[-1]
    return gaussian_log_density_and_tangent(x-m,tf.zeros([6,*x.shape],x.dtype),C,
        tf.zeros([6,1,d,d],x.dtype))[0]


def test_actual_two_component_proposal_matches_normalized_f_times_psi():
    dtype=tf.float64;Q=tf.constant([[.8,.1],[.1,.7]],dtype)
    V=tf.constant([[.5,-.1],[-.1,.6]],dtype);m=tf.constant([[.2,-.4]],dtype)
    center=tf.constant([1.,.5],dtype);floor=tf.constant(-2.,dtype)
    z=tf.constant([[.4,-.5],[1.,2.],[-2.,.2]],dtype)
    lt,_,p=normalizer(m,tf.zeros([6,1,2],dtype),Q,tf.zeros([6,2,2],dtype),center,V,floor)
    K=tf.transpose(tf.linalg.solve(Q+V,tf.transpose(Q)))
    cm=m+tf.einsum("ij,nj->ni",K,center-m);C=Q-K@Q
    logq=tf.reduce_logsumexp(tf.stack([tf.math.log(p)+logn(z,cm,C),
        tf.math.log(1-p)+logn(z,m,Q)]),axis=0)
    lp,_=gaussian_floor_log(z,tf.zeros([6,3,2],dtype),center,V,floor)
    tf.debugging.assert_near(logq,logn(z,m,Q)+lp-lt,atol=2e-12,rtol=2e-12)


def test_quadratic_fit_recovers_gaussian_and_rejects_unbounded_fit():
    x=tf.reshape(tf.linspace(tf.constant(-3.,tf.float64),tf.constant(4.,tf.float64),31),[-1,1])
    target=-.5*(x[:,0]-.7)**2/.8+2.
    (center,V,_),(valid,error,_)=fit_log_quadratic(x,target,.02)
    assert bool(valid);assert float(error)<2e-12
    tf.debugging.assert_near(center,tf.constant([.7],tf.float64),atol=2e-12)
    tf.debugging.assert_near(V,tf.constant([[.8]],tf.float64),atol=2e-12)
    (_,bad,_),(valid,_,_)=fit_log_quadratic(x,-target,.02)
    assert not bool(valid); assert bool(tf.reduce_any(tf.math.is_nan(bad)))


def test_recursive_fit_is_conditioned_and_frozen_score_matches_its_value():
    dtype=tf.float64;N=16;T=2
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype)
    obs=tf.constant([[.5],[-.3]],dtype)
    clouds=tf.random.stateless_normal([T,N,1],[91,1],dtype=dtype)
    fit=make_recursive_fit_kernel(1,1,N,T,.01)
    centers,V,floors,valid,errors=fit(theta,obs,clouds)
    assert bool(valid);assert bool(tf.reduce_all(tf.math.is_finite(errors)))
    _,_,H,_,_,_,_,_,_,_,R,_=parameterized_model(theta,1,1)
    tf.debugging.assert_near(centers[-1],obs[-1]/H[0,0],atol=2e-11)
    tf.debugging.assert_near(V[-1],R/(H[0,0]**2),atol=2e-11)
    kernel=make_fitted_twist_kernel(1,1,N,T)
    initial=tf.random.stateless_normal([N,1],[91,2],dtype=dtype)
    noise=tf.random.stateless_normal([T,N,1],[91,3],dtype=dtype)
    uniform=tf.random.stateless_uniform([T+1,N],[91,4],dtype=dtype)
    mix=tf.random.stateless_uniform([T,N],[91,5],dtype=dtype)
    args=(obs,initial,noise,uniform,mix,centers,V,floors)
    value,score,_=kernel(theta,*args)
    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
        h=tf.constant(1e-6,dtype)
        fd=(kernel(theta+h*direction,*args)[0]-kernel(theta-h*direction,*args)[0])/(2*h)
        tf.debugging.assert_near(tf.reduce_sum(score*direction),fd,rtol=3e-5,atol=2e-7)
    assert bool(tf.math.is_finite(value));assert kernel.experimental_get_tracing_count()==1
