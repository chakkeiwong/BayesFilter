"""Independent M10 quadrature repair candidate; not posterior admission.

Composite Gaussian quadrature follows both observation roots and each normal
density. Autodiff computes the total derivative of this finite reference
integral, including moving nodes. This is never a canonical DCTR score path.
"""
from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.score_study.gaussian_tf import parameterized_model
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import (
    source_theta, TRANSITION_CURVE, OBSERVATION_CURVE)

DTYPE = tf.float64
FP64_MIN_NORMAL = float.fromhex('0x1.0p-1022')


def predictive_integrals_valid(likelihood, predictive):
    # Relative error alone is undefined at zero and too strict for subnormal
    # sums. Preserve every computed value; give only the validity check a
    # minimum-normal scale. This admits about 2.23e-320 absolute rounding error,
    # far below any posterior-reference allowance, without clipping a result.
    rounding = tf.constant(1e-12, DTYPE)*tf.maximum(
        likelihood, tf.constant(FP64_MIN_NORMAL, DTYPE))
    return tf.reduce_all(tf.math.is_finite(predictive) & (predictive>=0)
                         & (predictive<=likelihood+rounding))


def quadrature_rule(order, *, normal):
    with tf.device('/CPU:0'):
        index = tf.range(1, order, dtype=DTYPE)
        off = tf.sqrt(index) if normal else index/tf.sqrt(4*index*index-1)
        jacobi = tf.linalg.diag(off,k=1,num_rows=order,num_cols=order)
        nodes,vectors = tf.linalg.eigh(jacobi+tf.transpose(jacobi))
        weights = tf.square(vectors[0])*(1. if normal else 2.)
        tf.debugging.assert_near(tf.reduce_sum(weights),tf.constant(1. if normal else 2.,DTYPE),atol=1e-12)
        tf.debugging.assert_near(tf.reduce_sum(weights*nodes*nodes),tf.constant(1. if normal else 2/3,DTYPE),atol=1e-12)
    return nodes,weights


@lru_cache(maxsize=12)
def make_reference(legendre_order=20, normal_order=128, convolution='auto', *, value_only=False):
    if (legendre_order not in (12,20,32) or normal_order not in (64,128)
            or convolution not in ('auto','initial','inverse')):
        raise ValueError('unplanned M10 quadrature configuration')
    nodes,weights = quadrature_rule(legendre_order,normal=False)
    gh,gw = quadrature_rule(normal_order,normal=True)
    normal_offsets = tf.constant([-10.,-6.,-3.,-1.,0.,1.,3.,6.,10.],DTYPE)
    root_offsets = tf.constant([-16.,-8.,-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.,8.,16.],DTYPE)
    c,b = tf.cast(TRANSITION_CURVE,DTYPE),tf.cast(OBSERVATION_CURVE,DTYPE)
    normalizer = tf.constant(math.sqrt(2*math.pi),DTYPE)

    def density(x,mean,sd):
        return tf.exp(-.5*((x-mean)/sd)**2)/(normalizer*sd)

    def compute(scales,observations):
        a,_,h,_,m,_,p,_,q,_,r,_ = parameterized_model(source_theta(scales),1,1)
        a,h,m,sp,sq,sr = a[0,0],h[0,0],m[0],tf.sqrt(p[0,0]),tf.sqrt(q[0,0]),tf.sqrt(r[0,0])
        def f(x):
            return a*x+c*tf.sin(x)

        def partition(lower,upper,mean,sd,y):
            # Both roots define partitions of the full state interval, never
            # a selected branch of the observation law.
            discriminant = tf.maximum(h*h+4*b*y,tf.constant(0.,DTYPE))
            spread = tf.sqrt(discriminant)
            roots = tf.stack([(-h-spread)/(2*b),(-h+spread)/(2*b)])
            width = 2*sr/(tf.sqrt(discriminant+4*b*sr)+spread)
            obs_breaks = tf.reshape(roots[:,None]+width*root_offsets[None,:],[-1])
            breaks = tf.concat([lower[...,None],upper[...,None],
                mean[...,None]+sd*normal_offsets,
                tf.broadcast_to(obs_breaks,tf.concat([tf.shape(mean),[26]],0)),
                tf.broadcast_to((-h/(2*b))[None],tf.concat([tf.shape(mean),[1]],0))],-1)
            breaks = tf.sort(tf.minimum(tf.maximum(breaks,lower[...,None]),upper[...,None]),axis=-1)
            half = (breaks[...,1:]-breaks[...,:-1])/2
            middle = (breaks[...,1:]+breaks[...,:-1])/2
            shape = tf.concat([tf.shape(mean),[-1]],0)
            return (tf.reshape(middle[...,None]+half[...,None]*nodes,shape),
                    tf.reshape(half[...,None]*weights,shape))

        outer_sd = tf.sqrt((a+tf.abs(c))**2*sp**2+sq**2)
        x1,w1 = partition(f(m-10*sp)-10*sq,f(m+10*sp)+10*sq,f(m),outer_sd,observations[0,0])

        def initial_convolution():
            initial = m+sp*gh
            p1 = tf.reduce_sum(gw[None,:]*density(x1[:,None],f(initial)[None,:],sq),-1)
            return p1,tf.constant(0.,DTYPE)

        def inverse_convolution():
            u = x1[:,None]-sq*gh[None,:]
            inverse = u/a
            # Fixed small root solver, unrolled within one stable graph. The
            # differentiated while-loop under tf.cond produces DT_VARIANT
            # FakeParam on this XLA version; unrolling retains the same finite
            # iterations and their total derivative without a backend fallback.
            for _ in range(16):
                inverse -= (f(inverse)-u)/(a+c*tf.cos(inverse))
            error = tf.reduce_max(tf.abs(f(inverse)-u)/(1+tf.abs(u)))
            p1 = tf.reduce_sum(gw[None,:]*density(inverse,m,sp)/(a+c*tf.cos(inverse)),-1)
            return p1,error

        use_initial = sq>(a+tf.abs(c))*sp
        if convolution == 'auto':
            p1,error = tf.cond(use_initial,initial_convolution,inverse_convolution)
        else:
            p1,error = initial_convolution() if convolution == 'initial' else inverse_convolution()
            use_initial = tf.constant(convolution == 'initial')
        mean2 = f(x1)
        x2,w2 = partition(mean2-10*sq,mean2+10*sq,mean2,sq,observations[1,0])
        inner_weights = w2*density(x2,mean2[:,None],sq)*density(observations[1,0],h*x2+b*x2*x2,sr)
        outer_weights = w1*p1*density(observations[0,0],h*x1+b*x1*x1,sr)
        likelihood = tf.reduce_sum(outer_weights*tf.reduce_sum(inner_weights,-1))
        if value_only:
            thresholds = tf.constant([.5,2.],DTYPE)
            cdf = .5*tf.math.erfc(((h*x2+b*x2*x2)[None,:,:]-thresholds[:,None,None])/
                                  (tf.sqrt(tf.constant(2.,DTYPE))*sr))
            predictive = tf.reduce_sum(cdf*inner_weights[None,:,:]*outer_weights[None,:,None],axis=[1,2])
            return likelihood,predictive,error,use_initial
        return tf.math.log(likelihood),error,use_initial

    @tf.function(input_signature=[tf.TensorSpec([2],DTYPE),tf.TensorSpec([2,1],DTYPE)],
                 jit_compile=True,autograph=False)
    def evaluate(scales,observations):
        if value_only:
            likelihood,predictive,residual,initial = compute(scales,observations)
            valid = (tf.math.is_finite(likelihood) & (likelihood>=0)
                & predictive_integrals_valid(likelihood,predictive) & (residual<=1e-11))
            return dict(likelihood=likelihood,predictive_integrals=predictive,
                        inverse_relative_residual=residual,initial_convolution=initial,valid=valid)
        with tf.GradientTape() as tape:
            tape.watch(scales)
            value,residual,initial = compute(scales,observations)
        score = tape.gradient(value,scales)
        valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score)) & (residual<=1e-11)
        return dict(log_likelihood=value,score=score,inverse_relative_residual=residual,
                    initial_convolution=initial,valid=valid)
    return evaluate
