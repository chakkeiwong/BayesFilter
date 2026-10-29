"""CPU-only independent numerical fixtures for the diagnostic reference."""
import math
import statistics
from types import SimpleNamespace

import numpy as np  # Independent quadrature/test oracle only.
import pytest
import tensorflow as tf

from bayesfilter.testing.nonlinear_trajectory_fisher_reference_tf import make_trajectory_fisher_kernel


def factory(curvature):
    def build(theta, direction):
        variance = tf.exp(2*theta[1])*tf.eye(1,dtype=tf.float64)
        return SimpleNamespace(
            transition_mean_fn=lambda theta,x: .7*x+curvature*tf.sin(x)+theta[0],
            transition_mean_tangent_fn=lambda theta,x,dx: (.7+curvature*tf.cos(x))*dx+direction[0],
            observation_fn=lambda x:x,
            observation_jacobian_fn=lambda x:tf.ones([tf.shape(x)[0],1,1],tf.float64),
            process_covariance=tf.constant([[.4]],tf.float64),
            observation_covariance=variance,
            process_covariance_tangent_fn=None,
            observation_covariance_tangent_fn=lambda theta:2*variance*direction[1],
            transition_log_density_tangent_fn=None,
            observation_log_density_tangent_fn=None),None
    return build


def quadrature(theta, curvature, order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    nodes=math.sqrt(2)*nodes
    weights=weights/math.sqrt(math.pi)
    x0=nodes[:,None,None]
    x1=.7*x0+curvature*np.sin(x0)+theta[0]+math.sqrt(.4)*nodes[None,:,None]
    x2=.7*x1+curvature*np.sin(x1)+theta[0]+math.sqrt(.4)*nodes[None,None,:]
    variance=math.exp(2*theta[1])
    density=np.exp(-.5*((.4-x1)**2+(-.2-x2)**2)/variance)/(2*math.pi*variance)
    return math.log(np.sum(density*weights[:,None,None]*weights[None,:,None]*weights[None,None,:]))


@pytest.mark.parametrize('curvature',[0.,.35])
def test_trajectory_reference_matches_independent_quadrature(curvature):
    theta=[.3,-.2]
    exact=quadrature(theta,curvature,48)
    assert abs(exact-quadrature(theta,curvature,64))<2e-9
    score=[]
    for k in range(2):
        plus,minus=theta.copy(),theta.copy()
        plus[k]+=1e-5;minus[k]-=1e-5
        score.append((quadrature(plus,curvature,64)-quadrature(minus,curvature,64))/2e-5)
    kernel=make_trajectory_fisher_kernel(factory(curvature),tf.zeros([1],tf.float64),2,1,2,8192,jit_compile=False)
    columns=[[],[],[]]
    for seed in range(700,708):
        result=kernel(tf.constant(theta,tf.float64),tf.constant([[.4],[-.2]],tf.float64),tf.constant(seed))
        assert bool(result['valid'])
        for column,value in zip(columns,[float(result['log_likelihood']),*result['score'].numpy().tolist()]):
            column.append(value)
        if curvature==0:
            assert abs(float(result['log_likelihood'])-exact)<1e-10
            assert abs(float(result['importance_ess'])-8192)<1e-6
            assert abs(float(result['log_weight_min']))<1e-10
            assert abs(float(result['log_weight_max']))<1e-10
    for column,target in zip(columns,[exact,*score]):
        mcse=statistics.stdev(column)/math.sqrt(len(column))
        assert abs(statistics.mean(column)-target)<6*mcse+.002
