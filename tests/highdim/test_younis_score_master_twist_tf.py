"""Normalizer telescoping, terminal/initial factors and full finite derivatives."""
import tensorflow as tf
from bayesfilter.score_study.twist_tf import make_twist_kernel
from bayesfilter.score_study.gaussian_tf import make_particle_kernel, make_gaussian_kernel, parameterized_model
from tests.highdim.test_younis_score_master_combinations_tf import normal_quadrature


def test_finite_discrete_twisted_path_measure_equals_original():
    mu=tf.constant([.2,.3,.5],tf.float64)
    f=tf.constant([[.4,.5,.1],[.1,.2,.7],[.3,.3,.4]],tf.float64)
    g=tf.constant([[.3,.8,.6],[.7,.2,.5]],tf.float64)
    psi=tf.constant([[1.2,.6,.4],[.2,.5,.8]],tf.float64)
    fpsi=f@psi[1,:,None]
    z0=tf.reduce_sum(mu*psi[0])
    twisted_mu=mu*psi[0]/z0
    twisted_f=f*psi[1,None,:]/fpsi
    g1=g[0]*fpsi[:,0]*z0/psi[0]
    g2=g[1]/psi[1]
    original=mu[:,None]*g[0,:,None]*f*g[1,None,:]
    twisted=twisted_mu[:,None]*g1[:,None]*twisted_f*g2[None,:]
    tf.debugging.assert_near(original,twisted,atol=1e-15)
    tf.debugging.assert_near(tf.reduce_sum(twisted_f,axis=1),tf.ones([3],tf.float64))


def test_constant_twist_recovers_bootstrap_and_all_six_derivatives():
    dtype=tf.float64
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype)
    y=tf.constant([[.5],[-.2]],dtype)
    initial=tf.random.stateless_normal([8,2],[7,1],dtype=dtype)
    noise=tf.random.stateless_normal([2,8,2],[7,2],dtype=dtype)
    u=tf.random.stateless_uniform([3,8],[7,3],dtype=dtype)
    args=(y,initial,noise,u)
    constant=make_twist_kernel(2,1,8,2,0.)
    baseline=make_particle_kernel(2,1,8,2,resampling=True)
    a=constant(theta,*args);b=baseline(theta,y,initial,noise,u[1:])
    tf.debugging.assert_near(a[0],b[0],atol=1e-12)
    tf.debugging.assert_near(a[1],b[1],atol=1e-12)
    kernel=make_twist_kernel(2,1,8,2,.6)
    value,score,ess=kernel(theta,*args)
    differences=[]
    h=tf.constant(1e-6,dtype)
    for direction in tf.unstack(tf.eye(6,dtype=dtype)):
        differences.append((kernel(theta+h*direction,*args)[0]-kernel(theta-h*direction,*args)[0])/(2*h))
    tf.debugging.assert_near(score,tf.stack(differences),atol=1e-6,rtol=1e-5)
    assert float(ess)>0 and bool(tf.math.is_finite(value))
    assert kernel.experimental_get_tracing_count()==1


def test_one_step_fully_adapted_initial_normalizer_integrates_to_exact_likelihood():
    dtype=tf.float64
    theta=tf.constant([.62,-.8,-.6,.9,.25,-.3],dtype)
    y=tf.constant([[.5]],dtype)
    kernel=make_twist_kernel(1,1,1,1,1.)
    nodes,weights=normal_quadrature(40)
    values=[]
    for node in tf.unstack(nodes):
        result=kernel(theta,y,tf.reshape(node,[1,1]),tf.zeros([1,1,1],dtype),tf.fill([2,1],tf.constant(.5,dtype)))
        values.append(tf.exp(result[0]))
    oracle=make_gaussian_kernel(1,1,6)(y,*parameterized_model(theta,1,1))[0]
    tf.debugging.assert_near(tf.reduce_sum(weights*tf.stack(values)),tf.exp(oracle),atol=1e-12)
