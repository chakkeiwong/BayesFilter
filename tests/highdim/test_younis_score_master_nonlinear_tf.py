"""Independent reference, refinement and nonlinear consumer identities."""
import pytest
import tensorflow as tf
from bayesfilter.score_study.nonlinear_tf import make_grid_reference,make_data_kernel,make_ledh_kernel,make_moment_filter,make_particle_filter
from bayesfilter.score_study.gaussian_tf import parameterized_model,make_gaussian_kernel,make_particle_kernel
from test_younis_score_master_canonical_tf import CONTROLS

THETA = [.62,-.8,-.6,.9,.25,-.3]


def test_grid_affine_limit_includes_all_initial_and_noise_tangents():
    theta=tf.constant(THETA,tf.float64)
    obs=tf.constant([[.5],[-.3],[.9]],tf.float64)
    grid=make_grid_reference(3,401,9.,0.,0.)(theta,obs)
    exact=make_gaussian_kernel(1,1,6)(obs,*parameterized_model(theta,1,1))
    for got,want in zip(grid[:4],(exact[0],exact[1],exact[2][0],exact[3][0,0])):
        tf.debugging.assert_near(got,want,atol=2e-10,rtol=2e-10)
    assert float(grid[4])<1e-12
    assert float(grid[5])<1e-12


def test_nonlinear_grid_refinement_and_total_derivative():
    theta=tf.constant(THETA,tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
    obs=make_data_kernel(3,.25,.15)(theta,tf.constant([197,31],tf.int32))
    kernel=make_grid_reference(3,601,10.,.25,.15)
    result=kernel(theta,obs)
    for points,radius in ((301,10.),(721,12.)):
        refined=make_grid_reference(3,points,radius,.25,.15)(theta,obs)
        for got,want in zip(result[:4],refined[:4]):
            tf.debugging.assert_near(got,want,atol=1e-8,rtol=1e-8)
    h=tf.constant(2e-5,tf.float64)
    fd=(kernel(theta+h*direction,obs)[0]-kernel(theta-h*direction,obs)[0])/(2*h)
    tf.debugging.assert_near(tf.reduce_sum(result[1]*direction),fd,atol=2e-8,rtol=2e-7)
    assert float(result[4])<1e-10
    assert float(result[5])<1e-10
    assert kernel.experimental_get_tracing_count()==1


@pytest.mark.parametrize("provider,control",[("ledh",1.),("kdm_covariance",.5),("sgqf",2.)])
def test_nonlinear_shared_consumer_total_derivative(provider,control):
    theta=tf.constant(THETA,tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
    obs=tf.constant([[.5],[-.3]],tf.float64)
    initial=tf.random.stateless_normal([8,1],[17,1],dtype=tf.float64)
    noise=tf.random.stateless_normal([2,8,1],[17,2],dtype=tf.float64)
    design=tf.random.stateless_normal([8,1],[17,3],dtype=tf.float64)
    kernel=make_ledh_kernel(8,2,tuple(sorted(CONTROLS.items())),.2,.12,provider,control)
    value,score=kernel(theta,direction,obs,initial,noise,design)
    h=tf.constant(2e-5,tf.float64)
    plus=kernel(theta+h*direction,direction,obs,initial,noise,design)[0]
    minus=kernel(theta-h*direction,direction,obs,initial,noise,design)[0]
    tf.debugging.assert_all_finite(value,"value")
    tf.debugging.assert_near(score,(plus-minus)/(2*h),rtol=3e-5,atol=3e-7)
    assert kernel.experimental_get_tracing_count()==1


@pytest.mark.parametrize("method",["ekf","ukf"])
def test_moment_baseline_affine_and_nonlinear_identity(method):
    theta=tf.constant(THETA,tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
    obs=tf.constant([[.5],[-.3],[.9]],tf.float64)
    ref=make_gaussian_kernel(1,1,6)(obs,*parameterized_model(theta,1,1))
    result=make_moment_filter(3,0.,0.,method)(theta,direction,obs)
    tf.debugging.assert_near(result[0],ref[0],atol=2e-11)
    tf.debugging.assert_near(result[1],tf.reduce_sum(direction*ref[1]),atol=2e-11)
    kernel=make_moment_filter(3,.2,.12,method)
    result=kernel(theta,direction,obs);h=tf.constant(2e-5,tf.float64)
    fd=(kernel(theta+h*direction,direction,obs)[0]-kernel(theta-h*direction,direction,obs)[0])/(2*h)
    tf.debugging.assert_near(result[1],fd,atol=2e-8,rtol=2e-7)


@pytest.mark.parametrize("adapted",[False,True])
def test_corrected_particle_affine_and_nonlinear_identity(adapted):
    theta=tf.constant(THETA,tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
    obs=tf.constant([[.5],[-.3],[.9]],tf.float64)
    initial=tf.random.stateless_normal([16,1],[19,1],dtype=tf.float64)
    noise=tf.random.stateless_normal([3,16,1],[19,2],dtype=tf.float64)
    uniforms=tf.random.stateless_uniform([3,16],[19,3],dtype=tf.float64)
    for resampling in (False,True):
        ref_kernel=make_particle_kernel(1,1,16,3,adapted=adapted,resampling=resampling)
        args=[theta,obs,initial,noise]+([uniforms] if resampling else [])
        ref=ref_kernel(*args)
        affine=make_particle_filter(16,3,0.,0.,adapted,resampling)(theta,obs,initial,noise,uniforms)
        for got,want in zip(affine,ref):tf.debugging.assert_near(got,want,atol=2e-11)
        kernel=make_particle_filter(16,3,.2,.12,adapted,resampling)
        result=kernel(theta,obs,initial,noise,uniforms);h=tf.constant(1e-6,tf.float64)
        fd=(kernel(theta+h*direction,obs,initial,noise,uniforms)[0]-kernel(theta-h*direction,obs,initial,noise,uniforms)[0])/(2*h)
        tf.debugging.assert_near(tf.reduce_sum(result[1]*direction),fd,atol=2e-7,rtol=2e-6)
