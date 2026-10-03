"""Independent identities and actual LEDH consumption of mixture covariance."""
import tensorflow as tf
import pytest
from bayesfilter.score_study.mixture_covariance_tf import (
    gaussian_mixture_initial,mixture_moments,mixture_schedule,make_mixture_covariance_kernel)
from bayesfilter.score_study.canonical_adapter_tf import gaussian_direction_inputs
from bayesfilter.score_study.gaussian_tf import parameterized_model,make_gaussian_kernel
from test_younis_score_master_canonical_tf import CONTROLS


THETA = [.62,-.8,-.6,.9,.25,-.3]


@pytest.mark.parametrize("stage", ["predict", "update"])
def test_shared_invalidity_reaches_actual_schedule_consumer(monkeypatch, stage):
    from bayesfilter.score_study import mixture_covariance_tf as module
    name = "quadrature_"+stage+"_with_parameter_tangent"
    original = getattr(module, name)

    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        return (*result[:4], tf.constant(False), *result[5:])

    monkeypatch.setattr(module, name, invalid)
    make_mixture_covariance_kernel.cache_clear()
    theta = tf.constant(THETA, tf.float64)
    kernel = make_mixture_covariance_kernel(2, 1, 8, 1, tuple(sorted(CONTROLS.items())), .5)
    result = kernel(theta, tf.ones([6], tf.float64), tf.constant([[.5]], tf.float64),
        tf.random.stateless_normal([8, 2], [82, 1], dtype=tf.float64),
        tf.random.stateless_normal([1, 8, 2], [82, 2], dtype=tf.float64),
        tf.random.stateless_normal([8, 2], [82, 3], dtype=tf.float64))
    assert not bool(result[2]["post_valid"][0])
    assert float(result[0]) == float("-inf")
    # The shared consumer's invalid-target contract is (-inf, zero score).
    assert float(result[1]) == 0.
    make_mixture_covariance_kernel.cache_clear()


def test_initial_mixture_matches_mean_covariance_and_total_tangents():
    theta = tf.constant(THETA,tf.float64)
    direction = tf.constant([.1,.2,.3,.4,.5,.6],tf.float64)
    components = gaussian_mixture_initial(theta,direction,2,1,.5)
    got = mixture_moments(*components,tf.zeros([4],tf.float64),tf.zeros([4],tf.float64))
    _,_,_,_,m,dm,P,dP,*_ = parameterized_model(theta,2,1)
    for a,b in zip(got,(m,P,tf.tensordot(direction,dm,1),tf.tensordot(direction,dP,1))):
        tf.debugging.assert_near(a,b,atol=2e-12,rtol=2e-12)


def test_identical_components_recover_kalman_and_conditioned_components_persist():
    theta = tf.constant(THETA,tf.float64); direction=tf.ones([6],tf.float64)
    obs=tf.constant([[.5],[-.3],[.9]],tf.float64)
    model,*_=gaussian_direction_inputs(theta,direction,tf.zeros([8,2],tf.float64),2,1)
    exact=mixture_schedule(model,theta,obs,*gaussian_mixture_initial(theta,direction,2,1,1.))
    kf=make_gaussian_kernel(2,1,6)
    for t in range(3):
        ref=kf(obs[:t+1],*parameterized_model(theta,2,1))
        tf.debugging.assert_near(exact["post_means"][t],ref[2],atol=2e-11)
        tf.debugging.assert_near(exact["post_covariances"][t],ref[3],atol=2e-11)
    mixed=mixture_schedule(model,theta,obs,*gaussian_mixture_initial(theta,direction,2,1,.5))
    changed=mixture_schedule(model,theta,-obs,*gaussian_mixture_initial(theta,direction,2,1,.5))
    assert float(tf.reduce_max(tf.abs(mixed["component_log_weights"]-changed["component_log_weights"])))>.01
    A,*_=parameterized_model(theta,2,1)
    predicted=tf.einsum("ij,nj->ni",A,mixed["component_means"][0])
    aggregate=tf.einsum("n,ni->i",tf.nn.softmax(mixed["component_log_weights"][0]),predicted)
    tf.debugging.assert_near(mixed["predicted_means"][1],aggregate,atol=2e-11)


def test_actual_ledh_consumes_schedule_and_total_derivative():
    theta=tf.constant(THETA,tf.float64);direction=tf.constant([.1,.2,-.3,.4,.5,-.2],tf.float64)
    obs=tf.constant([[.5],[-.3]],tf.float64)
    initial=tf.random.stateless_normal([8,2],[12,1],dtype=tf.float64)
    noise=tf.random.stateless_normal([2,8,2],[12,2],dtype=tf.float64)
    design=tf.random.stateless_normal([8,2],[12,3],dtype=tf.float64)
    kernel=make_mixture_covariance_kernel(2,1,8,2,tuple(sorted(CONTROLS.items())),.5)
    result=kernel(theta,direction,obs,initial,noise,design)
    for t,cov in enumerate(result[3]):
        tf.debugging.assert_near(cov,tf.broadcast_to(result[2]["predicted_covariances"][t],[8,2,2]),atol=2e-12)
    h=tf.constant(2e-5,tf.float64)
    plus=kernel(theta+h*direction,direction,obs,initial,noise,design)
    minus=kernel(theta-h*direction,direction,obs,initial,noise,design)
    tf.debugging.assert_near(result[1],(plus[0]-minus[0])/(2*h),rtol=2e-5,atol=2e-7)
    tf.debugging.assert_near(result[2]["post_d_covariances"],
        (plus[2]["post_covariances"]-minus[2]["post_covariances"])/(2*h),rtol=2e-5,atol=2e-8)
    assert kernel.experimental_get_tracing_count()==1
