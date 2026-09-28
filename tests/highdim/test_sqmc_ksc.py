"""CPU-only independent KSC reference and full callback derivative checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pytest
from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec, grid_reference, enumeration_reference, gaussian_approximation_reference, observation_terms, DTYPE

@pytest.mark.parametrize('horizon', [1,2])
def test_grid_matches_independent_exact_mixture(horizon):
    theta=tf.constant([1.5,-.2],DTYPE)
    observations=tf.constant([[-2.1],[.8]][:horizon],DTYPE)
    actual=grid_reference(801,40.)(theta,observations)
    exact=enumeration_reference(horizon)(theta,observations)
    for a,b in zip(actual,exact):
        tf.debugging.assert_near(a,b,atol=1e-9,rtol=1e-9)
    for i in range(2):
        step=tf.one_hot(i,2,dtype=DTYPE)*1e-5
        fd=(grid_reference(801,40.)(theta+step,observations)[0]-grid_reference(801,40.)(theta-step,observations)[0])/2e-5
        tf.debugging.assert_near(actual[1][i],fd,atol=1e-7,rtol=1e-7)

def test_moment_matched_kalman_is_not_exact_mixture():
    theta=tf.constant([1.5,0.],DTYPE)
    observations=tf.constant([[-4.],[1.]],DTYPE)
    exact=enumeration_reference(2)(theta,observations)
    gaussian=gaussian_approximation_reference(theta,observations)
    assert abs(float(exact[0]-gaussian[0])) > .01
    assert float(tf.linalg.norm(exact[1]-gaussian[1])) > .01

@pytest.mark.parametrize('coordinate',[0,1])
def test_complete_canonical_callbacks_match_total_finite_difference(coordinate):
    spec=KSCSpec()
    theta=spec.default_theta()
    direction=tf.one_hot(coordinate,2,dtype=DTYPE)
    points=tf.constant([[-1.2],[.3],[1.4]],DTYPE)
    dp=tf.constant([[.2],[-.4],[.1]],DTYPE)
    obs=tf.constant([-.7],DTYPE)
    model,_=spec.model(theta,direction)
    h=1e-5
    plus,_=spec.model(theta+h*direction)
    minus,_=spec.model(theta-h*direction)
    comparisons=[
        (model.transition_mean_tangent_fn(theta,points,dp),(plus.transition_mean_fn(theta+h*direction,points+h*dp)-minus.transition_mean_fn(theta-h*direction,points-h*dp))/(2*h)),
        (model.observation_tangent_fn(points,dp),(plus.observation_fn(points+h*dp)-minus.observation_fn(points-h*dp))/(2*h)),
        (model.observation_log_density_tangent_fn(theta,points,obs,dp),(plus.observation_log_density_fn(theta+h*direction,points+h*dp,obs)-minus.observation_log_density_fn(theta-h*direction,points-h*dp,obs))/(2*h)),
    ]
    for a,b in comparisons:
        tf.debugging.assert_near(a,b,atol=1e-8,rtol=1e-7)
    independent,_=observation_terms(theta,points[:,0],obs[0])
    tf.debugging.assert_near(model.observation_log_density_fn(theta,points,obs),independent,atol=1e-12,rtol=1e-12)
