"""CPU diagnostic checks of the optional shared reset/protection extension."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import pytest
import tensorflow as tf
from bayesfilter.highdim.higher_moment_contract_e import (
    identity_core_standardized_cap, higher_moment_shape_jvp,
)
from bayesfilter.highdim.ledh_unified_correction_tf import batched_higher_moment_shape_jvp
from bayesfilter.highdim.sqmc_campaign_tf import reset_design

D = tf.float64

@pytest.mark.parametrize('dimension', [1, 3, 10])
def test_quantile_coverage_and_whitening(dimension):
    n = 120
    points = reset_design(n, dimension, D, 'normal_quantiles')
    tf.debugging.assert_near(tf.reduce_mean(points, 0), tf.zeros([dimension], D), atol=2e-14)
    tf.debugging.assert_near(tf.transpose(points) @ points / n, tf.eye(dimension, dtype=D), atol=2e-13)
    assert int(tf.size(tf.unique(points[:, 0])[0])) == n
    reverse = reset_design(n, dimension, D, 'normal_quantiles_reversed')
    tf.debugging.assert_equal(reverse, tf.reverse(points, [0]))

def test_identity_core_bound_and_derivative():
    z = tf.constant([-100., -8.1, -8., -7., 0., 7., 8., 8.1, 100.], D)
    y, dy = identity_core_standardized_cap(z, radius=8., tail_fraction=.98)
    inside = tf.abs(z) <= 8.
    tf.debugging.assert_equal(tf.boolean_mask(y, inside), tf.boolean_mask(z, inside))
    tf.debugging.assert_equal(tf.boolean_mask(dy, inside), tf.ones([5], D))
    assert float(tf.reduce_max(tf.abs(y))) <= 8. * 1.98
    h = tf.constant(1e-5, D)
    plus = identity_core_standardized_cap(z+h, radius=8., tail_fraction=.98)[0]
    minus = identity_core_standardized_cap(z-h, radius=8., tail_fraction=.98)[0]
    tf.debugging.assert_near(dy, (plus-minus)/(2*h), atol=2e-9)

def inputs():
    source = tf.random.stateless_normal([24, 2], [17, 31], dtype=D)
    points = reset_design(24, 2, D, 'normal_quantiles')
    w = tf.nn.softmax(tf.linspace(tf.constant(-.3, D), tf.constant(.3, D), 24))
    ds = tf.random.stateless_normal([24, 2, 1], [19, 32], dtype=D)*.03
    dp = tf.random.stateless_normal([24, 2, 1], [29, 32], dtype=D)*.02
    dw = tf.linspace(tf.constant(-.002, D), tf.constant(.002, D), 24)[:, None]
    return source, w, ds, dw, points, dp

@pytest.mark.parametrize('radius', [0., 1., 8.])
def test_shared_batch_stages_and_total_derivative(radius):
    args = inputs()
    config = dict(correction_steps=2, strength=.12, diagonal_lm_damping=.01,
                  diagonal_trust_radius=.5, pairwise_correction_steps=1,
                  pairwise_strength=.03, pairwise_particle_rms_cap=2.,
                  coordinatewise_standardized_cap=.98,
                  coordinatewise_standardized_identity_radius=radius, return_stages=True)
    expected = higher_moment_shape_jvp(*args, **config)
    batch_args = (args[0][None], args[1][None], tf.transpose(args[2], [2,0,1])[:,None],
                  tf.transpose(args[3])[...,None,:], args[4][None], tf.transpose(args[5], [2,0,1])[:,None])
    actual = batched_higher_moment_shape_jvp(*batch_args, **config)
    assert bool(expected['valid']) and bool(actual['valid'][0])
    for key in ('particles', 'stage_raw_reset', 'stage_pre_cap', 'stage_post_cap'):
        tf.debugging.assert_near(actual[key][0], expected[key], atol=2e-12)
    h = tf.constant(1e-5, D)
    s,w,ds,dw,p,dp = args
    def shifted(sign):
        return higher_moment_shape_jvp(s+sign*h*ds[:,:,0], w+sign*h*dw[:,0], ds,dw,
                                      p+sign*h*dp[:,:,0],dp, **config)
    plus, minus = shifted(1.), shifted(-1.)
    for key in ('particles', 'stage_raw_reset', 'stage_pre_cap', 'stage_post_cap'):
        tangent_key = 'particles_tangent' if key == 'particles' else key+'_tangent'
        tf.debugging.assert_near(expected[tangent_key][:,:,0], (plus[key]-minus[key])/(2*h), atol=3e-7, rtol=3e-6)

def test_default_unchanged_when_optional_fields_are_explicit():
    args=inputs()
    config=dict(correction_steps=1, strength=.12, coordinatewise_standardized_cap=.98)
    a=higher_moment_shape_jvp(*args, **config)
    b=higher_moment_shape_jvp(*args, **config, coordinatewise_standardized_identity_radius=0., return_stages=False)
    for key in a:
        tf.debugging.assert_equal(a[key], b[key])

def test_healthy_complete_map_matches_uncapped_moment_correction():
    args=inputs()
    config=dict(correction_steps=2, strength=.12, diagonal_lm_damping=.01,
                diagonal_trust_radius=.5, pairwise_correction_steps=1,
                pairwise_strength=.03, pairwise_particle_rms_cap=2.)
    intended=higher_moment_shape_jvp(*args, **config)
    protected=higher_moment_shape_jvp(*args, **config,
        coordinatewise_standardized_cap=.98, coordinatewise_standardized_identity_radius=8.)
    assert bool(intended['valid']) and bool(protected['valid'])
    assert float(protected['fraction_coordinatewise_cap_active']) == 0.
    # Covariance restoration introduces rounding; the cap is exactly identity.
    for key in ('particles','particles_tangent'):
        tf.debugging.assert_near(protected[key],intended[key],atol=2e-12,rtol=2e-12)

def test_zero_iteration_batch_trace_preserves_existing_noop():
    args=inputs()
    config=dict(correction_steps=0,strength=.12,pairwise_correction_steps=0,return_stages=True)
    expected=higher_moment_shape_jvp(*args,**config)
    batch_args=(args[0][None],args[1][None],tf.transpose(args[2],[2,0,1])[:,None],
                tf.transpose(args[3])[...,None,:],args[4][None],tf.transpose(args[5],[2,0,1])[:,None])
    actual=batched_higher_moment_shape_jvp(*batch_args,**config)
    for key in ('particles','stage_raw_reset','stage_pre_cap','stage_post_cap'):
        tf.debugging.assert_equal(expected[key],args[4])
        tf.debugging.assert_equal(actual[key][0],args[4])
        tangent_key='particles_tangent' if key=='particles' else key+'_tangent'
        tf.debugging.assert_equal(expected[tangent_key],args[5])
