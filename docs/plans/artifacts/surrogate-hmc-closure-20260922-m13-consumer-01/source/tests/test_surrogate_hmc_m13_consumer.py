"""Independent M13 callback identities and executable canonical wiring."""
from dataclasses import replace
from types import MappingProxyType
import math

import pytest
import tensorflow as tf

import bayesfilter.highdim.ledh_range_bearing_target_tf as target_module
from bayesfilter.highdim.ledh_range_bearing_target_tf import (
    make_range_bearing_model, make_range_bearing_target, polar_observation,
    polar_observation_tangent, polar_jacobian, polar_jacobian_tangent)
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf


def test_polar_geometry_total_directional_derivatives():
    x = tf.constant([[1.2,.7,.18,-.06],[-.8,.01,.2,.3],[.001,-.002,0.,0.]],tf.float64)
    dx = tf.constant([[.1,-.2,.3,.4],[.4,.2,.1,-.1],[.0002,-.0001,.1,.1]],tf.float64)
    h = tf.constant(1e-6,tf.float64)
    tf.debugging.assert_near(polar_observation_tangent(x,dx),
        (polar_observation(x+h*dx)-polar_observation(x-h*dx))/(2*h),atol=2e-9,rtol=2e-9)
    tf.debugging.assert_near(polar_observation_tangent(x,dx),
        tf.einsum("noi,ni->no",polar_jacobian(x),dx),atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(polar_jacobian_tangent(x,dx),
        (polar_jacobian(x+h*dx)-polar_jacobian(x-h*dx))/(2*h),atol=1e-7,rtol=1e-8)
    assert not bool(tf.reduce_all(tf.math.is_finite(polar_jacobian(tf.zeros([1,4],tf.float64)))))


def test_all_parameter_rows_and_residual_covariance_terms_reach_shared_density(monkeypatch):
    calls = []
    original = target_module.wrapped_gaussian_batch_kernel
    def tracked_factory(**kwargs):
        kernel = original(**kwargs)
        def tracked(*args):
            calls.append(args[1])
            return kernel(*args)
        return tracked
    monkeypatch.setattr(target_module,"wrapped_gaussian_batch_kernel",tracked_factory)
    fixture = build_range_bearing_fixture_tf()
    model = make_range_bearing_model(fixture.A,fixture.Q,fixture.R,dtype=tf.float64)
    x = tf.constant([[1.2,.7,.18,-.06],[1.3,.8,.1,.0],[1.1,.6,.1,-.1]],tf.float64)
    dx = tf.constant([[.01,-.02,.0,.0],[-.03,.01,.0,.0],[.02,.03,.0,.0]],tf.float64)
    theta = tf.constant([[.2,-.4],[-.3,.2],[.5,.6]],tf.float64)
    direction = tf.constant([[.3,.2],[-.4,-.3],[.1,.5]],tf.float64)
    obs = fixture.observations[0]
    value = model.observation_log_density_fn(theta,x,obs)
    tangent = model.observation_log_density_tangent_fn(theta,x,obs,dx,direction)
    h = tf.constant(1e-6,tf.float64)
    plus = model.observation_log_density_fn(theta+h*direction,x+h*dx,obs)
    minus = model.observation_log_density_fn(theta-h*direction,x-h*dx,obs)
    tf.debugging.assert_near(tangent,(plus-minus)/(2*h),atol=1e-7,rtol=1e-7)
    # Independent diagonal normal law at these interior points (other aliases negligible).
    residual = obs-polar_observation(x)
    sigma = tf.sqrt(tf.linalg.diag_part(fixture.R))[None]*tf.exp(theta[:,1,None])
    oracle = tf.reduce_sum(-.5*(residual/sigma)**2-tf.math.log(sigma)-.5*math.log(2*math.pi),-1)
    tf.debugging.assert_near(value,oracle,atol=2e-12,rtol=2e-12)
    assert len(calls) == 4
    assert not bool(tf.reduce_all(calls[0][0] == calls[0][1]))
    qdot = model.process_covariance_tangent_fn(theta[0],direction[0])
    qfd = (model.process_covariance_fn(theta[0]+h*direction[0])
           -model.process_covariance_fn(theta[0]-h*direction[0]))/(2*h)
    tf.debugging.assert_near(qdot,qfd,atol=1e-12,rtol=1e-9)


def test_fixed_design_and_likelihood_only_canonical_consumer_wiring(monkeypatch):
    fixture = build_range_bearing_fixture_tf()
    fixture = replace(fixture, observations=fixture.observations[:2], states=fixture.states[:2])
    target = make_range_bearing_target(fixture,particle_count=16,controls=warm_start_controls(),dtype=tf.float64)
    design = target.controls["reset_design"]
    tf.debugging.assert_near(tf.reduce_mean(design,0),tf.zeros([4],tf.float64),atol=2e-8,rtol=0.)
    tf.debugging.assert_near(tf.transpose(design)@design/16,tf.eye(4,dtype=tf.float64),atol=8e-8,rtol=0.)
    calls = []
    def canonical(model,theta,directions,**kwargs):
        calls.append((model,theta,directions,kwargs))
        value = tf.constant([1.25,-2.5],tf.float64)
        score = tf.constant([[3.,4.],[5.,6.]],tf.float64)
        ok = tf.constant([True,True])
        return value,score,dict(value_valid=ok,force_valid=ok,program_valid=ok)
    monkeypatch.setattr(target_module,"canonical_batch_fused_value_score",canonical)
    points = tf.constant([[.2,-.3],[-.1,.4]],tf.float64)
    value,score,status = target.evaluate(points)
    tf.debugging.assert_equal(value,[1.25,-2.5])
    tf.debugging.assert_equal(score,[[3.,4.],[5.,6.]])
    tf.debugging.assert_equal(calls[0][2],tf.broadcast_to(tf.eye(2,dtype=tf.float64),[2,2,2]))
    assert calls[0][0] is target.model
    assert calls[0][3]["reset_policy"] == "contract_e"
    assert calls[0][3]["observations"].shape == [2,2]
    assert target.manifest["prior"] is None
    assert target.manifest["control_status"] == "diagnostic_warm_start_not_scope_tuned"
    assert not bool(tf.reduce_any(status["outside_declared_support"]))


def test_alias_numerical_failure_is_not_zero_density_or_support():
    fixture = build_range_bearing_fixture_tf()
    model = make_range_bearing_model(fixture.A,fixture.Q,fixture.R,dtype=tf.float64)
    x = fixture.m0[None]
    value = model.observation_log_density_fn(tf.constant([[0.,8.]],tf.float64),x,fixture.observations[0])
    assert bool(tf.reduce_all(tf.math.is_nan(value)))


def test_master_has_bounded_consumer_job_and_focused_review():
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs,PASS_CAPS
    selected = jobs("review,m13-consumer")
    assert [r["name"] for r in selected] == ["review","m13-consumer"]
    assert "tests/test_surrogate_hmc_m13_consumer.py" in selected[0]["command"]
    assert selected[1]["device"] == "gpu"
    assert selected[1]["timeout"] <= PASS_CAPS["m13-consumer"]["gpu_seconds"]
    with pytest.raises(ValueError):
        jobs("review,m13-consumer,m13-reference")
