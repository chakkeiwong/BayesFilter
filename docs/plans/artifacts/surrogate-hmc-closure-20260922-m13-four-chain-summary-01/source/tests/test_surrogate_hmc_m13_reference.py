"""Independent M13 reference mechanics; CPU-only focused checks."""
import math
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_m13_reference import (
    guided_particle_filter, intervals_overlap, likelihood_interval, particle_filter, systematic_indices,
)


def test_systematic_resampling_handles_zero_weights_and_batched_clouds():
    weights = tf.constant([[0., .25, .25, .5], [.25, .25, .25, .25],
                           [0., 0., 1., 0.]], tf.float64)
    actual = systematic_indices(weights, tf.constant([.5, 0., .9], tf.float64))
    assert actual.numpy().tolist() == [[1, 2, 3, 3], [0, 1, 2, 3], [2, 2, 2, 2]]


def test_likelihood_mean_is_not_the_mean_log_likelihood():
    interval = likelihood_interval([0., math.log(3.)], .1)
    assert interval["log_mean"] == pytest.approx(math.log(2.), abs=1e-14)
    assert interval["log_lower"] == pytest.approx(math.log(1.9), abs=1e-14)
    assert interval["log_upper"] == pytest.approx(math.log(2.1), abs=1e-14)


def test_nonpositive_mc_interval_and_nonfinite_estimates_cannot_qualify():
    assert not likelihood_interval([0., 10.], 10.)["interval_positive"]
    bad = likelihood_interval([0., math.nan], 2.)
    assert not bad["finite"]
    assert bad["log_mean"] is None
    assert not intervals_overlap(bad, likelihood_interval([0., 0.], 2.))


@pytest.mark.parametrize("method", ["prior_path", "bootstrap", "guided"])
def test_filter_propagates_before_observation_and_preserves_likelihood_normalization(method):
    dtype = tf.float64
    a = tf.constant([[1., 0., .1, 0.], [0., 1., 0., .1],
                     [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype)
    m0 = tf.constant([1.2, .7, 2., -1.], dtype)
    mean = m0
    observations = []
    for _ in range(2):
        mean = tf.linalg.matvec(a, mean)
        x, y = mean[:2].numpy().tolist()
        observations.append([math.sqrt(x*x+y*y+1e-12)+.1, math.atan2(y, x)-.03])
    fixture = SimpleNamespace(A=a, Q=tf.eye(4, dtype=dtype)*1e-24,
        P0=tf.eye(4, dtype=dtype)*1e-24, m0=m0,
        R=tf.linalg.diag(tf.constant([.12**2, .04**2], dtype)),
        observations=tf.constant(observations, dtype))
    evaluate = (guided_particle_filter(fixture, 8, replicates=4) if method == "guided"
                else particle_filter(fixture, 8, replicates=4, resample=method == "bootstrap"))
    result = evaluate(
        tf.zeros([2], dtype), tf.constant([20260923, 41], tf.int32))
    expected = -.5*((.1/.12)**2+(.03/.04)**2)-math.log(2*math.pi*.12*.04)
    assert bool(tf.reduce_all(result["valid"]).numpy())
    assert result["first"].numpy().tolist() == pytest.approx([expected]*4, abs=1e-9)
    assert result["log_likelihood"].numpy().tolist() == pytest.approx([2*expected]*4, abs=1e-9)


@pytest.mark.parametrize("stage", ["m13-reference", "m13-reference-guided", "m13-reference-trajectory"])
def test_master_requires_focused_review_and_charges_m13_reservation(stage):
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs, remaining_pass_reservation
    selected = jobs(stage)
    assert [j["name"] for j in selected] == ["review", stage]
    assert "tests/test_surrogate_hmc_m13_reference.py" in selected[0]["command"]
    receipt = dict(path="/tmp/surrogate-hmc-closure-20260922-m13-reference-01/accounting.json",
                   charged_this_run=dict(cpu_worker_seconds=123., gpu_seconds=45.))
    assert remaining_pass_reservation([receipt], "m13-reference") == dict(cpu_worker_seconds=2277., gpu_seconds=3555.)
    with pytest.raises(ValueError, match="separate"):
        jobs(stage+",wrapped")
    with pytest.raises(ValueError, match="separate"):
        jobs("m13-reference,m13-reference-guided")


def test_trajectory_prior_matches_constant_velocity_cross_covariance():
    from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_prior
    dtype = tf.float64
    dt = .1
    fixture = SimpleNamespace(
        A=tf.constant([[1.,0.,dt,0.],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]], dtype),
        m0=tf.constant([1.2,.7,.18,-.06], dtype),
        P0=tf.linalg.diag(tf.constant([.04,.04,.01,.01], dtype)),
        Q=tf.linalg.diag(tf.constant([.0015,.0015,.0008,.0008], dtype)))
    mean, initial, noise = trajectory_prior(fixture, 3)
    expected_mean = [v for t in range(1,4) for v in (1.2+t*dt*.18,.7-t*dt*.06)]
    assert mean.numpy().tolist() == pytest.approx(expected_mean, abs=1e-15)
    actual = (initial+4.*noise).numpy().tolist()
    for t in range(1,4):
        for s in range(1,4):
            shared = min(t,s)
            expected = (.04+t*s*dt*dt*.01+4.*(.0015*shared
                +.0008*dt*dt*sum((t-i)*(s-i) for i in range(1,shared+1))))
            for axis in range(2):
                assert actual[2*(t-1)+axis][2*(s-1)+axis] == pytest.approx(expected, abs=1e-15)
                assert actual[2*(t-1)+axis][2*(s-1)+1-axis] == 0.


def test_defensive_weight_uses_entire_normalized_mixture_density():
    from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import defensive_log_weights
    dtype = tf.float64
    sample, proposal_mean = [.7,.3], [.2,-.1]
    actual = defensive_log_weights(tf.constant([sample], dtype), tf.zeros([2],dtype),
        tf.eye(2,dtype=dtype), tf.constant(proposal_mean,dtype), 2*tf.eye(2,dtype=dtype),
        tf.constant([.3],dtype))
    logp = -.5*sum(x*x for x in sample)-math.log(2*math.pi)
    logq = (-.5*sum(((x-m)/2)**2 for x,m in zip(sample,proposal_mean))
            -2*math.log(2)-math.log(2*math.pi))
    expected = logp+.3-math.log((15/16)*math.exp(logq)+(1/16)*math.exp(logp))
    assert float(actual[0].numpy()) == pytest.approx(expected, abs=1e-14)
    matched = defensive_log_weights(tf.constant([sample],dtype),tf.zeros([2],dtype),
        tf.eye(2,dtype=dtype),tf.zeros([2],dtype),tf.eye(2,dtype=dtype),tf.constant([.3],dtype))
    assert float(matched[0].numpy()) == pytest.approx(.3,abs=1e-14)


def test_observation_density_broadcasts_over_time_without_swapping_observations():
    from bayesfilter.testing.surrogate_hmc_m13_reference import observation_log_density
    positions = tf.constant([[1.2,.7],[1.4,.8]],tf.float64)
    observations = tf.constant([[1.5,.6],[1.6,.45]],tf.float64)
    sigma = tf.constant([.12,.04],tf.float64)
    actual = observation_log_density(positions,observations,sigma)
    expected = tf.stack([observation_log_density(positions[i],observations[i],sigma) for i in range(2)])
    assert actual.numpy().tolist() == pytest.approx(expected.numpy().tolist(),abs=1e-14)
