"""Independent M13 reference mechanics; CPU-only focused checks."""
import math
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_m13_reference import (
    intervals_overlap, likelihood_interval, particle_filter, systematic_indices,
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


@pytest.mark.parametrize("resample", [False, True])
def test_filter_propagates_before_observation_and_preserves_likelihood_normalization(resample):
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
    result = particle_filter(fixture, 8, replicates=4, resample=resample)(
        tf.zeros([2], dtype), tf.constant([20260923, 41], tf.int32))
    expected = -.5*((.1/.12)**2+(.03/.04)**2)-math.log(2*math.pi*.12*.04)
    assert bool(tf.reduce_all(result["valid"]).numpy())
    assert result["first"].numpy().tolist() == pytest.approx([expected]*4, abs=1e-9)
    assert result["log_likelihood"].numpy().tolist() == pytest.approx([2*expected]*4, abs=1e-9)


def test_master_requires_focused_review_and_charges_m13_reservation():
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs, remaining_pass_reservation
    selected = jobs("m13-reference")
    assert [j["name"] for j in selected] == ["review", "m13-reference"]
    assert "tests/test_surrogate_hmc_m13_reference.py" in selected[0]["command"]
    receipt = dict(path="/tmp/surrogate-hmc-closure-20260922-m13-reference-01/accounting.json",
                   charged_this_run=dict(cpu_worker_seconds=123., gpu_seconds=45.))
    assert remaining_pass_reservation([receipt], "m13-reference") == dict(cpu_worker_seconds=2277., gpu_seconds=3555.)
    with pytest.raises(ValueError, match="separate"):
        jobs("m13-reference,wrapped")
