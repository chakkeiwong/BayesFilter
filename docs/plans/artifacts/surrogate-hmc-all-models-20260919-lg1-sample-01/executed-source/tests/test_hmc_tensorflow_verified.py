"""Independent references and fail-closed verified-tuner contracts (CPU diagnostic)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from bayesfilter.inference import VerifiedTensorFlowHMCConfig
from bayesfilter.inference.hmc_tensorflow_tuning import (
    TensorFlowHMCKernelTuningConfig, FourChainMeanBandAcceptancePolicy, _TensorChunk)
from bayesfilter.inference.hmc_tensorflow_verified import _health
from bayesfilter.inference.hmc_posterior_diagnostics import _rank_normalize, rank_normalized_split_rhat


def nomination():
    return TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2, evidence_role="diagnostic_candidate_screen",
        mass_window_results=(64,128), step_adaptation_results=128, verification_results=128,
        max_leapfrog_steps=4, initial_step_size=.2, budget_provenance="bounded qualification",
        initial_step_size_provenance="unit Gaussian scale hypothesis",
        geometry_provenance="unit Gaussian warm start", target_scope="verified-test",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
        target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2.,
        mass_shrinkage=.1, covariance_jitter=1e-9, eigenvalue_floor=1e-9,
        max_condition_number=1e8, seed=(20260919,300), use_xla=True)


def config():
    return VerifiedTensorFlowHMCConfig(nomination(), ((-2.,-1.),(-1.,2.),(1.,-2.),(2.,1.)),
                                      "dispersed Gaussian scale starts", 256, 1000, (20260919,400), 1000.)


def test_rank_transform_is_symmetric_and_matches_independent_blom_formula():
    values = np.array([-2.,-1.,-1.,0.,1.,1.,2.]).reshape(7,1,1)
    actual = _rank_normalize(tf.constant(values)).numpy().ravel()
    ranks = stats.rankdata(values.ravel())
    expected = stats.norm.ppf((ranks-.375)/(len(ranks)+.25))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual, -actual[::-1], atol=1e-14)


def test_rhat_rejects_stuck_separated_and_scale_mismatched_chains():
    rng = np.random.default_rng(913)
    for draws in (np.zeros((4,1000,2)), rng.normal(size=(4,1000,2)) + np.arange(4)[:,None,None]*3,
                  rng.normal(size=(4,1000,2))*np.array([.1,.5,2.,8.])[:,None,None]):
        maximum = rank_normalized_split_rhat(tf.constant(draws))["maximum"].numpy()
        assert not np.all(np.isfinite(maximum) & (maximum <= 1.01))


@pytest.mark.parametrize("changes", [
    {"verification_results":4}, {"verification_results":1001},
    {"fresh_seed":(20260919,300)}, {"warmup_results":0},
    {"initial_chain_positions":((0.,0.),)*4}, {"energy_error_max":float("nan")},
    {"nomination":dataclasses.replace(nomination(), use_xla=False)},
])
def test_configuration_does_not_admit_diagnostic_shortcuts(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(config(), **changes)


def test_xla_nomination_roundtrip_remains_diagnostic():
    cfg = config()
    recovered = VerifiedTensorFlowHMCConfig.from_payload(cfg.payload())
    assert recovered == cfg
    assert cfg.nomination.payload()["artifact_authority"] is False
    assert cfg.nomination.payload()["xla_mode"] == "diagnostic_xla"


def test_health_rejects_stuck_chain_and_numerical_endpoint_failure():
    states = tf.ones([4,4,2],tf.float64)
    status = tf.concat([tf.ones([4,4,1],tf.int32),tf.zeros([4,4,9],tf.int32)],axis=-1)
    chunk = _TensorChunk(states, tf.zeros([4,4],tf.float64), tf.zeros([4,4],tf.bool),
        tf.zeros([4,4],tf.bool), tf.zeros([4,4],tf.float64), tf.ones([4,4],tf.bool),
        states[-1], tf.constant(.1,tf.float64), status)
    healthy = _health(chunk, tf.zeros([4,2],tf.float64), 1000.)
    assert all(bool(x) for x in healthy.values())
    assert not bool(_health(chunk, states[0],1000.)["all_chains_moved"])
    bad = chunk._replace(endpoint_status=tf.tensor_scatter_nd_update(status, [[1,2,8]], [1]))
    assert not bool(_health(bad,tf.zeros([4,2],tf.float64),1000.)["endpoint_valid"])
