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
                                      "dispersed Gaussian scale starts", 2000, 1000, (20260919,400), 1000.)


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


def test_endpoint_rounding_flag_reaches_tensor_trace_without_changing_samples():
    from bayesfilter.inference.neural_force_hmc import (
        FrozenPositionOnlyForce, FrozenTargetPotential, TargetPotentialEvaluation,
        NeuralForceTransitionKernel, NeuralForceHMCConfig)
    from bayesfilter.inference.hmc_tensorflow_tuning import _run_tensor_steps, ENDPOINT_STATUS_TRACE_FIELDS
    force=FrozenPositionOnlyForce(lambda q:q,identity="rounding-test-force")
    def value(q):
        return .5*tf.reduce_sum(q*q,axis=-1)
    cfg=NeuralForceHMCConfig(step_size=.1,num_leapfrog_steps=2,inverse_mass_diagonal=(1.,))
    traces=[]
    for flagged in (False,True):
        target=FrozenTargetPotential((lambda q:TargetPotentialEvaluation(value(q),
            {"coordinate_boundary_rounding":tf.ones([4],tf.bool)})) if flagged else value,
            identity="rounding-test-endpoint")
        kernel=NeuralForceTransitionKernel(force=force,target=target,config=cfg)
        traces.append(_run_tensor_steps(kernel=kernel,initial_state=tf.zeros([4,1],tf.float64),
            num_results=8,seed=tf.constant([17,34]),adaptive=False))
    tf.debugging.assert_equal(traces[0].states,traces[1].states)
    index=ENDPOINT_STATUS_TRACE_FIELDS.index("coordinate_boundary_rounding")
    tf.debugging.assert_equal(traces[0].endpoint_status[...,index],tf.zeros([8,4],tf.int32))
    tf.debugging.assert_equal(traces[1].endpoint_status[...,index],tf.ones([8,4],tf.int32))


def test_retained_compilation_reuse_keeps_state_and_seed_as_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bayesfilter.inference import hmc_tensorflow_verified as verified
    artifact = tmp_path/"manifest.json"
    artifact.write_text('{"frozen": 1}')
    tuning = SimpleNamespace(artifact_manifest_path=str(artifact), nomination=object())
    calls = []

    def factory(nomination, binding, count):
        calls.append(count)
        @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([2], tf.int32)],
                     autograph=False, jit_compile=True)
        def graph(state, seed):
            return state+tf.random.stateless_normal([4, 2], seed, dtype=tf.float64)
        return graph

    monkeypatch.setattr(verified, "_chunk_graph", factory)
    runner = verified.VerifiedTensorFlowHMCRunner(tuning, object())
    first = runner._compiled_chunk_graph(tuning, 250)
    seed = tf.constant([11, 43], tf.int32)
    output = first(tf.zeros([4, 2], tf.float64), seed)
    second = runner._compiled_chunk_graph(tuning, 250)
    tf.debugging.assert_near(second(tf.ones([4, 2], tf.float64), seed), output+1.)
    assert first is second and first.experimental_get_tracing_count() == 1 and calls == [250]
    assert not bool(tf.reduce_all(second(tf.zeros([4, 2], tf.float64), seed+1) == output))
    runner._compiled_chunk_graph(tuning, 500)
    artifact.write_text('{"frozen": 2}')
    assert runner._compiled_chunk_graph(tuning, 250) is not first
    assert calls == [250, 500, 250]


def test_cached_actual_kernel_matches_fresh_graph_after_continuation(tmp_path):
    from types import SimpleNamespace
    from bayesfilter.inference import hmc_tensorflow_verified as verified
    from bayesfilter.inference.neural_force_hmc import (
        FrozenPositionOnlyForce, FrozenTargetPotential, bind_neural_force_hmc_tuning_runner)
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(lambda q: .9*q, identity="cache-parity-surrogate"),
        target=FrozenTargetPotential(lambda q: .5*tf.reduce_sum(q*q, -1), identity="cache-parity-endpoint"),
        target_scope="verified-test")
    nominated = SimpleNamespace(config=nomination(), center=tf.zeros([2], tf.float64),
        factor=tf.eye(2, dtype=tf.float64), adapter_signature="cache-parity",
        step_size=tf.constant(.2, tf.float64), num_leapfrog_steps=tf.constant(2))
    path = tmp_path/"frozen.json"
    path.write_text('{"mechanics_only": true}')
    tuning = SimpleNamespace(artifact_manifest_path=str(path), nomination=nominated)
    runner = verified.VerifiedTensorFlowHMCRunner(tuning, binding)
    graph = runner._compiled_chunk_graph(tuning, 8)
    first = graph(tf.zeros([4, 2], tf.float64), tf.constant([27, 40]))
    seed = tf.constant([27, 41])
    continued = runner._compiled_chunk_graph(tuning, 8)(first.final_state, seed)
    fresh = verified._chunk_graph(nominated, binding, 8)(first.final_state, seed)
    for actual, expected in zip(continued, fresh):
        tf.debugging.assert_equal(actual, expected)
    assert graph.experimental_get_tracing_count() == 1
