"""Tiny CPU reference and XLA mechanics; no posterior qualification."""
from dataclasses import replace

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_one_step_transition
from bayesfilter.inference.hmc import (
    FullChainHMCConfig, ReusableFullChainHMCRunner, _trace_fn_for_config,
)
from bayesfilter.inference.hmc_status import cache_hmc_target_status, cached_target_status
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
from tests.test_hmc_candidate_set_execution import GaussianTarget


class StatusGaussian(GaussianTarget):
    def __init__(self, *, invalid_radius=None):
        super().__init__()
        self.status_calls = 0
        self.invalid_radius = invalid_radius

    def target_status_telemetry(self, theta):
        self.status_calls += 1
        squared = tf.reduce_sum(theta * theta, axis=-1)
        valid = tf.math.is_finite(squared)
        if self.invalid_radius is not None:
            valid &= squared < self.invalid_radius**2
        return {"status_code": tf.where(valid, 0, 1),
                "valid_pre_regularized_score": valid,
                "floor_count_value": tf.reduce_sum(tf.cast(theta > 0., tf.int32), axis=-1),
                "min_innovation_eigenvalue": 1. + squared,
                "innovation_condition_estimate": 2. + squared}


def assert_same(left, right):
    tf.nest.assert_same_structure(left, right)
    for actual, expected in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
        tf.debugging.assert_equal(actual, expected)


def config(**overrides):
    result = FullChainHMCConfig(num_results=8, num_burnin_steps=0,
        step_size=.2, num_leapfrog_steps=3, seed=(73, 11), use_xla=False,
        target_status_trace_policy="per_chain_step", capture_candidate_health=True,
        target_scope="candidate-bridge-test")
    return replace(result, **overrides)


@pytest.mark.parametrize("step", [.2, 1.8, 12.])
def test_status_follows_exact_acceptance_mask_without_trace_reevaluation(step):
    target = StatusGaussian()
    raw = tfp.mcmc.HamiltonianMonteCarlo(
        reviewed_value_score_target_fn(target), step_size=tf.constant(step, tf.float64),
        num_leapfrog_steps=3)
    wrapped = cache_hmc_target_status(raw, target.target_status_telemetry)
    state = tf.constant([[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]], tf.float64)
    plain_results = raw.bootstrap_results(state)
    results = wrapped.bootstrap_results(state)
    assert target.status_calls == 1
    trace = _trace_fn_for_config(config(step_size=step), adapter=target)
    rejected = 0
    for index in range(8):
        seed = tf.constant([73, index], tf.int32)
        expected_state, plain_results = raw.one_step(state, plain_results, seed=seed)
        calls = target.status_calls
        next_state, results = wrapped.one_step(state, results, seed=seed)
        assert target.status_calls == calls + 1
        assert_same(next_state, expected_state)
        assert_same(results._replace(extra=[]), plain_results)
        calls = target.status_calls
        traced = trace(next_state, results)
        assert target.status_calls == calls
        assert_same(traced["target_status_telemetry"], target.target_status_telemetry(next_state))
        assert_same(traced["proposed_target_status_telemetry"],
                    target.target_status_telemetry(results.proposed_state))
        rejected += int(tf.reduce_sum(tf.cast(~results.is_accepted, tf.int32)))
        state = next_state
    if step == 12.:
        assert rejected == 32


def test_rejected_invalid_proposals_remain_visible_and_restart_preserves_status():
    target = StatusGaussian(invalid_radius=1.)
    kernel = cache_hmc_target_status(tfp.mcmc.HamiltonianMonteCarlo(
        reviewed_value_score_target_fn(target), step_size=tf.constant(12., tf.float64),
        num_leapfrog_steps=3), target.target_status_telemetry)
    state = tf.zeros([4, 2], tf.float64)
    state, results = kernel.one_step(state, kernel.bootstrap_results(state), seed=(3, 7))
    assert not bool(tf.reduce_any(results.is_accepted))
    assert bool(tf.reduce_all(cached_target_status(results)["status_code"] == 0))
    assert bool(tf.reduce_all(cached_target_status(results, proposed=True)["status_code"] == 1))
    restored = tf.nest.map_structure(
        lambda value: tf.io.parse_tensor(tf.io.serialize_tensor(value), out_type=value.dtype), results)
    copied = kernel.copy()
    assert_same(copied.one_step(state, restored, seed=(3, 8)),
                kernel.one_step(state, results, seed=(3, 8)))


@pytest.mark.parametrize("adaptive,guarded", [(False, False), (True, False), (False, True), (True, True)])
def test_public_reusable_runner_preserves_adaptation_and_finite_guard(monkeypatch, adaptive, guarded):
    import bayesfilter.inference.hmc as hmc
    target = StatusGaussian()
    cfg = config(require_finite_transitions=guarded)
    if adaptive:
        cfg = replace(cfg, num_burnin_steps=4,
            tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                num_adaptation_steps=4, target_accept_prob=.7, source=__file__))
    state = tf.ones([4, 2], tf.float64)
    cached = ReusableFullChainHMCRunner(target, state, cfg).run()
    monkeypatch.setattr(hmc, "cache_hmc_target_status", lambda kernel, *_a, **_kw: kernel)
    uncached = ReusableFullChainHMCRunner(target, state, cfg).run()
    assert_same(cached.samples, uncached.samples)
    assert_same(cached.trace, uncached.trace)
    assert cached.metadata["target_status_reuse"] == "accepted_status_from_tfp_metropolis_mask_v1"


def test_batched_xla_replay_dynamic_inputs_and_status_shape():
    target = StatusGaussian()
    runner = ReusableFullChainHMCRunner(target, tf.zeros([4, 2], tf.float64),
                                      config(use_xla=True), dynamic_num_leapfrog_steps=True)
    first = runner.run()
    assert_same(first.samples, runner.run().samples)
    changed = runner.run(seed=(73, 12), step_size=.3, num_leapfrog_steps=5)
    assert bool(tf.reduce_any(first.samples != changed.samples))
    assert bool(tf.reduce_any(first.samples[:, 0] != first.samples[:, 1]))
    assert first.trace["target_status_telemetry"]["floor_count_value"].shape == (8, 4)
    for name, states in (("target_status_telemetry", first.samples),
                         ("proposed_target_status_telemetry", first.trace["proposed_state"])):
        actual = first.trace[name]
        expected = target.target_status_telemetry(states)
        for field in actual:
            if actual[field].dtype.is_floating:
                tf.debugging.assert_near(actual[field], expected[field], atol=1e-12, rtol=1e-12)
            else:
                tf.debugging.assert_equal(actual[field], expected[field])
    assert runner._runner.experimental_get_tracing_count() == 1
    graph = runner._runner.get_concrete_function()
    assert graph.function_def.attr["_XlaMustCompile"].b
    definition = graph.graph.as_graph_def()
    nodes = list(definition.node) + [node for fn in definition.library.function for node in fn.node_def]
    assert not any("PyFunc" in node.op or "HostCompute" in node.op for node in nodes)


def test_one_step_preserves_bad_proposal_health_and_valid_large_energy():
    for invalid_radius, expected_health in ((None, True), (1., False)):
        primitive = build_fixed_transport_one_step_transition(
            StatusGaussian(invalid_radius=invalid_radius), state_shape=(4, 2),
            step_size=12., num_leapfrog_steps=3, use_xla=True, capture_health=True)
        result = primitive(tf.zeros([4, 2], tf.float64), tf.constant([3, 7], tf.int32))
        assert not bool(tf.reduce_any(result[1]))
        assert bool(result[5]) is expected_health


def test_missing_status_and_nonempty_extra_are_rejected():
    target = StatusGaussian()
    raw = tfp.mcmc.HamiltonianMonteCarlo(reviewed_value_score_target_fn(target),
                                      step_size=.2, num_leapfrog_steps=3)
    state = tf.ones([4, 2], tf.float64)
    wrapped = cache_hmc_target_status(raw, target.target_status_telemetry)
    with pytest.raises(ValueError, match="missing from previous"):
        wrapped.one_step(state, raw.bootstrap_results(state), seed=(3, 7))
    with pytest.raises(ValueError, match="unused TFP extra"):
        cache_hmc_target_status(wrapped, target.target_status_telemetry).bootstrap_results(state)
    with pytest.raises(ValueError, match="incomplete"):
        cache_hmc_target_status(raw, lambda _: {}).bootstrap_results(state)


def test_no_status_route_does_not_evaluate_telemetry():
    target = StatusGaussian()
    result = ReusableFullChainHMCRunner(target, tf.zeros([4, 2], tf.float64),
        config(target_status_trace_policy="none")).run()
    assert target.status_calls == 0
    assert result.metadata["target_status_reuse"] == "none"


def test_accepted_only_legacy_trace_does_not_add_proposal_telemetry(monkeypatch):
    import bayesfilter.inference.hmc as hmc
    def unexpected(*_args, **_kwargs):
        raise AssertionError("accepted-only consumers must not add proposal evaluations")
    monkeypatch.setattr(hmc, "cache_hmc_target_status", unexpected)
    result = ReusableFullChainHMCRunner(StatusGaussian(), tf.zeros([4, 2], tf.float64),
        config(capture_candidate_health=False)).run()
    assert "target_status_telemetry" in result.trace
    assert "proposed_target_status_telemetry" not in result.trace
    assert result.metadata["target_status_reuse"] == "none"


def test_status_implementation_is_part_of_public_execution_source_closure():
    from pathlib import Path
    from tests.test_hmc_candidate_set_execution import make_binding
    binding = make_binding()
    assert any(Path(path).name == "hmc_status.py" for path in binding._spec["source_closure"])
