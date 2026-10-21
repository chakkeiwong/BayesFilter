"""CPU mechanics diagnostics; short synthetic sequences make no inference claim."""
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference import hmc_tensorflow_resume as resume
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference.hmc_sequential_verification import combined_chunks
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    NeuralForceHMCConfig, NeuralForceTransitionKernel,
    bind_neural_force_hmc_tuning_runner, neural_force_proposal)


def components(kind="nan", semantics="deterministic_position_only_proposal_field"):
    def field(q):
        if kind == "nan":
            return tf.fill(tf.shape(q), tf.constant(float("nan"), tf.float64))
        if kind == "invalid":
            return PositionOnlyForceEvaluation(1000.*q, tf.zeros(tf.shape(q)[:-1], tf.bool))
        return tf.zeros_like(q)
    force = FrozenPositionOnlyForce(field, identity="fallback-test-"+kind, semantics=semantics)
    target = FrozenTargetPotential(lambda q: .5*tf.reduce_sum(q*q, -1), identity="fallback-test-Gaussian")
    cfg = NeuralForceHMCConfig(step_size=.1, num_leapfrog_steps=2, inverse_mass_diagonal=(1.,))
    binding = bind_neural_force_hmc_tuning_runner(force=force, target=target, target_scope="fallback-test")
    kernel = NeuralForceTransitionKernel(force=force, target=target, config=cfg)
    @tf.function(input_signature=[tf.TensorSpec([4,1], tf.float64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def graph(state, seed):
        return stage._run_tensor_steps(kernel=kernel, initial_state=state, num_results=16,
                                       seed=seed, adaptive=False)
    return binding, graph


@pytest.mark.parametrize("kind", ["nan", "invalid"])
def test_actual_fallback_kernel_matches_zero_field_with_endpoint_mh(kind):
    binding, graph = components(kind)
    _, zero = components("zero")
    start = tf.constant([[-.2], [.1], [.4], [-.3]], tf.float64)
    actual, expected = graph(start, tf.constant([17,23])), zero(start, tf.constant([17,23]))
    for name in ("states", "is_accepted", "delta_h", "log_accept_ratio", "divergence"):
        tf.debugging.assert_equal(getattr(actual, name), getattr(expected, name))
    assert bool(tf.reduce_all(actual.force_fallback))
    assert bool(tf.reduce_any(actual.is_accepted))
    assert bool(tf.reduce_any(tf.abs(actual.delta_h) > 0.))  # Independent energy correction executed.
    assert all(bool(x) for x in verified._health(actual, start, 1000., binding).values())
    assert not bool(verified._health(actual, start, 1000.)["no_force_fallback"])
    diagnostics = verified._force_diagnostics(actual, binding)
    assert diagnostics["force_fallback_count"] == 64
    assert diagnostics["force_fallback_count_by_chain"] == [16]*4
    assert diagnostics["force_fallback_fraction"] == 1.
    assert diagnostics["warnings"] == ["deterministic_proposal_force_fallback"]


def test_zero_extension_across_invalid_region_remains_an_involution():
    force = FrozenPositionOnlyForce(lambda q: PositionOnlyForceEvaluation(q, q[:,0] < 0.),
        identity="piecewise-invalid", semantics="deterministic_position_only_proposal_field")
    cfg = NeuralForceHMCConfig(step_size=.2, num_leapfrog_steps=4, inverse_mass_diagonal=(1.,))
    q, p = tf.constant([[-.2], [.3]], tf.float64), tf.constant([[1.], [-1.]], tf.float64)
    forward = neural_force_proposal(q, p, force, cfg)
    reverse = neural_force_proposal(forward.position, forward.momentum, force, cfg)
    assert bool(tf.reduce_all(forward.force_fallback))
    tf.debugging.assert_near(reverse.position, q, atol=1e-13, rtol=1e-13)
    tf.debugging.assert_near(reverse.momentum, p, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("failure", ["endpoint", "nonfinite", "divergence", "energy", "stuck"])
def test_warning_policy_does_not_remove_other_health_checks(failure):
    binding, graph = components()
    start = tf.zeros([4,1], tf.float64)
    chunk = graph(start, tf.constant([11,33]))
    if failure == "endpoint":
        chunk = chunk._replace(endpoint_status=tf.tensor_scatter_nd_update(chunk.endpoint_status, [[0,0,8]], [1]))
    elif failure == "nonfinite":
        chunk = chunk._replace(log_accept_ratio=tf.fill([16,4], tf.constant(float("nan"), tf.float64)))
    elif failure == "divergence":
        chunk = chunk._replace(divergence=tf.ones([16,4], tf.bool))
    elif failure == "energy":
        chunk = chunk._replace(delta_h=tf.fill([16,4], tf.constant(1001., tf.float64)))
    else:
        chunk = chunk._replace(states=tf.zeros([16,4,1], tf.float64), final_state=start)
    assert not all(bool(x) for x in verified._health(chunk, start, 1000., binding).values())


def test_gradient_runner_keeps_existing_fallback_veto():
    binding, graph = components(semantics="position_only_scalar_potential_gradient")
    start = tf.zeros([4,1], tf.float64)
    chunk = graph(start, tf.constant([19,23]))
    assert not bool(verified._health(chunk, start, 1000., binding)["no_force_fallback"])


def test_fresh_replay_resume_aggregate_and_retained_consumers_preserve_warnings(tmp_path, monkeypatch):
    binding, graph = components()
    cfg = SimpleNamespace(initial_chain_positions=((-.2,),(.1,),(.4,),(-.3,)),
        nomination=SimpleNamespace(parameter_dimension=1, target_scope="fallback-test", seed=(31,77)),
        energy_error_max=1000., warmup_results=16, warmup_max_results=32,
        warmup_check_window_results=16, verification_results=16, verification_max_results=32,
        check_every_results=16, chunk_results=16, warmup_rhat_max=1.05, fresh_seed=(31,79))
    nominated = SimpleNamespace(center=tf.zeros([1], tf.float64), factor=tf.eye(1, dtype=tf.float64),
        config=cfg.nomination, adapter_signature="fallback-mechanics", heuristic_screen_passed=True)
    monkeypatch.setattr(verified, "_chunk_graph", lambda *args: graph)
    # This test checks control flow, not convergence of 16 draws.
    monkeypatch.setattr(verified, "_rhat_graph", lambda *args: lambda x: dict(maximum=tf.ones([1], tf.float64)))
    result = verified._fresh_sequence(cfg, nominated, binding, tmp_path)
    assert result["passed"] and len(result["events"]) == 2
    assert all(e["diagnostics"]["force_fallback_count"] == 64 for e in result["events"])
    verified._fresh_sequence(cfg, nominated, binding, tmp_path, replay=result)
    replay = resume.replay_then_continue(config=cfg, nomination=nominated, binding=binding,
        source_root=tmp_path, records=result["chunks"])
    assert replay["events"] == result["events"]
    warmup, retained = (combined_chunks(result["phases"][phase]) for phase in ("warmup", "verification"))
    checks, _ = verified._fresh_checks(cfg, nominated, warmup, retained, binding)
    assert all(bool(x) for x in checks.values())
    artifact = tmp_path / "manifest.json"
    stage._write_json(artifact, dict(verification=verified._save_chunk(tmp_path, "retained-start", retained),
                                    sequence=dict(used_seeds=result["used_seeds"])))
    tuning = SimpleNamespace(config=cfg, nomination=nominated, artifact_manifest_path=str(artifact), passed=True)
    monkeypatch.setattr(verified, "load_verified_tensorflow_hmc_result", lambda *args, **kwargs: tuning)
    runner = verified.VerifiedTensorFlowHMCRunner(tuning, binding)
    first = runner.run(stage.BoundRetainedHMCArchiveConfig(16, (45,78), tmp_path/"retain1", "CPU mechanics"))
    second = runner.run(stage.BoundRetainedHMCArchiveConfig(16, (45,79), tmp_path/"retain2", "CPU mechanics",
                                                          continuation_manifest=first["archive_manifest_path"]))
    assert first["health_passed"] and second["health_passed"]
    saved = json.loads((tmp_path/"retain2/manifest.json").read_text())
    assert saved["force_diagnostics"]["force_fallback_count"] == 64
