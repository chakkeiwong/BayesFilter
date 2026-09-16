"""CPU reference/mechanics checks; no posterior or default-readiness evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np  # Independent assertions and finite-difference references only.
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig,
    PrecomputedMassArtifact, ValueScoreCapability, bind_hmc_candidate_set_execution,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
    load_hmc_candidate_retained_runner, run_typed_hmc_candidate_set, tune_hmc_kernel,
)
from bayesfilter.inference.hmc_candidate_set_execution import (
    _tensor_payload, _tensor_from_payload, _trace_from_payload,
)
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


class GaussianTarget:
    parameter_dim = 2

    def __init__(self, scale=1.0):
        self.scale = scale

    def adapter_signature(self):
        return _sha256({"target": "candidate-bridge-gaussian-v1"})

    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True, target_scope="candidate-bridge-test",
            runtime_backend="tensorflow", evidence_path=__file__,
            nonclaims=("Gaussian engineering fixture only",))

    def log_prob_and_grad(self, theta):
        theta = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * self.scale * tf.reduce_sum(theta * theta, axis=-1), -self.scale * theta

    def target_status_telemetry(self, theta):
        shape = tf.shape(theta)[:-1]
        return {"status_code": tf.zeros(shape, tf.int32),
                "valid_pre_regularized_score": tf.ones(shape, tf.bool),
                "floor_count_value": tf.zeros(shape, tf.int32)}


def execution_config(**overrides):
    # The wider test band reduces stochastic fixture brittleness. It is not a
    # recommended scientific policy; default policy decisions are tested below.
    values = dict(measurement_num_results=128, verification_num_results=128,
        num_warmup_steps=8, seed=(20260914, 11),
        acceptance_policy=HMCAcceptancePolicy(practical_region=(0.55, 0.85), repair_region=(0.50, 0.90)),
        target_status_trace_policy="per_chain_step", use_xla=False,
        non_xla_reason="CPU numerical reference and persistence regression")
    return HMCCandidateExecutionConfig(**(values | overrides))


def mass_for(target, factor=None, center=None):
    factor = tf.eye(2, dtype=tf.float64) if factor is None else tf.constant(factor, tf.float64)
    return PrecomputedMassArtifact(position=[0., 0.] if center is None else center,
        factor=factor, covariance=tf.matmul(factor, factor, transpose_b=True),
        adapter_signature=target.adapter_signature(), position_role="reference_center",
        covariance_source="explicit engineering fixture")


def make_binding(*, target=None, config=None, **overrides):
    target = GaussianTarget() if target is None else target
    values = dict(adapter=target, initial_position=tf.constant([[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]], tf.float64),
        target_scope="candidate-bridge-test", target_lineage={"model": "Gaussian", "data": "none", "prior": "standard_normal"},
        config=execution_config() if config is None else config, source_paths=[__file__],
        scope_id="bridge-test", search_id="search-1", epsilon_domain=(0.01, 1.95),
        repair_factor=1.1, max_repairs_per_family=0, mass_artifact=mass_for(target))
    return bind_hmc_candidate_set_execution(**(values | overrides))


@pytest.fixture(scope="module", params=("serial", "batched"))
def tuned(request):
    binding = make_binding(config=execution_config(chain_mode=request.param))
    config = HMCControllerConfig(primary_l_grid=(2, 3),
        epsilon_by_l=((2, (1.1, 1.3, 1.5)), (3, (1.1, 1.3, 1.5))),
        total_budget_units=40, repair_reserve_units=3)
    run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
        config=config, candidate_set_adapter=binding.typed_adapter)
    assert run.result.verified_candidate_ids, [r.payload() for r in run.result.verification_receipts]
    assert run.result.final_status == "complete"
    return binding, run.result


def test_real_tune_uses_all_pairs_and_fresh_candidate_specific_evidence(tuned):
    binding, result = tuned
    assert len(result.candidates) == 6
    assert not result.payload()["numerical_handoff_authority"]
    seen_seeds = set()
    for receipt in result.verification_receipts:
        evidence = binding._evidence[receipt.numerical_evidence_hash]
        assert evidence["candidate"]["epsilon"] == receipt.epsilon
        assert evidence["candidate"]["leapfrog_steps"] == receipt.exact_l
        assert receipt.numerical_evidence_hash == _sha256(evidence)
        assert tuple(evidence["seed"]) not in seen_seeds
        seen_seeds.add(tuple(evidence["seed"]))
        samples = _tensor_from_payload(evidence["samples"])
        work = next(w for w in result.work_items if w.work_item_id == evidence["work"]["work_item_id"])
        expected = (binding.config.num_warmup_steps
                    + binding.config.verification_num_results * work.evidence_multiplier)
        assert samples.shape == (expected, 4, 2)
        assert receipt.draw_range == (binding.config.num_warmup_steps, expected)
    assert len(result.verified_candidate_ids) >= 2


def test_member_export_reload_and_continuation_match_direct_runner(tuned, tmp_path):
    from bayesfilter.inference.hmc import (
        FullChainHMCConfig, ReusableFullChainHMCRunner,
        build_independent_chain_tfp_hmc_runner,
    )
    binding, result = tuned
    candidate_id = result.verified_candidate_ids[-1]  # Explicit member, not a nominee.
    runner = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=result, candidate_id=candidate_id, retained_binding=binding)
    assert runner.numerical_handoff_authority and not runner.posterior_convergence_authority
    member = runner.export(tmp_path / "member.json")
    first = runner.run(num_results=12, seed=(19, 41), output_dir=tmp_path / "first")
    reloaded = load_hmc_candidate_retained_runner(member, adapter=GaussianTarget())
    second = reloaded.run(num_results=12, seed=(19, 42), output_dir=tmp_path / "second",
                          previous_archive=first["archive_path"])
    payload = json.loads(Path(second["archive_path"]).read_text())
    np.testing.assert_array_equal(_tensor_from_payload(payload["initial_active_state"]), first["final_active_state"])
    assert not np.array_equal(first["final_active_state"], runner.initial_active_state)
    assert payload["predecessor"]["content_hash"] == first["content_hash"]
    assert not payload["warmup_draws_included"] and not payload["tuning_draws_included"]
    factory = (ReusableFullChainHMCRunner if binding.config.chain_mode == "batched"
               else build_independent_chain_tfp_hmc_runner)
    baseline = factory(binding._active_adapter, binding.initial_active_state,
        FullChainHMCConfig(num_results=12, num_burnin_steps=0, step_size=runner.candidate.epsilon,
            num_leapfrog_steps=runner.candidate.leapfrog_steps, seed=(19,42), use_xla=False,
            target_scope="candidate-bridge-test"))
    direct = baseline.run(current_state=first["final_active_state"],
                          **({} if binding.config.chain_mode == "batched" else {"mode": "serial"}))
    assert payload["runtime"]["execution_mode"] == binding.config.chain_mode
    np.testing.assert_array_equal(second["samples"], direct.samples)
    with pytest.raises(ValueError, match="fresh"):
        reloaded.run(num_results=8, seed=(19, 41), output_dir=tmp_path / "reuse", previous_archive=second["archive_path"])
    with pytest.raises(FileExistsError):
        runner.export(member)
    # A new interpreter has no live adapters, runner caches or evidence dict.
    code = """
import sys
from tests.test_hmc_candidate_set_execution import GaussianTarget
from bayesfilter.inference import load_hmc_candidate_retained_runner
runner = load_hmc_candidate_retained_runner(sys.argv[1], adapter=GaussianTarget())
runner.run(num_results=8, seed=(19,43), output_dir=sys.argv[3], previous_archive=sys.argv[2])
"""
    subprocess.run([sys.executable, "-c", code, str(member), second["archive_path"], str(tmp_path / "third")],
        check=True, timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
        capture_output=True, text=True)
    third = json.loads((tmp_path / "third" / "retained_archive.json").read_text())
    np.testing.assert_array_equal(_tensor_from_payload(third["initial_active_state"]), second["final_active_state"])


def test_three_builders_share_numerical_validation_and_policy_boundary(tuned):
    binding, result = tuned
    kwargs = dict(candidate_set_result=result, candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    one = build_retained_bound_hmc_archive_runner_from_candidate_set_result(**kwargs)
    two = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(**kwargs)
    assert one.member_hash == two.member_hash
    with pytest.raises(ValueError, match="mechanics-only"):
        build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(**kwargs)


def test_callback_qualified_label_cannot_issue_numerical_bridge(tuned):
    binding, _ = tuned
    fake = replace(binding.typed_adapter, observe=lambda *_: {"decision": "passed", "acceptance": .7})
    result = run_typed_hmc_candidate_set(fake, HMCControllerConfig(primary_l_grid=(2,), epsilon_by_l=((2,(.3,)),),
        total_budget_units=10, repair_reserve_units=3)).result
    with pytest.raises(ValueError, match="numerical evidence"):
        build_retained_bound_hmc_archive_runner_from_candidate_set_result(candidate_set_result=result,
            candidate_id=result.verified_candidate_ids[0], retained_binding=binding)


def test_default_policy_rejects_finite_out_of_band_acceptance_and_warmup_health():
    binding = make_binding(config=execution_config(acceptance_policy=HMCAcceptancePolicy()))
    samples = tf.random.stateless_normal([136, 4, 2], (51, 1), dtype=tf.float64)
    trace = {"is_accepted": tf.ones([136,4], tf.bool), "log_accept_ratio": tf.fill([136,4], tf.math.log(tf.constant(.99, tf.float64))),
        "target_log_prob": tf.zeros([136,4], tf.float64), "proposed_target_log_prob": tf.zeros([136,4], tf.float64),
        "target_score_finite": tf.ones([136,4], tf.bool), "proposed_state": samples,
        "initial_momentum": tf.ones_like(samples), "final_momentum": tf.ones_like(samples),
        "target_status_telemetry": GaussianTarget().target_status_telemetry(samples),
        "proposed_target_status_telemetry": GaussianTarget().target_status_telemetry(samples)}
    assert binding.analyze(binding.initial_active_state, samples, trace)["decision"] == "repair_step_higher"
    trace["log_accept_ratio"] = tf.fill([136,4], tf.math.log(tf.constant(.7, tf.float64)))
    assert binding.analyze(binding.initial_active_state, samples, trace)["decision"] == "passed"
    # A discarded warmup score failure still vetoes the candidate.
    trace["target_score_finite"] = tf.tensor_scatter_nd_update(trace["target_score_finite"], [[0,0]], [False])
    bad = binding.analyze(binding.initial_active_state, samples, trace)
    assert not bad["promotion_eligible"] and "nonfinite_target_score" in bad["hard_vetoes"]
    assert not bad["repair_eligible"]


@pytest.mark.parametrize("damage", ["missing_evidence", "tensor", "endpoint", "config", "source", "target", "scope"])
def test_durable_reload_rejects_drift_or_corruption(tuned, tmp_path, damage):
    binding, result = tuned
    runner = build_retained_bound_hmc_archive_runner_from_candidate_set_result(candidate_set_result=result,
        candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    path = runner.export(tmp_path / "member.json", portable=True)
    payload = json.loads(path.read_text())
    target = GaussianTarget()
    if damage == "target":
        target.scale = 2.0  # Same label, changed actual values/scores.
    elif damage == "missing_evidence":
        payload["numerical_evidence"] = {}
    elif damage == "tensor":
        next(iter(payload["numerical_evidence"].values()))["samples"]["tensor"] = "corrupt"
    elif damage == "endpoint":
        payload["verified_endpoint"] = _tensor_payload(tf.zeros([4,2], tf.float64))
    elif damage == "config":
        payload["execution"]["config"]["use_xla"] = True
    elif damage == "source":
        key = next(iter(payload["execution"]["source_closure"]))
        payload["execution"]["source_closure"][key] = "changed"
        payload["binding_hash"] = _sha256(payload["execution"])
    elif damage == "scope":
        payload["candidate_set_result"]["scope"]["mass_signature"] = "changed"
    payload.pop("content_hash")
    payload["content_hash"] = _sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_hmc_candidate_retained_runner(path, adapter=target)


def test_dispatch_rejects_wrong_target_or_start_before_work(tuned):
    binding, _ = tuned
    config = HMCControllerConfig(primary_l_grid=(2,), epsilon_by_l=((2,(.3,)),), total_budget_units=8, repair_reserve_units=3)
    with pytest.raises(ValueError, match="initial_active_state"):
        tune_hmc_kernel(adapter=GaussianTarget(), initial_position=tf.zeros([4,2],tf.float64),
                        config=config, candidate_set_adapter=binding.typed_adapter)


@pytest.mark.parametrize("overrides", [dict(measurement_num_results=63), dict(verification_num_results=0),
    dict(num_warmup_steps=-1), dict(seed=(1,)), dict(non_xla_reason=None), dict(use_xla="true"),
    dict(chain_mode="unknown")])
def test_execution_config_rejects_invalid_evidence_budgets(overrides):
    with pytest.raises((ValueError, TypeError)):
        execution_config(**overrides)


def test_batched_binding_xla_preserves_independent_rows_and_dynamic_inputs(monkeypatch):
    from bayesfilter.inference import hmc
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord

    def forbid_scalar_runners(*_args, **_kwargs):
        raise AssertionError("batched execution must not dispatch scalar chains")

    monkeypatch.setattr(hmc, "build_independent_chain_tfp_hmc_runner", forbid_scalar_runners)
    binding = make_binding(config=execution_config(chain_mode="batched", use_xla=True,
                                                   non_xla_reason=None))
    candidate = HMCTuningCandidateRecord.create(binding.scope, leapfrog_steps=2,
                                                epsilon=.2, creation_ordinal=1)
    starts = tf.zeros_like(binding.initial_active_state)
    first = binding._run(candidate, starts, 4, (31, 7))
    replay = binding._run(candidate, starts, 4, (31, 7))
    tf.debugging.assert_equal(first.samples, replay.samples)
    momenta = first.trace["initial_momentum"][0]
    assert bool(tf.reduce_all(tf.reduce_any(momenta[1:] != momenta[:1], axis=-1)))
    shifted = tf.tensor_scatter_nd_update(starts, [[0, 0]], [1.])
    independent = binding._run(candidate, shifted, 4, (31, 7))
    tf.debugging.assert_equal(first.samples[:, 1:], independent.samples[:, 1:])
    assert bool(tf.reduce_any(first.samples[:, 0] != independent.samples[:, 0]))
    changed_seed = binding._run(candidate, starts, 4, (31, 8))
    assert bool(tf.reduce_any(first.samples != changed_seed.samples))
    child = HMCTuningCandidateRecord.create(binding.scope, leapfrog_steps=2,
                                            epsilon=.3, creation_ordinal=2)
    changed_step = binding._run(child, starts, 4, (31, 7))
    assert bool(tf.reduce_any(first.samples != changed_step.samples))
    assert binding.health_failures(starts, first.samples, first.trace) == ()
    assert first.trace["proposed_target_status_telemetry"]["status_code"].shape == (4, 4)
    assert len(binding._runners) == 1
    compiled = next(iter(binding._runners.values()))._runner
    concrete = compiled.get_concrete_function()
    assert compiled.experimental_get_tracing_count() == 1
    assert concrete.function_def.attr["_XlaMustCompile"].b
    definition = concrete.graph.as_graph_def()
    nodes = list(definition.node) + [node for fn in definition.library.function for node in fn.node_def]
    assert not any("PyFunc" in node.op or "HostCompute" in node.op for node in nodes)


def test_finite_guard_and_candidate_health_wrappers_compose():
    from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner
    binding = make_binding()
    runner = build_independent_chain_tfp_hmc_runner(binding._active_adapter, binding.initial_active_state,
        FullChainHMCConfig(num_results=4, num_burnin_steps=0, step_size=.2, num_leapfrog_steps=2,
            seed=(3,4), require_finite_transitions=True, capture_candidate_health=True,
            target_scope="candidate-bridge-test"))
    result = runner.run(mode="serial")
    assert result.trace["target_score_finite"].shape == (4,4)
    assert binding.health_failures(binding.initial_active_state, result.samples,
        {**result.trace, "target_status_telemetry": GaussianTarget().target_status_telemetry(result.samples),
         "proposed_target_status_telemetry": GaussianTarget().target_status_telemetry(result.samples)}) == ()


def test_real_windowed_preparation_preserves_both_affine_layers(tmp_path):
    # Reuse the existing rotated-Gaussian preparation fixture, with real TF
    # operational warmup. Bootstrap's synthetic screen is preparation mechanics
    # only; candidate verification below always runs the numerical evaluator.
    from tests import test_hmc_kernel_tuning_windowed_mass as fixture
    from bayesfilter.inference import bind_hmc_candidate_set_execution_from_preparation
    from bayesfilter.inference.hmc_kernel_tuning import (
        build_operational_fixed_mass_hmc_adapter, run_hmc_windowed_mass_stage,
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    )
    adapter = fixture._RotatedGaussianAdapter()
    geometry = fixture.initialize_hmc_kernel_geometry(adapter=adapter, initial_position=[.4,-.3],
        initial_covariance=[[1.4,.3],[.3,.8]],
        config=fixture.HMCGeometryInitializationConfig(covariance_jitter=0.0))
    bootstrap = fixture.run_hmc_bootstrap_screen(adapter=adapter, geometry=geometry,
        run_full_chain=lambda _adapter, _state, config: fixture._runtime_shaped_result(
            warmup_steps=int(config.num_results), acceptance_trace=[True,True,False,True]*4))
    stage = run_hmc_windowed_mass_stage(adapter=adapter, geometry=geometry, bootstrap=bootstrap,
        config=fixture._stage_config(algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                                     chain_execution_mode="tf_function"),
        _attempt_budget_policy=fixture._operational_budget())
    assert stage.passed
    prepared = build_operational_fixed_mass_hmc_adapter(adapter=adapter, geometry=geometry,
        windowed_stage=stage, target_scope=stage.config.target_scope)
    prepared.update(geometry=geometry, windowed_stage=stage)
    kwargs = dict(adapter=adapter, preparation=prepared, target_lineage={"fixture": "rotated_Gaussian"},
        config=execution_config(target_status_trace_policy="none"), source_paths=[__file__, fixture.__file__],
        scope_id="prepared-test", search_id="search-1", epsilon_domain=(.001,10.), repair_factor=1.1,
        max_repairs_per_family=0)
    binding = bind_hmc_candidate_set_execution_from_preparation(**kwargs)
    assert binding.scope.adapter_signature == prepared["final_adapter_signature"]
    assert len(binding._transforms) == 2
    assert not np.allclose(geometry.mass_artifact.factor, np.eye(2))
    np.testing.assert_allclose(binding.position_samples(binding.initial_active_state),
        stage.operational_warmup_result.private_start_bank_theta, rtol=1e-10, atol=1e-10)
    probes = tf.constant([[.2,-.5],[-.3,.4]], tf.float64)
    value, score = binding._active_adapter.log_prob_and_grad(probes)
    expected_value, expected_score = prepared["final_adapter"].log_prob_and_grad(probes)
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_array_equal(score, expected_score)
    # Independent finite differences check the composed chain rule.
    for index in range(2):
        offset = np.zeros((2,2)); offset[:,index] = 1e-5
        plus = binding._active_adapter.log_prob_and_grad(probes + offset)[0]
        minus = binding._active_adapter.log_prob_and_grad(probes - offset)[0]
        np.testing.assert_allclose(score[:,index], (plus-minus)/(2e-5), atol=1e-8)
    factors = [np.asarray(layer.transform.factor) for layer in binding._transforms]
    composed = factors[0] @ factors[1]
    maximum_frequency = np.sqrt(np.linalg.eigvalsh(composed.T @ adapter.precision @ composed)[-1])
    bound = binding._spec["preparation"]["epsilon_proposal_bound"]
    final = stage.operational_warmup_result.final_kernel_state
    assert bound["coordinate_signature"] == final.transform.signature
    assert bound["metric_signature"] == final.momentum_metric.signature
    assert binding.scope.epsilon_domain[1] == bound["upper"]
    epsilons = tuple(bound["upper"] * factor for factor in (.6, .8, 1.))
    result = run_typed_hmc_candidate_set(binding.typed_adapter,
        HMCControllerConfig(primary_l_grid=(2,3), epsilon_by_l=((2,epsilons),(3,epsilons)),
            total_budget_units=40, repair_reserve_units=3), output_dir=tmp_path).result
    # Geometry/restart correctness does not require a particular stochastic
    # candidate to pass within the metric-derived proposal domain.
    assert result.observations
    from bayesfilter.inference import load_numerical_tuning_checkpoint
    restored, controller = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json", adapter=adapter)
    np.testing.assert_array_equal(restored.initial_active_state, binding.initial_active_state)
    np.testing.assert_array_equal(restored.position_samples(probes), binding.position_samples(probes))
    assert controller.result().verified_candidate_ids == result.verified_candidate_ids
    bad = dict(prepared, initial_position=tf.zeros([4,2],tf.float64))
    with pytest.raises(ValueError, match="start bank"):
        bind_hmc_candidate_set_execution_from_preparation(**(kwargs | {"preparation": bad}))


@pytest.mark.parametrize("kind", ["affine", "dense_iaf"])
def test_supported_frozen_transports_use_numerical_controller_and_durable_geometry(kind, tmp_path):
    target = GaussianTarget()
    if kind == "affine":
        payload = {"schema": "bayesfilter.neutra.frozen_affine_diag.v1", "transport_id": "test-affine",
            "dimension": 2, "target_signature": target.adapter_signature(), "log_jacobian_available": True,
            "shift": [.2,-.1], "raw_scale": [.1,.05]}
    else:
        from tests.test_dense_iaf_neutra_artifact_loader import _payload
        payload = _payload(target_signature=target.adapter_signature())
    binding = make_binding(mass_artifact=None, frozen_transport_payload=payload, start_coordinates="active")
    probes = tf.constant([[.2,-.5],[-.3,.4]], tf.float64)
    value, score = binding._active_adapter.log_prob_and_grad(probes)
    raw = binding.position_samples(probes)
    expected = target.log_prob_and_grad(raw)[0] + binding._active_adapter.log_abs_det_jacobian(probes)
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
    for index in range(2):
        offset = np.zeros((2,2)); offset[:,index] = 1e-5
        plus = binding._active_adapter.log_prob_and_grad(probes+offset)[0]
        minus = binding._active_adapter.log_prob_and_grad(probes-offset)[0]
        np.testing.assert_allclose(score[:,index], (plus-minus)/(2e-5), atol=1e-8)
    from bayesfilter.inference import tune_fixed_transport_hmc_kernel
    result = tune_fixed_transport_hmc_kernel(base_adapter=target, fixed_transport=binding.fixed_transport,
        initial_position=binding.initial_active_state, candidate_set_adapter=binding.typed_adapter,
        config=HMCControllerConfig(primary_l_grid=(2,3), epsilon_by_l=((2,(1.1,1.3,1.5)),(3,(1.1,1.3,1.5))),
            total_budget_units=40, repair_reserve_units=3)).result
    assert result.verified_candidate_ids, [(r.epsilon,r.exact_l,r.decision,r.acceptance,r.hard_vetoes) for r in result.verification_receipts]
    runner = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=result,
        candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    restored = load_hmc_candidate_retained_runner(runner.export(tmp_path / "transport.json"), adapter=GaussianTarget())
    np.testing.assert_array_equal(restored._binding.position_samples(probes), raw)
    retained = restored.run(num_results=8, seed=(55,82), output_dir=tmp_path / "retained")
    assert retained["samples"].shape == (8,4,2)


def test_unsupported_transport_and_fallback_force_fail_closed():
    with pytest.raises(ValueError):
        make_binding(mass_artifact=None, frozen_transport_payload={"schema": "arbitrary_force"}, start_coordinates="active")
    class Fallback(GaussianTarget):
        def value_score_capability(self):
            return ValueScoreCapability(value_score_authority="gradient_tape_fallback", xla_hmc_ready=False)
    with pytest.raises(ValueError, match="exact TensorFlow"):
        make_binding(target=Fallback())


def test_retained_predecessor_endpoint_and_ancestry_are_checked(tuned, tmp_path):
    binding, result = tuned
    runner = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=result,
        candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    first = runner.run(num_results=8, seed=(26,1), output_dir=tmp_path / "first")
    second = runner.run(num_results=8, seed=(26,2), output_dir=tmp_path / "second", previous_archive=first["archive_path"])
    path = Path(second["archive_path"])
    payload = json.loads(path.read_text())
    payload["predecessor"]["content_hash"] = "wrong"
    payload.pop("content_hash")
    payload["content_hash"] = _sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="predecessor"):
        runner.run(num_results=8, seed=(26,3), output_dir=tmp_path / "third", previous_archive=path)


def test_bad_rhat_is_persisted_without_changing_numerical_tuning_decision(tuned, monkeypatch):
    import bayesfilter.inference.hmc as hmc
    binding, result = tuned
    candidate = result.replay_candidate(result.verified_candidate_ids[0])
    work = next(work for work in result.work_items if work.candidate_id == candidate.candidate_id and work.stage == "verification")
    monkeypatch.setattr(hmc, "_rhat_summary_from_retained_samples", lambda *_args, **_kwargs: {
        "passed": False, "max_finite_rhat": 98.0, "nonfinite_rhat_count": 1})
    observation = binding.observe(work, candidate)
    assert observation["decision"] == "passed"
    evidence = binding._evidence[observation["numerical_evidence_hash"]]
    assert evidence["rhat_reporting_only"]["max_finite_rhat"] == 98.0


def test_proposed_status_and_retained_health_failure_are_preserved(tuned, tmp_path, monkeypatch):
    binding, result = tuned
    runner = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=result,
        candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    original_run = binding._run
    def fail_status(*args, **kwargs):
        actual = original_run(*args, **kwargs)
        telemetry = dict(actual.trace["proposed_target_status_telemetry"])
        telemetry["status_code"] = tf.tensor_scatter_nd_update(telemetry["status_code"], [[0,0]], [1])
        return replace(actual, trace={**actual.trace, "proposed_target_status_telemetry": telemetry})
    monkeypatch.setattr(binding, "_run", fail_status)
    with pytest.raises(ValueError, match="preserved archive"):
        runner.run(num_results=8, seed=(35,1), output_dir=tmp_path / "failed")
    path = tmp_path / "failed" / "retained_archive.json"
    assert "proposed_target_status_telemetry_failed" in json.loads(path.read_text())["health_failures"]
    with pytest.raises(ValueError, match="failed numerical health"):
        runner.run(num_results=8, seed=(35,2), output_dir=tmp_path / "next", previous_archive=path)


@pytest.mark.parametrize("field,value", [("epsilon",.123), ("leapfrog_steps",99), ("mass_signature","wrong"),
    ("backend","numpy"), ("dtype","float32"), ("use_xla",True), ("start_bank_signature","wrong")])
def test_rehashed_candidate_record_cannot_change_bound_scope_or_kernel(tuned, field, value):
    from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload
    binding, result = tuned
    payload = json.loads(json.dumps(candidate_set_result_payload(result)))
    candidate_id = result.verified_candidate_ids[0]
    record = next(c for c in payload["candidates"] if c["candidate_id"] == candidate_id)
    record[field] = value
    # Recompute outer checksum: rejection must come from scope/record/receipt
    # invariants, not just from a stale file checksum.
    payload.pop("result_hash")
    payload["result_hash"] = _sha256(payload)
    with pytest.raises(ValueError):
        build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=payload,
            candidate_id=candidate_id, retained_binding=binding)


def test_live_frozen_geometry_mutation_is_rejected_even_for_invariant_target():
    binding = make_binding()
    # Reflection preserves a standard Gaussian's value and score in active
    # coordinates but changes the active-to-model map. Probe-only validation
    # would miss it; exact geometry validation must reject it.
    transform = binding._active_adapter.transform
    object.__setattr__(transform, "factor", -tf.convert_to_tensor(transform.factor))
    with pytest.raises(ValueError, match="geometry changed"):
        binding.validate()


def test_source_dependency_drift_rejected_before_execution(tmp_path):
    source = tmp_path / "target_dependency.py"
    source.write_text("MODEL_VERSION = 1\n")
    binding = make_binding(source_paths=[__file__, source])
    source.write_text("MODEL_VERSION = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        binding.validate()


def test_numerically_repaired_child_bridges_with_exact_child_kernel(tmp_path):
    # L=3, epsilon=1.1 is near a return phase of the standard Gaussian and
    # gives high acceptance. The declared repair proposes 1.298, near the
    # independently checked 1.3 fixture; the child must still earn its own pass.
    binding = make_binding(config=execution_config(measurement_num_results=512,
        verification_num_results=512, num_warmup_steps=32), max_repairs_per_family=2,
        repair_factor=1.18)
    result = run_typed_hmc_candidate_set(binding.typed_adapter,
        HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3,(1.1,)),),
            total_budget_units=16, repair_reserve_units=6)).result
    assert result.verified_candidate_ids, [(r.epsilon,r.decision,r.acceptance) for r in result.verification_receipts]
    candidate = result.replay_candidate(result.verified_candidate_ids[0])
    assert candidate.parent_candidate_id is not None
    assert candidate.leapfrog_steps == 3 and candidate.epsilon != 1.1
    action = next(a for a in result.repair_actions if a.child_candidate_id == candidate.candidate_id)
    assert action.qualified_repair_status == "executed_and_verified"
    runner = build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=result,
        candidate_id=candidate.candidate_id, retained_binding=binding)
    restored = load_hmc_candidate_retained_runner(runner.export(tmp_path / "child.json"), adapter=GaussianTarget())
    assert restored.candidate == candidate
    assert restored.run(num_results=8, seed=(92,4), output_dir=tmp_path / "child_draws")["num_results"] == 8
    with pytest.raises(ValueError, match="not independently verified"):
        build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(candidate_set_result=result,
            candidate_id=candidate.parent_candidate_id, retained_binding=binding)
