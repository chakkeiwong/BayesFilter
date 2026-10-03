"""Regression counterexamples for the September 15 shared-tuner repair.

Numerical examples are small CPU/debug mechanics fixtures, not sampler rankings.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    HMCKernelTuningConfig, run_typed_hmc_candidate_set, tune_hmc_kernel,
    resume_hmc_candidate_set_tuning,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    load_hmc_candidate_retained_runner,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetController, HMCTuningScopeCollection,
)
from bayesfilter.inference.hmc_candidate_set_artifacts import (
    write_candidate_set_result, load_candidate_set_result_payload,
)
from tests.test_hmc_candidate_set_tuning import _scope, _config, _pass
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config


def _monotone(work, candidate):
    if candidate.epsilon < 1.6:
        return {"decision": "repair_step_higher", "acceptance": .9}
    if candidate.epsilon > 1.8:
        return {"decision": "repair_step_lower", "acceptance": .4}
    return _pass(work, candidate)


@pytest.mark.parametrize("pause", [None, 2, 3])
def test_directional_repairs_preserve_the_observed_interval_on_resume(pause):
    scope = replace(_scope(), epsilon_domain=(.01, 2.2), max_repairs_per_family=3)
    cfg = _config(grid=(3,), epsilons=((3, (1.,)),), evidence_rungs=(1,))
    controller = HMCTuningCandidateSetController(scope, cfg)
    result = controller.run(_monotone, max_work_items=pause)
    if pause is not None:
        result = HMCTuningCandidateSetController.from_result_payload(result.payload()).run(_monotone)
    assert result.verified_candidate_ids
    assert [c.epsilon for c in result.candidates] == pytest.approx([1., 2., 2.**.5, 2.**.75])
    assert result.scope.max_repairs_per_family == 3


def test_refinement_uses_unresolved_directional_evidence_without_survivors():
    scope = replace(_scope(), epsilon_domain=(.01, 2.2), max_repairs_per_family=0)
    cfg = _config(grid=(3,), epsilons=((3, (1.5, 1.9)),), evidence_rungs=(1,),
                  refinement_rounds=1)
    result = HMCTuningCandidateSetController(scope, cfg).run(_monotone)
    assert len(result.verified_candidate_ids) == 1
    assert len(result.candidates) == 3
    assert result.repair_actions == ()


def test_nonmonotone_reversals_remain_separate_measured_hypotheses():
    directions = {.5: "repair_step_lower", .9: "repair_step_higher", 1.3: "repair_step_lower"}
    def observe(work, candidate):
        decision = directions.get(candidate.epsilon)
        return ({"decision": decision, "acceptance": .4 if decision == "repair_step_lower" else .9}
                if decision else _pass(work, candidate))
    scope = replace(_scope(), max_repairs_per_family=0)
    cfg = _config(grid=(3,), epsilons=((3, (.5, .9, 1.3)),), evidence_rungs=(1,), refinement_rounds=1)
    result = HMCTuningCandidateSetController(scope, cfg).run(observe)
    assert result.completion_status == "complete"
    assert [result.replay_candidate(cid).epsilon for cid in result.verified_candidate_ids] == pytest.approx(
        [(.5 * .9)**.5, (.9 * 1.3)**.5])
    assert len(result.candidates) == 5


def test_rejected_acceptance_receipt_does_not_poison_verified_peer(tmp_path):
    def observe(work, candidate):
        return ({"decision": "passed", "acceptance": .7,
                 "promotion_vetoes": ("native_divergence_positive",)}
                if candidate.leapfrog_steps == 3 else _pass(work, candidate))
    result = HMCTuningCandidateSetController(_scope(), _config()).run(observe)
    path = tmp_path / "result.json"
    write_candidate_set_result(result, path)
    loaded = load_candidate_set_result_payload(path)
    assert len(loaded["verified_candidate_ids"]) == 1
    assert loaded["candidate_states"][result.candidates[0].candidate_id] == "promotion_failed"


def test_affordable_verification_runs_when_a_peer_measurement_cannot_fit():
    cfg = _config(grid=(3, 25), epsilons=((3, (.25,)), (25, (.25,))),
                  max_gradient_work=600)
    result = HMCTuningCandidateSetController(_scope(), cfg).run(_pass,
        work_cost=lambda w, c: {"gradient_work": 100 * c.leapfrog_steps})
    assert result.completion_status == "partial_budget"
    assert result.search_state["gradient_work"] == 600
    assert [result.replay_candidate(cid).leapfrog_steps for cid in result.verified_candidate_ids] == [3]
    assert any(w.status == "pending" and w.stage == "measurement" for w in result.work_items)


def test_resume_charges_attempted_chunks_and_preserves_completed_work(tmp_path):
    binding = make_binding(config=execution_config(chunk_max_results=64))
    # Two complete stages cost 4352. The injected failed native call is charged
    # conservatively for its 64*4*(3+1)=1024 attempted work, even before success.
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,),
                  max_gradient_work=5376)
    original = binding._run
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise tf.errors.ResourceExhaustedError(None, None, "injected native call interruption")
        return original(*args)
    binding._run = interrupted
    first = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path)
    assert first.result.completion_status == "paused_infrastructure"
    assert first.result.search_state["gradient_work"] == 2048
    result = resume_hmc_candidate_set_tuning(tmp_path / "tuning_checkpoint.json",
                                            adapter=GaussianTarget()).result
    assert result.completion_status == "complete"
    assert result.verified_candidate_ids
    assert result.search_state["gradient_work"] == 5376


def test_repeated_chunk_interruptions_keep_work_identity_across_restarts(tmp_path):
    from bayesfilter.inference import load_numerical_tuning_checkpoint
    binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,), max_gradient_work=5504)
    controller = None
    for _ in range(2):
        original = binding._run
        calls = 0
        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise tf.errors.ResourceExhaustedError(None, None, "second interrupted native call")
            return original(*args)
        binding._run = interrupted
        partial = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path,
                                             _resume_controller=controller).result
        assert partial.completion_status == "paused_infrastructure"
        binding, controller = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json",
                                                               adapter=GaussianTarget())
    final = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path,
                                       _resume_controller=controller).result
    assert final.completion_status == "complete"
    assert final.verified_candidate_ids
    assert final.search_state["gradient_work"] == 5504


def test_retained_sampling_rejects_every_tuning_chunk_seed_before_execution(tmp_path):
    binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,))
    result = run_typed_hmc_candidate_set(binding.typed_adapter, cfg).result
    assert result.verified_candidate_ids
    runner = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=result, candidate_id=result.verified_candidate_ids[0],
        retained_binding=binding)
    seeds = {tuple(c["seed"]) for e in binding._evidence.values() for c in e["chunks"]}
    with patch.object(binding, "_run", side_effect=AssertionError("seed reuse reached numerical work")):
        for seed in seeds:
            with pytest.raises(ValueError, match="fresh"):
                runner.run(num_results=4, seed=seed, output_dir=tmp_path)


def test_retained_export_keeps_partial_and_failed_chunk_seed_history(tmp_path):
    from bayesfilter.inference.hmc_candidate_runtime import chunk_seed
    binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,), refinement_rounds=1)
    original = binding._run
    peer_calls = []

    def interrupted(candidate, state, count, seed):
        if candidate.epsilon != 1.3:
            peer_calls.append(seed)
            if len(peer_calls) == 3:
                raise tf.errors.ResourceExhaustedError(None, None, "unfinished refinement peer")
        return original(candidate, state, count, seed)

    binding._run = interrupted
    result = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path / "tuning").result
    assert result.completion_status == "paused_infrastructure"
    assert result.verified_candidate_ids
    assert len(peer_calls) == 3
    assert peer_calls[1] == chunk_seed(peer_calls[0], 1)
    runner = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=result, candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
    reloaded = load_hmc_candidate_retained_runner(runner.export(tmp_path / "member.json"), adapter=GaussianTarget())
    for checked in (runner, reloaded):
        with patch.object(checked._binding, "_run", side_effect=AssertionError("seed reuse reached execution")):
            for seed in peer_calls:
                with pytest.raises(ValueError, match="fresh"):
                    checked.run(num_results=4, seed=seed, output_dir=tmp_path / "retained")


@pytest.mark.parametrize("reverse", [False, True])
def test_collection_resolves_each_search_and_completeness_without_order_dependence(reverse):
    first = HMCTuningCandidateSetController(_scope(), _config()).run(_pass, max_work_items=1)
    second = HMCTuningCandidateSetController(replace(_scope(), search_id="second"), _config()).run(_pass)
    rows = (second, first) if reverse else (first, second)
    collection = HMCTuningScopeCollection(rows, expected_scope_ids=("scope",))
    assert not collection.complete
    cid = second.verified_candidate_ids[0]
    assert collection.member("scope", cid).candidate_id == cid


@pytest.mark.parametrize("option", ["search_config", "execution_config"])
def test_invalid_config_is_rejected_before_preparation(option):
    with patch("bayesfilter.inference.hmc_preparation.prepare_operational_windowed_mass_handoff") as prepare:
        with pytest.raises(TypeError, match=option):
            tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=HMCKernelTuningConfig.smoke(target_scope="candidate-bridge-test"),
                **{option: object()})
        prepare.assert_not_called()


@pytest.mark.parametrize("grid_kind", ["primary", "refinement", "expansion"])
def test_all_supplied_l_grids_obey_the_preparation_cap_before_work(grid_kind):
    kwargs = {grid_kind + "_l_grid": (26,)} if grid_kind != "primary" else {}
    search = (_config(grid=(26,), epsilons=((26, (.25,)),)) if grid_kind == "primary"
              else _config(grid=(3,), epsilons=((3, (.25,)),), **kwargs))
    preparation = HMCKernelTuningConfig.smoke(target_scope="candidate-bridge-test")
    with patch("bayesfilter.inference.hmc_preparation.prepare_operational_windowed_mass_handoff") as prepare:
        with pytest.raises(ValueError, match="max_leapfrog_steps"):
            tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=preparation,
                search_config=search)
        prepare.assert_not_called()


@pytest.mark.parametrize("option", ["search_config", "execution_config", "target_lineage",
                                    "source_paths", "frozen_transport_payload"])
def test_bound_fixed_transport_rejects_redundant_options(option):
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import tune_fixed_transport_hmc_kernel
    with patch("bayesfilter.inference.hmc_tuning_dispatch.tune_hmc_kernel") as dispatch:
        with pytest.raises(ValueError, match=option):
            tune_fixed_transport_hmc_kernel(base_adapter=object(), fixed_transport=object(),
                initial_position=[0.], config=_config(), candidate_set_adapter=object(),
                **{option: object()})
        dispatch.assert_not_called()


@pytest.mark.parametrize("option", ["target_lineage", "source_paths"])
def test_bound_ordinary_rejects_redundant_provenance(option):
    with pytest.raises(ValueError, match=option):
        tune_hmc_kernel(adapter=object(), initial_position=[0.], config=_config(),
            candidate_set_adapter=SimpleNamespace(adapter_kind="ordinary"), **{option: object()})


@pytest.mark.parametrize("probability", [.2, .7])
def test_native_divergence_preserves_acceptance_role_and_veto(probability):
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence
    from bayesfilter.inference.hmc_candidate_decisions import HMCCandidateDecision
    binding = make_binding(config=execution_config(num_warmup_steps=0))
    samples = tf.random.stateless_normal([64, 4, 2], (20260915, 410), dtype=tf.float64)
    log_accept = tf.fill([64, 4], tf.math.log(tf.constant(probability, tf.float64)))
    accepted = tf.ones([64, 4], tf.bool)
    divergence = tf.tensor_scatter_nd_update(tf.zeros([64, 4], tf.bool), [[0, 0]], [True])
    trace = dict(is_accepted=accepted, log_accept_ratio=log_accept,
        target_log_prob=tf.zeros([64, 4], tf.float64), proposed_target_log_prob=tf.zeros([64, 4], tf.float64),
        target_score_finite=tf.ones([64, 4], tf.bool), proposed_state=samples,
        initial_momentum=tf.ones_like(samples), final_momentum=tf.ones_like(samples), divergence=divergence,
        target_status_telemetry=GaussianTarget().target_status_telemetry(samples),
        proposed_target_status_telemetry=GaussianTarget().target_status_telemetry(samples))
    evidence = evaluate_hmc_acceptance_evidence(samples=samples, log_accept_ratio=log_accept,
        is_accepted=accepted, policy=HMCAcceptancePolicy(), native_divergence_status="available",
        native_divergence_count=1)
    expected = HMCCandidateDecision.from_evidence(evidence)
    observed = binding.analyze(binding.initial_active_state, samples, trace)
    assert all(observed[k] == v for k, v in expected.payload().items())
    assert "native_divergence_positive" in observed["promotion_vetoes"]
    assert not observed["promotion_eligible"]
    assert observed["repair_eligible"] == (probability == .2)


def test_preparation_failure_is_preserved_and_requires_fresh_retry_directory(tmp_path):
    def fail(**kwargs):
        kwargs["progress_callback"]("geometry_completed", {"diagnostic": "fixture"})
        raise RuntimeError("injected preparation failure")
    with patch("bayesfilter.inference.hmc_preparation.prepare_operational_windowed_mass_handoff", fail):
        with pytest.raises(RuntimeError, match="injected preparation"):
            tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=HMCKernelTuningConfig.smoke(target_scope="candidate-bridge-test"), output_dir=tmp_path)
    record = json.loads((tmp_path / "preparation_progress.json").read_text())
    assert record["status"] == "failed"
    assert record["events"][-1]["phase"] == "geometry_completed"
    assert record["failure"]["type"] == "RuntimeError"
    with pytest.raises(FileExistsError):
        tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.], output_dir=tmp_path)


def test_preparation_deadline_stops_between_phases_and_persists_reason(tmp_path):
    from bayesfilter.inference.hmc_preparation import HMCPreparationProgress, HMCPreparationBudgetExceeded
    clock = [0.]
    with patch("bayesfilter.inference.hmc_preparation.time.monotonic", lambda: clock[0]):
        with pytest.raises(HMCPreparationBudgetExceeded, match="geometry"):
            with HMCPreparationProgress(tmp_path, max_wall_time_seconds=1.) as progress:
                clock[0] = 2.
                progress.phase("geometry_completed")
    record = json.loads((tmp_path / "preparation_progress.json").read_text())
    assert record["status"] == "deferred"
    assert record["failure"]["type"] == "HMCPreparationBudgetExceeded"
    assert record["elapsed_seconds"] == 2.


def test_position_field_pilot_uses_declared_count_and_shared_execution_policy(tmp_path):
    from bayesfilter.inference import FourChainMeanBandAcceptancePolicy
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    cfg = replace(position_config(), step_adaptation_results=96, verification_results=64,
        max_leapfrog_steps=5, acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(.65, .75), per_chain_band=(.55, .85)))
    search = _config(grid=(3,), epsilons=((3, (.2,)),), pilot_enabled=True, evidence_rungs=(1,))
    run = tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64), runner_binding=_binding(), config=cfg,
        search_config=search, output_dir=tmp_path, max_work_items=1)
    observation = run.result.observations[0]["observation"]
    assert sum(chunk["count"] for chunk in observation["chunks"]) == 96
    assert observation["draw_range"] == (0, 96)
    assert run.result.search_state["gradient_work"] == 96 * 4 * (3 + 1)
    assert not run.numerical_handoff_authority


def test_position_field_default_xla_and_explicit_debug_exception():
    from bayesfilter.inference import TensorFlowHMCKernelTuningConfig
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    assert TensorFlowHMCKernelTuningConfig.__dataclass_fields__["use_xla"].default is True
    cfg = replace(position_config(), use_xla=False, non_xla_reason=None,
                  verification_results=64, max_leapfrog_steps=5)
    with pytest.raises(ValueError, match="non_xla_reason"):
        tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
            parameter_scales=tf.ones([2], tf.float64), runner_binding=_binding(), config=cfg)


@pytest.mark.parametrize("fault", ["state", "log_accept_ratio", "delta_h", "divergence"])
def test_position_field_warmup_health_and_mixed_receipts_survive_restart(tmp_path, monkeypatch, fault):
    from bayesfilter.inference import HMCAcceptancePolicy
    from bayesfilter.inference.hmc_candidate_set_position_field import resume_position_field_candidate_tuning
    from bayesfilter.inference import hmc_tensorflow_tuning as numerical
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    cfg = replace(position_config(), step_adaptation_results=64, verification_results=64, max_leapfrog_steps=5)
    execution = execution_config(measurement_num_results=64, verification_num_results=64,
        acceptance_policy=HMCAcceptancePolicy(), target_status_trace_policy="none")
    search = _config(grid=(3,), epsilons=((3, (.2, .3)),), evidence_rungs=(1,))
    runner_binding = _binding()
    tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64), runner_binding=runner_binding, config=cfg,
        search_config=search, execution_config=execution, output_dir=tmp_path, max_work_items=0)

    def synthetic_steps(*, kernel, initial_state, num_results, seed, adaptive):
        assert not adaptive
        states = tf.random.stateless_normal([num_results, 4, 2], seed, dtype=tf.float64)
        log_accept = tf.fill([num_results, 4], tf.math.log(tf.constant(.7, tf.float64)))
        delta_h = -log_accept
        divergence = tf.zeros([num_results, 4], tf.bool)
        bad = kernel.config.step_size < .25
        injected = tf.where(bad, tf.constant(float("nan"), tf.float64), tf.constant(0., tf.float64))
        if fault == "state":
            states = tf.tensor_scatter_nd_update(states, [[0, 0, 0]], [injected])
        elif fault == "log_accept_ratio":
            log_accept = tf.tensor_scatter_nd_update(log_accept, [[0, 0]], [injected])
        elif fault == "delta_h":
            delta_h = tf.tensor_scatter_nd_update(delta_h, [[0, 0]], [injected])
        else:
            divergence = tf.tensor_scatter_nd_update(divergence, [[0, 0]], [bad])
        return numerical._TensorChunk(states, log_accept, divergence, tf.zeros_like(divergence),
            delta_h, tf.ones_like(divergence), states[-1], kernel.config.step_size)

    monkeypatch.setattr(numerical, "_run_tensor_steps", synthetic_steps)
    partial = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
        adapter=_Adapter(), runner_binding=runner_binding, max_work_items=2).result
    assert partial.completion_status == "partial_budget"
    assert not partial.verified_candidate_ids
    rejected = next(r for r in partial.verification_receipts if r.epsilon == .2)
    assert rejected.decision == "passed"  # Finite retained acceptance remains in band.
    assert not rejected.decision_evidence.promotion_eligible
    expected = "native_divergence_positive" if fault == "divergence" else "nonfinite_" + fault
    assert expected in (*rejected.hard_vetoes, *rejected.promotion_vetoes)
    result = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
        adapter=_Adapter(), runner_binding=runner_binding).result
    assert result.completion_status == "complete"
    assert [result.replay_candidate(cid).epsilon for cid in result.verified_candidate_ids] == [.3]
    loaded = load_candidate_set_result_payload(tmp_path / "candidate_set_result.json")
    assert loaded["verified_candidate_ids"] == list(result.verified_candidate_ids)


def test_registry_and_actual_position_field_route_use_the_same_broad_grid(tmp_path):
    from bayesfilter.inference import FourChainMeanBandAcceptancePolicy, HMC_TUNING_INTERFACE_CAPABILITIES
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    capability = next(c for c in HMC_TUNING_INTERFACE_CAPABILITIES
                      if c.interface_name == "bind_neural_force_hmc_tuning_runner")
    assert "shared broad L grid" in capability.trajectory_policy
    assert "powers-of-two" not in capability.trajectory_policy
    cfg = replace(position_config(), step_adaptation_results=64, verification_results=64,
        max_leapfrog_steps=25, acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(.65, .75), per_chain_band=(.55, .85)))
    run = tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64), runner_binding=_binding(), config=cfg,
        output_dir=tmp_path, max_work_items=0)
    assert tuple(c.leapfrog_steps for c in run.result.candidates) == (3, 5, 9, 13, 18, 25)
    assert cfg.payload()["candidate_policy_metadata_role"].startswith("historical_graph_helper_only")


def test_native_deadline_stop_does_not_charge_unattempted_chunks(tmp_path, monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,), max_gradient_work=4352)
    import bayesfilter.inference.hmc_candidate_set_execution as execution_module
    original = execution_module.before_numerical_chunk
    interrupted = False
    def between(runtime, work, candidate, **kwargs):
        nonlocal interrupted
        if kwargs["index"] == 1 and not interrupted:
            interrupted = True
            raise HMCInfrastructureFailure("interrupted between chunks, before any new native call")
        original(runtime, work, candidate, **kwargs)
    monkeypatch.setattr(execution_module, "before_numerical_chunk", between)
    first = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path)
    assert first.result.search_state["gradient_work"] == 1024
    final = resume_hmc_candidate_set_tuning(tmp_path / "tuning_checkpoint.json", adapter=GaussianTarget()).result
    assert final.completion_status == "complete"
    assert final.verified_candidate_ids
    assert final.search_state["gradient_work"] == 4352


def test_budget_deferral_survives_resume_without_charging_blocked_work():
    cfg = _config(grid=(3, 25), epsilons=((3, (.25,)), (25, (.25,))), max_gradient_work=600)
    cost = lambda w, c: {"gradient_work": 100 * c.leapfrog_steps}
    first = HMCTuningCandidateSetController(_scope(), cfg).run(_pass, work_cost=cost, max_work_items=1)
    second = HMCTuningCandidateSetController.from_result_payload(first.payload()).run(_pass, work_cost=cost)
    third = HMCTuningCandidateSetController.from_result_payload(second.payload()).run(_pass, work_cost=cost)
    assert third.completion_status == "partial_budget"
    assert second.verified_candidate_ids == third.verified_candidate_ids
    assert second.budget_used_units == third.budget_used_units == 2
    assert second.search_state["gradient_work"] == third.search_state["gradient_work"] == 600


def test_changed_controller_policy_is_readable_but_cannot_resume(tmp_path):
    import hashlib
    from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload, _canonical
    result = HMCTuningCandidateSetController(_scope(), _config()).run(_pass)
    payload = dict(candidate_set_result_payload(result))
    payload["search_state"]["controller_policy_version"] = 2
    payload.pop("result_hash")
    payload["result_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload))
    loaded = load_candidate_set_result_payload(path)
    with pytest.raises(ValueError, match="historical controller"):
        HMCTuningCandidateSetController.from_result_payload(loaded)
