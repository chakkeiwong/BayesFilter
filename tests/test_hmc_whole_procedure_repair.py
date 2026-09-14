"""Engineering counterexamples for the September 14 procedure repair.

Gaussian cases are deliberately CPU/reference fixtures, not sampler rankings.
"""
from dataclasses import replace
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    HMCControllerConfig, run_typed_hmc_candidate_set, resume_hmc_candidate_set_tuning,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, resume_hmc_candidate_set, write_candidate_set_result,
)
from tests.test_hmc_candidate_set_tuning import _scope, _config, _pass
from tests.test_hmc_candidate_set_execution import make_binding, execution_config


def test_movement_veto_rejects_parent_but_measures_smaller_step_child():
    def observe(work, candidate):
        if candidate.parent_candidate_id is None:
            return {"decision": "repair_step_lower", "promotion_vetoes": ("movement_gate_failed",),
                    "repair_eligible": True, "acceptance": 0.0}
        return _pass(work, candidate)
    result = HMCTuningCandidateSetController(_scope(), _config()).run(observe)
    assert len(result.repair_actions) == 2
    for action in result.repair_actions:
        child = result.replay_candidate(action.child_candidate_id)
        assert child.epsilon == action.old_epsilon / 2
        assert result.candidate_states[action.parent_candidate_id] == "promotion_failed"
        stages = [w.stage for w in result.work_items if w.candidate_id == child.candidate_id]
        assert stages == ["measurement", "verification"]


def test_shared_observation_invalidates_already_verified_scope():
    controller = HMCTuningCandidateSetController(_scope(), _config())
    def observe(work, candidate):
        if work.stage == "verification" and candidate.leapfrog_steps == 5:
            return {"decision": "failed", "evidence_validity": "shared_execution_invalid",
                    "engineering_invalidity_reasons": ("accepted_draws_corrupt",)}
        return _pass(work, candidate)
    result = controller.run(observe)
    assert result.completion_status == "shared_invalidity"
    assert not result.verified_candidate_ids
    assert "accepted_draws_corrupt" in json.dumps(result.payload())


@pytest.mark.parametrize("decision", ["inconclusive_evidence", "inconclusive_conflict"])
def test_inconclusive_resume_extends_same_candidate_and_reaches_explicit_cap(tmp_path, decision):
    cfg = _config(evidence_rungs=(1, 3), candidate_reserve_units=2)
    def observe(work, candidate):
        return {"decision": decision} if work.stage == "verification" else _pass(work, candidate)
    controller = HMCTuningCandidateSetController(_scope(), cfg)
    first = controller.run(observe, max_work_items=4)
    assert first.completion_status == "partial_budget"
    assert first.resume_pending_work_item_ids
    path = tmp_path / "partial.json"
    write_candidate_set_result(first, path)
    second = resume_hmc_candidate_set(path, observe)
    assert second.completion_status == "complete"
    assert all(s == "inconclusive_at_cap" for s in second.candidate_states.values())
    assert len(second.candidates) == 2
    assert [w.evidence_multiplier for w in second.work_items if w.stage == "verification"] == [1, 1, 3, 3]


@pytest.mark.parametrize("factor", [.5, 1.0, 0.0])
def test_repair_factor_rejects_reversed_or_unchanged_proposals(factor):
    with pytest.raises(ValueError):
        replace(_scope(), repair_factor=factor)


@pytest.mark.parametrize("kwargs", [
    {"primary_l_grid": (3.5,), "epsilon_by_l": ((3.5, (.2,)),)},
    {"total_budget_units": True}, {"candidate_reserve_units": 1},
    {"refinement_rounds": 1.5}, {"max_candidates": False},
])
def test_invalid_search_numbers_fail_before_execution(kwargs):
    values = dict(primary_l_grid=(3,), epsilon_by_l=((3, (.2,)),)) | kwargs
    with pytest.raises(ValueError):
        HMCControllerConfig(**values)


def test_reverse_direction_does_not_cycle_or_duplicate_pair():
    def observe(work, candidate):
        if candidate.epsilon == .25:
            return {"decision": "repair_step_higher"}
        if candidate.epsilon == .5:
            return {"decision": "repair_step_lower"}
        return _pass(work, candidate)
    controller = HMCTuningCandidateSetController(_scope(), _config(grid=(3,), epsilons=((3, (.25,)),)))
    duplicate_id = controller.add_exploration_candidate(3, .25)
    result = controller.run(observe)
    assert duplicate_id == result.candidates[0].candidate_id
    assert len(result.candidates) == len({(c.leapfrog_steps, c.epsilon) for c in result.candidates}) == 3
    assert .25 < result.replay_candidate(result.verified_candidate_ids[0]).epsilon < .5


def test_pilot_and_refinement_cover_all_survivors_with_no_nominee():
    cfg = _config(pilot_enabled=True, refinement_rounds=1,
                  epsilon_refinement_factors=(.8, 1.25), refinement_l_grid=(4,), max_candidates=20)
    result = HMCTuningCandidateSetController(_scope(), cfg).run(_pass)
    assert len(result.verified_candidate_ids) == 8
    assert result.payload()["nominee_id"] is None
    assert [w.stage for w in result.work_items[:2]] == ["pilot", "pilot"]
    assert len([c for c in result.candidates if c.leapfrog_steps == 4]) == 2
    parents = {p for e in result.accounting_events if e["event"] == "refinement_proposed" for p in e["requesting_parents"]}
    assert parents == {c.candidate_id for c in result.candidates[:2]}


def test_gradient_cap_preserves_long_l_work_without_reducing_evidence():
    cfg = _config(max_gradient_work=500)
    def cost(work, c):
        return {"gradient_work": 100 * c.leapfrog_steps}
    result = HMCTuningCandidateSetController(_scope(), cfg).run(_pass, work_cost=cost)
    assert result.completion_status == "partial_budget"
    assert result.search_state["gradient_work"] == 300
    assert result.work_items[1].status == "pending"


def test_directional_repair_measures_unvisited_epsilon_domain_boundary():
    controller = HMCTuningCandidateSetController(replace(_scope(), epsilon_domain=(.1, .4)),
        _config(grid=(3,), epsilons=((3, (.3,)),)))
    result = controller.run(lambda w, c: {"decision": "repair_step_higher"} if c.epsilon == .3 else _pass(w, c))
    assert result.replay_candidate(result.verified_candidate_ids[0]).epsilon == .4


def test_checkpoint_rejects_observation_that_disagrees_with_numerical_evidence(tmp_path):
    from bayesfilter.inference.hmc_candidate_set_artifacts import _canonical
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference import load_numerical_tuning_checkpoint
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    import hashlib
    binding = make_binding()
    run_typed_hmc_candidate_set(binding.typed_adapter, _config(), output_dir=tmp_path, max_work_items=1)
    path = tmp_path / "tuning_checkpoint.json"
    payload = json.loads(path.read_text())
    result = payload["result"]
    result["observations"][0]["observation"]["acceptance"] = .123
    result.pop("result_hash")
    result["result_hash"] = hashlib.sha256(_canonical(result)).hexdigest()
    payload.pop("content_hash")
    payload["content_hash"] = _sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="observation differs"):
        load_numerical_tuning_checkpoint(path, adapter=GaussianTarget())


def test_real_oversized_step_is_repaired_and_failed_measurement_is_durable(tmp_path):
    binding = make_binding(epsilon_domain=(.01, 4.), repair_factor=2., max_repairs_per_family=2)
    cfg = _config(grid=(3,), epsilons=((3, (3.,)),), evidence_rungs=(1,))
    run = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path)
    assert run.result.repair_actions
    assert run.result.repair_actions[0].new_epsilon == 1.5
    checkpoint = json.loads((tmp_path / "tuning_checkpoint.json").read_text())
    first = checkpoint["result"]["observations"][0]["observation"]
    assert first["decision"] == "repair_step_lower"
    assert "movement_gate_failed" in first["promotion_vetoes"]
    assert (tmp_path / "numerical_evidence" / (first["numerical_evidence_hash"] + ".json")).exists()


def test_numerical_chunk_failure_resumes_and_exports_verified_member(tmp_path, monkeypatch):
    binding = make_binding(config=execution_config(chunk_max_results=64))
    cfg = _config(grid=(3,), epsilons=((3, (1.3,)),))
    original = binding._run
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise tf.errors.ResourceExhaustedError(None, None, "injected resource failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(binding, "_run", fail_second)
    first = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path)
    assert first.result.completion_status == "paused_infrastructure"
    checkpoint = json.loads((tmp_path / "tuning_checkpoint.json").read_text())
    assert len(next(iter(checkpoint["partial_chunks"].values()))) == 1
    second = resume_hmc_candidate_set_tuning(tmp_path / "tuning_checkpoint.json", adapter=binding._base_adapter)
    assert second.result.verified_candidate_ids
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=second.result, candidate_id=second.result.verified_candidate_ids[0],
        retained_binding=second.adapter._execution_binding)
    member.export(tmp_path / "member.json")
    assert second.result.budget_used_units > 2  # Failed dispatch consumes the call budget.


def test_rhat_reporting_error_does_not_erase_or_veto_evidence(tmp_path, monkeypatch):
    import bayesfilter.inference.hmc as hmc
    binding = make_binding()
    monkeypatch.setattr(hmc, "_rhat_summary_from_retained_samples",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("report unavailable")))
    run = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(3,), epsilons=((3, (1.3,)),)), output_dir=tmp_path)
    assert run.result.verified_candidate_ids
    assert all("RuntimeError" in e["rhat_reporting_only"]["unavailable_reason"] for e in binding._evidence.values())
    with pytest.raises(FileExistsError):
        run_typed_hmc_candidate_set(binding.typed_adapter, run.result.config, output_dir=tmp_path)


def test_public_ordinary_configuration_uses_common_search_and_retains_multiple_members(tmp_path, monkeypatch):
    from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
    import bayesfilter.inference.hmc_kernel_tuning as legacy
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    monkeypatch.setattr(legacy, "_run_canonical_hmc_tuning",
                        lambda **kw: pytest.fail("public call entered historical selection"))
    search = _config(grid=(2, 3), epsilons=((2, (1.1, 1.3, 1.5)), (3, (1.1, 1.3, 1.5))))
    run = tune_hmc_kernel(adapter=GaussianTarget(),
        initial_position=[[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]],
        config=HMCKernelTuningConfig.smoke(mass_policy="fixed_identity", use_xla=False,
            target_scope="candidate-bridge-test", target_status_trace_policy="per_chain_step"),
        search_config=search, execution_config=execution_config(), output_dir=tmp_path)
    assert len(run.result.verified_candidate_ids) >= 2
    assert {(c.leapfrog_steps, c.epsilon) for c in run.result.candidates[:6]} == {
        (l, e) for l in (2, 3) for e in (1.1, 1.3, 1.5)}
    for cid in run.result.verified_candidate_ids[:2]:
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=run.result, candidate_id=cid, retained_binding=run.adapter._execution_binding)
        member.export(tmp_path / (cid.rsplit(":", 1)[-1] + ".json"))


def test_public_windowed_preparation_feeds_automatic_broad_pilot(tmp_path):
    from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    run = tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[.2, -.3],
        parameter_scales=[1., 1.],
        config=HMCKernelTuningConfig.smoke(target_scope="candidate-bridge-test"),
        target_lineage={"model":"standard Gaussian", "prior":"standard normal", "data":"none"},
        source_paths=[__file__], output_dir=tmp_path, max_work_items=1)
    assert run.result.config.primary_l_grid == (3, 5, 9, 13, 18, 25)
    assert run.result.config.pilot_enabled and run.result.config.refinement_rounds == 1
    assert run.result.work_items[0].stage == "pilot"
    assert len(run.result.observations) == 1
    binding = run.adapter._execution_binding
    assert len(binding._transforms) == 2
    assert binding._spec["preparation"]["epsilon_proposal_bound"]["upper"] == binding.scope.epsilon_domain[1]


def test_public_fixed_transport_config_enters_common_scheduler(tmp_path, monkeypatch):
    from bayesfilter.inference import FixedTransportHMCKernelTuningConfig, tune_fixed_transport_hmc_kernel
    import bayesfilter.inference.fixed_transport_hmc_tuning_tf as legacy
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    target = GaussianTarget()
    payload = {"schema": "bayesfilter.neutra.frozen_affine_diag.v1", "transport_id": "public-test",
        "target_signature": target.adapter_signature(), "dimension": 2, "shift": [.0, .0], "raw_scale": [.0, .0],
        "log_jacobian_available": True}
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=target.adapter_signature())
    monkeypatch.setattr(legacy, "_run_historical_fixed_transport_hmc_tuning",
                        lambda **kw: pytest.fail("public call entered historical selection"))
    run = tune_fixed_transport_hmc_kernel(base_adapter=target, fixed_transport=loaded.transport,
        frozen_transport_payload=payload, initial_position=[[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]],
        config=FixedTransportHMCKernelTuningConfig(initial_step_size=1.3, leapfrog_grid=(3, 5),
            use_xla=False, target_status_trace_policy="per_chain_step", target_scope="candidate-bridge-test"),
        execution_config=execution_config(), search_config=_config(grid=(3,), epsilons=((3, (1.3,)),)),
        output_dir=tmp_path)
    assert run.adapter.adapter_kind == "fixed_transport"
    assert run.result.verified_candidate_ids


@pytest.mark.parametrize("target", [.7, .8])
def test_position_field_public_route_uses_common_result_and_resumes(tmp_path, target):
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    from bayesfilter.inference import FourChainMeanBandAcceptancePolicy, tune_hmc_kernel, resume_position_field_candidate_tuning
    cfg = replace(position_config(), verification_results=64, max_leapfrog_steps=5, target_accept_prob=target,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(target-.05,target+.05), per_chain_band=(target-.15,target+.15)))
    search = _config(grid=(3, 5), epsilons=((3, (1.3,)), (5, (1.3,))), evidence_rungs=(1,))
    first = tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4,2], tf.float64),
        parameter_scales=tf.ones([2], tf.float64), config=cfg, search_config=search,
        runner_binding=_binding(), output_dir=tmp_path, max_work_items=1)
    assert first.result.completion_status == "partial_budget"
    second = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
                                                    adapter=_Adapter(), runner_binding=_binding())
    assert second.result.completion_status == "complete"
    assert not second.numerical_handoff_authority
    assert {c.leapfrog_steps for c in second.result.candidates} == {3, 5}
    assert len(second.result.observations) >= 2


def test_position_field_recovers_completed_chunk_after_interruption(tmp_path, monkeypatch):
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config
    from bayesfilter.inference import tune_hmc_kernel, resume_position_field_candidate_tuning, FourChainMeanBandAcceptancePolicy
    from bayesfilter.inference.hmc_candidate_set_position_field import _MechanicsCheckpoint
    cfg = replace(position_config(), verification_results=320, max_leapfrog_steps=3,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.55,.85), per_chain_band=(.5,.9)))
    search = _config(grid=(3,), epsilons=((3, (1.3,)),), evidence_rungs=(1,))
    original = _MechanicsCheckpoint.write_checkpoint
    interrupted = False
    def checkpoint(runtime, result, root):
        nonlocal interrupted
        original(runtime, result, root)
        if any(runtime.partial.values()) and not interrupted:
            interrupted = True
            raise RuntimeError("simulated process interruption after durable chunk")
    monkeypatch.setattr(_MechanicsCheckpoint, "write_checkpoint", checkpoint)
    with pytest.raises(RuntimeError, match="simulated process"):
        tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4,2], tf.float64),
            parameter_scales=tf.ones([2], tf.float64), config=cfg, search_config=search,
            runner_binding=_binding(), output_dir=tmp_path)
    payload = json.loads((tmp_path / "controller_checkpoint.json").read_text())
    assert len(next(iter(payload["partial_chunks"].values()))) == 1
    resumed = resume_position_field_candidate_tuning(tmp_path / "controller_checkpoint.json",
                                                    adapter=_Adapter(), runner_binding=_binding())
    first = resumed.result.observations[0]["observation"]
    assert [p["count"] for p in first["chunks"]] == [256, 64]
    assert resumed.result.completion_status == "complete"
    assert resumed.result.search_state["elapsed_seconds"] >= first["elapsed_seconds"]


def test_preparation_timing_cannot_change_candidate_identity_or_random_stream():
    first = make_binding(config=execution_config(preparation_elapsed_seconds=1.))
    later = make_binding(config=execution_config(preparation_elapsed_seconds=19.))
    left = HMCTuningCandidateSetController(first.scope, _config()).result()
    right = HMCTuningCandidateSetController(later.scope, _config()).result()
    assert first.scope == later.scope
    assert left.candidates == right.candidates
    assert first.work_seed(left.work_items[0]) == later.work_seed(right.work_items[0])


def test_legacy_custom_repair_option_fails_before_preparation():
    from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
    with pytest.raises(ValueError, match="trajectory_window_lower_multiplier"):
        tune_hmc_kernel(adapter=object(), initial_position=[0.],
            config=HMCKernelTuningConfig.smoke(trajectory_window_lower_multiplier=.2))


def test_consumer_exports_every_verified_member_for_later_explicit_selection(tmp_path, monkeypatch):
    import bayesfilter.inference as public
    from bayesfilter.inference.neutra_end_to_end import _export_verified_tuning_members
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    binding = make_binding()
    run = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(2,3), epsilons=((2,(1.1,1.3,1.5)),(3,(1.1,1.3,1.5)))))
    assert len(run.result.verified_candidate_ids) >= 2
    # CPU composition exception: the GPU admission guard has separate tests.
    monkeypatch.setattr(public, "build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result",
                        public.build_retained_bound_hmc_archive_runner_from_candidate_set_result)
    members = _export_verified_tuning_members(run, tmp_path)
    assert set(members) == set(run.result.verified_candidate_ids)
    for cid, path in members.items():
        restored = public.load_hmc_candidate_retained_runner(path, adapter=GaussianTarget())
        assert restored.candidate.candidate_id == cid


def test_resume_materializes_repair_when_crash_follows_last_cohort_receipt():
    controller = HMCTuningCandidateSetController(_scope(), _config(grid=(3,), epsilons=((3, (.25,)),)))
    snapshots = []
    def checkpoint(result):
        if result.repair_actions and all(w.status == "completed" for w in result.work_items):
            snapshots.append(result)
    controller.run(lambda work,c: {"decision": "repair_step_higher"} if c.parent_candidate_id is None else _pass(work,c),
                   checkpoint=checkpoint)
    interrupted = snapshots[0]
    resumed = HMCTuningCandidateSetController.from_result_payload(interrupted.payload()).run(_pass)
    assert resumed.verified_candidate_ids
    assert resumed.repair_actions[0].qualified_repair_status == "executed_and_verified"


def test_numerical_tuning_resumes_in_fresh_process_and_exports_two_members(tmp_path):
    import os
    import subprocess
    import sys
    binding = make_binding()
    cfg = _config(grid=(2,3), epsilons=((2,(1.1,1.3,1.5)),(3,(1.1,1.3,1.5))))
    first = run_typed_hmc_candidate_set(binding.typed_adapter, cfg, output_dir=tmp_path, max_work_items=1)
    assert len(first.result.observations) == 1
    script = '''
import json, sys
from pathlib import Path
from tests.test_hmc_candidate_set_execution import GaussianTarget
from bayesfilter.inference import resume_hmc_candidate_set_tuning, build_retained_bound_hmc_archive_runner_from_candidate_set_result
root=Path(sys.argv[1])
run=resume_hmc_candidate_set_tuning(root/'tuning_checkpoint.json', adapter=GaussianTarget())
assert len(run.result.verified_candidate_ids)>=2
for index, cid in enumerate(run.result.verified_candidate_ids[:2]):
    member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(candidate_set_result=run.result, candidate_id=cid, retained_binding=run.adapter._execution_binding)
    member.export(root/f'member-{index}.json')
(root/'resume-check.json').write_text(json.dumps(run.result.payload()))
'''
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES":"-1", "TF_FORCE_GPU_ALLOW_GROWTH":"true"},
        capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "member-1.json").is_file()
    resumed = json.loads((tmp_path / "resume-check.json").read_text())
    assert resumed["observations"][0] == json.loads(json.dumps(first.result.observations[0]))
    measurements = {row["candidate_id"] for row in resumed["observations"] if row["stage"] == "measurement"}
    verifications = {row["candidate_id"] for row in resumed["observations"] if row["stage"] == "verification"}
    assert len(measurements) == 6
    assert set(resumed["verified_candidate_ids"]) <= verifications


def test_selected_member_runs_actual_sequential_posterior_controller(tmp_path):
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig
    binding = make_binding()
    run = run_typed_hmc_candidate_set(binding.typed_adapter, _config(grid=(3,), epsilons=((3,(1.3,)),)))
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0], retained_binding=binding)
    # Tiny mechanics cap: a posterior rejection is a legitimate outcome and
    # must not erase the separately verified tuning member.
    cfg = SequentialNeuTraHMCConfig(step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
        warmup_seed=(20260915,5), retained_seed=(20260915,6), jit_compile=False,
        warmup_chunk_results=16, warmup_min_results=16, warmup_check_window_results=16, warmup_max_results=16,
        retained_chunk_results=16, retained_min_results=16, retained_max_results=16)
    result = member.run_sequential(config=cfg, parameter_names=("x","y"))
    assert isinstance(result, dict)
    assert run.result.verified_candidate_ids
    assert member.candidate.candidate_id in run.result.verified_candidate_ids
    with pytest.raises(ValueError, match="preserve"):
        member.run_sequential(config=replace(cfg, step_size=.4), parameter_names=("x","y"))
