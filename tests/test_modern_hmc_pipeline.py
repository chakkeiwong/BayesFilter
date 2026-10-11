"""Complete public tuning/reload mechanics for optional corrected profiles.

The deliberately broad engineering acceptance floor prices a wiring test;
it is not a proposed scientific tuning default or convergence certificate.
"""
from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    HMCControllerConfig, tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    load_hmc_candidate_retained_runner,
)
from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile, candidate_gradient_work
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config


def setup(*, family="hmc", transport=False):
    policy = HMCReplicatedAcceptancePolicy(trial_num_results=64,discarded_prefix=0,
        base_repetitions=16,max_repetitions=16,max_candidates=2,
        search_family_alpha=.05,verification_family_alpha=.05,diagnostic_family_alpha=.05,
        temporal_tolerance=.1,qualification_mode="lower_bound",qualification_region=(.1,1.),
        min_normalized_return_displacement=0.)
    execution=execution_config(measurement_num_results=64,verification_num_results=64,
        num_warmup_steps=0,acceptance_policy=policy,chain_mode="batched",chunk_max_results=64,
        transition_profile=HMCTransitionProfile(family,"uniform_duration"))
    extra={}
    if transport:
        target=GaussianTarget()
        extra=dict(target=target,mass_artifact=None,start_coordinates="active",
            frozen_transport_payload={"schema":"bayesfilter.neutra.frozen_affine_diag.v1",
                "transport_id":"analytic-mechanics-map", "dimension":2,
                "target_signature":target.adapter_signature(),"log_jacobian_available":True,
                "shift":[.2,-.1],"raw_scale":[.1,.05]})
    binding=make_binding(config=execution,**extra)
    search=HMCControllerConfig(primary_l_grid=(3,5),epsilon_by_l=((3,(.2,)),(5,(.2,))),
        total_budget_units=10,repair_reserve_units=1,max_candidates=2,evidence_rungs=(1,),
        replicated_acceptance_policy=policy)
    return binding, search


@pytest.mark.parametrize("family", ["hmc","mams"])
@pytest.mark.parametrize("transport", [False,True])
def test_both_public_entrypoints_retain_and_reload_every_modern_member(family,transport,tmp_path):
    from bayesfilter.inference.hmc_modern_kernels import ReusableModernHMCRunner
    binding,search=setup(family=family,transport=transport)
    args=dict(initial_position=binding.initial_active_state,config=search,
        candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/"tuning")
    if transport:
        run=tune_fixed_transport_hmc_kernel(base_adapter=binding._base_adapter,
            fixed_transport=binding.fixed_transport,**args)
    else:
        run=tune_hmc_kernel(adapter=binding._base_adapter,**args)
    result=run.result
    assert len(result.verified_candidate_ids)==2, result.candidate_states
    assert result.payload()["nominee_id"] is None
    candidates={c.candidate_id:c for c in result.candidates}
    works={w.work_item_id:w for w in result.work_items}
    for event in result.accounting_events:
        if event["event"]=="numerical_chunk_charged":
            assert event["gradient_work"]==event["transitions"]*candidate_gradient_work(candidates[works[event["work_item_id"]].candidate_id])
    assert binding._runners
    assert all(isinstance(r,ReusableModernHMCRunner) for r in binding._runners.values())
    for i,cid in enumerate(result.verified_candidate_ids):
        member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=result,candidate_id=cid,retained_binding=binding)
        path=member.export(tmp_path/f"member-{i}.json")
        loaded=load_hmc_candidate_retained_runner(path,adapter=binding._base_adapter)
        assert loaded.candidate.transition_spec.profile.family==family
        kw=dict(num_results=6,seed=(999,2+i))
        first=member.run(**kw,output_dir=tmp_path/f"first-{i}")
        replay=loaded.run(**kw,output_dir=tmp_path/f"replay-{i}")
        np.testing.assert_array_equal(first["samples"],replay["samples"])
        direct=binding._run(member.candidate,member.initial_active_state,6,(999,2+i))
        np.testing.assert_array_equal(first["samples"],direct.samples)
        intended=ReusableModernHMCRunner(binding._active_adapter,member.initial_active_state,
            profile=binding.config.transition_profile,num_results=6,use_xla=False,
            target_status_trace_policy=binding.config.target_status_trace_policy).run(
                current_state=member.initial_active_state,seed=(999,2+i),
                step_size=member.candidate.epsilon,num_leapfrog_steps=member.candidate.leapfrog_steps)
        np.testing.assert_array_equal(first["samples"],intended.samples)
        assert direct.metadata["transition_spec"]==member.candidate.transition_spec.payload()
        assert len(set(direct.trace["trajectory_steps"].numpy().ravel()))>1
        if family=="mams":
            np.testing.assert_allclose(tf.linalg.norm(direct.trace["initial_momentum"],axis=-1),1.,atol=1e-14)
    from bayesfilter.inference import build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result
    with pytest.raises(ValueError,match="modern transition profiles are experimental"):
        build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(
            candidate_set_result=result,candidate_id=result.verified_candidate_ids[0],retained_binding=binding)


def test_new_execution_rejects_unsupported_topology_and_policy_before_work():
    binding,_=setup()
    with pytest.raises(ValueError,match="batched"):
        replace(binding.config,chain_mode="serial")
    with pytest.raises(ValueError,match="one-sided"):
        replace(binding.config,acceptance_policy=replace(binding.config.acceptance_policy,
            qualification_mode="two_sided",qualification_region=(.1,.99)))


def test_modern_profile_cannot_inherit_automatic_hmc_step_bound():
    from bayesfilter.inference.hmc_candidate_set_public import _execution
    binding,_=setup(family="mams")
    with pytest.raises(ValueError,match="explicitly prepared geometry"):
        _execution(binding.config,seed=binding.config.seed,policy=binding.config.acceptance_policy,
            status=binding.config.target_status_trace_policy,use_xla=binding.config.use_xla)


@pytest.mark.parametrize("defect",["missing_length","missing_work","wrong_calls"])
def test_legacy_or_inconsistent_native_trace_cannot_validate_as_modern(defect):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    binding,_=setup()
    candidate=HMCTuningCandidateRecord.create(binding.scope,leapfrog_steps=3,epsilon=.2,creation_ordinal=1)
    result=binding._run(candidate,binding.initial_active_state,2,(941,778))
    trace=dict(result.trace)
    if defect=="missing_length": del trace["trajectory_steps"]
    elif defect=="missing_work": del trace["integration_work"]
    else: trace["value_score_calls_per_chain"]=tf.zeros([2,4],tf.int32)
    with pytest.raises(ValueError,match="modern transition"):
        binding.health_failures(binding.initial_active_state,result.samples,trace)


def test_modern_cost_preflight_rejects_budget_that_only_funds_legacy_work(tmp_path,monkeypatch):
    binding, search = setup()
    search = replace(search, max_gradient_work=100000)
    def forbidden(*args, **kwargs):
        pytest.fail("an underfunded scope must fail before native work")
    monkeypatch.setattr(binding, "_run", forbidden)
    with pytest.raises(ValueError, match="modern transition primary cohort"):
        tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
            config=search, candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path)


def test_interrupted_modern_trial_reuses_identity_without_double_evidence(tmp_path,monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_checkpoint import resume_hmc_candidate_set_tuning
    binding,search=setup()
    original=binding._run
    calls=[]
    def failing(candidate,state,count,seed):
        calls.append(seed)
        if len(calls)==3:
            raise tf.errors.UnavailableError(None,None,"injected modern native interruption")
        return original(candidate,state,count,seed)
    monkeypatch.setattr(binding,"_run",failing)
    run=tune_hmc_kernel(adapter=binding._base_adapter,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert len(run.result.verified_candidate_ids)<2
    recovered=resume_hmc_candidate_set_tuning(tmp_path/"tuning_checkpoint.json",adapter=binding._base_adapter)
    assert len(recovered.result.verified_candidate_ids)==2
    assert all(c.transition_spec is not None for c in recovered.result.candidates)


@pytest.mark.parametrize("family",["hmc","mams"])
def test_actual_nonlinear_ssm_tuning_and_retained_continuation(family,tmp_path):
    from bayesfilter.inference import bind_hmc_candidate_set_execution,PrecomputedMassArtifact
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    fixture=make_nonlinear_comparison_target("NL-ACC",jit_compile=False)
    template,search=setup(family=family)
    adapter=fixture.adapter
    factor=tf.eye(3,dtype=tf.float64)
    binding=bind_hmc_candidate_set_execution(adapter=adapter,initial_position=fixture.initial_state,
        config=replace(template.config,target_status_trace_policy="none"),
        target_scope=adapter.value_score_capability().target_scope,
        target_lineage={"target":"NL-ACC","data":fixture.metadata["data_sha256"],"purpose":"mechanics"},
        source_paths=fixture.source_paths,scope_id="nonlinear-modern",search_id="mechanics",
        epsilon_domain=(.001,.1),repair_factor=1.1,max_repairs_per_family=0,
        mass_artifact=PrecomputedMassArtifact(position=[0.]*3,factor=factor,covariance=factor,
            adapter_signature=adapter.adapter_signature(),position_role="reference_center",
            covariance_source="explicit identity mechanics hypothesis"))
    search=replace(search,primary_l_grid=(2,),epsilon_by_l=((2,(.01,)),))
    run=tune_hmc_kernel(adapter=adapter,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/"tuning")
    assert len(run.result.verified_candidate_ids)==1,run.result.candidate_states
    member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result,candidate_id=run.result.verified_candidate_ids[0],retained_binding=binding)
    path=member.export(tmp_path/"member.json")
    first=member.run(num_results=4,seed=(673,411),output_dir=tmp_path/"first")
    loaded=load_hmc_candidate_retained_runner(path,adapter=adapter)
    second=loaded.run(num_results=4,seed=(673,412),previous_archive=first["archive_path"],output_dir=tmp_path/"second")
    direct=binding._run(member.candidate,first["final_active_state"],4,(673,412))
    np.testing.assert_array_equal(second["samples"],direct.samples)
    assert np.all(np.isfinite(second["samples"]))


def test_full_status_adapter_preserves_actual_predator_prey_value_score_and_status():
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    fixture=make_nonlinear_comparison_target("NL-PP-UKF",jit_compile=False)
    wrapper=fixture.adapter
    value,score,status=wrapper.base.neutra_batch_log_prob_and_grad_status(fixture.initial_state)
    actual=wrapper.log_prob_and_grad_status(fixture.initial_state)
    np.testing.assert_array_equal(actual[0],value)
    np.testing.assert_array_equal(actual[1],score)
    assert set(actual[2])==set(status)
    assert {"floor_count_value","innovation_condition_estimate"} <= set(status)
    for key in status:
        np.testing.assert_array_equal(wrapper.target_status_telemetry(fixture.initial_state)[key],status[key])
