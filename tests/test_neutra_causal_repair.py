"""CPU reference checks of actual repair boundaries; no posterior claims."""
import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy, _acceptance_policy_from_payload,
    _acceptance_decision_from_summary, temporal_block_conflicts)
from bayesfilter.testing.neutra_warm_start_policy import fit_repair_decision, teacher_repair_settings
from bayesfilter.testing.neutra_warm_start_campaign import make_transport, TrainingBlock, write_json, read_tensor
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, F64
from bayesfilter.testing.neutra_warm_start_closure import FrozenMapTarget, LatentDefensiveProposal
from bayesfilter.inference.neutra_warm_start_tf import AnnealedSMC, SMCConfig


def test_optional_temporal_policy_preserves_legacy_codec_and_binds_new_semantics():
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
    for method,schema in [('raw_block_crossing_v1','v5'),('paired_chain_block_contrasts_v1','v6')]:
        policy=HMCAcceptancePolicy(temporal_conflict_method=method)
        payload=policy.payload()
        assert payload['schema'].endswith(schema)
        assert _acceptance_policy_from_payload(payload)==policy
        execution=HMCCandidateExecutionConfig(measurement_num_results=64,verification_num_results=64,
            num_warmup_steps=0,seed=(11,9),acceptance_policy=policy,
            target_status_trace_policy='none')
        assert HMCCandidateExecutionConfig.from_payload(execution.payload())==execution
    bad=HMCAcceptancePolicy().payload();bad['schema']='bayesfilter.hmc_acceptance_policy.v6'
    with pytest.raises(ValueError):_acceptance_policy_from_payload(bad)


def test_temporal_contrasts_detect_common_drift_without_raw_single_chain_crossings():
    blocks=tf.constant([[[.68,.79,.635,.768],[.7,.7,.7,.7],[.69,.71,.70,.70],[.71,.69,.70,.70]],
                        [[.2,.4,.8,.95]]*4],F64)
    assert temporal_block_conflicts(blocks,practical_width=.1).numpy().tolist()==[False,True]
    common=dict(pooled_mean=.7,chain_means=tf.reduce_mean(blocks[0],1),block_means=blocks[0],
        uncertainty_interval=(.66,.73),movement=tf.ones([4],F64),repeated=tf.zeros([4],F64),
        normalized_return=tf.ones([4],F64),path_return=tf.zeros([4],F64))
    assert _acceptance_decision_from_summary(policy=HMCAcceptancePolicy(),**common)=='inconclusive_evidence'
    assert _acceptance_decision_from_summary(policy=HMCAcceptancePolicy(
        temporal_conflict_method='paired_chain_block_contrasts_v1'),**common)=='passed'


@pytest.mark.parametrize('target_name',['gaussian','mixture','funnel'])
def test_real_qualification_passes_map_draws_and_effective_repairs_to_public_tuner(tmp_path,monkeypatch,target_name):
    from bayesfilter.testing import neutra_warm_start_qualification as consumer
    target=WarmStartTarget(target_name,jit_compile=False)
    flow=make_transport(target,2*target.parameter_dim,(11,91))
    training=tmp_path/'training';training.mkdir()
    write_json(training/'selected-frozen.json',flow.frozen_payload(target_signature=target.signature))
    captured={}
    class ReachedTuner(Exception):pass
    def tuner(**kwargs):
        captured.update(kwargs)
        raise ReachedTuner()
    monkeypatch.setattr(consumer,'tune_fixed_transport_hmc_kernel',tuner)
    cfg={'hmc':{'leapfrogs':[3],'initial_epsilon':.5,'max_candidates':36,'work_units':144,
                'job_wall_seconds':30,'fixed_grid_max_attempts':12,
                'temporal_conflict_method':'paired_chain_block_contrasts_v1'}}
    out=tmp_path/'result'
    with pytest.raises(ReachedTuner):
        consumer.qualify(target_name,23,tmp_path/'deliberately_missing_modes',training,out,cfg,
                         frozen_filename='selected-frozen.json',discovered_starts=True)
    expected=tf.random.stateless_normal([4,target.parameter_dim],[23,702],dtype=F64)
    tf.debugging.assert_equal(captured['initial_position'],expected)
    tf.debugging.assert_near(read_tensor(out/'initial-physical.tensor'),flow.forward_batch(expected))
    assert captured['config'].fixed_grid_max_attempts==12
    assert captured['execution_config'].acceptance_policy.temporal_conflict_method=='paired_chain_block_contrasts_v1'
    assert json.loads((out/'initialization.json').read_text())['source']=='frozen_map_standard_normal'


@pytest.mark.parametrize('kind',['forward','gabrie'])
def test_real_adam_restoration_produces_same_next_update(kind):
    target=WarmStartTarget('mixture',jit_compile=False)
    flow=make_transport(target,4,(11,91))
    options=dict(batch=8,learning_rate=.001,clip=100.,kind=kind,walkers=4,walk_steps=2,jit_compile=False)
    block=TrainingBlock(flow,target,**options)
    pool=target.reference_sample(32,tf.constant([41,2]));weights=tf.zeros([32],F64)
    current=tf.zeros([4,2],F64)
    first=block.run(tf.constant([11,4]),tf.constant(3),pool,weights,current,tf.constant(.01,F64))
    current=first[1]
    state=block.checkpoint()
    restored_flow=make_transport(target,4,(11,91));restored=TrainingBlock(restored_flow,target,**options)
    restored.restore(state)
    assert restored.checkpoint()['state_hash']==state['state_hash']
    a=block.run(tf.constant([11,9]),tf.constant(2),pool,weights,current,tf.constant(.01,F64))
    b=restored.run(tf.constant([11,9]),tf.constant(2),pool,weights,current,tf.constant(.01,F64))
    tf.debugging.assert_near(a[2],b[2],atol=1e-12)
    tf.debugging.assert_near(a[1],b[1],atol=1e-12)
    assert restored.checkpoint()['step']==5
    for x,y in zip(flow.trainable_variables,restored_flow.trainable_variables):tf.debugging.assert_equal(x,y)


def test_repair_decisions_distinguish_progress_plateau_invalid_and_budget():
    row={'assessment':{'finite':True,'passed':False,'nonlinear_learning_passed':True},
         'paired_progress':{'mean_log_density_gain':.1,'standard_error':.01}}
    assert fit_repair_decision([row])=='continue_checkpoint'
    assert fit_repair_decision([row],budget_available=False)=='budget_limited_unfinished'
    row['paired_progress']['standard_error']=.1
    assert fit_repair_decision([row])=='shape_failure_without_supported_progress'
    row['assessment']['nonlinear_learning_passed']=False
    assert fit_repair_decision([row])=='nonlinear_plateau_repair'
    row['assessment']['finite']=False
    assert fit_repair_decision([row])=='numerical_failure'


def test_mutation_repairs_change_actual_work_before_particle_count():
    rows=[teacher_repair_settings(i) for i in range(5)]
    assert [(r['particles'],r['mutation_steps']) for r in rows]==[(1024,4),(1024,16),(1024,64),(4096,64),(8192,64)]
    with pytest.raises(ValueError):teacher_repair_settings(5)


def test_mobility_diagnostic_is_scale_invariant_and_flags_collapsed_coordinates():
    from bayesfilter.testing.neutra_warm_start_closure import scaled_displacement
    jump=tf.constant([.2,3.,0.],F64);variance=tf.constant([2.,12.,0.],F64)
    first=scaled_displacement(jump,variance)
    scaled=scaled_displacement(jump*tf.constant([4.,.01,9.],F64),
                               variance*tf.constant([4.,.01,9.],F64))
    tf.debugging.assert_near(first['variance_scaled_displacement'][:2],
                             scaled['variance_scaled_displacement'][:2])
    assert first['scale_valid'].numpy().tolist()==[True,True,False]
    assert bool(tf.math.is_nan(first['variance_scaled_displacement'][2]).numpy())


def test_continuation_price_uses_actual_updates_and_remaining_budget(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/continue_neutra_causal_repair.py'
    spec=importlib.util.spec_from_file_location('causal_followup_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=object.__new__(module.Followup)
    controller.state={'attempts':[],'active_job':None}
    controller.base={'cpu_threads':2,'gpu_process_seconds':1000.,'cpu_core_seconds':2000.}
    parent=tmp_path/'worker'/'candidate';parent.mkdir(parents=True)
    write_json(parent/'history.json',[{'updates':100,'paired_progress':{'previous_updates':50}}])
    write_json(parent.parent/'manifest.json',{'wall_seconds':10.})
    seconds,enough=controller.price(parent,200)
    assert seconds==70. and enough
    controller.base['gpu_process_seconds']=50.
    assert not controller.price(parent,200)[1]


def test_followup_uses_only_latest_unresolved_continuations(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/continue_neutra_causal_repair.py'
    spec=importlib.util.spec_from_file_location('causal_pending_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=object.__new__(module.Followup)
    pending={'status':'improving_fit_at_cap','evidence':'unused'}
    controller.record={'outcomes':{'a':pending,'a-continue':{'status':'posterior_screen_passed'},
        'b':pending,'b-progress':pending,'c':pending}}
    assert [name for name,_ in controller.pending_continuations()]==['b-progress','c']
    parent=tmp_path/'previous'/'candidate';parent.mkdir(parents=True)
    teacher=tmp_path/'exact-teacher'
    write_json(parent.parent/'manifest.json',{'input_sha256':{
        str(teacher/'teacher-particles.tensor'):'a',str(teacher/'teacher-log-weights.tensor'):'b'}})
    assert controller.teacher_for(parent)==teacher
    write_json(parent.parent/'manifest.json',{'input_sha256':{}})
    with pytest.raises(ValueError,match='missing or ambiguous'):controller.teacher_for(parent)


def test_historical_failed_case_enumeration_includes_warped_gabrie(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/continue_neutra_causal_repair.py'
    spec=importlib.util.spec_from_file_location('causal_case_coverage_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=object.__new__(module.Followup);controller.root=tmp_path
    parent=tmp_path/'attempts/closure-fit-warped_mixture-gabrie-s37-v0-r1/w8-lr0.001'
    write_json(parent/'history.json',[{'updates':8192,'assessment':{
        'finite':True,'nonlinear_learning_passed':True}}])
    controller.historical={'outcomes':[{'status':'nonlinear_fit_repair_exhausted',
        'target':'warped_mixture','arm':'gabrie','seed':37}]}
    controller.state={};controller.base={}
    monkeypatch.setattr(module.master,'recover_active',lambda *a:None)
    controller.calibrate=lambda:None;controller.prepare=lambda target:tmp_path/'prepared'
    controller.refresh=lambda *a,**k:None
    teacher_calls=[];controller.phase=lambda *a,**k:teacher_calls.append((a,k))
    calls=[];controller.extend=lambda *a,**k:calls.append((a,k))
    controller.remaining_audit_cases()
    assert len(calls)==1 and calls[0][0][1:3]==('warped_mixture',37)
    assert calls[0][0][-1]==parent and calls[0][1]['arm']=='gabrie'
    assert teacher_calls[0][1]['level']==2


def test_supervisor_rejects_failed_refinement_before_reference_generation(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/run_neutra_causal_repair_master.py'
    spec=importlib.util.spec_from_file_location('causal_refine_veto_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=object.__new__(module.Controller)
    controller.record={'outcomes':{}};controller.refresh=lambda *a,**k:None
    write_json(tmp_path/'phase.json',{'passed':False,'reason':'numerical_failure'})
    assert controller.qualify('invalid','mixture',11,tmp_path,tmp_path,'smc') is False
    assert controller.record['outcomes']['invalid']['status']=='numerical_failure'


def test_frozen_coordinate_change_includes_jacobian_and_correct_importance_ratio():
    target=WarmStartTarget('gaussian',jit_compile=False)
    flow=make_transport(target,4,(11,91));pulled=FrozenMapTarget(target,flow)
    proposal=LatentDefensiveProposal(2)
    z=tf.constant([[-1.,.2],[.1,1.],[2.,-1.]],F64)
    x,ld=flow.forward_and_logdet(z)
    physical_log_q=proposal.log_prob(z)-ld
    tf.debugging.assert_near(pulled.log_prob(z)-proposal.log_prob(z),target.log_prob(x)-physical_log_q,atol=1e-12)
    with tf.GradientTape() as tape:
        tape.watch(z);value=pulled.log_prob(z)
    analytic=tape.gradient(value,z)
    direction=tf.constant([[.3,.7],[.4,-.3],[.2,.5]],F64);eps=tf.constant(1e-5,F64)
    finite_difference=(pulled.log_prob(z+eps*direction)-pulled.log_prob(z-eps*direction))/(2*eps)
    tf.debugging.assert_near(tf.reduce_sum(analytic*direction,1),finite_difference,atol=1e-7)


def test_smc_schedule_is_validated_and_exact_identity_bridge_keeps_normalizer():
    proposal=LatentDefensiveProposal(2)
    target=SimpleNamespace(parameter_dim=2,log_prob=proposal.log_prob)
    cfg=SMCConfig(32,4,16,.8,.5,.01,jit_compile=False,step_size_schedule=((0.,.01),(.5,.1)))
    result=AnnealedSMC(target,proposal,cfg).run(tf.constant([3,9]))
    assert result['complete']
    assert abs(float(result['log_normalizer'].numpy()))<1e-12
    with pytest.raises(ValueError):replace(cfg,step_size_schedule=((.5,.1),))


def test_public_tuner_actually_reaches_below_old_repair_boundary(tmp_path):
    """Stiff Gaussian CPU reference: actual HMC proposals, not a fake evaluator."""
    from bayesfilter.inference import FixedTransportHMCKernelTuningConfig, tune_fixed_transport_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    class StiffGaussian:
        parameter_dim=2
        def adapter_signature(self):return hashlib.sha256(b'causal_repair_stiff_gaussian_precision10000_v1').hexdigest()
        def value_score_capability(self):
            return ValueScoreCapability(value_score_authority='graph_native',xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,runtime_backend='tensorflow',
                target_scope='causal_stiff_reference',nonclaims=('CPU engineering reference, not trained map evidence',))
        def log_prob_and_grad(self,x):
            x=tf.convert_to_tensor(x,F64)
            return -.5*10000.*tf.reduce_sum(x*x,axis=-1),-10000.*x
    target=StiffGaussian()
    flow=make_transport(WarmStartTarget('gaussian',jit_compile=False),4,(11,91))
    payload=flow.frozen_payload(target_signature=target.adapter_signature())
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    flow=load_frozen_neutra_artifact(payload,expected_target_signature=target.adapter_signature()).transport
    execution=HMCCandidateExecutionConfig(measurement_num_results=64,verification_num_results=64,
        num_warmup_steps=0,seed=(31,12),acceptance_policy=HMCAcceptancePolicy(),
        target_status_trace_policy='none',use_xla=False,
        non_xla_reason='tiny stiff Gaussian search-domain reference',chain_mode='batched')
    starts=tf.constant([[-.01,-.005],[-.003,.002],[.004,-.002],[.01,.005]],F64)
    minima=[]
    for repairs in (3,8):
        run=tune_fixed_transport_hmc_kernel(base_adapter=target,fixed_transport=flow,
            frozen_transport_payload=payload,initial_position=starts,
            config=FixedTransportHMCKernelTuningConfig(initial_step_size=.5,maximum_candidate_step_size=.5,
                leapfrog_grid=(3,),fixed_grid_max_attempts=repairs,use_xla=False),
            search_config=HMCControllerConfig(primary_l_grid=(3,),initial_epsilon=.5,pilot_enabled=True,
                evidence_rungs=(1,),refinement_rounds=0,max_candidates=16,total_budget_units=24,repair_reserve_units=6),
            execution_config=execution,output_dir=tmp_path/f'repairs-{repairs}')
        minima.append(min(c.epsilon for c in run.result.candidates))
        assert run.result.scope.max_repairs_per_family==repairs
    assert minima[0]==.0625
    assert minima[1]<minima[0]
