"""Trial-batch evidence, lost-call accounting and real-filter process recovery."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf


def setup(case='gaussian', *, batch_size=2, positive=False, route='ordinary'):
    from tests.test_hmc_candidate_set_execution import execution_config, make_binding, GaussianTarget
    from tests.test_hmc_acceptance_ssm_recovery import setup as ssm_setup
    from tests.test_hmc_acceptance_protocol import policy
    from bayesfilter.inference import bind_hmc_candidate_set_execution
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    if case != 'gaussian':
        target, previous, old_search = ssm_setup(case)
        p = replace(previous.config.acceptance_policy,base_repetitions=3,max_repetitions=6,
                    min_normalized_return_displacement=0.)
        config = replace(previous.config,acceptance_policy=p,chunk_max_results=68,
                         replicated_trial_batch_size=batch_size)
        from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding
        spec = previous._spec
        binding = _issue_binding(adapter=target,layers=spec['layers'],
            initial_active_state=previous.initial_active_state,target_scope=spec['target_scope'],
            target_lineage=spec['target_lineage'],preparation=spec['preparation'],config=config,
            source_paths=[__file__],scope_id='batch-recovery-'+case,search_id='same-stream',
            epsilon_domain=previous.scope.epsilon_domain,repair_factor=1.3,max_repairs_per_family=0)
        return target,binding,replace(old_search,replicated_acceptance_policy=p)
    p=policy(base_repetitions=32 if positive else 3,max_repetitions=128 if positive else 6,
             max_candidates=2,min_normalized_return_displacement=0.)
    config=execution_config(measurement_num_results=65,verification_num_results=65,
        num_warmup_steps=3,acceptance_policy=p,chunk_max_results=68,chain_mode='batched',
        use_xla=True,non_xla_reason=None,reuse_leapfrog_graphs=True,replicated_trial_batch_size=batch_size)
    target=GaussianTarget();extra={}
    if route=='fixed_transport':
        extra=dict(mass_artifact=None,start_coordinates='active',frozen_transport_payload={
            'schema':'bayesfilter.neutra.frozen_affine_diag.v1','transport_id':'batch-identity',
            'dimension':2,'target_signature':target.adapter_signature(),'log_jacobian_available':True,
            'shift':[0.,0.],'raw_scale':[0.,0.]})
    binding=make_binding(target=target,config=config,source_paths=[__file__],**extra)
    search=HMCControllerConfig(primary_l_grid=(1,3) if positive else (2,),
        epsilon_by_l=((1,(1.35,)),(3,(1.4,))) if positive else ((2,(1.3,)),),
        total_budget_units=20,repair_reserve_units=1,max_candidates=2,
        evidence_rungs=(1,2,4) if positive else (1,2),replicated_acceptance_policy=p)
    return target,binding,search


@pytest.mark.parametrize('route',['ordinary','fixed_transport'])
def test_batch_multiple_members_export_reload_and_attempted_seed_exclusion(tmp_path,route):
    from bayesfilter.inference import (tune_hmc_kernel,tune_fixed_transport_hmc_kernel,
        export_hmc_candidate_retained_runners,load_hmc_candidate_retained_runners)
    from bayesfilter.testing.acceptance_decision_models import evidence_accounting
    target,binding,search=setup(batch_size=8,positive=True,route=route)
    kwargs=dict(initial_position=binding.initial_active_state,config=search,
                candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/'tuning')
    run=(tune_hmc_kernel(adapter=target,**kwargs) if route=='ordinary' else
         tune_fixed_transport_hmc_kernel(base_adapter=target,fixed_transport=binding.fixed_transport,**kwargs))
    assert run.result.completion_status=='complete'
    assert len(run.result.verified_candidate_ids)==2
    paths=export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding,output_dir=tmp_path/'members')
    restored=load_hmc_candidate_retained_runners(paths.values(),adapter=target)
    assert set(restored)==set(run.result.verified_candidate_ids)
    charged={tuple(e['seed']) for e in run.result.accounting_events if e['event']=='numerical_chunk_charged'}
    for member in restored.values():assert charged <= member._tuning_seeds()
    accounting=evidence_accounting(binding,run.result)
    assert accounting['attempted_transitions']==accounting['completed_trial_transitions']
    assert accounting['unique_complete_trials']==accounting['independent_trial_streams']


@pytest.mark.parametrize('case',['gaussian','lgssm_qr','nonlinear'])
@pytest.mark.parametrize('boundary',['lost_batch','one_saved_row'])
def test_batch_recovers_same_trial_values_in_fresh_process(case,boundary,tmp_path,monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    from tests.test_hmc_acceptance_ssm_recovery import numerical_summary
    target,baseline,search=setup(case)
    complete=tune_hmc_kernel(adapter=target,initial_position=baseline.initial_active_state,
        config=search,candidate_set_adapter=baseline.typed_adapter)
    expected=numerical_summary(baseline,complete.result)
    _,binding,_=setup(case)
    with monkeypatch.context() as patch:
        if boundary=='lost_batch':
            def interrupt(*args):
                raise tf.errors.UnavailableError(None,None,'injected lost native batch')
            patch.setattr(binding,'_run_replicated_batch',interrupt)
        else:
            from bayesfilter.inference import hmc_candidate_set_execution as execution
            original=execution._json_copy; seen=[]
            def interrupt(value):
                if isinstance(value,dict) and 'trial_ordinal' in value:
                    seen.append(value['trial_ordinal'])
                    if len(seen)==2:
                        raise HMCInfrastructureFailure('injected after first saved trial row')
                return original(value)
            patch.setattr(execution,'_json_copy',interrupt)
        paused=tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
            config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/'tuning')
    assert paused.result.completion_status=='paused_infrastructure'
    code='''
import json,sys
from pathlib import Path
from tests.test_hmc_replicated_batch_integration import setup
from tests.test_hmc_acceptance_ssm_recovery import numerical_summary
from bayesfilter.inference import resume_hmc_candidate_set_tuning
target,_,_=setup(sys.argv[1])
run=resume_hmc_candidate_set_tuning(Path(sys.argv[2])/'tuning/tuning_checkpoint.json',adapter=target)
Path(sys.argv[2],'resumed.json').write_text(json.dumps(numerical_summary(run.adapter._execution_binding,run.result)))
'''
    env={k:os.environ[k] for k in ('PATH','LANG','LD_LIBRARY_PATH') if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',BAYESFILTER_PRELOAD_CUSTOM_OP='0',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (tmp_path/'child.log').open('w') as log:
        child=subprocess.run([sys.executable,'-c',code,case,str(tmp_path)],env=env,
                             stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert child.returncode==0,(tmp_path/'child.log').read_text()[-5000:]
    actual=json.loads((tmp_path/'resumed.json').read_text());expected=json.loads(json.dumps(expected))
    for field in ('states','trials','decisions'):assert actual[field]==expected[field]
    extra_trials=2 if boundary=='lost_batch' else 1
    expected_extra=extra_trials*68*4*(search.primary_l_grid[0]+1)
    assert actual['accounting']['unique_complete_trials']==6
    assert actual['accounting']['gradient_work']==expected['accounting']['gradient_work']+expected_extra
    assert actual['accounting']['attempted_work_outside_complete_trials']==expected_extra


@pytest.mark.parametrize('damage',[dict(replicated_trial_batch_size=33),
    dict(replicated_trial_batch_size=True),dict(chain_mode='serial'),dict(chunk_max_results=34)])
def test_batch_configuration_rejects_unsupported_paths(damage):
    _,binding,_=setup()
    with pytest.raises(ValueError):replace(binding.config,**damage)


def test_default_execution_payload_retains_legacy_shape():
    from bayesfilter.inference import HMCCandidateExecutionConfig
    _,binding,_=setup(batch_size=1)
    payload=binding.config.payload()
    assert 'replicated_trial_batch_size' not in payload
    assert HMCCandidateExecutionConfig.from_payload(payload).payload()==payload


def test_batch_seed_metadata_tampering_is_detected():
    from copy import deepcopy
    from bayesfilter.inference import tune_hmc_kernel
    target,binding,search=setup()
    tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter)
    row=deepcopy(next(iter(binding._evidence.values())))
    row['chunks'][0]['runtime']['trial_batch_seeds'][0][0]+=1
    with pytest.raises(ValueError,match='trial batch streams'):
        binding.evidence_analysis(row)


def test_classified_failed_batch_preserves_every_attempt_and_no_scores(monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference.hmc_acceptance_trials import initialize_seed_registry
    target,binding,search=setup()
    monkeypatch.setattr(target,'classify_target_exception',lambda error:isinstance(error,ArithmeticError),raising=False)
    def fail(*args):raise ArithmeticError('injected target failure in an unspecified batch row')
    monkeypatch.setattr(binding,'_run_replicated_batch',fail)
    run=tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter)
    row=next(iter(binding._evidence.values()))
    assert row['failed_batch_size']==2 and row['chunks']==[]
    assert row['analysis']['trial_outcomes']['failed_batch_trial_ordinals']==[0,1]
    assert row['analysis']['trial_outcomes']['unstarted_trials']==1
    assert not run.result.verified_candidate_ids
    assert json.loads(json.dumps(binding.evidence_analysis(row)))==row['analysis']
    initialize_seed_registry(binding,run.result)
    events=list(run.result.accounting_events)
    missing=next(i for i,event in enumerate(events) if event['event']=='numerical_chunk_charged')
    events.pop(missing)
    with pytest.raises(ValueError,match='failed trial or batch.*uncharged'):
        initialize_seed_registry(binding,replace(run.result,accounting_events=tuple(events)))


def test_partial_batch_cannot_hide_charge_for_unsaved_row(tmp_path,monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference import hmc_candidate_set_execution as execution
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    from bayesfilter.inference.hmc_acceptance_trials import initialize_seed_registry
    target,binding,search=setup();original=execution._json_copy;seen=[]
    def interrupt(value):
        if isinstance(value,dict) and 'trial_ordinal' in value:
            seen.append(value['trial_ordinal'])
            if len(seen)==2:
                raise HMCInfrastructureFailure('injected after one row')
        return original(value)
    monkeypatch.setattr(execution,'_json_copy',interrupt)
    run=tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert run.result.completion_status=='paused_infrastructure'
    events=tuple(event for event in run.result.accounting_events
                 if not (event['event']=='numerical_chunk_charged' and event['trial_ordinal']==1))
    with pytest.raises(ValueError,match='batch has an uncharged row'):
        initialize_seed_registry(binding,replace(run.result,accounting_events=events))


def test_serialization_failure_flushes_charged_batch_prefix(tmp_path,monkeypatch):
    """A second-row write failure leaves the first row and all charges durable."""
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference import hmc_candidate_set_execution as execution
    target,binding,search=setup(batch_size=2)
    original=execution._json_copy
    seen=[]
    def fail_second(value):
        if isinstance(value,dict) and 'trial_ordinal' in value:
            seen.append(value['trial_ordinal'])
            if len(seen)==2:
                raise RuntimeError('injected second-row serialization failure')
        return original(value)
    monkeypatch.setattr(execution,'_json_copy',fail_second)
    with pytest.raises(RuntimeError,match='second-row'):
        tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
            config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    checkpoint=json.loads((tmp_path/'tuning_checkpoint.json').read_text())
    result=checkpoint['result']
    charged=[event for event in result['accounting_events']
             if event['event']=='numerical_chunk_charged']
    assert len(charged)==2
    assert len(checkpoint['partial_chunks'])==1
    assert len(next(iter(checkpoint['partial_chunks'].values())))==1


@pytest.mark.parametrize('ending',['return','raise'])
def test_dispatch_exit_revalidates_previously_persisted_history(tmp_path,monkeypatch,ending):
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
    target,binding,search=setup()
    original=HMCTuningCandidateSetController.run
    def mutate_after_dispatch(controller,*args,**kwargs):
        result=original(controller,*args,**kwargs)
        next(iter(binding._evidence.values()))['elapsed_seconds']+=1
        if ending=='raise':
            raise RuntimeError('injected dispatch exit failure')
        return result
    monkeypatch.setattr(HMCTuningCandidateSetController,'run',mutate_after_dispatch)
    with pytest.raises(ValueError,match='corrupt live numerical evidence'):
        tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
            config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert binding._checkpoint_callback is None
    assert binding._charge_chunk is None
    assert not binding._incremental_checkpoint


def test_grouped_save_is_incremental_and_all_charges_precede_native_call(tmp_path,monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel,hmc_candidate_set_checkpoint as checkpoint
    target,binding,search=setup()
    original=binding._run_replicated_batch
    original_save=checkpoint.write_numerical_tuning_checkpoint
    snapshots=[]
    def save(runtime,result,*args,**kwargs):
        snapshots.append((kwargs.get('_incremental',False),
            sum(e['event']=='numerical_chunk_charged' for e in result.accounting_events)))
        return original_save(runtime,result,*args,**kwargs)
    def checked_call(candidate,take,seeds):
        durable=json.loads((tmp_path/'tuning_checkpoint.json').read_text())
        charged={tuple(e['seed']) for e in durable['result']['accounting_events']
                 if e['event']=='numerical_chunk_charged'}
        assert set(map(tuple,seeds))<=charged
        return original(candidate,take,seeds)
    monkeypatch.setattr(checkpoint,'write_numerical_tuning_checkpoint',save)
    monkeypatch.setattr(binding,'_run_replicated_batch',checked_call)
    tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert snapshots[-1][0] is False
    assert all(incremental for incremental,_ in snapshots[:-1])
    assert not any(charges==1 for _,charges in snapshots)


def test_interrupted_grouped_charge_flushes_prefix_without_native_call(tmp_path,monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel,hmc_acceptance_trials as trials
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    target,binding,search=setup();original=trials.before_numerical_chunk
    def stop_after_first(*args,**kwargs):
        original(*args,**kwargs)
        raise HMCInfrastructureFailure('injected during grouped charging')
    def unexpected_native(*args):pytest.fail('no native batch may precede the complete charge group')
    monkeypatch.setattr(trials,'before_numerical_chunk',stop_after_first)
    monkeypatch.setattr(binding,'_run_replicated_batch',unexpected_native)
    run=tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert run.result.completion_status=='paused_infrastructure'
    durable=json.loads((tmp_path/'tuning_checkpoint.json').read_text())
    assert sum(e['event']=='numerical_chunk_charged' for e in durable['result']['accounting_events'])==1
    assert not binding._defer_checkpoint and not binding._incremental_checkpoint
