"""CPU consumer regressions; mocked HMC outcomes do not certify inference."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.testing import neutra_warm_start_qualification as consumer
from bayesfilter.testing.neutra_warm_start_campaign import make_transport,TrainingBlock,write_json
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64


@pytest.mark.parametrize('passes,reference_pass,expected_ids,expected_reads',[
    ([False,True,True],True,['a','c'],1),
    ([True,True,True],False,['a'],1),
    ([False,False,False],True,['a','c','d'],0),
])
def test_actual_qualification_selects_verified_members_before_final_reference(
        tmp_path,monkeypatch,passes,reference_pass,expected_ids,expected_reads):
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(11,91))
    training=tmp_path/'training';training.mkdir()
    write_json(training/'selected-frozen.json',flow.frozen_payload(target_signature=target.signature))
    candidates=[SimpleNamespace(candidate_id=i,leapfrog_steps=l,epsilon=e)
                for i,l,e in [('a',3,.25),('b',3,.2),('c',9,.17),('d',18,.15),('unverified',25,.1)]]
    result=SimpleNamespace(candidates=candidates,verified_candidate_ids=['a','b','c','d'],completion_status='complete')
    run=SimpleNamespace(result=result,adapter=SimpleNamespace(_execution_binding=object()))
    captured={};calls=[];reads=[]
    def tuner(**kwargs):captured.update(kwargs);return run
    monkeypatch.setattr(consumer,'tune_fixed_transport_hmc_kernel',tuner)
    def builder(**kwargs):
        candidate_id=kwargs['candidate_id'];assert candidate_id in result.verified_candidate_ids
        c=next(c for c in candidates if c.candidate_id==candidate_id)
        def sequential(**options):
            assert not reads and 'retained_diagnostic_fn' not in options
            assert options['config'].step_size==c.epsilon
            assert options['config'].num_leapfrog_steps==c.leapfrog_steps
            calls.append(candidate_id)
            draws=tf.ones([16,4,2],F64)
            options['archive_callback'](stage='retained',chunk_index=0,model_samples=draws)
            return {'passed':passes[len(calls)-1],'private_retained_raw':draws,
                'private_warmup_raw':draws,'warmup_results_per_chain':2000,'retained_results_per_chain':10000}
        return SimpleNamespace(candidate=c,step_size=c.epsilon,num_leapfrog_steps=c.leapfrog_steps,
            export=lambda path:write_json(path,{'candidate_id':candidate_id}),run_sequential=sequential)
    monkeypatch.setattr(consumer,'build_retained_bound_hmc_archive_runner_from_candidate_set_result',builder)
    def final_check(target,draws,path):reads.append(str(path));return {'passed':reference_pass}
    monkeypatch.setattr(consumer,'final_reference_check',final_check)
    def forbidden_read(*args):raise AssertionError('holdout read before selected final check')
    monkeypatch.setattr(consumer,'read_tensor',forbidden_read)
    cfg={'hmc':{'leapfrogs':[3,9,18],'initial_epsilon':.5,'max_candidates':36,'work_units':144,
        'job_wall_seconds':60,'measurement_num_results':256,'verification_num_results':256}}
    output=tmp_path/'result'
    report=consumer.qualify('mixture',11,tmp_path/'unopened-reference',training,output,cfg,
                             frozen_filename='selected-frozen.json')
    assert calls==expected_ids and len(reads)==expected_reads
    assert report['heldout_consumed']==bool(expected_reads)
    assert report['qualified']==(bool(expected_reads) and reference_pass)
    assert captured['execution_config'].measurement_num_results==256
    assert captured['execution_config'].verification_num_results==256
    assert len(list(output.glob('member-*/verified-member.json')))==len(calls)
    if expected_reads and not reference_pass:
        assert report['reason']=='final_reference_failed'
        assert not json.loads((output/'posterior.json').read_text())['passed']


def test_actual_v4_consumer_installs_retained_only_event_requirement(tmp_path,monkeypatch):
    from bayesfilter.testing.neutra_warm_start_diagnostics import RARE_EVENT_DIAGNOSTIC_PROFILE
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(11,91))
    write_json(tmp_path/'selected-frozen.json',flow.frozen_payload(target_signature=target.signature))
    candidate=SimpleNamespace(candidate_id='a',epsilon=.1,leapfrog_steps=9)
    result=SimpleNamespace(candidates=[candidate],verified_candidate_ids=['a'],completion_status='complete')
    monkeypatch.setattr(consumer,'tune_fixed_transport_hmc_kernel',lambda **k:SimpleNamespace(
        result=result,adapter=SimpleNamespace(_execution_binding=object())))
    called=[]
    def sequential(**kwargs):
        assert 'retained_diagnostic_fn' in kwargs
        assert 'warmup_diagnostic_fn' not in kwargs
        draws=tf.ones([16,4,2],F64)*5
        event=kwargs['retained_diagnostic_fn'](draws)
        assert not event['passed'];called.append(True)
        return {'passed':False,'private_retained_raw':draws,'private_warmup_raw':draws,
            'warmup_results_per_chain':2000,'retained_results_per_chain':10000}
    monkeypatch.setattr(consumer,'build_retained_bound_hmc_archive_runner_from_candidate_set_result',
        lambda **k:SimpleNamespace(candidate=candidate,step_size=.1,num_leapfrog_steps=9,
            export=lambda p:write_json(p,{}),run_sequential=sequential))
    cfg={'hmc':{'leapfrogs':[3,9,18],'initial_epsilon':.5,'max_candidates':36,'work_units':144,
        'job_wall_seconds':60,'diagnostic_profile':RARE_EVENT_DIAGNOSTIC_PROFILE}}
    report=consumer.qualify('mixture',11,tmp_path/'unopened',tmp_path,tmp_path/'result',cfg,
        frozen_filename='selected-frozen.json')
    assert called==[True] and not report['qualified'] and not report['heldout_consumed']


def test_initializer_override_changes_only_declared_variance_and_survives_checkpoint():
    target=WarmStartTarget('mixture')
    base=make_transport(target,8,(11,91));candidate=make_transport(target,8,(11,91),variance_scale=.2)
    assert candidate.config==replace(base.config,iaf_variance_scale=.2)
    assert any(not bool(tf.reduce_all(a==b).numpy()) for a,b in zip(base.trainable_variables,candidate.trainable_variables))
    block=TrainingBlock(candidate,target,batch=8,learning_rate=.001,clip=100.,kind='gabrie',walkers=4,walk_steps=2)
    assert block.checkpoint()['transport_config']['iaf_variance_scale']==.2


def test_training_measure_uses_actual_sampler_rows_not_placeholder_pool(monkeypatch):
    from bayesfilter.testing import neutra_warm_start_campaign as campaign
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(11,91))
    physical=tf.constant([[-5.,1.],[5.,2.],[5.,3.],[5.,4.]]*2,F64)
    class Sampler:
        def __init__(self,*args,**kwargs):pass
        def run(self,current,dt,seed):return current,physical,tf.constant(.5,F64),tf.constant(.9,F64),tf.constant(0)
    monkeypatch.setattr(campaign,'GabrieProgram',Sampler)
    block=TrainingBlock(flow,target,batch=8,learning_rate=.001,clip=100.,kind='gabrie',walkers=4,walk_steps=2)
    block.run(tf.constant([11,8]),tf.constant(3),tf.ones([16,2],F64)*99,
              tf.zeros([16],F64),tf.zeros([4,2],F64),tf.constant(.1,F64))
    summary=block.measure_summary()
    assert int(summary['rows'].numpy())==24
    tf.debugging.assert_near(summary['mean'][:2],tf.reduce_mean(physical,axis=0))
    assert float(summary['mean'][-1].numpy())==0.
    assert float(summary['mean'][-2].numpy())==.75


def test_forward_loss_directional_check_restores_all_parameters():
    from bayesfilter.testing.neutra_gap_diagnostics import forward_directional_check
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(11,91))
    rows=target.reference_sample(16,tf.constant([72,13]))
    before=[tf.identity(v) for v in flow.trainable_variables]
    report=forward_directional_check(flow,rows,11)
    assert report['passed']
    for v,b in zip(flow.trainable_variables,before):tf.debugging.assert_equal(v,b)


def test_controller_does_not_retry_on_final_reference_or_precision_failure(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/run_neutra_gap_closure.py'
    spec=importlib.util.spec_from_file_location('gap_controller_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for verified,consumed,expected in (([],False,2),(['a'],False,1),(['a'],True,1)):
        controller=object.__new__(module.Controller)
        controller.record={'outcomes':{}};controller.root=tmp_path
        controller.refresh=lambda *args,**kwargs:None
        calls=[]
        def phase(name,phase,*args,**kwargs):
            output=tmp_path/name;output.mkdir(exist_ok=True)
            if phase=='qualify':
                calls.append(kwargs['closure'])
                write_json(output/'qualification.json',{'verified_members':verified,'heldout_consumed':consumed})
            return output,{'passed':False,'reason':'controlled_failure'}
        controller.phase=phase
        assert not controller.qualify('case','mixture',11,tmp_path)
        assert len(calls)==expected
        if expected==2:assert calls[-1]['larger_hmc_evidence']


def test_continuation_targets_supported_fit_progress_and_preserves_downstream_failure(tmp_path):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/continue_neutra_gap_closure.py'
    spec=importlib.util.spec_from_file_location('gap_continuation_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=object.__new__(module.Continuation)
    arm='init-w8-v0.2-s11'
    controller.record={'outcomes':{},'jobs':{
        arm:{'result':{'passed':False,'repair':'continue_checkpoint','continuation_candidates':[str(tmp_path)]}},
        'init-w8-v1-s11':{'result':{'passed':False,'repair':'continue_checkpoint','continuation_candidates':['later']}}}}
    assert controller.progressing_parent(11)==(arm,tmp_path)
    controller.record['jobs'][arm]['result']['repair']='nonlinear_plateau_repair'
    assert controller.progressing_parent(11)[0]=='init-w8-v1-s11'
    controller.record['jobs']['init-w16-v0.2-s11']={'result':{'passed':True}}
    # Even if its sampler failed before producing a target-labelled outcome,
    # a shape-passing map must not trigger this fit-only continuation.
    assert controller.progressing_parent(11) is None
    controller.record['jobs'].pop('init-w16-v0.2-s11')
    controller.record['outcomes']['qualified']={'target':'mixture','seed':11,'passed':True}
    assert controller.progressing_parent(11) is None


def test_worker_launch_preserves_configuration_when_shared_config_changes(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/run_neutra_warm_start_master.py'
    spec=importlib.util.spec_from_file_location('gap_config_snapshot_test',script)
    master=importlib.util.module_from_spec(spec);spec.loader.exec_module(master)
    cfg=master.configuration();cfg['job_wall_seconds']['closure_reference']=10.
    master.write(tmp_path/'config.json',{'mutable':'before'})
    captured={}
    class Process:
        pid=123
        def __init__(self,command,**kwargs):
            path=Path(command[command.index('--config')+1]);captured['path']=path
            assert json.loads(path.read_text())==cfg
            master.write(tmp_path/'config.json',{'mutable':'after'})
        def wait(self,**kwargs):return 0
    monkeypatch.setattr(master.subprocess,'Popen',Process)
    monkeypatch.setattr(master,'source_hashes',lambda:{})
    state={'attempts':[],'active_job':None}
    assert master.run_one(tmp_path,state,cfg,'config-check','closure_reference',[])
    assert captured['path'].parent==tmp_path/'attempts/config-check-r1'
    assert json.loads(captured['path'].read_text())==cfg
    assert json.loads((tmp_path/'config.json').read_text())=={'mutable':'after'}
