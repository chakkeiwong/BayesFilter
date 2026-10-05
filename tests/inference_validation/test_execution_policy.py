"""Execution flags and cumulative time survive the actual process dispatch path."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from bayesfilter.testing.inference_validation import execution, fit_process
from bayesfilter.testing.inference_validation.storage import read_json,write_json


def suite_of(*designs):
    return {"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"execution-policies",
            "profile":"test","profiles":{"test":["stopping"]},
            "designs":[d.payload() for d in designs]}


def test_actual_dispatch_carries_unused_time_and_graph_option(design,tmp_path,monkeypatch):
    first=design("stopping","gaussian","ordinary",budget_seconds=10)
    second=replace(first,design_id="second")
    monkeypatch.setattr(execution,"source_state",lambda:{"identity":"fixture"})
    from bayesfilter.testing.inference_validation import reporting
    monkeypatch.setattr(reporting,"report",lambda root:None)
    calls=[]
    def execute(d,command,log,remaining,attempt):
        calls.append((d.design_id,command,remaining))
        write_json(Path(log).parent/f"attempt-{attempt:03d}-result.json",{})
        return {"exit_code":0,"status":"complete","elapsed_seconds":2 if d==first else 17}
    monkeypatch.setattr(execution,"_execute_job",execute)
    suite=suite_of(first,second)
    index=execution.run_suite(suite,tmp_path/'run',share_unused_budget=True,reuse_leapfrog_graphs=True)
    assert [r[2] for r in calls]==[10,18]
    assert [float(r[1][6]) for r in calls]==[10,18]
    assert all('--reuse-leapfrog-graphs' in r[1] for r in calls)
    again=execution.run_suite(suite,tmp_path/'run',resume=True,share_unused_budget=True,reuse_leapfrog_graphs=True)
    assert again['jobs']==index['jobs'] and len(calls)==2
    with pytest.raises(ValueError,match="execution options"):
        execution.run_suite(suite,tmp_path/'run',resume=True)
    with pytest.raises(ValueError,match="sequential"):
        execution.run_suite(suite,tmp_path/'parallel',share_unused_budget=True,max_workers=2)


@pytest.mark.parametrize('later_status',['complete','failed','timed_out','interrupted'])
def test_spent_credit_cannot_be_reclaimed_on_earlier_cell_retry(design,later_status):
    first=design("stopping","gaussian","ordinary",budget_seconds=10)
    second=replace(first,design_id="second")
    third=replace(first,design_id="third")
    plan=execution.plan_suite(suite_of(first,second,third))
    index={'jobs':{first.design_id:{'status':'failed','attempts':[{'elapsed_seconds':2}]},
                   second.design_id:{'status':later_status,'attempts':[{'elapsed_seconds':18}]}}}
    assert execution.shared_remaining_seconds(first,plan,index)==0
    assert execution.shared_remaining_seconds(third,plan,index)==10
    index['jobs'][second.design_id]['attempts'][0]['elapsed_seconds']=15
    assert execution.shared_remaining_seconds(first,plan,index)==3
    assert execution.shared_remaining_seconds(third,plan,index)==13
    index['jobs'][first.design_id]['attempts'].append({'elapsed_seconds':float('nan')})
    with pytest.raises(ValueError,match="receipt"):
        execution.shared_remaining_seconds(third,plan,index)


def test_interrupted_shared_reservation_is_charged_before_retry(design,tmp_path,monkeypatch):
    d=design("stopping","gaussian","ordinary",budget_seconds=10)
    suite=suite_of(d)
    monkeypatch.setattr(execution,"source_state",lambda:{"identity":"fixture"})
    from bayesfilter.testing.inference_validation import reporting
    monkeypatch.setattr(reporting,"report",lambda root:None)
    plan=execution.plan_suite(suite)
    root=tmp_path/'run'
    write_json(root/'run_index.json',{'suite_identity':plan['suite_identity'],'source':{'identity':'fixture'},
        'plan':plan,'execution_options':{'reuse_leapfrog_graphs':False,'share_unused_budget':True},
        'jobs':{d.design_id:{'status':'running','reserved_seconds':10,'attempts':[]}}})
    monkeypatch.setattr(execution,'_execute_job',lambda *a:pytest.fail('interrupted allowance reused'))
    index=execution.run_suite(suite,root,resume=True,share_unused_budget=True)
    assert index['jobs'][d.design_id]['status']=='unfunded'
    assert index['jobs'][d.design_id]['attempts'][0]['elapsed_seconds']==10


def test_earlier_retry_cannot_reclaim_later_interrupted_credit_with_max_jobs(design,tmp_path,monkeypatch):
    first=design('stopping','gaussian','ordinary',budget_seconds=10)
    second=replace(first,design_id='later_interrupted')
    suite=suite_of(first,second)
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    from bayesfilter.testing.inference_validation import reporting
    monkeypatch.setattr(reporting,'report',lambda root:None)
    plan=execution.plan_suite(suite)
    root=tmp_path/'run'
    write_json(root/'run_index.json',{'suite_identity':plan['suite_identity'],'source':{'identity':'fixture'},
        'plan':plan,'execution_options':{'reuse_leapfrog_graphs':False,'share_unused_budget':True},
        'jobs':{first.design_id:{'status':'failed','attempts':[{'elapsed_seconds':2}]},
                second.design_id:{'status':'running','reserved_seconds':18,'attempts':[]}}})
    monkeypatch.setattr(execution,'_execute_job',lambda *a:pytest.fail('spent credit reused'))
    for _ in range(2):
        index=execution.run_suite(suite,root,resume=True,share_unused_budget=True,max_jobs=1)
        assert index['jobs'][first.design_id]['status']=='unfunded'
        assert index['jobs'][second.design_id]['status']=='interrupted'
        assert [a['elapsed_seconds'] for a in index['jobs'][second.design_id]['attempts']]==[18]


@pytest.mark.parametrize('reuse',[False,True])
def test_fit_worker_forwards_graph_policy_to_real_pipeline_boundary(design,tmp_path,monkeypatch,reuse):
    from bayesfilter.testing.inference_validation.engines import pipeline
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    monkeypatch.setattr(execution,'configure_worker',lambda d:{})
    monkeypatch.setattr(fit_process,'resource_snapshot',lambda:{})
    received=[]
    monkeypatch.setattr(pipeline,'run_replication',lambda *a,**kw:received.append(kw))
    path=write_json(tmp_path/'design.json',design('stopping','gaussian','ordinary').payload())
    assert fit_process.fit_worker(path,tmp_path,0,10,1,reuse_leapfrog_graphs=reuse)==0
    assert received==([{'reuse_leapfrog_graphs':True}] if reuse else [{}])
    assert read_json(tmp_path/'replication-0000/process-attempt-001-manifest.json')['reuse_leapfrog_graphs']==reuse


def test_cell_limited_fit_resumes_only_unspent_fit_budget(design,tmp_path,monkeypatch):
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    d=design('stopping','gaussian','ordinary',replications=1,
        options={'isolate_fits':True,'fit_process_timeout_seconds':10})
    calls=[]
    def run(command,log,seconds,device,**kw):
        calls.append((command,seconds))
        if len(calls)==1:
            write_json(tmp_path/'replication-0000/tuning/preparation_progress.json',
                       {'status':'running','artifact_authority':False})
            return {'status':'timed_out','exit_code':-15,'elapsed_seconds':.03}
        assert not (tmp_path/'replication-0000/tuning').exists()
        assert (tmp_path/'replication-0000/interrupted-preparation-before-attempt-002/preparation_progress.json').exists()
        write_json(tmp_path/'replication-0000/independent_assessment.json',{
            'replication':0,'inventory':{'failures':[]},'members':[],
            'tuning_completion':'complete','pipeline':'fixture'})
        return {'status':'complete','exit_code':0,'elapsed_seconds':.1}
    monkeypatch.setattr(fit_process,'_supervise',run)
    fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+.05,reuse_leapfrog_graphs=True)
    first=read_json(tmp_path/'replication-0000/process-attempt-001-exit.json')
    assert first['allocation_limiter']=='cell'
    fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+20,reuse_leapfrog_graphs=True)
    assert calls[1][1]==pytest.approx(9.97)
    assert all('--reuse-leapfrog-graphs' in call[0] for call in calls)
    with pytest.raises(ValueError,match='identical'):
        fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+20)
    fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+20,reuse_leapfrog_graphs=True)
    assert len(calls)==2


@pytest.mark.parametrize('extra',['tuning_checkpoint.json','unexpected.json'])
def test_preparation_restart_preserves_real_checkpoint_and_unknown_files(tmp_path,extra):
    write_json(tmp_path/'tuning/preparation_progress.json',{'status':'running','artifact_authority':False})
    write_json(tmp_path/'tuning'/extra,{'preserve':True})
    assert fit_process.archive_interrupted_preparation(tmp_path,2) is None
    assert (tmp_path/'tuning'/extra).exists()


@pytest.mark.parametrize('failure_type,archived',[
    ('HMCPreparationBudgetExceeded',True),('HMCPreparationFailure',False),('ValueError',False)])
def test_preparation_retry_distinguishes_time_limit_from_numerical_failure(tmp_path,failure_type,archived):
    progress={'status':'deferred','artifact_authority':False,'failure':{'type':failure_type}}
    write_json(tmp_path/'tuning/preparation_progress.json',progress)
    result=fit_process.archive_interrupted_preparation(tmp_path,2)
    assert bool(result)==archived
    assert read_json((Path(result) if result else tmp_path/'tuning')/'preparation_progress.json')==progress


@pytest.mark.parametrize('target',['gaussian','beta_binomial'])
def test_actual_isolated_pipeline_graph_reuse_matches_numerics(design,tmp_path,target):
    from hashlib import sha256
    d=design('stopping',target,'ordinary',replications=1,budget_seconds=160,
        l_grid=(3,5),posterior_cap=128,options={
            'isolate_fits':True,'fit_process_timeout_seconds':150,
            'posterior_members':'selected','member_rule':'first_verified',
            'acceptance_policy':{'practical_region':(.41,.99),'repair_region':(.405,.995)},
            'search':{'pilot_enabled':False,'refinement_rounds':0,'total_budget_units':24,
                      'repair_reserve_units':4,'evidence_rungs':(1,)}})
    outputs=[]
    for reuse in (False,True):
        root=tmp_path/str(reuse)
        index=execution.run_suite(suite_of(d),root,reuse_leapfrog_graphs=reuse)
        assert index['jobs'][d.design_id]['status']=='complete'
        fit=root/d.design_id/'replication-0000'
        payload=read_json(fit/'pipeline.json')
        assert payload['verified_candidate_ids']
        tensors={str(p.relative_to(fit)):sha256(p.read_bytes()).hexdigest()
                 for pattern in ('*.tensor','*.bin') for p in fit.rglob(pattern)}
        assert tensors
        outputs.append((payload['verified_candidate_ids'],tensors))
    assert outputs[0]==outputs[1]
