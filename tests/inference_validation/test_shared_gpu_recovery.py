"""Shared-resource scheduling and native checkpoint recovery, CPU diagnostics."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from bayesfilter.testing.inference_validation import fit_process, timeout_policy
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation.timeout_policy import TimeoutPolicy


@pytest.mark.parametrize('trusted,free,admitted',[(True,True,True),(False,True,False),(True,False,False)])
def test_shared_admission_accepts_busy_device_but_requires_valid_capacity(monkeypatch,trusted,free,admitted):
    clock=[0.]
    monkeypatch.setattr(timeout_policy,'time',SimpleNamespace(monotonic=lambda:clock[0],
        sleep=lambda n:clock.__setitem__(0,clock[0]+n)))
    result=timeout_policy.wait_for_gpu_admission(device='gpu',deadline=2.,
        policy=TimeoutPolicy(gpu_admission_mode='shared',gpu_admission_wait_seconds=2.,telemetry_interval_seconds=1.),
        workload_probe=lambda *a:{'trusted_for_extension':trusted,'contended':True,
            'gpu':{'available':True,'trusted':trusted,'memory_used_mib':100 if free else 1000,
                   'memory_total_mib':1000}})
    assert result['admitted'] is admitted
    assert clock[0]==(0. if admitted else 2.)


def recovery_design(design):
    return design('stopping','gaussian','ordinary',replications=1,budget_seconds=20.,
        options={'isolate_fits':True,'fit_process_timeout_seconds':10.,
            'timeout_policy':{'max_extension_seconds':5.,'max_contention_retries':1,
                              'shutdown_grace_seconds':.1}})


@pytest.mark.parametrize('status,contended,assessment,expected',[
    ('budget_exhausted',True,False,2), ('timed_out',True,False,2),
    ('budget_exhausted',False,False,1), ('failed',True,False,1),
    ('budget_exhausted',True,True,1),
])
def test_automatic_recovery_only_retries_incomplete_contention_stops(
        design,tmp_path,monkeypatch,status,contended,assessment,expected):
    from bayesfilter.testing.inference_validation import execution
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    calls=[]
    def run(command,log,seconds,device,**kwargs):
        calls.append((command,seconds,kwargs['timeout_policy'].max_extension_seconds))
        path=tmp_path/'replication-0000'
        if len(calls)==2 or assessment:
            write_json(path/'independent_assessment.json',{'replication':0,
                'inventory':{'failures':[]},'members':[],'tuning_completion':'complete','pipeline':'fixture'})
        if len(calls)==1:
            write_json(path/'partial-marker.json',{'preserved':True})
            return {'status':status,'exit_code':75,'elapsed_seconds':10.,
                    'observed_contention_seconds':3. if contended else 0.}
        assert read_json(path/'partial-marker.json')['preserved']
        return {'status':'complete','exit_code':0,'elapsed_seconds':1.}
    monkeypatch.setattr(fit_process,'_supervise',run)
    d=recovery_design(design)
    fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+30.)
    assert len(calls)==expected
    if expected==2:
        assert [c[1:] for c in calls]==[(10.,5.),(5.,0.)]
        assert [int(c[0][8]) for c in calls]==[1,2]
        fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+30.)
        assert len(calls)==2


def test_retry_limits_survive_reentry_and_never_reset_hard_budget(design,tmp_path,monkeypatch):
    from bayesfilter.testing.inference_validation import execution
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    calls=[]
    def exhausted(command,log,seconds,device,**kwargs):
        calls.append(seconds)
        return {'status':'budget_exhausted','exit_code':75,'elapsed_seconds':seconds,
                'observed_contention_seconds':1.}
    monkeypatch.setattr(fit_process,'_supervise',exhausted)
    d=recovery_design(design)
    for _ in range(2):
        fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+30.)
    assert calls==[10.,5.]
    assert read_json(tmp_path/'replication-0000/process-attempt-002-exit.json')['cumulative_fit_seconds']==15.
    assert fit_process.contention_retry_budget([
        {'status':'timed_out','observed_contention_seconds':4.,'elapsed_seconds':15.}],10.,
        TimeoutPolicy(max_extension_seconds=5.,max_contention_retries=1)) is None


def test_recovery_cannot_cross_cell_deadline(design,tmp_path,monkeypatch):
    from bayesfilter.testing.inference_validation import execution
    clock=[0.]
    monkeypatch.setattr(fit_process,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    calls=[]
    def exhausted(command,log,seconds,device,**kwargs):
        calls.append(seconds); clock[0]+=seconds
        return {'status':'budget_exhausted','exit_code':75,'elapsed_seconds':seconds,
                'observed_contention_seconds':1.}
    monkeypatch.setattr(fit_process,'_supervise',exhausted)
    fit_process.run_isolated_replications(recovery_design(design),tmp_path,deadline=12.)
    assert calls==[10.,2.] and clock[0]==12.


def test_cell_limited_recovery_cannot_reset_attempt_cap(design,tmp_path,monkeypatch):
    from bayesfilter.testing.inference_validation import execution
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    monkeypatch.setattr(execution,'source_state',lambda:{'identity':'fixture'})
    clock=[0.];calls=[]
    monkeypatch.setattr(fit_process,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    def exhausted(command,log,seconds,device,**kwargs):
        spent=min(1.,seconds);calls.append(spent);clock[0]+=spent
        return {'status':'budget_exhausted','exit_code':75,'elapsed_seconds':spent,
                'observed_contention_seconds':.5}
    monkeypatch.setattr(fit_process,'_supervise',exhausted)
    d=recovery_design(design)
    fit_process.run_isolated_replications(d,tmp_path,deadline=2.)
    fit_process.run_isolated_replications(d,tmp_path,deadline=30.)
    assert calls==[1.,1.]


# Inject one cooperative stop immediately after a real committed numerical
# chunk. The second process uses the ordinary, unmodified CLI and source.
STOP_AFTER_CHECKPOINT = '''
import json, sys
from pathlib import Path
from bayesfilter.inference import hmc_candidate_set_checkpoint as checkpoint
from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded
original=checkpoint._write
def interrupted(payload,path,**kwargs):
    original(payload,path,**kwargs)
    if path.name=='tuning_checkpoint.json' and (payload['numerical_evidence_hashes'] or any(payload['partial_chunks'].values())):
        path.with_name('injected-stop-checkpoint.json').write_text(json.dumps(payload))
        raise ExecutionBudgetExceeded('test-only interruption after committed numerical evidence')
checkpoint._write=interrupted
from bayesfilter.testing.inference_validation.__main__ import main
raise SystemExit(main(sys.argv[1:]))
'''


@pytest.mark.parametrize('target',['ssm_campaign_location','ssm_campaign_nonlinear'])
def test_actual_state_space_child_resumes_native_checkpoint_after_contention(
        tmp_path,monkeypatch,target):
    from tests.inference_validation.test_ssm_public_pipeline import _design
    from bayesfilter.testing.inference_validation.fit_supervision import supervise_fit
    monkeypatch.setattr(fit_process,'sys',SimpleNamespace(modules={},executable=sys.executable))
    d=_design(target,'prepared')
    d=replace(d,budget_seconds=250.,options={**d.options,
        'isolate_fits':True,'fit_process_timeout_seconds':120.,'timeout_policy':{
            'max_extension_seconds':120.,'max_contention_retries':1,
            'extension_mode':'observed_intervals','telemetry_interval_seconds':.2,
            'poll_interval_seconds':.1,'shutdown_grace_seconds':1.}})
    calls=[]
    immutable={}
    def run(command,log,seconds,device,**kwargs):
        calls.append(command)
        if len(calls)==1:
            command=[sys.executable,'-c',STOP_AFTER_CHECKPOINT,*command[3:]]
        kwargs['workload_probe']=lambda *a:{'trusted_for_extension':True,'contended':True}
        receipt=supervise_fit(command,log,seconds,device,**kwargs)
        if len(calls)==1:
            assert receipt['status']=='budget_exhausted',receipt
            assert receipt['observed_contention_seconds']>0
            for path in (tmp_path/'replication-0000/tuning').glob('numerical_*/*.json'):
                immutable[str(path)]=file_hash(path)
            assert immutable
        return receipt
    monkeypatch.setattr(fit_process,'_supervise',run)
    result=fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+250.,reuse_leapfrog_graphs=True)
    assert len(calls)==2,result
    assert result['completed']==1,result
    assert all(file_hash(path)==sha for path,sha in immutable.items())
    fit=tmp_path/'replication-0000'
    assert read_json(fit/'process-attempt-002-exit.json')['contention_recovery']
    pipeline=read_json(fit/'pipeline.json')
    assert pipeline['verified_candidate_ids']
    assert any(m.get('status')=='assessed' and m['warmup_exclusion_matches'] for m in pipeline['members'])
    fit_process.run_isolated_replications(d,tmp_path,deadline=time.monotonic()+250.,reuse_leapfrog_graphs=True)
    assert len(calls)==2
