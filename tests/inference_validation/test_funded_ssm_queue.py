"""Capacity waits must not spend numerical allowances or reset grant/deadlines."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_hmc_funded_ssm_queue as queue
from bayesfilter.testing.inference_validation.storage import read_json,write_json,file_hash


def setup_queue(tmp_path,monkeypatch):
    clock=[0.]
    monkeypatch.setattr(queue,"time",SimpleNamespace(monotonic=lambda:clock[0],time=lambda:1000.+clock[0],
        sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)))
    monkeypatch.setattr(queue.campaign,"c1_is_active",lambda:False)
    ledger=tmp_path/"ledger.json"
    write_json(ledger,{"grant_gpu_seconds":200000.,"remaining_gpu_seconds":199900.,"charged_gpu_seconds":100.,
        "allocations":{"canonical_neutra_pricing_reserved_only":1200.},
        "records":[{"receipt":queue.campaign.C1_RECEIPT,"elapsed_seconds":100.}]})
    config={"root":str(tmp_path/"output"),"ledger":str(ledger),"service":"queue-test",
        "campaign_started_epoch":1000.,"prior_ssm_receipts":[],"queue_cap_seconds":60.,
        "ssm_gpu_cap_seconds":129600.}
    return config,clock


def test_busy_unknown_then_idle_preserves_device_and_charges_once(tmp_path,monkeypatch):
    config,clock=setup_queue(tmp_path,monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-before")
    calls=[]
    def probe(*args):
        device=os.environ['CUDA_VISIBLE_DEVICES'];calls.append(device)
        return {"trusted_for_extension":device=='GPU-B',"contended":clock[0]==0.}
    selected=queue.wait_for_capacity(config,"first",["GPU-A","GPU-B"],probe=probe)
    assert selected=='GPU-B' and calls==['GPU-A','GPU-B','GPU-A','GPU-B']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-before'
    ledger=read_json(config['ledger'])
    assert ledger['charged_gpu_seconds']==130. and 'active_reservation' not in ledger
    assert len(ledger['records'])==2
    with pytest.raises(ValueError,match='already exists'):
        queue.wait_for_capacity(config,'first',['GPU-A'],probe=probe)


def test_wait_exhaustion_never_launches_and_cannot_reset_allowance(tmp_path,monkeypatch):
    config,clock=setup_queue(tmp_path,monkeypatch)
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:pytest.fail('no numerical process while waiting'))
    probe=lambda *a:{'trusted_for_extension':True,'contended':True}
    assert queue.wait_for_capacity(config,'first',['GPU-A'],probe=probe) is None
    assert clock[0]==60.
    assert queue.wait_for_capacity(config,'second',['GPU-A'],probe=probe) is None
    assert len(read_json(config['ledger'])['records'])==2


def test_queue_interrupt_settles_elapsed_wait_and_restores_environment(tmp_path,monkeypatch):
    config,clock=setup_queue(tmp_path,monkeypatch)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    def probe(*args):
        clock[0]+=2.
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):queue.wait_for_capacity(config,'interrupted',['GPU-A'],probe=probe)
    ledger=read_json(config['ledger'])
    assert ledger['charged_gpu_seconds']==102. and 'active_reservation' not in ledger
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ


def test_original_deadline_and_prior_spending_remain_binding(tmp_path,monkeypatch):
    config,clock=setup_queue(tmp_path,monkeypatch)
    clock[0]=42*3600.
    with pytest.raises(ValueError,match='original 42/46'):queue.check_wall_time(config)
    clock[0]=0.
    config['ssm_gpu_cap_seconds']=120.
    ledger=read_json(config['ledger'])
    ledger['records'].append({'receipt':'previous-ssm','elapsed_seconds':100.})
    ledger['remaining_gpu_seconds']-=100.
    write_json(config['ledger'],ledger)
    config['prior_ssm_receipts']=['previous-ssm']
    with pytest.raises(ValueError,match='cumulative SSM'):queue.check_ssm_allocation(config,30.)


def test_recovery_spending_is_not_double_counted_as_ssm(tmp_path,monkeypatch):
    config,_=setup_queue(tmp_path,monkeypatch)
    root=Path(config['root'])
    ledger={'records':[{'receipt':str(root/'recovery-r2/recovery-execution.json'),'elapsed_seconds':400.},
        {'receipt':str(root/'queue-first-execution.json'),'elapsed_seconds':30.},
        {'receipt':str(root/'prepared/preflight-mechanics-execution.json'),'elapsed_seconds':200.}]}
    assert queue.charged_ssm_seconds(config,ledger)==230.


def test_ssm_charge_includes_separate_stage_roots_and_prior_attempts_once(tmp_path,monkeypatch):
    config,_ = setup_queue(tmp_path,monkeypatch)
    config.update(ssm_root=str(tmp_path/'prepared'),
        ssm_accounting_roots=[str(tmp_path/'older'),str(tmp_path/'older/prepared')],
        prior_ssm_receipts=[str(tmp_path/'older/prepared/price-execution.json')])
    records = [
        ('prepared/preflight-execution.json',20.),
        ('older/prepared/price-execution.json',30.),
        ('older/runtime/queue-execution.json',10.),
        ('older/runtime/recovery-r2/recovery-execution.json',400.),
        ('unrelated/preflight-execution.json',50.),
    ]
    ledger = {'records':[{'receipt':str(tmp_path/path),'elapsed_seconds':seconds}
        for path,seconds in records]}
    assert queue.charged_ssm_seconds(config,ledger)==60.


def completed_recovery_fixture(tmp_path,monkeypatch):
    source = tmp_path/'source'
    original = tmp_path/'original'
    recovered = tmp_path/'recovered'
    monkeypatch.setattr(queue.recovery,'original_paths',lambda _: (tmp_path,source,original))
    numerical = source/'bayesfilter/example.py'
    numerical.parent.mkdir(parents=True)
    numerical.write_text('original = 1\n')
    files = {'bayesfilter/example.py':file_hash(numerical)}
    write_json(source/'source_snapshot.json',{'git_commit':'test','files':files})
    for cell in (original,recovered):
        write_json(cell/'isolated_design.json',{'target':'original','seed':71})
    rows=[]
    for index in queue.recovery.INDICES:
        relative = Path(f'replication-{index:04d}')
        write_json(original/relative/'fit_identity.json',{'replication':index})
        write_json(recovered/relative/'fit_identity.json',{'replication':index})
        assessment = write_json(recovered/relative/'independent_assessment.json',{'passed':False})
        write_json(recovered/relative/'process-attempt-001-manifest.json',{'runtime':{
            'jit_compile':True,'gpu_tensor_device':'/GPU:0','memory_policy':{
                'configured_before_logical_device_initialization':True,
                'all_physical_devices_memory_growth':True,'physical_devices':['GPU-test']}}})
        receipt = write_json(recovered/relative/'process-attempt-001-exit.json',{
            'status':'complete','exit_code':0,'assessment_sha256':file_hash(assessment)})
        rows.append({'replication':index,'status':'complete','receipt':str(receipt)})
    result = write_json(tmp_path/'recovery-result.json',{'planned':10,'completed':10,
        'numerical_source_identity':queue.recovery.digest(files),'outcomes':rows})
    return result


def test_complete_recovery_reuse_checks_evidence_without_selecting_posterior_success(tmp_path,monkeypatch):
    result = completed_recovery_fixture(tmp_path,monkeypatch)
    assert queue.check_completed_recovery(result,tmp_path)['completed']==10


@pytest.mark.parametrize('changed',['missing','duplicate','source','identity','assessment','growth','design'])
def test_complete_recovery_reuse_rejects_changed_or_missing_evidence(tmp_path,monkeypatch,changed):
    result = completed_recovery_fixture(tmp_path,monkeypatch)
    payload = read_json(result)
    if changed=='missing':
        payload['outcomes'].pop()
    elif changed=='duplicate':
        payload['outcomes'][0]=payload['outcomes'][1]
    elif changed=='source':
        (tmp_path/'source/bayesfilter/example.py').write_text('changed = 1\n')
    elif changed in {'identity','assessment'}:
        filename = 'fit_identity.json' if changed=='identity' else 'independent_assessment.json'
        write_json(tmp_path/'recovered/replication-0078'/filename,{'changed':True})
    elif changed=='growth':
        path = tmp_path/'recovered/replication-0078/process-attempt-001-manifest.json'
        manifest = read_json(path)
        manifest['runtime']['memory_policy']['all_physical_devices_memory_growth']=False
        write_json(path,manifest)
    else:
        write_json(tmp_path/'recovered/isolated_design.json',{'changed':True})
    write_json(result,payload)
    with pytest.raises(ValueError):
        queue.check_completed_recovery(result,tmp_path)


@pytest.mark.parametrize('reuse_completed',[False,True])
@pytest.mark.parametrize('failure_stage',[None,'preflight-pipeline'])
def test_sequence_preserves_order_device_and_stops_on_invalid_stage(tmp_path,monkeypatch,failure_stage,reuse_completed):
    config,_=setup_queue(tmp_path,monkeypatch)
    prepared=tmp_path/'prepared'
    config.update(ssm_root=str(prepared),repo=str(tmp_path),previous_cell=str(tmp_path/'prior'),
        gpu_uuids=['GPU-A','GPU-B'],plan_file='reviewed-plan.md')
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH','true')
    waits=[];stages=[]
    def capacity(config,label,devices):
        waits.append((label,devices))
        return 'GPU-A' if label=='recovery' else 'GPU-B'
    monkeypatch.setattr(queue,'wait_for_capacity',capacity)
    def recover(args):
        assert args.previous_cell==Path(config['previous_cell'])
        assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-A'
        return tmp_path/'old',tmp_path/'new',tmp_path/'original-source'
    monkeypatch.setattr(queue.recovery,'run_recovery',recover)
    def summarize_old(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='-1'
        assert command[-1]==config['previous_cell']
    monkeypatch.setattr(queue.subprocess,'run',summarize_old)
    if reuse_completed:
        config['completed_recovery_result'] = str(completed_recovery_fixture(tmp_path,monkeypatch))
        monkeypatch.setattr(queue.recovery,'run_recovery',lambda *a:pytest.fail('C1 must not rerun'))
        monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:pytest.fail('C1 must not rerun'))
    monkeypatch.setattr(queue,'plan_suite',lambda value:{'maximum_worker_seconds':10.})
    monkeypatch.setattr(queue.campaign,'price',lambda *a,**kw:None)
    summaries=[]
    monkeypatch.setattr(queue.campaign,'summarize',lambda path:summaries.append(path))
    for stage in ('preflight-mechanics','preflight-pipeline','pricing','main'):
        write_json(prepared/(stage+'.json'),{})
    def stage(path,name,ledger,**kwargs):
        stages.append(name)
        assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-B'
        return {'exit_code':1 if name==failure_stage else 0}
    monkeypatch.setattr(queue.campaign,'run_stage',stage)
    assert queue.run(config)==(1 if failure_stage else 0)
    assert stages==(['preflight-mechanics','preflight-pipeline'] if failure_stage else
        ['preflight-mechanics','preflight-pipeline','pricing','main'])
    assert all(devices==['GPU-B'] for label,devices in waits[2:])
    assert ('recovery' not in [label for label,_ in waits]) is reuse_completed
    assert read_json(prepared/'checkpoint.json')['started_epoch']==1000.
    assert summaries==[prepared]
    with pytest.raises(ValueError,match='already attempted'):queue.run(config)


def test_shared_queue_selects_capacity_without_requiring_foreign_jobs_to_exit(tmp_path,monkeypatch):
    config,clock = setup_queue(tmp_path,monkeypatch)
    config['gpu_admission_mode']='shared'
    calls=[]
    def probe(*args):
        device=os.environ['CUDA_VISIBLE_DEVICES']; calls.append(device)
        return {'trusted_for_extension':True,'contended':True,'gpu':{
            'available':True,'trusted':True,'utilization_gpu_pct':80 if device=='GPU-A' else 30,
            'memory_used_mib':100,'memory_total_mib':1000}}
    assert queue.wait_for_capacity(config,'shared',['GPU-A','GPU-B'],probe=probe)=='GPU-B'
    assert calls==['GPU-A','GPU-B'] and clock[0]==0
    receipt=read_json(Path(config['root'])/'queue-shared-execution.json')
    assert receipt['gpu_admission_mode']=='shared'


@pytest.mark.parametrize('status,outer,allowed',[
    ('budget_exhausted','failed',True),('timed_out','failed',True),
    ('failed','failed',False),('supervision_failed','failed',False),
    (None,'timed_out',True),(None,'failed',False)])
def test_pricing_resource_failure_does_not_discard_independent_complete_lanes(tmp_path,status,outer,allowed):
    write_json(tmp_path/'pricing.json',{'designs':[{'design_id':'good'},{'design_id':'bad'}]})
    write_json(tmp_path/'pricing/run_index.json',{'jobs':{'good':{'status':'complete'},
        'bad':{'status':outer,'attempts':[{'status':outer}]}}})
    if status:
        write_json(tmp_path/'pricing/bad/replication-0000/process-attempt-001-exit.json',{'status':status})
    assert queue.pricing_only_resource_failures(tmp_path) is allowed
