"""Fault-injection tests of observation/retry boundaries, with no GPU work."""
import json
from pathlib import Path

import pytest

from scripts import monitor_hmc_v7_confirmation as monitor


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path):
    source=tmp_path/'source-tree/source';source.mkdir(parents=True)
    (source/'checked.py').write_text('frozen\n')
    dump(source.parent/'source-manifest.json',{'checked.py':monitor.sha(source/'checked.py')})
    signature=monitor.sha(source.parent/'source-manifest.json')
    root=tmp_path/'primary';root.mkdir()
    (root/'runner.py').write_text('worker\n')
    slots=[dict(slot_id='s1',family='lgssm_qr',seed=[1,2]),dict(slot_id='s2',family='nonlinear',seed=[3,4])]
    design=dict(source_manifest_sha256=signature,gpu_uuid='GPU-fixture',slots=slots,
        search_cap_seconds=1800,closeout_cap_seconds=1200,readiness=dict(wait_cap_seconds=60))
    design_path=tmp_path/'frozen/design.json';dump(design_path,design);dump(root/'design.json',design)
    rows=[]
    for slot in slots:
        sid=slot['slot_id'];directory=root/sid
        cfg=dict(seed=slot['seed'],classification='confirmation')
        dump(directory/'configuration.json',cfg)
        dump(design_path.parent/(sid+'-config.json'),cfg)
        dump(directory/'resource.json',dict(ready=False,gpu_initialized=False,disposition='resource_deferred'))
        rows.append(dict(**slot,exit_code=3,source_manifest_sha256=signature))
    dump(root/'progress.json',dict(outcomes=rows))
    dump(root/'result.json',dict(status='execution_complete',original_denominator=2,outcomes=rows,wall_seconds=120))
    request=dict(run_root=str(root),source=str(source),design=str(design_path),design_sha256=monitor.sha(design_path),
        runner_sha256=monitor.sha(root/'runner.py'),stale_warning_seconds=1200,disk_stop_floor_bytes=100,
        cleanup_seconds=10,primary_start_monotonic_seconds=1000,gpu_reservation_seconds=5000)
    return request,root,source,rows


def completed_slot(request,root,rows):
    rows[0]['exit_code']=0
    d=root/'s1'
    dump(d/'manifest.json',dict(configuration_sha256=monitor.sha(d/'configuration.json'),
        source_manifest_sha256=rows[0]['source_manifest_sha256'],runner_sha256=request['runner_sha256'],
        gpu_uuid='GPU-fixture',jit_compile=True,memory_policy=dict(
            all_physical_devices_memory_growth=True,configured_before_logical_device_initialization=True)))
    dump(d/'model/result.json',dict(sampling_streams=[1,2],classification='confirmation',
        completion_status='complete',checkpoint_recomputed=True,expectation_met=True,
        verified_candidate_ids=['c1','c2'],candidate_states={'c1':'verified','c2':'verified'}))
    for cid in ('c1','c2'):dump(d/'model'/(cid+'-member.json'),dict(candidate_id=cid))
    dump(d/'device_result.json',dict(sample_devices=['/device:GPU:0']))
    dump(root/'progress.json',dict(outcomes=rows))
    dump(root/'result.json',dict(status='execution_complete',original_denominator=2,outcomes=rows,wall_seconds=120))


def test_finished_service_is_not_release_or_all_success(setup):
    request,root,source,rows=setup
    completed_slot(request,root,rows)
    snapshot=monitor.inspect_run(request,{'ActiveState':'inactive'},free_bytes=1000)
    assert snapshot['status']=='finished_pending_audit'
    assert snapshot['complete_delivery_count']==1 and snapshot['resource_deferred_slots']==['s2']
    assert not snapshot['release_ready']


@pytest.mark.parametrize('damage',['source','runner','design','seed','memory','member','device','outcome_order','denominator'])
def test_invalid_evidence_is_detected(setup,damage):
    request,root,source,rows=setup;completed_slot(request,root,rows)
    if damage=='source':(source/'checked.py').write_text('changed')
    if damage=='runner':(root/'runner.py').write_text('changed')
    if damage=='design':dump(root/'design.json',{})
    if damage=='seed':
        p=root/'s1/model/result.json';j=monitor.read(p);j['sampling_streams']=[7,8];dump(p,j)
    if damage=='memory':
        p=root/'s1/manifest.json';j=monitor.read(p);j['memory_policy']['all_physical_devices_memory_growth']=False;dump(p,j)
    if damage=='member':(root/'s1/model/c2-member.json').unlink()
    if damage=='device':dump(root/'s1/device_result.json',dict(sample_devices=['/device:CPU:0']))
    if damage=='outcome_order':dump(root/'progress.json',dict(outcomes=list(reversed(rows))))
    if damage=='denominator':
        p=root/'result.json';j=monitor.read(p);j['original_denominator']=1;dump(p,j)
    with pytest.raises(ValueError):monitor.inspect_run(request,{'ActiveState':'inactive'},free_bytes=1000)


def test_missing_terminal_and_low_disk_are_reported(setup):
    request,root,source,rows=setup;(root/'result.json').unlink()
    result=monitor.inspect_run(request,{'ActiveState':'failed'},free_bytes=99)
    assert result['status']=='error_requires_diagnosis' and result['disk_stop_required']


def test_stale_live_checkpoint_is_a_warning_not_a_failed_outcome(setup):
    request,root,source,rows=setup;(root/'result.json').unlink()
    dump(root/'progress.json',dict(outcomes=[]))
    stamp=(root/'s1/resource.json').stat().st_mtime
    result=monitor.inspect_run(request,{'ActiveState':'active'},now_epoch=stamp+1201,free_bytes=1000)
    assert result['status']=='running' and result['active_slot']['warning']
    assert not result['other_failures'] and not result['resource_deferred_slots']


def test_storage_forecast_warning_does_not_become_stop(setup):
    request,root,source,rows=setup
    request['allocated_bytes_by_family']={'lgssm_qr':1000,'nonlinear':1000}
    result=monitor.inspect_run(request,{'ActiveState':'inactive'},free_bytes=1000)
    assert result['storage_projection_warning']
    assert result['projected_remaining_allocated_bytes']==2000
    assert not result['disk_stop_required']


@pytest.mark.parametrize('reason',['live','tried','budget','sampled','initialized','wrong_seed','missing_result','harness_failure'])
def test_only_untouched_resource_deferrals_can_retry(setup,reason):
    request,root,source,rows=setup
    kwargs=dict(primary_active=False,tried=set(),remaining_seconds=4000)
    assert monitor.recovery_eligible(request,'s1',**kwargs)
    if reason=='live':kwargs['primary_active']=True
    if reason=='tried':kwargs['tried']={'s1'}
    if reason=='budget':kwargs['remaining_seconds']=3069
    if reason=='sampled':(root/'s1/model').mkdir()
    if reason=='initialized':dump(root/'s1/resource.json',dict(ready=False,gpu_initialized=True,disposition='resource_deferred'))
    if reason=='wrong_seed':dump(root/'s1/configuration.json',dict(seed=[9,9],classification='confirmation'))
    if reason=='missing_result':(root/'result.json').unlink()
    if reason=='harness_failure':
        p=root/'result.json';j=monitor.read(p);j['status']='harness_failure';dump(p,j)
    if reason=='wrong_seed':
        with pytest.raises(ValueError):monitor.recovery_eligible(request,'s1',**kwargs)
    else:assert not monitor.recovery_eligible(request,'s1',**kwargs)


def test_cost_uses_enclosing_service_not_sum_of_children(setup):
    request,root,source,rows=setup
    terminal=dict(wall_seconds=119,status='execution_complete')
    result=monitor.primary_charge(request,{'ExecMainExitTimestampMonotonic':'1120000000','ExecMainStatus':'0'},terminal,now_monotonic=1300)
    assert result['wall_seconds']==120
    result=monitor.primary_charge(request,{},terminal,now_monotonic=1300)
    assert result['wall_seconds']==300


def test_retry_preserves_primary_and_original_config(setup,tmp_path,monkeypatch):
    request,root,source,rows=setup
    request.update(python='/python')
    output=tmp_path/'monitor';output.mkdir()
    state=dict(recoveries=[])
    primary=(root/'result.json').read_bytes();configuration=(root/'s1/configuration.json').read_bytes()
    class Child:
        def __init__(self,command,**kwargs):
            self.command=command;self.pid=123
            assert kwargs['env']['TF_FORCE_GPU_ALLOW_GROWTH']=='true'
            assert command[command.index('--design')+1]==request['design']
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(monitor.subprocess,'Popen',Child)
    assert monitor.recover(request,output,state,'s1',4000)>=0
    assert (root/'result.json').read_bytes()==primary
    assert (output/'recoveries/s1/configuration.json').read_bytes()==configuration
    assert state['recoveries'][0]['secondary_evidence_only']
    assert state['recoveries'][0]['primary_outcome_unchanged']
    assert not monitor.recovery_eligible(request,'s1',primary_active=False,tried={'s1'},remaining_seconds=4000)


def test_notification_failure_is_recorded_without_losing_event(tmp_path,monkeypatch):
    state=dict(notified=[],notification_errors=[])
    def absent(*args,**kwargs):raise FileNotFoundError('notification service absent')
    monkeypatch.setattr(monitor.subprocess,'run',absent)
    monitor.notify(tmp_path,state,'error','test','message')
    monitor.notify(tmp_path,state,'error','test','message')
    assert len(state['notification_errors'])==1
    assert len((tmp_path/'events.jsonl').read_text().splitlines())==1


def test_recovery_timeout_kills_only_its_owned_process_group(setup,tmp_path,monkeypatch):
    request,root,source,rows=setup;request.update(python='/python')
    output=tmp_path/'observer';output.mkdir()
    state=dict(recoveries=[]);killed=[]
    class Child:
        pid=123
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout=None):
            if timeout==3060:raise monitor.subprocess.TimeoutExpired('child',timeout)
            return -15
    monkeypatch.setattr(monitor.subprocess,'Popen',Child)
    monkeypatch.setattr(monitor.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    monitor.recover(request,output,state,'s1',4000)
    assert killed==[(123,monitor.signal.SIGTERM)]
    assert state['recoveries'][0]['exit_code']==124
    assert monitor.read(root/'result.json')['outcomes']==rows


def test_settlement_is_idempotent_preserves_other_charges_and_releases_reservation(tmp_path):
    output=tmp_path/'observer';output.mkdir()
    allocation=tmp_path/'allocation.json'
    original=dict(resource='cpu',seconds=7,receipt='earlier/receipt.json',exit_code=0)
    ledger=dict(charges=[original],active_reservations=[
        dict(resource='gpu',seconds=5000,artifact='primary'),
        dict(resource='cpu',seconds=1800,artifact='observer'),
        dict(resource='cpu',seconds=100,artifact='unrelated')],
        gpu_seconds_reserved=10000,cpu_worker_seconds_reserved=5000)
    dump(allocation,ledger)
    dump(output/'state.json',dict(recoveries=[]))
    dump(output/'primary-receipt.json',dict(wall_seconds=120,exit_code=0))
    dump(output/'recoveries/s1/receipt.json',dict(wall_seconds=20,exit_code=0))
    dump(output/'receipt.json',dict(wall_seconds=3,exit_code=0))
    request=dict(allocation=str(allocation),repo=str(tmp_path),gpu_reservation_seconds=5000,
        cleanup_seconds=10,primary_reservation_artifact='primary',monitor_reservation_artifact='observer')
    monitor.settle_allocation(request,output,3)
    monitor.settle_allocation(request,output,3)
    result=monitor.read(allocation)
    assert result['charges'][0]==original and len(result['charges'])==4
    assert result['gpu_seconds_charged']==140 and result['cpu_seconds_charged']==10
    assert result['active_reservations']==[dict(resource='cpu',seconds=100,artifact='unrelated')]
    assert result['cpu_release_uncommitted_seconds']==4890


def test_missing_retry_receipt_never_loses_attempted_work(tmp_path):
    output=tmp_path/'observer';output.mkdir();allocation=tmp_path/'allocation.json'
    dump(allocation,dict(charges=[],active_reservations=[],gpu_seconds_reserved=10000,cpu_worker_seconds_reserved=5000))
    dump(output/'state.json',dict(recoveries=[dict(slot_id='s1',output=str(output/'recoveries/s1'),process_cap_seconds=3060)]))
    dump(output/'primary-receipt.json',dict(wall_seconds=120,exit_code=0))
    dump(output/'receipt.json',dict(wall_seconds=3,exit_code=1))
    request=dict(allocation=str(allocation),repo=str(tmp_path),gpu_reservation_seconds=5000,
        cleanup_seconds=10,primary_reservation_artifact='primary',monitor_reservation_artifact='observer')
    monitor.settle_allocation(request,output,3)
    result=monitor.read(allocation)
    assert result['gpu_seconds_charged']==3190
    assert monitor.read(output/'unsettled-s1.json')['wall_seconds']==3070
