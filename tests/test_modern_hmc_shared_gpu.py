"""CPU-only scheduling mechanics; no CUDA initialization or sampler claims."""
from types import SimpleNamespace
import json
import pytest
from scripts import run_modern_hmc_filter_health_gpu as policy


def gpu(index, load=25., free=12000., **updates):
    return dict(nvidia_smi_index=index, uuid='GPU-'+str(index), name=policy.HARDWARE,
        utilization_gpu_pct=load, memory_free_mib=free, memory_used_mib=32768.-free,
        memory_total_mib=32768., **updates)


def test_busy_devices_are_eligible_and_lowest_load_wins():
    assert policy.select_shared_gpu([gpu(0,30),gpu(1,45)])['nvidia_smi_index']==0
    assert policy.select_shared_gpu([gpu(0,50),gpu(1,95)]) is None


@pytest.mark.parametrize('load,eligible',[(0.,True),(49.999,True),(50.,False),(100.,False)])
def test_owner_strict_below_50_launch_boundary(load,eligible):
    assert (policy.select_shared_gpu([gpu(0,load)]) is not None)==eligible


def test_load_precedes_memory_then_uuid_breaks_ties():
    assert policy.select_shared_gpu([gpu(0,10,5000),gpu(1,20,20000)])['nvidia_smi_index']==0
    assert policy.select_shared_gpu([gpu(0,10,5000),gpu(1,10,20000)])['nvidia_smi_index']==1
    assert policy.select_shared_gpu([gpu(1),gpu(0)])['uuid']=='GPU-0'


def test_memory_screen_applies_even_to_idle_gpu():
    assert policy.select_shared_gpu([gpu(0,0,4095),gpu(1,49,4096)])['uuid']=='GPU-1'
    assert policy.select_shared_gpu([gpu(0,0,1),gpu(1,0,2)]) is None


def test_out_of_scope_gpu_does_not_expand_device_authority():
    assert policy.select_shared_gpu([gpu(2,0)]) is None


def test_reload_preserves_uuid_even_when_another_device_is_less_loaded():
    assert policy.select_shared_gpu([gpu(0,0),gpu(1,40)],required_uuid='GPU-1')['uuid']=='GPU-1'
    assert policy.select_shared_gpu([gpu(0,0),gpu(1,50)],required_uuid='GPU-1') is None
    assert policy.select_shared_gpu([gpu(0,0),gpu(1,0,100)],required_uuid='GPU-1') is None
    assert policy.select_shared_gpu([gpu(2,0)],required_uuid='GPU-2') is None


def test_reload_uuid_comes_from_recorded_physical_runtime(tmp_path):
    manifest=dict(environment=dict(CUDA_VISIBLE_DEVICES='GPU-1',TF_FORCE_GPU_ALLOW_GROWTH='true'),
        physical_gpu=dict(uuid='GPU-1'))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    assert policy.reload_gpu_uuid(tmp_path)=='GPU-1'
    manifest['physical_gpu']['uuid']='GPU-0'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='recorded single physical GPU'):
        policy.reload_gpu_uuid(tmp_path)
    with pytest.raises(ValueError,match='requires source'):
        policy.reload_gpu_uuid(None)


@pytest.mark.parametrize('change',[{'utilization_gpu_pct':float('nan')},
    {'utilization_gpu_pct':101.}, {'memory_free_mib':float('inf')},
    {'memory_used_mib':-1.}, {'memory_free_mib':32769.},
    {'name':'different hardware'}, {'uuid':'missing'}, {'nvidia_smi_index':True}])
def test_bad_inventory_rejected(change):
    with pytest.raises(ValueError):
        policy.select_shared_gpu([gpu(0)|change])


def test_duplicate_identities_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        policy.select_shared_gpu([gpu(0),gpu(0)])


def test_30_minute_monitor_recommends_next_gpu_without_migration():
    monitor=policy.Reassessment()
    assert monitor.due(100.)
    first=monitor.observe(100.,[gpu(0,10),gpu(1,20)])
    assert first['selected_for_next_worker']['uuid']=='GPU-0'
    assert not monitor.due(1899.)
    assert monitor.due(1900.)
    second=monitor.observe(1900.,[gpu(0,100),gpu(1,5)],active_uuid='GPU-0')
    assert second['selected_for_next_worker']['uuid']=='GPU-1'
    assert second['active_uuid']=='GPU-0' and not second['active_worker_migrated']


def test_missing_active_device_rejected():
    with pytest.raises(ValueError,match='disappeared'):
        policy.Reassessment().observe(0,[gpu(1)],active_uuid='GPU-0')


def test_waiting_and_running_consume_same_budget():
    assert policy.remaining_seconds(100,200,450)==350
    assert policy.remaining_seconds(100,600,450)==0


def test_waiting_does_not_consume_worker_cap_but_counts_toward_total():
    assert policy.execution_remaining(0,1800,1800,4000,2100)==2100
    assert policy.execution_remaining(0,1800,2000,4000,2100)==1900
    assert policy.execution_remaining(0,1800,3900,4000,2100)==0
    assert policy.execution_remaining(0,1800,3950,4000,None)==50


@pytest.mark.parametrize('opening', [60., 1800.])
def test_delayed_launch_keeps_full_worker_allowance(tmp_path,monkeypatch,opening):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy.time,'sleep',lambda s:tick.__setitem__(0,tick[0]+s))
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:[gpu(0,100 if tick[0]<opening else 40)])
    class Process:
        pid=123
        returncode=None
        def poll(self):
            if tick[0]>=opening+1300:self.returncode=0
            return self.returncode
    def launch(command,**kwargs):
        assert tick[0]==opening
        assert float(command[command.index('--wall-seconds')+1])==2090
        out=tmp_path/'delayed/evidence';out.mkdir()
        (out/'result.json').write_text(json.dumps({'status':'completed','execution':{'completion_status':'complete'}}))
        return Process()
    monkeypatch.setattr(policy.subprocess,'Popen',launch)
    monkeypatch.setattr(policy.os,'killpg',lambda *a:pytest.fail('healthy worker killed'))
    args=SimpleNamespace(output=tmp_path/'delayed',inputs=tmp_path/'inputs.json',
        task='public-tune',max_wall_seconds=4000.,worker_wall_seconds=2100.)
    assert policy.supervise(args)==0
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['capacity_wait_seconds']==opening
    assert result['elapsed_seconds']==opening+1300 and not result['active_run']
    assert len(result['observations'])==int(opening/60)+1


def test_wait_checks_are_frequent_without_changing_active_cadence():
    monitor=policy.Reassessment()
    monitor.observe(0,[gpu(0,100)])
    assert not monitor.due(59,waiting=True)
    assert monitor.due(60,waiting=True)
    assert not monitor.due(60)
    assert monitor.due(1800)


def test_does_not_start_underfunded_worker_after_wait(tmp_path,monkeypatch):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy.time,'sleep',lambda s:tick.__setitem__(0,tick[0]+s))
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:[gpu(0,100)])
    monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:pytest.fail('underfunded launch'))
    args=SimpleNamespace(output=tmp_path/'short',inputs=tmp_path/'inputs.json',
        max_wall_seconds=2100.,worker_wall_seconds=2000.)
    assert policy.supervise(args)==1
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['status']=='insufficient_worker_allowance_after_wait'
    assert not result['numerical_worker_launched'] and not result['active_run']


def test_waiting_termination_preserves_terminal_receipt(tmp_path,monkeypatch):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:[gpu(0,100)])
    previous=policy.signal.getsignal(policy.signal.SIGTERM)
    def interrupt(s):
        policy.signal.getsignal(policy.signal.SIGTERM)(policy.signal.SIGTERM,None)
    monkeypatch.setattr(policy.time,'sleep',interrupt)
    args=SimpleNamespace(output=tmp_path/'interrupted',inputs=tmp_path/'inputs.json',max_wall_seconds=2000.)
    assert policy.supervise(args)==1
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['error_type']=='InterruptedError'
    assert not result['numerical_worker_launched'] and not result['active_run']
    assert policy.signal.getsignal(policy.signal.SIGTERM)==previous


def test_no_memory_wait_terminates_without_worker(tmp_path,monkeypatch):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy.time,'sleep',lambda s:tick.__setitem__(0,tick[0]+s))
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:[gpu(0,0,1)])
    monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:pytest.fail('worker launched'))
    args=SimpleNamespace(output=tmp_path/'wait',inputs=tmp_path/'inputs.json',max_wall_seconds=10.)
    assert policy.supervise(args)==1
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['status']=='budget_exhausted_waiting_for_capacity'
    assert not result['numerical_worker_launched'] and not result['active_run']


def test_long_worker_is_not_killed_or_migrated_at_reassessment(tmp_path,monkeypatch):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy.time,'sleep',lambda s:tick.__setitem__(0,tick[0]+s))
    inventories=iter([[gpu(0,10),gpu(1,20)],[gpu(0,99),gpu(1,0)]])
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:next(inventories))
    class Process:
        pid=123
        returncode=None
        def poll(self):
            if tick[0]>=1802.:self.returncode=0
            return self.returncode
    def launch(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-0'
        assert kwargs['env']['TF_FORCE_GPU_ALLOW_GROWTH']=='true'
        out=tmp_path/'run/evidence';out.mkdir()
        (out/'result.json').write_text(json.dumps({'status':'completed','execution':{'completion_status':'complete'}}))
        return Process()
    monkeypatch.setattr(policy.subprocess,'Popen',launch)
    monkeypatch.setattr(policy.os,'killpg',lambda *a:pytest.fail('killed a progressing worker'))
    args=SimpleNamespace(output=tmp_path/'run',inputs=tmp_path/'inputs.json',max_wall_seconds=1810.)
    assert policy.supervise(args)==0
    result=json.loads((args.output/'manifest.json').read_text())
    assert len(result['observations'])==2
    assert result['selected_uuid']=='GPU-0'
    assert result['observations'][-1]['selected_for_next_worker']['uuid']=='GPU-1'


def test_total_cap_stops_worker_and_records_incomplete_attempt(tmp_path,monkeypatch):
    tick=[0.]
    monkeypatch.setattr(policy.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(policy.time,'sleep',lambda s:tick.__setitem__(0,tick[0]+s))
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',lambda:[gpu(0)])
    class Process:
        pid=123
        returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15;return self.returncode
    process=Process()
    monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:process)
    kills=[]
    monkeypatch.setattr(policy.os,'killpg',lambda pid,sig:kills.append((pid,sig)))
    args=SimpleNamespace(output=tmp_path/'timeout',inputs=tmp_path/'inputs.json',max_wall_seconds=10.)
    assert policy.supervise(args)==1
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['status']=='wall_budget_exhausted'
    assert result['numerical_worker_launched'] and not result['active_run']
    assert len(kills)==1 and result['elapsed_seconds']<=args.max_wall_seconds


def test_driver_failure_does_not_get_shared_fallback(tmp_path,monkeypatch):
    def fail():raise RuntimeError('driver failure')
    monkeypatch.setattr(policy,'query_nvidia_smi_gpus',fail)
    monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:pytest.fail('worker launched'))
    args=SimpleNamespace(output=tmp_path/'driver',inputs=tmp_path/'inputs.json',max_wall_seconds=10.)
    assert policy.supervise(args)==1
    result=json.loads((args.output/'manifest.json').read_text())
    assert result['status']=='error' and result['error']=='driver failure'
    assert not result['numerical_worker_launched']


def test_progress_counts_new_trials_not_cumulative_rung_totals(tmp_path):
    import json
    from scripts.run_modern_hmc_filter_health_gpu import checkpoint_progress
    path = tmp_path/'evidence/tuning/tuning_checkpoint.json'
    path.parent.mkdir(parents=True)
    observations = []
    works = []
    for work_id, stage, start, end, completed in (
            ('m1', 'measurement', 0, 16, 16),
            ('m2', 'measurement', 16, 32, 32),
            ('v1', 'verification', 0, 16, 16),
            ('bad', 'measurement', 0, 16, 5)):
        works.append({'work_item_id': work_id, 'trial_range': [start, end]})
        observations.append({'work_item_id': work_id, 'stage': stage,
            'observation': {'acceptance_evidence': {'completed_trials': completed}}})
    path.write_text(json.dumps({'result': {'observations': observations,
        'work_items': works, 'completion_status': 'partial_budget'},
        'partial_chunks': {'pending': ['one', 'two']}}))
    progress = checkpoint_progress(tmp_path)
    assert progress['complete_work_trials'] == 53
    assert progress['partial_chunk_count'] == 2
    assert progress['completed_stage_counts'] == {'measurement': 3, 'verification': 1}
