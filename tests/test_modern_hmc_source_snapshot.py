"""Source isolation and budget mechanics; no scientific or GPU claims."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import modern_hmc_source_snapshot as snapshot
from scripts.continue_modern_hmc_isolated import stage_budget


def fixture(tmp_path, monkeypatch):
    repo, consumer = tmp_path/'repo', tmp_path/'consumer'
    (repo/'bayesfilter').mkdir(parents=True)
    (repo/'scripts').mkdir()
    consumer.mkdir()
    (repo/'bayesfilter/__init__.py').write_text('value=1\n')
    (repo/'scripts/run.py').write_text('value=2\n')
    (repo/'inputs.json').write_text('{}')
    for name in snapshot.CONSUMER_FILES:
        (consumer/name).write_text('value=3\n')
    monkeypatch.setattr(snapshot.subprocess,'check_output',lambda *a,**k:'commit\n')
    return repo,consumer


def test_copy_is_stable_after_shared_checkout_changes(tmp_path,monkeypatch):
    repo,consumer=fixture(tmp_path,monkeypatch)
    manifest=snapshot.capture(repo,consumer,tmp_path/'copy',extra_paths=['inputs.json'])
    (repo/'bayesfilter/__init__.py').write_text('changed\n')
    (consumer/snapshot.CONSUMER_FILES[0]).write_text('changed\n')
    paths=snapshot.validate_snapshot(manifest,expected_root=manifest['source_root'])
    assert len(paths)==7
    assert all(p.is_relative_to(tmp_path/'copy') for p in paths)
    with pytest.raises(ValueError,match='declared source copy'):
        snapshot.validate_snapshot(manifest,expected_root=repo)


def test_copy_detects_concurrent_change(tmp_path,monkeypatch):
    repo,consumer=fixture(tmp_path,monkeypatch)
    copy=snapshot.shutil.copy2
    def changing(origin,target):
        result=copy(origin,target)
        if str(origin).endswith('__init__.py'):
            Path(origin).write_text('changed during copy\n')
        return result
    monkeypatch.setattr(snapshot.shutil,'copy2',changing)
    with pytest.raises(ValueError,match='concurrent source edit'):
        snapshot.capture(repo,consumer,tmp_path/'copy')


@pytest.mark.parametrize('name,path_type',[
    ('bayesfilter','package'),('scripts.run','script'),('run','script'),
    ('one_country_affine_macro_phase9_hmc','consumer')])
def test_loaded_import_must_be_from_unchanged_copy(tmp_path,monkeypatch,name,path_type):
    repo,consumer=fixture(tmp_path,monkeypatch)
    manifest=snapshot.capture(repo,consumer,tmp_path/'copy')
    source=Path(manifest['source_root']); external=Path(manifest['consumer_root'])
    old,new={'package':(repo/'bayesfilter/__init__.py',source/'bayesfilter/__init__.py'),
        'script':(repo/'scripts/run.py',source/'scripts/run.py'),
        'consumer':(consumer/snapshot.CONSUMER_FILES[0],external/snapshot.CONSUMER_FILES[0])}[path_type]
    assert snapshot.audit_imports(manifest,modules={name:SimpleNamespace(__file__=str(new))})
    with pytest.raises(ValueError,match='escaped'):
        snapshot.audit_imports(manifest,modules={name:SimpleNamespace(__file__=str(old))})
    new.write_text('changed snapshot\n')
    with pytest.raises(ValueError,match='changed'):
        snapshot.audit_imports(manifest,modules={name:SimpleNamespace(__file__=str(new))})


def test_stage_budget_is_cumulative_and_keeps_closeout():
    assert stage_budget(5000,1800,1020)==1800
    assert stage_budget(2000,1800,1020)==980
    for remaining in (1020,1000,float('nan'),float('inf')):
        with pytest.raises(ValueError):stage_budget(remaining,1800,1020)


def test_frozen_sources_never_read_old_live_binding(tmp_path,monkeypatch):
    from scripts import run_modern_hmc_macrofinance_frozen_diagnostic as frozen
    repo,consumer=fixture(tmp_path,monkeypatch)
    manifest=snapshot.capture(repo,consumer,tmp_path/'copy')
    campaign=tmp_path/'campaign';campaign.mkdir()
    (campaign/'isolated-source.json').write_text(json.dumps(manifest))
    for name in ('frozen-cohort.json','proposal-design.json'):(campaign/name).write_text('{}')
    monkeypatch.setattr(frozen,'ROOT',Path(manifest['source_root']))
    paths,reconciliation=frozen.source_paths(campaign)
    assert reconciliation is None
    assert all(not p.is_relative_to(repo) for p in paths)
    assert Path(manifest['source_root'])/'bayesfilter/__init__.py' in paths


@pytest.mark.parametrize('outcome',['completed','resource_stopped','preflight_error'])
def test_isolated_orchestrator_keeps_prerequisites_and_total_budget(tmp_path,monkeypatch,outcome):
    from scripts import continue_modern_hmc_isolated as campaign
    from scripts import run_modern_hmc_filter_health_gpu as scheduler
    from bayesfilter.runtime import gpu_provenance
    repo,consumer=fixture(tmp_path,monkeypatch)
    manifest=snapshot.capture(repo,consumer,tmp_path/'copy')
    out=tmp_path/'campaign';out.mkdir()
    (out/'isolated-source.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(campaign,'ROOT',Path(manifest['source_root']))
    row=dict(nvidia_smi_index=0,uuid='GPU-test',name=scheduler.HARDWARE,
        utilization_gpu_pct=20,memory_free_mib=10000,memory_used_mib=100,memory_total_mib=10100)
    monkeypatch.setattr(gpu_provenance,'query_nvidia_smi_gpus',lambda:[row])
    calls=[]
    class Process:
        pid=1
        def __init__(self,command,**kwargs):
            calls.append((command,kwargs))
            self.returncode=1 if outcome=='preflight_error' else 0
            if 'run_modern_hmc_macrofinance_frozen_diagnostic.py' in command[1]:
                campaign.write(out/'qualification-01/manifest.json',dict(status=outcome))
        def poll(self):return self.returncode
    monkeypatch.setattr(campaign.subprocess,'Popen',Process)
    monkeypatch.setattr(scheduler,'stop_worker',lambda _:None)
    code=campaign.run(out,5000)
    saved=json.loads((out/'manifest.json').read_text())
    assert saved['active_run'] is False
    assert all(kwargs['env']['TF_FORCE_GPU_ALLOW_GROWTH']=='true' for _,kwargs in calls)
    if outcome=='preflight_error':
        assert len(calls)==1 and code==1 and saved['status']=='error'
    else:
        assert len(calls)==3
        assert '--allow-partial-budget' in calls[-1][0]
        assert 1020 < float(calls[-1][0][calls[-1][0].index('--wall-seconds')+1]) < 5000
        assert saved['status']==outcome
        assert code==(0 if outcome=='completed' else 1)
