"""Framework-free controller recovery: existing work must not be duplicated."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest


@pytest.fixture
def controlled_module(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('controlled_master_test',scripts/'run_neutra_controlled_repair_master.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_resume_adopts_existing_worker_and_reconciles_once(tmp_path,controlled_module):
    output=tmp_path/'worker';output.mkdir()
    code="""import json,pathlib,time,sys
p=pathlib.Path(sys.argv[1]);time.sleep(.1)
(p/'result.json').write_text('{"passed":false}')
(p/'manifest.json').write_text('{"status":"complete","wall_seconds":0.1,"cpu_core_seconds":0.01}')
"""
    process=subprocess.Popen([sys.executable,'-c',code,str(output)],start_new_session=True)
    row={'pid':process.pid,'output':str(output),'started_unix':time.time(),'wall_limit':10.}
    controller=object.__new__(controlled_module.Controller);controller.state={'active':{'existing':row}}
    calls=[]
    controller.finish=lambda name,record,exit_code,manifest:calls.append((name,exit_code,manifest))
    controller.recover();process.wait(timeout=5)
    assert len(calls)==1 and calls[0][0:2]==('existing',0)
    # Infrastructure completion never invents a scientific pass.
    assert json.loads((output/'result.json').read_text())=={'passed':False}


def test_completion_excludes_settled_failures_and_requires_incomplete_work(tmp_path,controlled_module):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    completed=checkpoint/'complete.json';interrupted=checkpoint/'interrupted.json'
    completed.write_text(json.dumps({'passed':False,'retained_cap_hit':True,'hard_vetoes':[]}))
    interrupted.write_text(json.dumps({'passed':False,'hard_vetoes':['campaign_resource_cap']}))
    (checkpoint/'member-screen.json').write_text(json.dumps([
        {'step_size':.5,'leapfrog_steps':3,'posterior':str(completed)},
        {'step_size':.25,'leapfrog_steps':9,'posterior':str(interrupted)}]))
    result={'selection_screen_passed':False,'checkpoint_screen':[{
        'path':str(checkpoint),'reason':'qualification_budget_exhausted'}]}
    (tmp_path/'result.json').write_text(json.dumps(result))
    request=controlled_module.resource_completion_request(tmp_path)
    assert request=={'excluded_kernel_pairs':[[.5,3]],'qualification_seconds':900.}
    result['selection_screen_passed']=True
    (tmp_path/'result.json').write_text(json.dumps(result))
    assert controlled_module.resource_completion_request(tmp_path) is None
    result['selection_screen_passed']=False
    result['checkpoint_screen'][0]['reason']='no_verified_member_passed_posterior_screen'
    interrupted.write_text(json.dumps({'passed':False,'warmup_cap_hit':True,'hard_vetoes':[]}))
    (tmp_path/'result.json').write_text(json.dumps(result))
    assert controlled_module.resource_completion_request(tmp_path) is None


def test_completion_time_override_is_local_and_preserves_total_budget(tmp_path,controlled_module):
    controller=object.__new__(controlled_module.Controller)
    cfg={'gpu_process_seconds':43200.,'cpu_core_seconds':86400.,
         'hmc_wall_seconds':300.,'hmc':{'job_wall_seconds':270.,'posterior_member_limit':3}}
    controller.cfg=cfg;controller.state={'jobs':{}}
    (tmp_path/'result.json').write_text('{"qualified":false}')
    observed=[]
    controller.batch=lambda specs:observed.append((specs,controller.cfg))
    controller.done=lambda name:tmp_path
    _,result=controller.one('qualify','mixture','completion',qualification_seconds=900.,
                            excluded_kernel_pairs=[[.5,3]])
    actual=observed[0][1]
    assert actual['hmc_wall_seconds']==930.
    assert actual['hmc']['job_wall_seconds']==900.
    assert actual['hmc']['excluded_kernel_pairs']==[[.5,3]]
    assert actual['hmc']['posterior_member_limit']==3
    assert actual['gpu_process_seconds']==cfg['gpu_process_seconds']
    assert controller.cfg is cfg and cfg['hmc']['job_wall_seconds']==270.
    assert result=={'qualified':False}
    def fail(_):raise RuntimeError('fixture failure')
    controller.batch=fail
    with pytest.raises(RuntimeError,match='fixture failure'):
        controller.one('qualify','mixture','completion',qualification_seconds=900.)
    assert controller.cfg is cfg
