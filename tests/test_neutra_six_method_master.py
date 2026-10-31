"""CPU-only controller regressions; no numerical or scientific evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def master():
    path=Path(__file__).resolve().parents[1]/'scripts/run_neutra_six_method_master.py'
    spec=importlib.util.spec_from_file_location('six_method_master_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(master,tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir()
    shared=tmp_path/'shared';shared.mkdir()
    source=root/'source-fixture';source.mkdir()
    (source/'fixture.py').write_text('# independent controller fixture\n')
    hashes={'fixture.py':hashlib.sha256((source/'fixture.py').read_bytes()).hexdigest()}
    master.write(source/'source.json',{'sha256':hashes,'git_commit':'test-commit',
        'snapshot_id':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()})
    state={'attempts':[],'active':None,'limits':master.LIMITS,
           'status':'engineering_review_required','next_action':'preserved review'}
    ledger={'attempts':[],'active_job':None}

    def add(job,device,method=None,target=None):
        output=root/(job+'-r1');output.mkdir()
        row={'job':job,'attempt':1,'source':str(source),'output':str(output),
             'status':'complete','device':device,'command':['fixture',job],
             'wall_seconds':1.,'cpu_core_seconds':1.,
             'gpu_process_seconds':1. if device=='gpu' else 0.,
             'method':method,'target':target}
        manifest={**row,'source':str(source/'source.json'),'execution_cwd':str(source),
                  'git_commit':'test-commit','gpu_devices_intentionally_hidden':device=='cpu',
                  'TF_FORCE_GPU_ALLOW_GROWTH':'true','jit_compile':True,'dtype':'float64_reference',
                  'memory_policy':{'all_physical_devices_memory_growth':True,
                                   'configured_before_logical_device_initialization':True}}
        master.write(output/'manifest.json',manifest)
        if method:
            result={'method':method,'target_name':target,'scientific_promotion':False,
                    'sampling_quality':'not_established','downstream_hmc':'not_run',
                    'target_signature':'test-target','teacher_admitted':False,
                    'status':'mechanics_control_completed'}
            if method=='fab' and target!='gaussian':
                result.update(status='prerequisite_failed',native_training='not_run',reason='tail fixture')
            else:
                for name in ('bank.tensor','log-weights.tensor','student-checkpoint.json',
                             'native-summary.json','reference-summary.json','target.json','spec.json','stdout.log'):
                    (output/name).write_text('controller fixture')
                master.write(output/'student-frozen.json',{'target_signature':'test-target'})
                master.write(output/'post-training-1000.json',{
                    'complete':True,'finite':True,'rows':1000,'valid_rows':1000,
                    'geometry_role':'explanatory_only_no_calibrated_finite_cutoff',
                    'score_residual_norm':{'median':3.}})
        else:
            result={'status':'passed','device':'GPU:0' if device=='gpu' else 'CPU:0'}
        master.write(output/'result.json',result)
        state['attempts'].append(row)
        ledger['attempts'].append({**row,'job':'six-method-20261003-'+job+'-r1'})
        return row

    add('check','cpu');add('preflight','gpu')
    for method in master.METHODS:
        for target in master.TARGETS:add(f'control-{target}-{method}','gpu',method,target)
    master.write(root/'state.json',state)
    master.write(shared/'state.json',ledger)
    master.write(shared/'config.json',{'gpu_process_seconds':100000.,'cpu_core_seconds':100000.})
    monkeypatch.setattr(master,'CAMPAIGN',root)
    monkeypatch.setattr(master,'SHARED',shared)
    return root,shared,state,ledger,source


def test_terminal_resume_after_live_edits_audits_without_workers(master,campaign,monkeypatch):
    root,_,state,_,_=campaign
    def forbidden(*args,**kwargs):
        pytest.fail('terminal resume attempted a source snapshot or worker')
    monkeypatch.setattr(master,'source_snapshot',forbidden)
    controller=master.Controller()
    assert master.read(root/'state.json')['next_action']==state['next_action']
    monkeypatch.setattr(controller,'execute',forbidden)
    assert controller.run()==0
    saved=master.read(root/'state.json')
    assert len(saved['attempts'])==20
    audit=master.read(saved['terminal_audit'])
    assert audit['status']=='passed'
    assert sum(r['status']=='prerequisite_failed' for r in audit['cells'])==2
    assert audit['scientific_promotion'] is False
    assert controller.run()==0
    assert (root/'audit-r1/result.json').is_file() and (root/'audit-r2/result.json').is_file()


def test_corrupt_terminal_artifact_fails_audit_without_relaunch(master,campaign,monkeypatch):
    root,_,_,_,_=campaign
    (root/'control-gaussian-smc-r1/post-training-1000.json').unlink()
    controller=master.Controller()
    monkeypatch.setattr(controller,'execute',lambda *a,**k:pytest.fail('unrequested GPU relaunch'))
    assert controller.run()==1
    assert master.read(root/'state.json')['status']=='engineering_audit_failed'


@pytest.mark.parametrize('failure',['missing_cell','new_failed_attempt'])
def test_incomplete_matrix_continues_to_checks(master,campaign,monkeypatch,failure):
    root,_,state,_,_=campaign
    if failure=='missing_cell':state['attempts'].pop()
    else:state['attempts'].append({**state['attempts'][-1],'status':'failed','attempt':2})
    master.write(root/'state.json',state)
    controller=master.Controller();jobs=[]
    def failed_check(job,**kwargs):
        jobs.append(job);return {'status':'failed'}
    monkeypatch.setattr(controller,'execute',failed_check)
    assert controller.run()==1 and jobs==['check']


@pytest.mark.parametrize('fault',['source','duplicate_charge','missing_charge','promotion','probe','live_check','gpu_policy'])
def test_audit_detects_provenance_accounting_and_false_completion(master,campaign,fault):
    root,_,_,ledger,source=campaign
    if fault=='source':(source/'fixture.py').write_text('changed\n')
    elif fault=='duplicate_charge':ledger['attempts'].append(ledger['attempts'][-1])
    elif fault=='missing_charge':ledger['attempts'].pop()
    else:
        path=(root/'check-r1/manifest.json' if fault=='live_check' else
              root/'control-gaussian-smc-r1'/('post-training-1000.json' if fault=='probe' else
                                            'manifest.json' if fault=='gpu_policy' else 'result.json'))
        value=master.read(path)
        if fault=='promotion':value['scientific_promotion']=True
        elif fault=='probe':value['valid_rows']=999
        elif fault=='live_check':value['execution_cwd']='/live'
        else:value['memory_policy']['all_physical_devices_memory_growth']=False
        master.write(path,value)
    result=master.audit_campaign(root,ledger)
    assert result['status']=='failed' and result['errors']


def test_shared_consumer_crash_stops_dependent_repetition(master,campaign,monkeypatch):
    root,_,state,_,_=campaign
    state['attempts']=[];master.write(root/'state.json',state)
    controller=master.Controller();jobs=[]
    def execute(job,**kwargs):
        jobs.append(job)
        return {'status':'complete' if job in ('check','preflight') else 'failed'}
    monkeypatch.setattr(controller,'execute',execute)
    assert controller.run()==1
    assert jobs==['check','preflight','control-gaussian-fab']


def test_check_executes_frozen_tests_and_records_parent_cost(master,campaign,monkeypatch):
    root,_,_,_,source=campaign
    controller=master.Controller()
    controller.source=source
    controller.state['attempts']=[]
    launches=[]
    class Child:
        pid=12345
        def __init__(self,command,**kwargs):launches.append((command,kwargs))
        def wait(self,timeout):return 0
    monkeypatch.setattr(master.subprocess,'Popen',Child)
    row=controller.execute('snapshot-check',device='cpu')
    command,options=launches[0]
    assert options['cwd']==source and options['env']['PYTHONPATH']==str(source)
    assert options['env']['CUDA_VISIBLE_DEVICES']=='-1'
    assert options['env']['WARM_START_AUTHOR_FIXTURE']==str(source/master.AUTHOR_FIXTURE)
    assert all(name in command for name in master.TEST_PATHS)
    manifest=master.read(Path(row['output'])/'manifest.json')
    assert manifest['execution_cwd']==str(source)
    assert manifest['cpu_core_seconds']==row['cpu_core_seconds']
    assert manifest['process_cpu_accounting']=='parent_wait_includes_shutdown'


@pytest.mark.parametrize('arguments',[[],['unknown'],['status','extra'],['run','--spec','/tmp/anything']])
def test_fixed_launcher_rejects_unapproved_argument_forms(arguments):
    wrapper=Path(__file__).resolve().parents[1]/'scripts/run_neutra_six_method_campaign.sh'
    process=subprocess.run(['bash',str(wrapper),*arguments],capture_output=True,text=True,timeout=5)
    assert process.returncode==2
