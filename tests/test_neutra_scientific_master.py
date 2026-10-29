"""Controller regressions: completeness, honest selection and cumulative budgets."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def master():
    path=Path(__file__).resolve().parents[1]/'scripts/run_neutra_scientific_campaign_master.py'
    spec=importlib.util.spec_from_file_location('scientific_master_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def controller(master,tmp_path,monkeypatch):
    root=tmp_path/'campaign';shared=tmp_path/'shared';shared.mkdir()
    fixture_limits={'gpu_process_seconds':6000.,'cpu_core_seconds':12000.}
    monkeypatch.setattr(master,'LIMITS',fixture_limits)
    master.write(shared/'config.json',fixture_limits)
    master.write(shared/'state.json',{'attempts':[]})
    monkeypatch.setattr(master,'CAMPAIGN',root);monkeypatch.setattr(master,'SHARED',shared)
    monkeypatch.setattr(master,'PREVIOUS',())
    return master.Controller()


def test_selection_requires_same_profile_to_pass_both_targets(master,controller,monkeypatch):
    outcomes={}
    for i,t in enumerate(master.DEV_TARGETS):
        p=master.profile_candidates('smc')[i]
        outcomes[f'calibrate-smc-{t}-{p["profile_id"]}']={'status':'passed','teacher_admitted':True}
    monkeypatch.setattr(controller,'result',lambda job:outcomes.get(job,{}))
    student=master.student_candidates()[0]
    assert 'smc' not in controller.select_profiles(student)
    p=master.profile_candidates('smc')[0]
    outcomes[f'calibrate-smc-{master.DEV_TARGETS[1]}-{p["profile_id"]}']={'status':'passed','teacher_admitted':True}
    assert controller.select_profiles(student)['smc']['profile_id']==p['profile_id']
    outcomes[f'calibrate-smc-{master.DEV_TARGETS[1]}-{p["profile_id"]}']['status']='map_failed'
    assert 'smc' not in controller.select_profiles(student)


def test_student_selection_rejects_one_favorable_target(master,controller,monkeypatch):
    p=master.student_candidates()[0]
    monkeypatch.setattr(controller,'result',lambda job:{'status':'passed' if master.DEV_TARGETS[0] in job else 'map_failed'})
    assert controller.select_student() is None


def test_previous_campaign_charges_are_not_reset(master,controller,tmp_path,monkeypatch):
    previous=tmp_path/'previous'
    master.write(previous/'state.json',{'attempts':[{'gpu_process_seconds':100.,'cpu_core_seconds':200.}]})
    monkeypatch.setattr(master,'PREVIOUS',(previous,))
    assert controller.remaining()=={'gpu_process_seconds':5900.,'cpu_core_seconds':11800.}


def test_pending_external_charge_settles_exactly_once_even_after_partial_write(master,controller):
    output=master.CAMPAIGN/'external-r1'
    row={'job':'external','phase':'reference','device':'cpu','output':str(output),
         'status':'complete','wall_seconds':3.,'gpu_process_seconds':0.,'cpu_core_seconds':5.,
         'accounting_status':'pending_shared_lock_release'}
    master.write(output/'pending-charge.json',row)
    # Simulate interruption after the local ledger update but before the shared one.
    controller.state['attempts'].append(row)
    controller.settle_pending_charges()
    controller.settle_pending_charges()
    assert len(controller.state['attempts'])==1
    assert len(master.read(master.SHARED/'state.json')['attempts'])==1
    assert controller.remaining()['cpu_core_seconds']==11995.
    assert master.read(output/'pending-charge.json')['accounting_status']=='settled'


def test_resume_of_failed_smoke_cannot_fall_back_to_historical_campaign(master,controller,monkeypatch):
    import sys
    from types import SimpleNamespace
    master.write(master.CAMPAIGN/'forward-reverse-smoke.json',{'status':'failed'})
    controller.state['status']='running'
    calls=[]
    monkeypatch.setattr(master,'RemedyController',lambda:controller)
    monkeypatch.setattr(controller,'run',lambda:pytest.fail('historical campaign launched'))
    monkeypatch.setitem(sys.modules,'neutra_forward_reverse_campaign',
        SimpleNamespace(smoke=lambda c:calls.append(c) or 0,
                        run_campaign=lambda c:pytest.fail('unvalidated calibration launched')))
    monkeypatch.setattr(sys,'argv',['master','resume'])
    assert master.main()==0
    assert calls==[controller]


def test_plan_refresh_settles_pending_checks_without_worker_launch(master,controller,monkeypatch,capsys):
    import sys
    from types import SimpleNamespace
    output=master.CAMPAIGN/'pending-readiness-r1'
    master.write(output/'pending-charge.json',dict(job='readiness',phase='reference',device='cpu',
        output=str(output),status='complete',wall_seconds=3.,gpu_process_seconds=0.,
        cpu_core_seconds=5.,accounting_status='pending_shared_lock_release'))
    controller.state.update(status='forward_reverse_prepared',next_action='calibration pending')
    monkeypatch.setattr(master,'RemedyController',lambda:controller)
    monkeypatch.setattr(controller,'execute',lambda *a,**k:pytest.fail('readiness launched a worker'))
    monkeypatch.setattr(controller,'ensure_source',lambda:pytest.fail('readiness froze a source'))
    monkeypatch.setitem(sys.modules,'neutra_forward_reverse_campaign',
        SimpleNamespace(program=lambda remaining:{'remaining':remaining}))
    monkeypatch.setattr(sys,'argv',['master','forward-reverse-plan'])
    assert master.main()==0
    assert master.read(master.CAMPAIGN/'forward-reverse-program.json')['remaining']['cpu_core_seconds']==11995.


def test_empty_matrix_cannot_be_reported_complete(master,controller):
    assert controller.report()==1
    result=master.read(master.CAMPAIGN/'scientific-result.json')
    assert result['status']=='scientific_incomplete'
    assert len(result['cells'])==144 and all(c['status']=='not_run' for c in result['cells'])


def test_terminal_resume_audits_without_freezing_or_launching(master,controller,monkeypatch):
    controller.state['status']='scientific_terminal_screen_complete'
    monkeypatch.setattr(controller,'execute',lambda *a,**k:pytest.fail('terminal GPU relaunch'))
    monkeypatch.setattr(controller,'ensure_source',lambda:pytest.fail('terminal source refresh'))
    assert controller.run()==1  # Deliberately empty terminal fixture must fail completeness.


def test_shared_worker_failure_stops_following_cells(master,controller,monkeypatch):
    calls=[]
    def fail(job,**kwargs):
        calls.append(job)
        raise master.CampaignStop('injected shared evaluator error')
    monkeypatch.setattr(controller,'execute',fail)
    assert controller.run()==1
    assert calls==['check']
    assert 'injected shared' in master.read(master.CAMPAIGN/'scientific-result.json')['stop_reason']


@pytest.mark.parametrize('control_passed',(False,True))
def test_development_common_failure_stops_but_candidate_failure_continues(master,controller,monkeypatch,control_passed):
    remedy=master.RemedyController()
    rows=[{'target':master.DEV_TARGETS[0],'seed':seed,'result':str(master.CAMPAIGN/f'fit-{seed}'/'result.json'),
           'endpoints':{'lr-lower':{'coarse_screen_passed':True}}} for seed in (11,37)]
    master.write(master.CAMPAIGN/'representation-result.json',{'status':'remedy_representation_complete','errors':[],'rows':rows})
    calls=[]
    def execute(job,**kwargs):
        calls.append(job)
        output=master.CAMPAIGN/(job+'-r1')
        # Every map fails its candidate screen; only the common control may pass.
        result={'passed':control_passed if 'gaussian-control' in job else False,
                'status':'development_not_confirmed','tuning_wall_seconds':1.}
        master.write(output/'result.json',result)
        master.write(output/'manifest.json',{'artifact_sha256':{}})
        attempt={'job':job,'phase':kwargs['phase'],'device':kwargs.get('device','gpu'),
                 'output':str(output),'status':'complete','wall_seconds':2.,
                 'gpu_process_seconds':0.,'cpu_core_seconds':0.}
        remedy.state['attempts'].append(attempt)
        return attempt
    monkeypatch.setattr(remedy,'execute',execute)
    code=remedy.run_development_hmc()
    if control_passed:
        assert code==0
        assert calls[-2:]==[f'development-hmc-{master.DEV_TARGETS[0]}-s{s}' for s in (11,37)]
        result=master.read(master.CAMPAIGN/'development-hmc-result.json')
        assert all(not row['passed'] for row in result['rows'])
    else:
        assert code==1
        assert calls==['check','development-hmc-gaussian-control']


def test_completed_development_resume_reads_evidence_without_launching(master,controller,monkeypatch):
    remedy=master.RemedyController()
    target=master.DEV_TARGETS[0]
    rows=[{'target':target,'seed':11,'endpoints':{'lr-lower':{'coarse_screen_passed':True}}}]
    master.write(master.CAMPAIGN/'representation-result.json',{'status':'remedy_representation_complete','errors':[],'rows':rows})
    output=master.CAMPAIGN/'completed-map-r1'
    master.write(output/'result.json',{'status':'development_screen_passed','passed':True})
    master.write(output/'manifest.json',{'artifact_sha256':{}})
    remedy.state['attempts'].append({'job':f'development-hmc-{target}-s11','output':str(output),
        'status':'complete','gpu_process_seconds':2.,'cpu_core_seconds':3.})
    monkeypatch.setattr(remedy,'execute',lambda *a,**k:pytest.fail('completed validation relaunched'))
    monkeypatch.setattr(remedy,'ensure_source',lambda:pytest.fail('completed validation resnapshotted'))
    assert remedy.run_development_hmc()==0
    assert master.read(master.CAMPAIGN/'development-hmc-result.json')['status']=='development_hmc_complete'


def test_design_import_does_not_initialize_tensorflow():
    import subprocess,sys,os
    env={**os.environ,'BAYESFILTER_PRELOAD_CUSTOM_OP':'0','CUDA_VISIBLE_DEVICES':'-1'}
    result=subprocess.run([sys.executable,'-c',"import sys; import bayesfilter.testing.neutra_scientific_design as d; assert 'tensorflow' not in sys.modules; assert len(d.target_catalog())==8"],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
