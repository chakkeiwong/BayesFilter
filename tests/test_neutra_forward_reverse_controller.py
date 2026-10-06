"""Framework-free tests of continuation, holdout separation and resource guards."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def queue():
    path=Path(__file__).resolve().parents[1]/'scripts/neutra_forward_reverse_campaign.py'
    spec=importlib.util.spec_from_file_location('forward_reverse_queue_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fake(queue,tmp_path,monkeypatch,*,reject_first=False,teacher_fail=False,budget=1e7):
    payloads={}
    class Controller:
        def __init__(self):self.state={'attempts':[{'output':str(tmp_path/'prior')}]};self.calls=[]
        def remaining(self):return {'gpu_process_seconds':budget,'cpu_core_seconds':budget}
        def sync(self,*args):pass
        def execute(self,job,**kw):
            self.calls.append((job,kw));output=tmp_path/(job+'-r1');output.mkdir()
            row={'job':job,'output':str(output),'source':'fixture','status':'complete',
                 'wall_seconds':10.,'cpu_core_seconds':12.,**kw}
            if kw['phase']=='forward_reverse_teacher':
                result={'status':'teacher_failed' if teacher_fail else 'teacher_passed'}
            else:
                p=kw['profile'];result={'status':'fit_complete','branches':[
                    {'rate':rate,'updates':rung,'passed':not(reject_first and 'smc-native_0' in job)}
                    for rate in p['rkl_rates'] for rung in p['rkl_rungs']]}
            payloads[str(output)]=result
            return row
    monkeypatch.setattr(queue,'checked',lambda row:payloads[row['output']])
    return Controller(),payloads


def test_full_queue_freezes_recipe_before_six_holdouts_and_resume_is_idle(queue,tmp_path,monkeypatch):
    c,_=fake(queue,tmp_path,monkeypatch)
    assert queue.run_campaign(c)==0
    teachers=[kw for _,kw in c.calls if kw['phase']=='forward_reverse_teacher']
    fits=[kw for _,kw in c.calls if kw['phase']=='forward_reverse_fit']
    assert len(teachers)==len(fits)==24
    assert all(kw['device']=='cpu' for kw in teachers)
    assert all(kw['device']=='gpu' and kw['profile']['gpu_index']==2 for kw in fits)
    assert set(kw['target'] for kw in fits[:6])==set(queue.CALIBRATION_TARGETS)
    assert all(len(kw['profile']['rkl_rates'])==1 for kw in fits[6:])
    count=len(c.calls);assert queue.run_campaign(c)==0 and len(c.calls)==count


def test_failed_candidate_continues_declared_teacher_ladder(queue,tmp_path,monkeypatch):
    c,_=fake(queue,tmp_path,monkeypatch,reject_first=True)
    assert queue.run_campaign(c)==0
    result=json.loads((tmp_path/'forward-reverse-result.json').read_text())
    assert result['selection']['method']=='ais'
    assert 'smc/native_0' in json.loads((tmp_path/'forward-reverse-state.json').read_text())['rejections']


def test_teacher_failure_never_launches_fit_or_holdout(queue,tmp_path,monkeypatch):
    c,_=fake(queue,tmp_path,monkeypatch,teacher_fail=True)
    assert queue.run_campaign(c)==2
    assert all(kw['phase']=='forward_reverse_teacher' for _,kw in c.calls)
    assert len(c.calls)==4


def test_larger_teacher_profile_does_not_inherit_smaller_profile_timing(queue,tmp_path,monkeypatch):
    c,payloads=fake(queue,tmp_path,monkeypatch)
    def checked(row):
        if row['phase']=='forward_reverse_teacher' and row['profile']['profile_id']=='native_0':
            return {'status':'teacher_failed'}
        return payloads[row['output']]
    monkeypatch.setattr(queue,'checked',checked)
    assert queue.run_campaign(c)==0
    larger=next(kw for _,kw in c.calls if kw['phase']=='forward_reverse_teacher'
                and kw['profile']['profile_id']=='native_1')
    assert larger['profile']['worker_wall_limit']==600
    assert larger['profile']['worker_cpu_limit']==1200


def test_failed_fixed_stage_keeps_random_geometry_untouched_and_resume_idle(queue,tmp_path,monkeypatch):
    c,payloads=fake(queue,tmp_path,monkeypatch)
    def checked(row):
        result=payloads[row['output']]
        if row['phase']=='forward_reverse_fit' and row['role']=='fixed':
            result={**result,'branches':[{**b,'passed':False} for b in result['branches']]}
        return result
    monkeypatch.setattr(queue,'checked',checked)
    assert queue.run_campaign(c)==2
    assert not any(kw['role']=='random' for _,kw in c.calls)
    result=json.loads((tmp_path/'forward-reverse-result.json').read_text())
    assert result['status']=='fixed_screen_failed' and result['random']=='not_exposed'
    count=len(c.calls)
    assert queue.run_campaign(c)==2 and len(c.calls)==count


def test_budget_failure_precedes_launch_and_preserves_resume_state(queue,tmp_path,monkeypatch):
    c,_=fake(queue,tmp_path,monkeypatch,budget=100.)
    with pytest.raises(RuntimeError,match='reservation'):queue.run_campaign(c)
    assert not c.calls
    assert json.loads((tmp_path/'forward-reverse-state.json').read_text())['status']=='under_budgeted'


def test_same_reverse_setting_must_pass_all_calibration_pairs(queue):
    rows=[{'status':'fit_complete','branches':[{'rate':.001,'updates':256,'passed':True},
                                             {'rate':.0003,'updates':256,'passed':False}]},
          {'status':'fit_complete','branches':[{'rate':.001,'updates':256,'passed':False},
                                             {'rate':.0003,'updates':256,'passed':True}]}]
    assert queue.successful_branch(rows) is None


def test_plan_keeps_unimplemented_methods_visible_without_false_failures(queue):
    plan=queue.program({'gpu_process_seconds':1.,'cpu_core_seconds':2.})
    assert set(plan['method_status'])=={'fab','gabrie','ais','smc','aft','craft'}
    assert plan['default']=='naf_dsf/author_cmade'
    assert set(plan['calibration_targets']).isdisjoint(plan['final_targets'])
    assert 'requires_measured' in plan['full_campaign_affordability']


def test_failed_smoke_preserves_recovery_route_without_launching_calibration(queue,tmp_path,monkeypatch):
    c,_=fake(queue,tmp_path,monkeypatch,teacher_fail=True)
    with pytest.raises(RuntimeError,match='native teacher screen failed'):
        queue.smoke(c)
    assert len(c.calls)==1 and c.calls[0][1]['phase']=='forward_reverse_teacher'
    result=json.loads((tmp_path/'forward-reverse-smoke.json').read_text())
    assert result['status']=='failed' and not result['scientific_promotion']
    next_phase=json.loads((tmp_path/'next-phase.json').read_text())
    assert next_phase['resume_command'].endswith(' forward-reverse-smoke')


def legacy_fixed_state(queue,tmp_path,monkeypatch):
    c,payloads=fake(queue,tmp_path,monkeypatch)
    selected=dict(method='smc',teacher=queue.forward_reverse_teachers()[0],
        reverse={'rate':.0001,'updates':256},student=queue.forward_reverse_student(),
        calibration_seeds=list(queue.CALIBRATION_SEEDS))
    state=dict(policy=queue.FORWARD_REVERSE_POLICY,status='fixed_screen_failed',
        target_catalog=queue.target_catalog(),selection=selected,completed=[],rejections=[],fixed_results=[])
    for role,targets,seeds in [('calibration',queue.CALIBRATION_TARGETS,queue.CALIBRATION_SEEDS),
                               ('fixed',queue.FINAL_TARGETS[:2],queue.FIT_SEEDS)]:
        for target in targets:
            for seed in seeds:
                for phase,suffix in [('forward_reverse_teacher','teacher'),('forward_reverse_fit','fit')]:
                    job=f'forward-reverse-{role}-smc-native_0-{target}-s{seed}-{suffix}'
                    row=c.execute(job,phase=phase,role=role,target=target,seed=seed,
                        profile={**selected['student'],'profile_id':'native_0'})
                    state['completed'].append(row)
                    r=payloads[row['output']]
                    if suffix=='fit':
                        r['forward']={'passed':role=='calibration','warm_start':{'passed':True}}
                        for b in r['branches']:b['endpoint']={'passed':True}
                        if role=='fixed':state['fixed_results'].append(dict(
                            target=target,seed=seed,status='fit_complete',passed=False))
                    (Path(row['output'])/'result.json').write_text(json.dumps(r))
    for label,obj in [('state',state),('selection',selected),('result',{'status':'fixed_screen_failed'})]:
        queue.write(tmp_path/f'forward-reverse-{label}.json',obj)
    c.calls=[]
    return c,payloads,state


def test_revision_rechecks_frozen_recipe_and_preserves_files_then_runs_only_random(queue,tmp_path,monkeypatch):
    c,payloads,state=legacy_fixed_state(queue,tmp_path,monkeypatch)
    old_state=(tmp_path/'forward-reverse-state.json').read_bytes()
    worker_bytes={key:(Path(key)/'result.json').read_bytes() for key in payloads}
    assert queue.run_campaign(c)==0
    assert len(c.calls)==24 and all(kw['role']=='random' for _,kw in c.calls)
    assert (tmp_path/'forward-reverse-state-legacy-v1.json').read_bytes()==old_state
    for key,data in worker_bytes.items():assert (Path(key)/'result.json').read_bytes()==data
    revision=queue.read(tmp_path/'forward-reverse-criterion-revision-v2.json')
    assert len(revision['calibration_rechecked'])==6
    assert len(revision['source_results'])==24 and revision['retrospective']
    assert all(x['passed'] and not x['legacy_passed'] for x in revision['new_results'])
    assert queue.read(tmp_path/'forward-reverse-state.json')['selection']==state['selection']
    count=len(c.calls);assert queue.run_campaign(c)==0 and len(c.calls)==count


def test_reassessment_cannot_hide_final_failure_or_invalid_training(queue,tmp_path,monkeypatch):
    c,payloads,state=legacy_fixed_state(queue,tmp_path,monkeypatch)
    row=next(x for x in state['completed'] if x['role']=='fixed' and x['phase']=='forward_reverse_fit')
    b=payloads[row['output']]['branches'][0]
    b['passed']=False  # Models a final or training-path veto from assess_pair_criteria.
    assert queue.run_campaign(c)==2 and not c.calls
    result=queue.read(tmp_path/'forward-reverse-result.json')
    assert result['random']=='not_exposed' and not result['fixed'][0]['passed']


def test_invalid_revised_calibration_cannot_reselect_after_inspecting_fixed_data(queue,tmp_path,monkeypatch):
    c,payloads,state=legacy_fixed_state(queue,tmp_path,monkeypatch)
    row=next(x for x in state['completed'] if x['role']=='calibration' and x['phase']=='forward_reverse_fit')
    payloads[row['output']]['branches'][0]['passed']=False
    with pytest.raises(RuntimeError,match='frozen recipe fails revised calibration'):
        queue.run_campaign(c)
    assert not c.calls
