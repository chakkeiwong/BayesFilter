"""Resumable native-teacher NAF queue using the existing campaign accountant."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from bayesfilter.testing.neutra_scientific_design import (
    FORWARD_REVERSE_POLICY, forward_reverse_student, forward_reverse_teachers,
    target_catalog, CALIBRATION_TARGETS, FINAL_TARGETS, FIT_SEEDS,
)
from bayesfilter.testing.neutra_forward_reverse_criteria import (
    PAIR_CRITERION, CRITERION_PLAN, assess_pair_criteria,
)

PLAN='docs/plans/bayesfilter-neutra-naf-forward-reverse-master-2026-10-06.md'
CALIBRATION_SEEDS=(601,607,613)
METHOD_STATUS={
    'smc':'executable native posterior teacher',
    'ais':'executable native posterior teacher; no resampling',
    'fab':'deferred: auxiliary target integrability not established',
    'gabrie':'deferred: frozen-map control is not the complete adaptive teacher',
    'aft':'deferred: complete source-controller equivalence not established',
    'craft':'deferred: complete source-controller equivalence not established',
}


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def program(remaining):
    return dict(policy=FORWARD_REVERSE_POLICY,plan=PLAN,default='naf_dsf/author_cmade',
        criterion=PAIR_CRITERION,criterion_plan=CRITERION_PLAN,
        method_status=METHOD_STATUS,student=forward_reverse_student(),
        teacher_candidates=forward_reverse_teachers(),target_catalog=target_catalog(),
        calibration_targets=CALIBRATION_TARGETS,calibration_seeds=CALIBRATION_SEEDS,
        final_targets=FINAL_TARGETS,final_seeds=FIT_SEEDS,
        stage_order=['native_teacher_calibration','forward_and_reverse_calibration',
                     'freeze_recipe','fixed_unwarped_and_warped','untouched_random_geometries'],
        remaining=remaining,full_campaign_affordability='requires_measured_stage_reservations',
        readiness='executable_calibration_protocol_not_yet_scientifically_validated',
        command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh forward-reverse',
        resume_command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh resume')


def checked(row):
    root=Path(row['output']); manifest=read(root/'manifest.json')
    if manifest['status']!='complete': raise RuntimeError('incomplete native-study worker')
    for name,digest in manifest['artifact_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('native-study artifact changed: '+name)
    result=read(root/'result.json')
    if result.get('status')=='fit_complete':
        specification=read(root/'spec.json')['specification']
        result=assess_pair_criteria(result,len(specification.get('weights',[])))
    return result


def successful_branch(rows):
    if not rows or any(r.get('status')!='fit_complete' for r in rows): return None
    first=rows[0]
    for branch in first['branches']:
        key=(branch['rate'],branch['updates'])
        if all(any((b['rate'],b['updates'])==key and b['passed'] for b in r['branches']) for r in rows):
            return dict(rate=key[0],updates=key[1])
    return None


def _reassess_saved_fixed(state, root):
    """Apply the owner-approved warm-start rule to preserved fixed artifacts.

    Worker files remain immutable. The old controller decision is copied before
    the queue view is updated, so the criterion change is auditable.
    """
    if not state.get('fixed_results') or state.get('criterion') == PAIR_CRITERION:
        return False
    selected = state.get('selection')
    if not selected:
        raise RuntimeError('cannot reassess fixed results before recipe selection')
    if state.get('random_results') or any(r.get('role')=='random' for r in state['completed']):
        raise RuntimeError('criterion migration requires unexposed random cases')
    evidence=[]

    def saved(role,target,seed,phase):
        suffix='teacher' if phase=='forward_reverse_teacher' else 'fit'
        job=f'forward-reverse-{role}-{selected["method"]}-{selected["teacher"]["profile_id"]}-{target}-s{seed}-{suffix}'
        matches=[row for row in state['completed'] if row['job']==job]
        if len(matches)!=1:
            raise RuntimeError('expected one preserved worker: '+job)
        result=checked(matches[0])
        path=Path(matches[0]['output'])/'result.json'
        evidence.append(dict(job=job,result_file=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return result

    def selected_branch(result):
        if result.get('status')!='fit_complete':
            raise RuntimeError('preserved fit is incomplete')
        branch=next((b for b in result['branches']
                     if b['rate']==selected['reverse']['rate'] and
                     b['updates']==selected['reverse']['updates']),None)
        if branch is None:
            raise RuntimeError('selected reverse branch missing')
        return branch

    calibration=[]
    for target in CALIBRATION_TARGETS:
        for seed in selected['calibration_seeds']:
            teacher=saved('calibration',target,seed,'forward_reverse_teacher')
            fit=saved('calibration',target,seed,'forward_reverse_fit')
            passed=teacher['status']=='teacher_passed' and selected_branch(fit)['passed']
            if not passed:
                raise RuntimeError('frozen recipe fails revised calibration: '+target+' '+str(seed))
            calibration.append(dict(target=target,seed=seed,passed=True))
    old = list(state['fixed_results'])
    revised=[]
    for target in FINAL_TARGETS[:2]:
        for seed in FIT_SEEDS:
            teacher=saved('fixed',target,seed,'forward_reverse_teacher')
            result=saved('fixed',target,seed,'forward_reverse_fit')
            branch=selected_branch(result)
            prior=next(r for r in old if r['target']==target and r['seed']==seed)
            revised.append(dict(target=target,seed=seed,status=result['status'],
                passed=bool(teacher['status']=='teacher_passed' and branch['passed']),
                legacy_passed=prior['passed'],legacy_forward_passed=bool(
                    result['forward'].get('legacy_full_accuracy_passed',
                                          result['forward'].get('passed',False))),
                warm_start=result['forward']['warm_start'],
                final_endpoint_passed=branch['endpoint']['passed']))
    archives=[]
    for label in ('state','result','selection'):
        source=root/f'forward-reverse-{label}.json'
        destination=root/f'forward-reverse-{label}-legacy-v1.json'
        if source.exists() and not destination.exists():
            shutil.copyfile(source,destination)
        if destination.exists():archives.append(str(destination))
    revision=dict(criterion=PAIR_CRITERION,criterion_plan=CRITERION_PLAN,
        legacy_archives=archives,calibration_rechecked=calibration,
        frozen_reverse_recipe=selected['reverse'],source_results=evidence,
        old_results=old,new_results=revised,worker_artifacts_unchanged=True,
        retrospective=True,scientific_promotion=False)
    revision_path=root/'forward-reverse-criterion-revision-v2.json'
    write(revision_path,revision)
    state['legacy_fixed_results']=old
    state['fixed_results']=revised
    state['criterion']=PAIR_CRITERION
    state['criterion_revision']=str(revision_path)
    write(root/'forward-reverse-result.json',dict(status='criterion_reassessed',
        policy=FORWARD_REVERSE_POLICY,criterion=PAIR_CRITERION,selection=selected,
        fixed=revised,random='not_exposed',criterion_revision=str(revision_path),
        scientific_promotion=False,scope='retrospective_fixed_reassessment_random_tests_pending',
        method_status=METHOD_STATUS))
    return True


def smoke(controller):
    """Bounded native CPU -> GPU/XLA call-chain check, not a research result."""
    root=Path(controller.state['attempts'][0]['output']).parent
    path=root/'forward-reverse-smoke.json'
    write(path,dict(status='running',scientific_promotion=False,
        scope='native_CPU_to_GPU_XLA_mechanics_only'))
    try:
        return _smoke(controller,root)
    except Exception as error:
        write(path,dict(status='failed',error=str(error),scientific_promotion=False,
            scope='native_CPU_to_GPU_XLA_mechanics_only'))
        controller.sync('forward_reverse_smoke_failed','repair or rerun the bounded smoke before calibration')
        write(root/'next-phase.json',dict(phase='forward_reverse_smoke_failed',
            next_action='inspect preserved failure and rerun bounded mechanics smoke',
            resume_command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh forward-reverse-smoke',
            remaining=controller.remaining()))
        raise


def _smoke(controller,root):
    spec={'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]}
    teacher_profile=dict(profile_id='mechanics_smoke',particles=256,stages=2,mutation_steps=2,
        step_size=.05,mode_starts=8,mode_iterations=50,replications=2,reference_rows=256,
        method='smc',jit_compile=True,worker_wall_limit=180,worker_cpu_limit=300,
        role='tiny_mechanics_no_scientific_claim')
    teacher_row=controller.execute('forward-reverse-smoke-teacher',phase='forward_reverse_teacher',
        device='cpu',method='smc',target='gaussian_smoke',seed=619,profile=teacher_profile,
        role=teacher_profile['role'],specification=spec)
    teacher=checked(teacher_row)
    if teacher['status']!='teacher_passed':raise RuntimeError('smoke native teacher screen failed')
    student={**forward_reverse_student(),'width':4,'batch':4,'forward_updates':[2,2],
        'rkl_rates':[.0001],'rkl_rungs':[2,4],'teacher_output':teacher_row['output'],
        'worker_wall_limit':240,'worker_cpu_limit':360,'gpu_index':2,
        'role':'tiny_mechanics_no_scientific_claim'}
    row=controller.execute('forward-reverse-smoke-fit',phase='forward_reverse_fit',device='gpu',
        target='gaussian_smoke',seed=619,profile=student,role=student['role'],specification=spec)
    result=checked(row)
    if result['status']!='fit_complete':raise RuntimeError('forward/reverse mechanics incomplete')
    if not all(b['finite'] and b['jit_compile'] and 'GPU' in b['device'] and not b['samplewise_loop']
               for b in result['training']):raise RuntimeError('smoke did not execute batched GPU/XLA')
    write(root/'forward-reverse-smoke.json',dict(status='passed',teacher=teacher_row['output'],
        fit=row['output'],scientific_promotion=False,scope='native_CPU_to_GPU_XLA_mechanics_only'))
    controller.sync('forward_reverse_prepared','mechanics checked; native calibration and pricing pending')
    write(root/'next-phase.json',dict(phase='forward_reverse_prepared',
        next_action='execute native calibration and measure complete-stage affordability',
        resume_command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh forward-reverse',
        remaining=controller.remaining()))
    return 0


def run_campaign(controller, *, prepare_only=False):
    root=Path(controller.state['attempts'][0]['output']).parent
    write(root/'forward-reverse-program.json',program(controller.remaining()))
    if prepare_only:
        print(json.dumps(program(controller.remaining()),indent=2)); return 0
    path=root/'forward-reverse-state.json'
    state=read(path) if path.exists() else dict(policy=FORWARD_REVERSE_POLICY,criterion=PAIR_CRITERION,status='prepared',
        completed=[],created_unix=time.time(),target_catalog=target_catalog(),rejections=[])
    if state['policy']!=FORWARD_REVERSE_POLICY or state['target_catalog']!=target_catalog():
        raise RuntimeError('frozen forward/reverse scientific design changed')

    # The owner changed only the intermediate warm-start criterion. Reassess
    # preserved fixed endpoints once, then continue to untouched random cases.
    _reassess_saved_fixed(state,root)

    def sync(status,action):
        state.update(status=status,next_action=action,remaining=controller.remaining(),updated_unix=time.time())
        write(path,state)
        controller.sync('forward_reverse_'+status,action)
        write(root/'next-phase.json',dict(phase='forward_reverse_'+status,next_action=action,
            remaining=controller.remaining(),resume_command=program({})['resume_command']))

    def reserve(gpu,cpu,reason):
        available=controller.remaining()
        if gpu>available['gpu_process_seconds'] or cpu>available['cpu_core_seconds']:
            sync('under_budgeted',reason)
            raise RuntimeError('full native-study reservation unavailable: '+reason)

    def execute(job,phase,target,seed,profile,device):
        saved=next((x for x in state['completed'] if x['job']==job),None)
        if saved: return checked(saved),saved
        wall=profile['worker_wall_limit']; cpu=profile['worker_cpu_limit']
        reserve(wall if device=='gpu' else 0,cpu,job)
        row=controller.execute(job,phase=phase,device=device,method=profile.get('method','naf'),
            target=target,seed=seed,profile=profile,role=profile.get('role','development'),
            specification=state['target_catalog'][target])
        if row['status']!='complete':
            sync('repair_required','inspect worker '+job); raise RuntimeError('native-study worker failed')
        state['completed'].append(row)
        sync('running','completed '+job)
        return checked(row),row

    def costs(phase,profile=None):
        rows=[x for x in state['completed'] if x['phase']==phase and
              (profile is None or x['profile']['profile_id']==profile['profile_id'])]
        return (max(x['wall_seconds'] for x in rows),max(x['cpu_core_seconds'] for x in rows)) if rows else None

    def limits(phase,profile=None):
        measured=costs(phase,profile)
        # Engineering ceilings from attribution plus explicit CPU sampling
        # hypothesis. Actual full workers replace them; never a fit criterion.
        fallback=(600,1200) if phase=='forward_reverse_teacher' else (2400,3000)
        if not measured:return dict(worker_wall_limit=fallback[0],worker_cpu_limit=fallback[1])
        return dict(worker_wall_limit=max(30,math.ceil(1.5*measured[0])),
                    worker_cpu_limit=max(30,math.ceil(1.5*measured[1])))

    def reserve_stage(cases,label,teacher_profile):
        if not cases:return
        t=limits('forward_reverse_teacher',teacher_profile); f=limits('forward_reverse_fit')
        reserve(cases*f['worker_wall_limit'],cases*(t['worker_cpu_limit']+f['worker_cpu_limit']),label)

    def trial(method,teacher_profile,target,seed,student,role):
        stem=f'forward-reverse-{role}-{method}-{teacher_profile["profile_id"]}-{target}-s{seed}'
        teacher,row=execute(stem+'-teacher','forward_reverse_teacher',target,seed,
            {**teacher_profile,**limits('forward_reverse_teacher',teacher_profile),'method':method,'role':role},'cpu')
        if teacher['status']!='teacher_passed':return teacher
        fitted,_=execute(stem+'-fit','forward_reverse_fit',target,seed,
            {**student,**limits('forward_reverse_fit'),'teacher_output':row['output'],
             'method':method,'gpu_index':2,'role':role},'gpu')
        return fitted

    if state['status']=='complete':
        for row in state['completed']:checked(row)
        return 0
    sync('prepared','calibrate native teacher and objective switch')
    try:
        if not state.get('selection'):
            for teacher_profile in forward_reverse_teachers():
                for method in ('smc','ais'):
                    candidate=method+'/'+teacher_profile['profile_id']
                    if candidate in state['rejections']:continue
                    rows=[]
                    cases=[(t,s) for t in CALIBRATION_TARGETS for s in CALIBRATION_SEEDS]
                    for index,(target,seed) in enumerate(cases):
                        rows.append(trial(method,teacher_profile,target,seed,forward_reverse_student(),'calibration'))
                        if not successful_branch(rows):break
                        # The first complete pair prices the remaining stage;
                        # later observations can increase the reservation.
                        reserve_stage(len(cases)-index-1,'remaining calibration '+candidate,teacher_profile)
                    if len(rows)==len(cases) and successful_branch(rows):
                        state['selection']=dict(method=method,teacher=teacher_profile,
                            reverse=successful_branch(rows),student=forward_reverse_student(),
                            criterion=PAIR_CRITERION,
                            calibration_seeds=CALIBRATION_SEEDS,selected_unix=time.time(),
                            selection_rule='first complete recipe with suitable warm start and passing final map')
                        write(root/'forward-reverse-selection.json',state['selection'])
                        sync('recipe_frozen','test fixed targets then unseen geometries')
                        break
                    state['rejections'].append(candidate)
                    sync('calibrating','candidate rejected; continue the declared ladder')
                if state.get('selection'):break
        if not state.get('selection'):
            sync('under_calibrated','no complete native FKL/RKL recipe passed; inspect stage failures')
            return 2
        selected=state['selection']
        student={**selected['student'],'rkl_rates':[selected['reverse']['rate']],
                 'rkl_rungs':[selected['reverse']['updates']]}
        for phase,targets in (('fixed',FINAL_TARGETS[:2]),('random',FINAL_TARGETS[2:])):
            if phase=='random' and not all(r['passed'] for r in state['fixed_results']):
                write(root/'forward-reverse-result.json',dict(status='fixed_screen_failed',
                    policy=FORWARD_REVERSE_POLICY,selection=state['selection'],
                    fixed=state['fixed_results'],random='not_exposed',remaining=controller.remaining(),
                    criterion=state.get('criterion','legacy_v1'),
                    scientific_promotion=False,scope='frozen_recipe_rejected_development_repair_required',
                    method_status=METHOD_STATUS))
                sync('fixed_screen_failed','repair development protocol; preserve randomized holdouts')
                return 2
            key=phase+'_results'
            if key in state:continue
            # Reserve all outstanding cases before exposing a new target stage.
            pending=sum(not any(x['job'].startswith(f'forward-reverse-{phase}-') and
                x['target']==t and x['seed']==s and x['phase']=='forward_reverse_fit'
                for x in state['completed']) for t in targets for s in FIT_SEEDS)
            reserve_stage(pending,phase+' holdout stage',selected['teacher'])
            rows=[]
            for target in targets:
                for seed in FIT_SEEDS:
                    result=trial(selected['method'],selected['teacher'],target,seed,student,phase)
                    rows.append(dict(target=target,seed=seed,status=result['status'],
                        passed=successful_branch([result]) is not None))
            state[key]=rows
            sync(phase+'_complete','preserve all holdout outcomes without retuning')
        result=dict(status='complete',policy=FORWARD_REVERSE_POLICY,criterion=PAIR_CRITERION,selection=state['selection'],
            fixed=state['fixed_results'],random=state['random_results'],remaining=controller.remaining(),
            criterion_revision=state.get('criterion_revision'),
            scientific_promotion=False,scope='bounded approximate_FKL_to_RKL transfer study',
            method_status=METHOD_STATUS)
        write(root/'forward-reverse-result.json',result)
        sync('complete','review training and transfer results; no automatic HMC or q20 promotion')
        return 0
    except Exception:
        if state['status']!='under_budgeted':sync('repair_required','inspect preserved worker/artifact failure')
        raise
