#!/usr/bin/env python3
"""Outcome-driven controlled NeuTra repair using the existing budget controller."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import run_neutra_rare_region_master as base
RARE_CONFIGURATION=base.config

ROOT=Path(__file__).resolve().parents[1]
PLAN='docs/plans/bayesfilter-neutra-controlled-repair-master-2026-10-02.md'
CAMPAIGN=base.LIVE_ROOT/'docs/plans/artifacts/neutra-controlled-repair-2026-10-02/campaign-r1'


def configuration():
    cfg=RARE_CONFIGURATION()
    cfg.update(plan=PLAN,result_file='docs/plans/bayesfilter-neutra-controlled-repair-results-2026-10-02.md',
        gpu_process_seconds=43200.,cpu_core_seconds=86400.,gpu=1,gpu_workers=1,
        teacher_particles=16384,teacher_seeds=[101,211,307,401,503,601,701,809],
        oracle_per_stratum=4096,reference_count=65536,train_wall_seconds=420.,
        training_rungs=[2048,8192,32768],pilot_updates=1024,plateau_delta=.001,
        worker_wall_seconds=540.,hmc_wall_seconds=300.,attempts_per_job=3)
    cfg['resume_command']='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_controlled_repair_campaign.sh resume'
    cfg['hmc'].update(job_wall_seconds=270.,posterior_member_limit=3,broad_relative_mcse=.2)
    return cfg


def worker(spec_path):
    spec=base.read(spec_path);cfg=spec['config'];out=Path(spec['output']);gpu=spec['device']=='gpu'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(cfg['gpu']) if gpu else '-1',
        TF_NUM_INTRAOP_THREADS=str(spec['cpu_threads']),TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS=str(spec['cpu_threads']),TF_CPP_MIN_LOG_LEVEL='2',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if gpu and os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true':raise RuntimeError('memory growth required')
    resource.setrlimit(resource.RLIMIT_CPU,(spec['cpu_limit'],spec['cpu_limit']))
    start=time.monotonic();usage0=resource.getrusage(resource.RUSAGE_SELF)
    source=base.read(ROOT/'source.json')
    manifest={'command':[sys.executable,str(Path(__file__).resolve()),'worker','--spec',str(spec_path)],
        'git_commit':source['git_commit'],'source_manifest':str(ROOT/'source.json'),'plan':PLAN,
        'environment':sys.executable,'seed':spec.get('seed'),'teacher_seeds':cfg['teacher_seeds'],
        'output':str(out),'data_version':'exact WarmStartTarget signature','device':spec['device'],
        'gpu_devices_intentionally_hidden':not gpu,'jit_compile':True,'tf32':True,
        'dtype':'float64 diagnostic reference; no throughput/default promotion','started_unix':time.time(),
        'input_sha256':{}}
    import hashlib
    for field in ('prepared','training','parent','previous_control'):
        if spec.get(field):
            for p in Path(spec[field]).rglob('*'):
                if p.is_file() and p.suffix in ('.json','.tensor'):
                    manifest['input_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    base.write(out/'manifest.json',manifest);code=0
    try:
        sys.path.insert(0,str(ROOT))
        import tensorflow as tf
        if gpu:
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
            manifest['memory_policy']=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
            manifest['trust_basis']='trusted_escalated_fixed_campaign_wrapper'
        else:manifest['memory_policy']={'mode':'cpu_only_gpu_hidden'}
        tf.config.experimental.enable_tensor_float_32_execution(True)
        manifest['tensorflow_version']=tf.__version__;base.write(out/'manifest.json',manifest)
        from bayesfilter.testing import neutra_controlled_repair as phases
        action=spec['action'];target=spec['target'];seed=spec.get('seed',3011)
        if spec.get('repair') and action in ('qualify','exact_hmc'):
            cfg={**cfg,'hmc':{**cfg['hmc'],'leapfrogs':[3,5,9,13,18,25],
                'max_candidates':36,'work_units':144,'fixed_grid_max_attempts':12,'refinement_rounds':2}}
        if action=='exact_hmc' and spec.get('repair')==2:
            prior=base.read(Path(spec['previous_control'])/'result.json')
            cfg={**cfg,'hmc':{**cfg['hmc'],'posterior_member_limit':7,
                'excluded_kernel_pairs':[[r['step_size'],r['leapfrog_steps']] for r in prior['member_screen']]}}
        if action=='prepare':result=phases.prepare(target,out,cfg,spec.get('repair',0))
        elif action=='iid_reference':
            reports={}
            for name in cfg['targets']:
                reports[name]=phases.iid_assessment_control(name,out/name,cfg)
                phases.write_json(out/name/'result.json',reports[name])
            result={'reports':reports,'passed':all(r['all_inputs_finite'] and r['joint_pass_count']>0 for r in reports.values()),
                'pass_role':'limited feasibility only; no diagnostic reliability certification'}
        elif action=='recover_confirmed':result=phases.recover_confirmed_control(spec['previous_control'])
        elif action=='reference':result=phases.reference_check(target,out,cfg)
        elif action=='train':result=phases.train(target,spec['prepared'],out,cfg,seed,spec['teacher'],
            parent=spec.get('parent'),arm=spec.get('arm','parent'),repair=spec.get('repair',0),continuation=spec.get('continuation',False))
        elif action=='exact_hmc':result=phases.exact_hmc(target,spec['prepared'],out,cfg,seed,spec.get('repair',0))
        elif action=='qualify':result=phases.qualify_training(target,spec['prepared'],spec['training'],out,cfg,seed,
            confirm=spec.get('confirm',False),repair=spec.get('repair',0),frozen_filename=spec.get('frozen_filename'))
        else:raise ValueError('unsupported controlled phase')
        phases.write_json(out/'result.json',result);manifest['status']='complete'
        if gpu:manifest['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    except Exception as exc:
        traceback.print_exc();manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc));code=1
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF);wall=time.monotonic()-start
        manifest.update(wall_seconds=wall,gpu_process_seconds=wall if gpu else 0.,
            cpu_core_seconds=usage.ru_utime+usage.ru_stime-usage0.ru_utime-usage0.ru_stime,finished_unix=time.time())
        base.write(out/'manifest.json',manifest)
    return code


def resource_completion_request(qualification):
    """Complete interrupted work once; do not retry settled failed pairs."""
    result=base.read(Path(qualification)/'result.json')
    if result.get('selection_screen_passed') or result.get('qualified'):return None
    interrupted=False;excluded=[]
    for checkpoint in result.get('checkpoint_screen',[]):
        interrupted |= checkpoint.get('reason')=='qualification_budget_exhausted'
        directory=Path(checkpoint['path'])
        member_file=directory/'member-screen.json'
        if not member_file.exists():continue
        for member in base.read(member_file):
            posterior=base.read(member['posterior'])
            resource_stop='campaign_resource_cap' in posterior.get('hard_vetoes',[])
            interrupted |= resource_stop
            settled=(posterior.get('warmup_cap_hit') or posterior.get('retained_cap_hit')
                     or posterior.get('hard_vetoes'))
            if not posterior.get('passed') and settled and not resource_stop:
                excluded.append([member['step_size'],member['leapfrog_steps']])
    return {'excluded_kernel_pairs':excluded,'qualification_seconds':900.} if interrupted else None


class Controller(base.Controller):
    def worker_entry(self):return 'scripts/run_neutra_controlled_repair_master.py'

    def recover(self):
        # The shared lock prevents two coordinators. Adopt this campaign's
        # surviving worker instead of requiring another user-triggered resume.
        for name,row in list(self.state['active'].items()):
            manifest_path=Path(row['output'])/'manifest.json'
            announced=False
            while True:
                manifest=base.read(manifest_path) if manifest_path.exists() else {}
                if manifest.get('status') in ('complete','failed'):break
                pid=row.get('pid')
                try:os.kill(pid,0)
                except ProcessLookupError:break
                status=Path(f'/proc/{pid}/status')
                if status.exists() and '\nState:\tZ' in status.read_text():break
                if not announced:
                    print({'event':'adopt_existing_worker','job':name,'pid':pid},flush=True);announced=True
                if time.time()-row['started_unix']>row['wall_limit']:
                    os.killpg(pid,signal.SIGKILL);break
                time.sleep(1)
            code=0 if manifest.get('status')=='complete' and (Path(row['output'])/'result.json').exists() else 125
            self.finish(name,row,code,manifest)

    def decision(self,stage,result,next_action):
        self.state.setdefault('decisions',[]).append({'stage':stage,'result':result,'next':next_action,'unix':time.time()})
        self.sync(stage,next_action)

    def one(self,action,target,name,device='gpu',**kwargs):
        spec={'action':action,'target':target,'name':'controlled-'+name,'device':device,**kwargs}
        attempts=self.state['jobs'].get(spec['name'],[])
        if action=='exact_hmc' and attempts and attempts[-1]['status']=='failed':
            previous=Path(attempts[-1]['output']);manifest=base.read(previous/'manifest.json')
            if (manifest.get('error_type')=='FileExistsError' and
                    str(previous/'verified-member.json') in manifest.get('error','')):
                spec.update(action='recover_confirmed',device='cpu',previous_control=str(previous))
        original_cfg=self.cfg
        seconds=kwargs.get('qualification_seconds')
        if seconds is not None:
            if action!='qualify' or seconds!=900.:
                raise ValueError('only the reviewed900-second qualification completion is supported')
            self.cfg={**self.cfg,'hmc_wall_seconds':seconds+30.,'hmc':{**self.cfg['hmc'],
                'job_wall_seconds':seconds,'excluded_kernel_pairs':kwargs.get('excluded_kernel_pairs',[])}}
        try:self.batch([spec])
        finally:self.cfg=original_cfg
        path=self.done(spec['name'])
        return path,base.read(path/'result.json')

    def run(self):
        prepared={};controls={};fits={};qualifications={};confirmations={}
        p,r=self.one('iid_reference','all','iid-reference','cpu')
        if not r['passed']:raise RuntimeError('independent exact-sample assessment control needs diagnosis')
        for target in self.cfg['targets']:
            p,r=self.one('reference',target,'reference-'+target,'cpu')
            if not r['passed']:
                self.decision('control_invalid',r,'repair analytic control before dependent training')
                raise RuntimeError('exact numerical transform control failed')
            p,r=self.one('prepare',target,'prepare-'+target,'cpu')
            if not r['passed']:
                self.decision('teacher_repair',r['teacher_screen'],'double independent teacher samples')
                p,r=self.one('prepare',target,'prepare-'+target+'-repair','cpu',repair=1)
            if not r['passed']:raise RuntimeError('teacher remains inadequate; preserve oracle/teacher distinction')
            prepared[target]=str(p)
            self.decision('teacher_qualified',{'target':target,'path':str(p)},'exact Gaussian HMC control')
            p,r=self.one('exact_hmc',target,'exact-hmc-'+target,prepared=prepared[target],seed=30011)
            if not r['qualified']:
                self.decision('control_search_repair',{'target':target,'result':str(p)},'broaden exact-control kernel search')
                p,r=self.one('exact_hmc',target,'exact-hmc-'+target+'-repair',prepared=prepared[target],seed=31011,repair=1)
            if not r['qualified'] and r.get('verified_members') and r.get('member_screen'):
                self.decision('control_remaining_members',{'target':target,'result':str(p)},
                    'fresh bounded search and remaining exact pairs, unchanged posterior criteria')
                p,r=self.one('exact_hmc',target,'exact-hmc-'+target+'-remaining-members',
                    prepared=prepared[target],seed=31011,repair=2,previous_control=str(p))
            controls[target]={'path':str(p),'qualified':r['qualified']}
            if not r['qualified']:
                self.decision('downstream_control_unresolved',controls[target],'diagnose exact control before dependent learned fits')
                raise RuntimeError('known-correct downstream control did not qualify')
        self.decision('controls_complete',controls,'matched parent and objective branches')
        for target in self.cfg['targets']:
            for teacher in ('oracle','estimated'):
                for seed in self.cfg['training_seeds']:
                    group=f'{target}-{teacher}-s{seed}'
                    parent,pr=self.one('train',target,'parent-'+group,prepared=prepared[target],teacher=teacher,seed=seed)
                    candidates=[]
                    for arm in ('continue','forward','reverse','joint'):
                        p,r=self.one('train',target,arm+'-'+group,prepared=prepared[target],teacher=teacher,
                            seed=seed,parent=str(parent),arm=arm)
                        fits[group+'/'+arm]={'path':str(p),'finite':r['finite'],'stop':r['stop_reason'],'coverage':r['coverage_screen']}
                        candidates.append((arm,p,r))
                    joint=next(item for item in candidates if item[0]=='joint')
                    if not joint[2]['coverage_screen'] and joint[2]['stop_reason']=='operational_plateau':
                        self.decision('capacity_repair',{'group':group},'width64 and halved learning-rate pilot, explicit reset')
                        p,r=self.one('train',target,'joint-repair-'+group,prepared=prepared[target],teacher=teacher,
                            seed=seed,parent=str(parent),arm='joint',repair=1)
                        candidates.insert(0,('joint_repair',p,r))
                    # Order is a declared hypothesis, not loss-based superiority.
                    order={'joint_repair':0,'joint':1,'continue':2,'forward':3,'reverse':4,'parent':5}
                    candidates.append(('parent',parent,pr));candidates.sort(key=lambda x:order[x[0]])
                    selected=None;screen=[];terminal_training=joint[1];terminal_result=joint[2]
                    for arm,p,r in candidates:
                        if not r['finite']:continue
                        q,qr=self.one('qualify',target,'qualify-'+arm+'-'+group,prepared=prepared[target],training=str(p),seed=seed+40000)
                        screen.append({'arm':arm,'path':str(q),'passed':qr.get('selection_screen_passed',False)})
                        if qr.get('selection_screen_passed'):
                            selected=(arm,p,qr['selected_frozen']);break
                    if selected is None:
                        # A targeted wider search is distinct from retraining.
                        arm,p,r=next((c for c in candidates if c[0]=='joint'),candidates[0])
                        q,qr=self.one('qualify',target,'qualify-repair-'+group,prepared=prepared[target],training=str(p),seed=seed+50000,repair=1)
                        screen.append({'arm':arm,'path':str(q),'passed':qr.get('selection_screen_passed',False),'repair':True})
                        if qr.get('selection_screen_passed'):selected=(arm,p,qr['selected_frozen'])
                    if selected is None and joint[2]['stop_reason']=='budget_limited':
                        self.decision('joint_continuation',{'group':group,'parent':str(joint[1])},
                            'preserve joint Adam and teacher; continue incomplete optimization before geometric intervention')
                        p,r=self.one('train',target,'joint-continuation-'+group,prepared=prepared[target],teacher=teacher,
                            seed=seed,parent=str(joint[1]),arm='joint',continuation=True)
                        fits[group+'/joint_continuation']={'path':str(p),'finite':r['finite'],'stop':r['stop_reason'],'coverage':r['coverage_screen']}
                        terminal_training,terminal_result=p,r
                        if r['finite']:
                            q,qr=self.one('qualify',target,'qualify-joint-continuation-'+group,prepared=prepared[target],training=str(p),seed=seed+55000,repair=1)
                            screen.append({'arm':'joint_continuation','path':str(q),'passed':qr.get('selection_screen_passed',False)})
                            if qr.get('selection_screen_passed'):selected=('joint_continuation',p,qr['selected_frozen'])
                    if selected is None and terminal_result['finite']:
                        self.decision('terminal_checkpoint_allocation',{'group':group,'training':str(terminal_training)},
                            'one full270-second qualification for the final trained joint checkpoint after preserved-map screening')
                        q,qr=self.one('qualify',target,'qualify-terminal-allocation-'+group,prepared=prepared[target],
                            training=str(terminal_training),seed=seed+58000,repair=1,frozen_filename=terminal_result['selected_frozen'])
                        screen.append({'arm':'joint_terminal_allocation','path':str(q),'passed':qr.get('selection_screen_passed',False)})
                        if qr.get('selection_screen_passed'):selected=('joint_terminal_allocation',terminal_training,qr['selected_frozen'])
                        completion=resource_completion_request(q)
                        if selected is None and completion is not None:
                            self.decision('qualification_resource_completion',{'group':group,'previous':str(q),**completion},
                                'one longer final-map check; completed failed kernel pairs excluded')
                            q,qr=self.one('qualify',target,'qualify-resource-completion-'+group,prepared=prepared[target],
                                training=str(terminal_training),seed=seed+59000,repair=1,
                                frozen_filename=terminal_result['selected_frozen'],**completion)
                            screen.append({'arm':'joint_resource_completion','path':str(q),
                                'passed':qr.get('selection_screen_passed',False),'completion':completion})
                            if qr.get('selection_screen_passed'):selected=('joint_resource_completion',terminal_training,qr['selected_frozen'])
                    qualifications[group]=screen
                    if selected:
                        arm,p,frozen_filename=selected
                        confirmation_options={'qualification_seconds':900.} if arm=='joint_resource_completion' else {}
                        q,qr=self.one('qualify',target,'confirmation-'+group,prepared=prepared[target],training=str(p),seed=seed+60000,
                            confirm=True,frozen_filename=frozen_filename,**confirmation_options)
                        confirmations[group]={'arm':arm,'path':str(q),'qualified':qr['qualified']}
                    else:confirmations[group]={'qualified':False,'reason':'no learned map qualified within the declared work',
                        'geometric_extension':'not triggered unless adequate stationary density fitting is demonstrated; inconclusive optimization remains a competing cause'}
                    self.decision('case_complete',{'group':group,'confirmation':confirmations[group]},'next independent case')
                    base.write(self.root/'summary.json',{'controls':controls,'fits':fits,'qualifications':qualifications,
                        'confirmations':confirmations,'remaining':self.remaining(),'ranking_established':False})
        self.sync('completed','terminal result review, geometry phase decision and monograph update')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','resume','status','check','worker'));p.add_argument('--spec')
    a=p.parse_args()
    if a.action=='worker':return worker(a.spec)
    if a.action=='status':
        import json
        print(json.dumps(base.read(CAMPAIGN/'next-phase.json') if (CAMPAIGN/'next-phase.json').exists() else {'status':'not_started'},indent=2));return 0
    if a.action in ('run','resume') and (CAMPAIGN/'state.json').exists():
        state=base.read(CAMPAIGN/'state.json')
        if state.get('status') in ('terminal_review_complete','completed'):
            import json as json_module
            print(json_module.dumps({'status':state['status'],'next_action':state.get('next_action'),
                                     'message':'campaign is terminal; start a new versioned campaign for new evidence'},indent=2))
            return 0
    if a.action=='check':
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',TF_CPP_MIN_LOG_LEVEL='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
        return subprocess.call([sys.executable,'-m','pytest','-q','tests/test_neutra_controlled_repair.py'],cwd=base.LIVE_ROOT,env=env)
    base.PLAN=PLAN;base.CAMPAIGN=CAMPAIGN;base.config=configuration
    with (base.SHARED/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller()
        try:controller.run()
        except Exception as exc:
            status='budget_exhausted' if str(exc)=='insufficient bounded worker budget' else 'repair_required'
            controller.sync(status,str(exc));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
