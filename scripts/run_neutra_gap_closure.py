#!/usr/bin/env python3
"""Finite diagnosis and repair of the three unresolved simpler-model cases."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
PLAN='docs/plans/bayesfilter-neutra-gap-closure-plan-2026-10-01.md'


def read(path):return json.loads(Path(path).read_text())


def load_master(root):
    spec=importlib.util.spec_from_file_location('gap_worker_master',root/'scripts/run_neutra_warm_start_master.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def freeze_sources(destination, master):
    manifest=destination/'frozen-source.json'
    if not manifest.exists():
        destination.mkdir(parents=True,exist_ok=False)
        paths=set(master.SOURCES)
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'bayesfilter').rglob('*.py'))
        paths.update((PLAN,'docs/reference/hmc-tuning-interface.md'))
        hashes={}
        for name in sorted(paths):
            source=ROOT/name;saved=destination/name;saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,saved);hashes[name]=hashlib.sha256(saved.read_bytes()).hexdigest()
        master.write(manifest,{'git_commit':master.git_head(),'created_unix':time.time(),
            'source_sha256':hashes,'purpose':'executed Python snapshot; dirty workspace preserved'})
    receipt=read(manifest)
    for name,digest in receipt['source_sha256'].items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen execution source changed: {name}')
    return receipt


class BudgetStop(RuntimeError):pass


class Controller:
    def __init__(self):
        self.root=CAMPAIGN;self.output=self.root/'gap-closure-20261001-r1'
        self.output.mkdir(exist_ok=True)
        self.master=load_master(ROOT)
        self.state=read(self.root/'state.json');self.original=read(self.root/'config.json')
        if self.state.get('active_job'):raise RuntimeError('Shared campaign has an active or unreconciled worker')
        self.record=read(self.output/'state.json') if (self.output/'state.json').exists() else {
            'plan':PLAN,'started_unix':time.time(),'jobs':{},'outcomes':{},
            'budget_at_start':self.master.remaining(self.state,self.original),
            'subcap':{'gpu_process_seconds':21600.,'cpu_core_seconds':43200.},
            'numerical_job_cap':24,'infrastructure_retry_cap':2,'infrastructure_retries':0}
        self.snapshot=self.output/'source'
        freeze_sources(self.snapshot,self.master)
        self.master=load_master(self.snapshot)
        self.base=copy.deepcopy(self.original)
        self.base['closure'].update(plan=PLAN)
        for name in ('teacher_transport','escape_rkl_updates','fit_rungs','reset_forward_optimizer'):
            self.base['closure'].pop(name,None)
        self.base['hmc'].update(posterior_member_limit=3,measurement_num_results=64,verification_num_results=64)
        self.base['job_wall_seconds'].update(closure_fit=1800.,closure_qualify=1200.,closure_diagnose=1200.)
        self.refresh('ready')

    def usage(self):
        remaining=self.master.remaining(self.state,self.original)
        return {k:self.record['budget_at_start'][k]-remaining[k] for k in self.record['subcap']}

    def refresh(self,status,**details):
        self.record.update(status=status,updated_unix=time.time(),used=self.usage(),
            remaining=self.master.remaining(self.state,self.original),**details)
        self.master.write(self.output/'state.json',self.record)
        self.master.write(self.root/'repair-next-phase.json',{
            'plan':PLAN,'status':status,'evidence':str(self.output/'state.json'),
            'active_job':self.record.get('active_job'),'remaining':self.record['remaining'],
            'next_phase':self.record.get('next_phase'),
            'resume_command':f'{sys.executable} scripts/run_neutra_gap_closure.py campaign'})

    def phase(self,name,phase,target,seed,*,prepared=None,training=None,parent=None,closure=None):
        if name in self.record['jobs']:
            row=self.record['jobs'][name];return Path(row['output']),row['result']
        used=self.usage()
        if any(used[k]>=v for k,v in self.record['subcap'].items()):raise BudgetStop('repair sub-cap exhausted')
        count=sum(row['phase']!='reference' for row in self.record['jobs'].values())
        if phase!='reference' and count>=self.record['numerical_job_cap']:raise BudgetStop('numerical phase cap exhausted')
        freeze_sources(self.snapshot,self.master)
        cfg=copy.deepcopy(self.base);cfg['closure'].update(closure or {})
        if phase=='qualify' and closure and closure.get('larger_hmc_evidence'):
            cfg['hmc'].update(measurement_num_results=256,verification_num_results=256,evidence_rungs=[1,2,4])
        # Enforce the cycle sub-cap at the existing worker resource boundary.
        for key,cap in self.record['subcap'].items():
            already=self.original[key]-self.record['budget_at_start'][key]
            cfg[key]=min(self.original[key],already+cap)
        extra=['--target',target,'--seed',str(seed),'--arm','gabrie']
        for option,path in (('--prepared',prepared),('--training',training),('--warm-parent',parent)):
            if path is not None:extra += [option,str(path)]
        job='gap-20261001-'+name
        while True:
            self.refresh('executing',active_job=job,next_phase=phase)
            self.master.write(self.root/'config.json',cfg)
            try:
                self.master.run_one(self.root,self.state,cfg,job,'closure_'+phase,extra)
            finally:
                self.master.write(self.root/'config.json',self.original)
            path=self.master.completed_path(self.root,self.state,job)
            if path and (path/'phase.json').exists():break
            last=next((r for r in reversed(self.state['attempts']) if r['job']==job),{})
            if last.get('status')=='budget_limited' and self.record['infrastructure_retries']<2:
                self.record['infrastructure_retries']+=1
                cfg['job_wall_seconds']['closure_'+phase]*=2
                self.refresh('timeout_repair',active_job=None,last_attempt=last)
                if any(self.usage()[k]>=cap for k,cap in self.record['subcap'].items()):raise BudgetStop('worker exhausted sub-cap')
                continue
            raise RuntimeError(f'Worker missing phase evidence: {job}, {last.get("status")}')
        result=read(path/'phase.json')
        self.record['jobs'][name]={'phase':phase,'output':str(path),'result':result}
        self.refresh('phase_complete',active_job=None,last_result=result)
        if result.get('continuation_veto') and result.get('failure_class')!='candidate_numerical':
            raise RuntimeError(f'Shared validity veto: {path}')
        return path,result

    def prepared(self,target):return self.root/'attempts'/f'closure-prepare-{target}-v0-r1'

    def parent(self,seed):
        return self.root/'attempts'/f'causal-20261001-mixture-gabrie-s{seed}-reset0-escape0-r1/w8-lr0.001'

    def diagnose(self):
        for seed in (11,37):
            for level,(burn,steps) in enumerate(((256,1024),(1024,4096))):
                path,result=self.phase(f'measure-s{seed}-v{level}','diagnose','mixture',seed,
                    prepared=self.prepared('mixture'),parent=self.parent(seed),
                    closure={'diagnostic_burn':burn,'diagnostic_steps':steps})
                if result['passed']:break
            self.record.setdefault('measure',{})[str(seed)]={'passed':result['passed'],'evidence':str(path)}
        self.refresh('diagnostics_complete',active_job=None,next_phase='continuation and matched initialization if measure passes')

    def qualify(self,name,target,seed,candidate):
        for level in (0,1):
            # Fresh reference identity is fixed before selection, opened only
            # once after a precision-qualified member has been selected.
            final_seed=int(hashlib.sha256(f'gap-v1:{name}:{seed}:{level}'.encode()).hexdigest()[:7],16)
            reference,_=self.phase(name+f'-reference-v{level}','reference',target,final_seed,
                                  prepared=self.prepared(target))
            path,result=self.phase(name+f'-qualify-v{level}','qualify',target,final_seed,
                prepared=reference,training=candidate,closure={'larger_hmc_evidence':bool(level)})
            payload=read(path/'qualification.json') if (path/'qualification.json').exists() else {}
            if result['passed'] or payload.get('verified_members') or payload.get('heldout_consumed'):break
        self.record['outcomes'][name]={'target':target,'seed':seed,'passed':result['passed'],
            'status':result['reason'],'evidence':str(path)}
        self.refresh('candidate_assessed',active_job=None)
        return result['passed']

    def finish_map(self,name,seed,warm):
        path,sampling=self.phase(name+'-sampler','sampler','mixture',seed,
            prepared=self.prepared('mixture'),training=warm)
        if not sampling['passed']:
            self.record['outcomes'][name]={'passed':False,'status':'frozen_sampler_failed','evidence':str(path)}
            self.refresh('candidate_assessed',active_job=None);return False
        refined,result=self.phase(name+'-refine','refine','mixture',seed,
            prepared=self.prepared('mixture'),training=warm)
        if not result['passed']:
            self.record['outcomes'][name]={'passed':False,'status':result['reason'],'evidence':str(refined)}
            self.refresh('candidate_assessed',active_job=None);return False
        return self.qualify(name,'mixture',seed,refined)

    def campaign(self):
        self.diagnose()
        viable={};resolved=set()
        for seed in (11,37):
            if not self.record['measure'][str(seed)]['passed']:continue
            name=f'control-s{seed}'
            path,result=self.phase(name,'fit','mixture',seed,prepared=self.prepared('mixture'),
                training=self.prepared('mixture'),parent=self.parent(seed),
                closure={'fit_rungs':[16384,32768],'plateau_min_updates':32768})
            if result['passed']:viable[seed]=(name,path)
            self.record.setdefault('training',{})[name]={'evidence':str(path),'passed':result['passed']}
        for seed,(name,path) in viable.items():
            if self.finish_map(name,seed,path):resolved.add(seed)
        # Matched arms run for both seeds even if one passes, until both have
        # viable fits. Fixed arm order is an operational choice, no ranking.
        for width,variance in ((8,.02),(8,.2),(8,1.),(16,.2)):
            if len(viable)==2:break
            for seed in (11,37):
                if not self.record['measure'][str(seed)]['passed']:continue
                name=f'init-w{width}-v{variance:g}-s{seed}'
                original_widths=self.base['widths_2d'];original_rates=self.base['learning_rates']
                self.base['widths_2d']=[width];self.base['learning_rates']=[.001]
                try:
                    path,result=self.phase(name,'fit','mixture',seed,prepared=self.prepared('mixture'),
                        training=self.prepared('mixture'),closure={'iaf_variance_scale':variance,
                        'fit_rungs':[4096,16384,65536],'plateau_min_updates':16384})
                finally:self.base['widths_2d']=original_widths;self.base['learning_rates']=original_rates
                self.record.setdefault('training',{})[name]={'evidence':str(path),'passed':result['passed']}
                if result['passed'] and seed not in viable:viable[seed]=(name,path)
            for seed,(name,path) in viable.items():
                if seed not in resolved and name not in self.record['outcomes']:
                    if self.finish_map(name,seed,path):resolved.add(seed)
        for seed in (11,37):
            if seed not in viable:
                self.record['outcomes'][f'mixture-s{seed}']={'target':'mixture','seed':seed,'passed':False,
                    'status':'finite_training_protocol_exhausted' if self.record['measure'][str(seed)]['passed'] else 'finite_measure_unresolved'}
        wiggle=self.root/'attempts/closure-refine-wiggle-gabrie-s23-r1'
        self.qualify('wiggle-gabrie-s23','wiggle',23,wiggle)
        self.refresh('bounded_protocol_complete',active_job=None,
            next_phase='terminal code/math/result review; unresolved cases remain explicit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('diagnose','campaign','status'))
    args=parser.parse_args()
    if args.stage=='status':
        print(json.dumps(read(CAMPAIGN/'gap-closure-20261001-r1/state.json'),indent=2));return
    with (CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller()
        try:getattr(controller,args.stage)()
        except BudgetStop as exc:
            controller.refresh('bounded_budget_stop',active_job=None,error=str(exc))
        except BaseException as exc:
            controller.refresh('stopped_with_evidence',error=str(exc));raise


if __name__=='__main__':main()
