#!/usr/bin/env python3
"""Follow measured training progress and exercise the Gabrié restoration lane."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('causal_master',ROOT/'scripts/run_neutra_causal_repair_master.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,master=base.read,base.master


class Followup(base.Controller):
    resume_command='/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/continue_neutra_causal_repair.py'

    @staticmethod
    def teacher_for(parent):
        inputs=read(parent.parent/'manifest.json')['input_sha256']
        particles=[Path(p) for p in inputs if Path(p).name=='teacher-particles.tensor']
        weights=[Path(p) for p in inputs if Path(p).name=='teacher-log-weights.tensor']
        if len(particles)!=1 or len(weights)!=1 or particles[0].parent!=weights[0].parent:
            raise ValueError(f'Continuation teacher identity is missing or ambiguous: {parent}')
        return particles[0].parent

    def pending_continuations(self):
        outcomes=self.record['outcomes']
        return [(name,dict(row)) for name,row in outcomes.items()
                if row['status']=='improving_fit_at_cap' and 'evidence' in row
                and name+'-continue' not in outcomes and name+'-progress' not in outcomes]

    def price(self,parent,next_total):
        history=read(parent/'history.json');last=history[-1]
        previous=last.get('paired_progress',{}).get('previous_updates',
                    history[-2]['updates'] if len(history)>1 else 0)
        completed=max(1,last['updates']-previous)
        manifest=read(parent.parent/'manifest.json')
        # Twofold headroom and 30s setup allowance are engineering hypotheses;
        # preserve the estimate for checking against actual recorded worker cost.
        seconds=2*float(manifest['wall_seconds'])*(next_total-last['updates'])/completed+30
        remaining=master.remaining(self.state,self.base)
        enough=seconds<remaining['gpu_process_seconds'] and seconds*self.base['cpu_threads']<remaining['cpu_core_seconds']
        return seconds,enough

    def finish_map(self,name,target,seed,warm,prepared,arm):
        if arm=='gabrie':
            for level in range(2):
                sampled,screen=self.phase(name+f'-sampler-v{level}','sampler',target,seed,
                    prepared=prepared,training=warm,arm=arm,level=level)
                if screen['passed']:break
            if not screen['passed']:
                self.record['outcomes'][name]={'status':'frozen_sampler_screen_failed',
                    'evidence':str(sampled),'target':target,'seed':seed,'arm':arm}
                self.refresh('candidate_resolved',active_job=None);return
        refined,_=self.phase(name+'-refine','refine',target,seed,prepared=prepared,training=warm,arm=arm)
        self.qualify(name,target,seed,refined,prepared,arm)

    def extend(self,name,target,seed,teacher,prepared,parent,arm='smc'):
        while True:
            total=max(int(p.name.split('-')[1]) for p in parent.glob('warm-*-checkpoint.json'))
            cap=2*total
            estimate,enough=self.price(parent,cap)
            if not enough:
                self.record['outcomes'][name]={'status':'priced_continuation_exceeds_remaining_budget',
                    'parent':str(parent),'proposed_updates':cap,'predicted_gpu_seconds':estimate,
                    'target':target,'seed':seed,'arm':arm}
                self.refresh('continuation_budget_limited',active_job=None);return
            self.base['job_wall_seconds']['closure_fit']=max(1800.,estimate)
            self.record.setdefault('continuation_prices',[]).append({'name':name,'parent':str(parent),
                'next_total_updates':cap,'predicted_gpu_seconds':estimate})
            path,result=self.phase(name+f'-n{cap}','fit',target,seed,prepared=prepared,training=teacher,
                parent=parent,arm=arm,closure={'fit_rungs':[cap]})
            if result['passed']:
                self.finish_map(name,target,seed,path,prepared,arm);return
            self.record['outcomes'][name]={'status':result['reason'],'evidence':str(path),'target':target,'seed':seed,'arm':arm}
            choices=result.get('continuation_candidates',[])
            if not choices:
                self.refresh('continuation_resolved',active_job=None);return
            parent=Path(choices[0])

    def nominated_gabrie(self,seed,target='mixture'):
        rows=[]
        for p in (self.root/'attempts').glob(f'closure-fit-{target}-gabrie-s{seed}-v*-r*/w*/history.json'):
            history=read(p);a=history[-1]['assessment']
            if not a['finite']:continue
            width,lr=p.parent.name[1:].split('-lr')
            rows.append((not a['nonlinear_learning_passed'],int(width),float(lr),str(p.parent),p.parent))
        return sorted(rows)[0] if rows else None

    def remaining_audit_cases(self):
        master.recover_active(self.root,self.state,self.base)
        self.calibrate()
        # Complete the declared 4 -> 16 -> 64 physical mutation control.
        # Transported populations may never resample and cannot test this work.
        self.phase('funnel-mobility-observed-s11-v2','teacher','funnel',11,
            prepared=self.prepare('funnel'),level=2)
        # Enumerate the historical case ledger, including non-mixture fits,
        # rather than relying on a hand-picked target-name checklist.
        for row in self.historical['outcomes']:
            if (row['status']!='nonlinear_fit_repair_exhausted' or
                    row['arm']!='gabrie' or row['target']=='mixture'):
                continue
            target,seed=row['target'],row['seed']
            nominee=self.nominated_gabrie(seed,target)
            if nominee is None:raise RuntimeError(f'Missing audited fit parent: {target}/{seed}')
            plateau,_,_,_,parent=nominee
            if plateau:raise RuntimeError(f'Unplanned non-mixture plateau: {target}/{seed}')
            prepared=self.prepare(target)
            self.extend(f'{target}-gabrie-continue-s{seed}',target,seed,prepared,prepared,parent,arm='gabrie')
        self.refresh('all_audited_cases_attempted',active_job=None)

    def run(self):
        master.recover_active(self.root,self.state,self.base)
        self.calibrate()
        # Snapshot the unresolved set; later successful stages must not erase
        # their historical parent outcome or make this traversal recursive.
        for name,row in self.pending_continuations():
            report=read(Path(row['evidence'])/'phase.json')
            parents=report.get('continuation_candidates',[])
            if not parents:continue
            seed=row['seed'];target=row['target'];parent=Path(parents[0]);arm=row.get('arm','smc')
            prepared=self.prepare(target)
            teacher=prepared if arm=='gabrie' else self.teacher_for(parent)
            self.extend(name+'-progress',target,seed,teacher,prepared,parent,arm=arm)

        for seed in (11,23,37):
            parent=self.mixture_parent(seed,plateau=True)
            if parent is None:continue
            teacher=self.teacher_for(parent)
            name=f'mixture-reset-only-s{seed}'
            warm,result=self.phase(name,'fit','mixture',seed,prepared=self.prepare('mixture'),training=teacher,
                parent=parent,closure={'fit_rungs':[4096,8192],'reset_forward_optimizer':True})
            self.record['outcomes'][name]={'status':result['reason'],'evidence':str(warm),'target':'mixture','seed':seed}
            if result['passed']:self.finish_map(name,'mixture',seed,warm,self.prepare('mixture'),'smc')
            elif result.get('continuation_candidates'):
                self.extend(name+'-progress','mixture',seed,teacher,self.prepare('mixture'),Path(result['continuation_candidates'][0]))

        for seed in (11,23,37):
            nominee=self.nominated_gabrie(seed)
            if nominee is None:continue
            plateau,_,_,_,parent=nominee;prepared=self.prepare('mixture')
            if not plateau:
                self.extend(f'mixture-gabrie-continue-s{seed}','mixture',seed,prepared,prepared,parent,arm='gabrie')
                continue
            for reset,escape in ((False,0),(True,0),(True,256)):
                name=f'mixture-gabrie-s{seed}-reset{int(reset)}-escape{escape}'
                warm,result=self.phase(name,'fit','mixture',seed,prepared=prepared,training=prepared,
                    parent=parent,arm='gabrie',closure={'fit_rungs':[4096,8192],
                        'reset_forward_optimizer':reset,'escape_rkl_updates':escape})
                self.record['outcomes'][name]={'status':result['reason'],'evidence':str(warm),'target':'mixture','seed':seed,'arm':'gabrie'}
                if result['passed']:self.finish_map(name,'mixture',seed,warm,prepared,'gabrie')
                elif result.get('continuation_candidates'):
                    self.extend(name+'-progress','mixture',seed,prepared,prepared,Path(result['continuation_candidates'][0]),arm='gabrie')
        for level in (0,1):
            self.phase(f'funnel-mobility-observed-s11-v{level}','teacher','funnel',11,
                prepared=self.prepare('funnel'),level=level)
        self.refresh('followup_repairs_resolved',active_job=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remaining-audit-cases',action='store_true')
    args=parser.parse_args()
    with (base.CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Followup()
        try:
            if args.remaining_audit_cases:
                controller.resume_command+=' --remaining-audit-cases'
                controller.remaining_audit_cases()
            else:controller.run()
        except BaseException as exc:
            controller.refresh('stopped_with_evidence',error=str(exc));raise


if __name__=='__main__':main()
