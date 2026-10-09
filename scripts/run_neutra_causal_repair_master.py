#!/usr/bin/env python3
"""Execute audited repairs through the existing workers and shared budget."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
PLAN='docs/plans/bayesfilter-neutra-causal-repair-plan-2026-10-01.md'
spec=importlib.util.spec_from_file_location('warm_master',ROOT/'scripts/run_neutra_warm_start_master.py')
master=importlib.util.module_from_spec(spec);spec.loader.exec_module(master)


def read(path):return json.loads(Path(path).read_text())


class Controller:
    resume_command='/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_causal_repair_master.py campaign'

    def __init__(self,root=CAMPAIGN):
        self.root=Path(root)
        self.output=self.root/'causal-repair-20261001-r1';self.output.mkdir(exist_ok=True)
        self.state=read(self.root/'state.json')
        self.base=read(self.root/'config.json')
        self.base['closure'].update(plan=PLAN,causal_repair=True)
        self.base['closure'].pop('teacher_transport',None)
        self.base['closure'].pop('escape_rkl_updates',None)
        self.base['closure'].pop('fit_rungs',None)
        self.base['closure'].pop('reset_forward_optimizer',None)
        self.base['hmc'].update(fixed_grid_max_attempts=8,retry_fixed_grid_max_attempts=12,
            max_candidates=36,work_units=144,job_wall_seconds=1100.,
            evidence_rungs=[1,2,4],refinement_rounds=2,explore_failed_intervals=True,
            temporal_conflict_method='paired_chain_block_contrasts_v1')
        self.base['job_wall_seconds'].update(closure_temporal=600.,closure_fit=1800.,closure_qualify=1200.)
        self.record=read(self.output/'state.json') if (self.output/'state.json').exists() else {
            'plan':PLAN,'started_unix':time.time(),'jobs':{},'outcomes':{},
            'budget_at_start':master.remaining(self.state,self.base)}
        old=self.output/'historical-repair-result.json'
        if not old.exists():master.write(old,read(self.root/'repair-master-result.json'))
        self.historical=read(old)

    def refresh(self,status,**details):
        if 'error' in self.record and 'error' not in details:
            self.record.setdefault('incidents',[]).append({
                'error':self.record.pop('error'),'cleared_on_resume_unix':time.time()})
        self.record.update(status=status,updated_unix=time.time(),remaining=master.remaining(self.state,self.base),**details)
        master.write(self.output/'state.json',self.record)
        master.write(self.root/'repair-next-phase.json',{'plan':PLAN,'status':status,**details,
            'remaining':self.record['remaining'],'evidence':str(self.output/'state.json'),
            'resume_command':self.resume_command})

    def phase(self,name,phase,target,seed,*,prepared=None,training=None,parent=None,arm='smc',level=0,closure=None):
        cfg=copy.deepcopy(self.base);cfg['closure'].update(closure or {})
        # Calibration must succeed before the optional diagnostic can govern HMC.
        if phase=='qualify' and not self.record.get('temporal_calibration_passed'):
            raise RuntimeError('temporal calibration required before repaired qualification')
        extra=['--target',target,'--seed',str(seed),'--arm',arm,'--repair-level',str(level)]
        for option,path in (('--prepared',prepared),('--training',training),('--warm-parent',parent)):
            if path is not None:extra += [option,str(path)]
        job='causal-20261001-'+name
        self.refresh('executing',active_job=job,phase=phase)
        for attempt in range(2):
            master.write(self.root/'config.json',cfg)
            okay=master.run_one(self.root,self.state,cfg,job,'closure_'+phase,extra)
            master.write(self.root/'state.json',self.state)
            output=master.completed_path(self.root,self.state,job)
            if output and (output/'phase.json').exists():
                result=read(output/'phase.json')
                self.record['jobs'][name]={'output':str(output),'result':result}
                self.refresh('phase_complete',active_job=None,phase=phase,last_result=result,last_output=str(output))
                if result.get('continuation_veto') and result.get('failure_class')!='candidate_numerical':
                    raise RuntimeError(f'Numerical continuation veto: {output}')
                return output,result
            latest=next((row for row in reversed(self.state['attempts']) if row['job']==job),{})
            if latest.get('status')=='budget_limited' and attempt==0:
                cfg['job_wall_seconds']['closure_'+phase]*=2
                self.refresh('local_timeout_repair',active_job=None,evidence=latest.get('output'))
                continue
            raise RuntimeError(f'Worker did not produce required phase evidence: {job}; {latest.get("status")}; okay={okay}')

    def old(self,pattern,*,passed=True):
        paths=sorted((self.root/'attempts').glob(pattern))
        selected=[p for p in paths if (p/'phase.json').exists() and (not passed or read(p/'phase.json').get('passed'))]
        if not selected:raise RuntimeError(f'Missing preserved input: {pattern}')
        return selected[-1]

    def prepare(self,target):return self.old(f'closure-prepare-{target}-v0-r*')

    def qualify(self,name,target,seed,candidate,prepared,arm):
        selection=read(candidate/'phase.json')
        if not selection.get('passed'):
            self.record['outcomes'][name]={'status':selection.get('reason','upstream_candidate_invalid'),
                'evidence':str(candidate),'target':target,'arm':arm,'seed':seed}
            self.refresh('candidate_resolved',active_job=None)
            return False
        for level in range(2):
            transport=read(candidate/'selected-frozen.json')['transport_hash']
            stream=f'causal-repair-v1:{name}:{seed}:{level}:{transport}'
            final_seed=int(hashlib.sha256(stream.encode()).hexdigest()[:7],16)
            reference,_=self.phase(name+f'-reference-v{level}','reference',target,final_seed,
                prepared=prepared,training=candidate,arm=arm)
            path,result=self.phase(name+f'-qualify-v{level}','qualify',target,final_seed,
                prepared=reference,training=candidate,arm=arm,level=level)
            if result['passed']:break
        self.record['outcomes'][name]={'status':'posterior_screen_passed' if result['passed'] else 'bounded_qualification_failed',
            'evidence':str(path),'target':target,'arm':arm,'seed':seed}
        self.refresh('candidate_resolved',active_job=None)
        return result['passed']

    def calibrate(self):
        path,result=self.phase('temporal','temporal','gaussian',11)
        self.record['temporal_calibration_passed']=result['passed']
        self.refresh('temporal_assessed',active_job=None)
        if not result['passed']:raise RuntimeError(f'Temporal calibration failed: {path}')

    def existing_hmc(self,*,canary=False):
        rows=[row for row in self.historical['outcomes'] if row['status']=='fresh_final_check_failed']
        rows.sort(key=lambda row:(row['target']!='funnel',row['seed'],row['arm']))
        if canary:rows=[row for row in rows if row['seed']==11 and (row['target']=='funnel' or (row['target']=='wiggle' and row['arm']=='smc'))]
        for row in rows:
            target,arm,seed=row['target'],row['arm'],row['seed']
            candidate=self.old(f'closure-refine-{target}-{arm}-s{seed}-r*')
            self.qualify(f'preserved-{target}-{arm}-s{seed}',target,seed,candidate,self.prepare(target),arm)

    def mixture_parent(self,seed,*,plateau=False):
        candidates=[]
        for parent in (self.root/'attempts').glob(f'closure-fit-mixture-smc-s{seed}-v*-r*/w*-lr*'):
            if not (parent/'history.json').exists():continue
            history=read(parent/'history.json')
            if not history or not history[-1].get('assessment',{}).get('finite'):continue
            learned=history[-1]['assessment']['nonlinear_learning_passed']
            if learned==plateau:continue
            if not plateau and (len(history)<2 or history[-1]['assessment']['heldout_forward_kl']>=history[-2]['assessment']['heldout_forward_kl']):continue
            width,lr=parent.name[1:].split('-lr')
            candidates.append((int(width),float(lr),str(parent),parent))
        return sorted(candidates)[0][-1] if candidates else None

    def continue_fit(self,name,target,seed,teacher,prepared,parent,*,escape=0,qualify=True):
        for cap in (16384,32768,65536):
            path,result=self.phase(name+f'-n{cap}','fit',target,seed,prepared=prepared,training=teacher,
                parent=parent,closure={'fit_rungs':[cap],'escape_rkl_updates':escape})
            if result['passed']:
                refined,_=self.phase(name+'-refine','refine',target,seed,prepared=prepared,training=path)
                if qualify:self.qualify(name,target,seed,refined,prepared,'smc')
                return path
            self.record['outcomes'][name]={'status':result['reason'],'evidence':str(path),'target':target,'seed':seed}
            parents=result.get('continuation_candidates',[])
            if not parents:break
            parent=Path(parents[0]);escape=0
        self.refresh('training_repair_resolved',active_job=None)
        return None

    def mixtures(self,*,canary=False):
        for seed in ((37,) if canary else (11,23,37)):
            parent=self.mixture_parent(seed)
            if parent:
                self.continue_fit(f'mixture-continue-s{seed}','mixture',seed,
                    self.old(f'closure-teacher-mixture-s{seed}-v*-r*'),self.prepare('mixture'),parent)
        # A matched finite intervention on plateaus is separate from continuation.
        for seed in ((11,) if canary else (11,23,37)):
            parent=self.mixture_parent(seed,plateau=True)
            if parent is None:continue
            teacher=self.old(f'closure-teacher-mixture-s{seed}-v*-r*')
            for escape in (0,256):
                name=f'mixture-plateau-s{seed}-escape{escape}'
                path,result=self.phase(name,'fit','mixture',seed,prepared=self.prepare('mixture'),
                    training=teacher,parent=parent,closure={'fit_rungs':[4096,8192],'escape_rkl_updates':escape})
                self.record['outcomes'][name]={'status':result['reason'],'evidence':str(path),'target':'mixture','seed':seed}
                if result['passed']:
                    refined,_=self.phase(name+'-refine','refine','mixture',seed,prepared=self.prepare('mixture'),training=path)
                    self.qualify(name,'mixture',seed,refined,self.prepare('mixture'),'smc')
                elif result.get('continuation_candidates'):
                    self.continue_fit(name+'-continue','mixture',seed,teacher,self.prepare('mixture'),Path(result['continuation_candidates'][0]))

    def teachers(self,*,canary=False):
        for seed in ((11,) if canary else (11,23,37)):
            prepared=self.prepare('funnel')
            # Controlled mutation-only comparison retains the physical proposal.
            if seed==11:
                for level in (0,1):self.phase(f'funnel-physical-s{seed}-v{level}','teacher','funnel',seed,prepared=prepared,level=level)
            transport=self.old(f'closure-refine-funnel-gabrie-s{seed}-r*')/'selected-frozen.json'
            for level in range(5):
                path,result=self.phase(f'funnel-transported-s{seed}-v{level}','teacher','funnel',seed,
                    prepared=prepared,level=level,closure={'teacher_transport':str(transport)})
                if result['passed']:break
            name=f'funnel-smc-teacher-s{seed}'
            self.record['outcomes'][name]={'status':result['reason'],'evidence':str(path),'target':'funnel','seed':seed}
            if result['passed']:
                warm,fit=self.phase(f'funnel-rescue-fit-s{seed}','fit','funnel',seed,prepared=prepared,training=path)
                if fit['passed']:
                    refined,_=self.phase(f'funnel-rescue-refine-s{seed}','refine','funnel',seed,prepared=prepared,training=warm)
                    self.qualify(f'funnel-smc-rescue-s{seed}','funnel',seed,refined,prepared,'smc')
                elif fit.get('continuation_candidates'):
                    self.continue_fit(f'funnel-smc-rescue-s{seed}','funnel',seed,path,prepared,Path(fit['continuation_candidates'][0]))
                else:self.record['outcomes'][f'funnel-smc-rescue-s{seed}']={'status':fit['reason'],'evidence':str(warm)}
            self.refresh('teacher_repair_resolved',active_job=None)

    def run(self,stage):
        master.recover_active(self.root,self.state,self.base)
        self.calibrate()
        if stage=='calibration':return
        canary=stage=='canaries'
        self.existing_hmc(canary=canary)
        self.mixtures(canary=canary)
        self.teachers(canary=canary)
        self.refresh('canaries_complete' if canary else 'declared_repairs_resolved',active_job=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('calibration','canaries','campaign','status'))
    args=parser.parse_args()
    if args.stage=='status':
        path=CAMPAIGN/'causal-repair-20261001-r1/state.json'
        result=read(path) if path.exists() else {'status':'not_started'}
        print(json.dumps({k:result.get(k) for k in ('status','active_job','remaining','outcomes')},indent=2));return
    with (CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller()
        try:controller.run(args.stage)
        except BaseException as exc:
            controller.refresh('stopped_with_evidence',error=str(exc));raise


if __name__=='__main__':main()
