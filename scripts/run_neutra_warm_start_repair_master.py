#!/usr/bin/env python3
"""Reviewed phase controller sharing the existing campaign budget and workers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
PLAN='docs/plans/bayesfilter-neutra-warm-start-repair-master-2026-09-30.md'
spec=importlib.util.spec_from_file_location('warm_start_master',ROOT/'scripts/run_neutra_warm_start_master.py')
master=importlib.util.module_from_spec(spec);spec.loader.exec_module(master)


def load(path):
    return json.loads(Path(path).read_text())


def configure(root):
    cfg=load(root/'config.json')
    if 'closure' not in cfg:
        master.write(root/'config-before-closure-20260930.json',cfg)
        cfg['closure']={'schema':'neutra.initialization_repair.v1','plan':PLAN,
            'targets':['gaussian','mixture','warped_mixture','wiggle','funnel'],
            'seeds':[11,23,37],'arms':['smc','gabrie'],'phase_repairs':2,
            'full_author_controller_equivalence':False,
            'aft_craft':'quarantined_pending_objective_specific_calibration_and_state_archival'}
        for phase in ('prepare','teacher','fit','refine','sampler','reference','qualify'):
            cfg['job_wall_seconds']['closure_'+phase]=600.
        master.write(root/'config.json',cfg)
    return cfg


class Controller:
    def __init__(self,root,*,execute=None):
        self.root=Path(root);self.cfg=configure(self.root)
        self.state=load(self.root/'state.json')
        self.path=self.root/'repair-master-state.json'
        self.record=load(self.path) if self.path.exists() else {
            'schema':'neutra.initialization_repair_state.v1','plan':PLAN,
            'started_unix':time.time(),'jobs':{},'decisions':[],'outcomes':[],
            'budget_basis':'existing shared campaign; no additional allocation'}
        self.execute=execute or master.run_one

    def refresh(self,phase,*,job=None,decision=None,evidence=None):
        self.record.update(status=phase,active_job=job,updated_unix=time.time(),
            remaining=master.remaining(self.state,self.cfg),pid=os.getpid())
        if decision:
            self.record['decisions'].append({'time':time.time(),'job':job,'decision':decision,'evidence':evidence})
        master.write(self.path,self.record)
        next_phase={'phase':phase,'job':job,'decision':decision,'evidence':evidence,
            'resume_command':'bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_warm_start_repair_campaign.sh start',
            'remaining':self.record['remaining'],'plan':PLAN}
        master.write(self.root/'repair-next-phase.json',next_phase)
        text=['# NeuTra repair campaign: current phase','',f'Phase: `{phase}`.',
            f'Job: `{job}`.',f'Decision: {decision or "execute reviewed phase"}.',
            '',f'GPU process seconds remaining: {self.record["remaining"]["gpu_process_seconds"]:.2f}.',
            f'CPU core seconds remaining: {self.record["remaining"]["cpu_core_seconds"]:.2f}.',
            '',f'Plan: `{PLAN}`.','',
            'Resume: `bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_warm_start_repair_campaign.sh start`.',
            'Status: `bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_warm_start_repair_campaign.sh status`.','']
        (self.root/'repair-next-phase.md').write_text('\n'.join(text))

    def phase(self,name,phase,target,seed,*,prepared=None,training=None,arm=None,repair_level=0):
        job='closure-'+name
        args=['--target',target,'--seed',str(seed),'--repair-level',str(repair_level)]
        if prepared:args+=['--prepared',str(prepared)]
        if training:args+=['--training',str(training)]
        if arm:args+=['--arm',arm]
        if phase=='prepare' and repair_level:args+=['--repair']
        self.refresh('ready',job=job,decision=f'execute {phase}; repair level {repair_level}')
        for infrastructure_attempt in range(2):
            okay=self.execute(self.root,self.state,self.cfg,job,'closure_'+phase,args)
            master.write(self.root/'state.json',self.state)
            output=master.completed_path(self.root,self.state,job)
            if output and (output/'phase.json').exists():
                result=load(output/'phase.json')
                self.record['jobs'][job]={'output':str(output),'result':result}
                self.refresh('phase_assessed',job=job,decision=result['reason'],evidence=str(output/'phase.json'))
                if result.get('continuation_veto'):
                    raise RuntimeError(f'Continuation veto: {job}: {result["reason"]}')
                return output,result
            if self.state['status'] in ('budget_exhausted','attempt_cap'):
                self.refresh(self.state['status'],job=job,decision='preserve pending phase and stop')
                raise RuntimeError(self.state['status'])
            recent=next((r for r in reversed(self.state['attempts']) if r['job']==job),{})
            if recent.get('status')=='budget_limited' and infrastructure_attempt==0:
                self.cfg['job_wall_seconds']['closure_'+phase]*=2
                master.write(self.root/'config.json',self.cfg)
                self.refresh('infrastructure_repair',job=job,decision='extend phase time once within unchanged total budget')
                continue
            self.refresh('harness_repair_required',job=job,decision='inspect preserved traceback before dependent work',evidence=recent.get('output'))
            raise RuntimeError(f'Worker failure: {job}, status={recent.get("status")}, okay={okay}')
        raise RuntimeError(f'Unresolved phase: {job}')

    def candidate_outcome(self,target,arm,seed,status,evidence):
        row={'target':target,'arm':arm,'seed':seed,'status':status,'evidence':evidence}
        key=(target,arm,seed)
        self.record['outcomes']=[r for r in self.record['outcomes'] if (r['target'],r['arm'],r['seed'])!=key]+[row]
        self.refresh('candidate_resolved',decision=status,evidence=evidence)

    def run(self,targets=None,seeds=None,arms=None):
        master.recover_active(self.root,self.state,self.cfg)
        selected=targets or self.cfg['closure']['targets'];seeds=seeds or self.cfg['closure']['seeds'];arms=arms or self.cfg['closure']['arms']
        for target in selected:
            prepared=None
            for level in range(2):
                path,report=self.phase(f'prepare-{target}-v{level}','prepare',target,400+level,repair_level=level)
                if report['passed']:prepared=path;break
            if prepared is None:
                self.candidate_outcome(target,'all',0,'discovery_repair_exhausted',str(path));continue
            for seed in seeds:
                teacher=None
                for level in range(3):
                    path,report=self.phase(f'teacher-{target}-s{seed}-v{level}','teacher',target,seed,
                                           prepared=prepared,repair_level=level)
                    if report['passed']:teacher=path;break
                if teacher is None:
                    self.candidate_outcome(target,'smc',seed,'teacher_repair_exhausted',str(path))
                for arm in arms:
                    if arm=='smc' and teacher is None:continue
                    warm=None
                    for level in range(2):
                        path,report=self.phase(f'fit-{target}-{arm}-s{seed}-v{level}','fit',target,seed,
                            prepared=prepared,training=teacher if arm=='smc' else prepared,arm=arm,repair_level=level)
                        if report['passed']:warm=path;break
                    if warm is None:
                        self.candidate_outcome(target,arm,seed,'nonlinear_fit_repair_exhausted',str(path));continue
                    if arm=='gabrie':
                        sampler_ok=False
                        for level in range(2):
                            path,report=self.phase(f'sampler-{target}-{arm}-s{seed}-v{level}','sampler',target,seed,
                                prepared=prepared,training=warm,repair_level=level)
                            if report['passed']:sampler_ok=True;break
                        if not sampler_ok:
                            self.candidate_outcome(target,arm,seed,'frozen_gabrie_sampler_repair_exhausted',str(path));continue
                    refined,report=self.phase(f'refine-{target}-{arm}-s{seed}','refine',target,seed,
                        prepared=prepared,training=warm,arm=arm)
                    if not report['passed']:
                        self.candidate_outcome(target,arm,seed,'refinement_harness_failure',str(refined));continue
                    candidate=refined
                    for final_round in range(2):
                        if final_round and report.get('selected_stage','warm')!='warm':
                            candidate=warm
                        frozen=load(candidate/'selected-frozen.json')
                        stream=f'{target}:{arm}:{seed}:{final_round}:{frozen["transport_hash"]}'
                        final_seed=int(hashlib.sha256(stream.encode()).hexdigest()[:7],16)
                        reference,_=self.phase(f'final-reference-{target}-{arm}-s{seed}-v{final_round}','reference',target,final_seed,
                            prepared=prepared,training=candidate,arm=arm)
                        qualified,q=self.phase(f'qualify-{target}-{arm}-s{seed}-v{final_round}','qualify',target,final_seed,
                            prepared=reference,training=candidate,arm=arm,repair_level=final_round)
                        if q['passed']:break
                        self.refresh('final_check_repair',decision='preserve failed holdout; widen kernel search or qualify preserved warm map with new reference',evidence=str(qualified))
                    self.candidate_outcome(target,arm,seed,'posterior_screen_passed' if q['passed'] else 'fresh_final_check_failed',str(qualified))
                    # A consumed final holdout cannot become a tuning set for retry.
        self.refresh('declared_matrix_resolved',decision='review successes and exhausted candidate repairs; no default or q20 promotion')
        self.write_results()

    def write_results(self):
        rows=self.record['outcomes']
        passed=[r for r in rows if r['status']=='posterior_screen_passed']
        result={'plan':PLAN,'outcomes':rows,'passed_candidates':len(passed),
            'remaining':master.remaining(self.state,self.cfg),'statistically_supported_ranking':False,
            'default_readiness':False,'full_author_program_equivalence':False,
            'status':self.record['status'],'decisions':self.record['decisions']}
        master.write(self.root/'repair-master-result.json',result)
        lines=['# NeuTra repair master: results','',f'Status: `{result["status"]}`.','',
            '| Target | Generator | Seed | Outcome |','|---|---|---|---|']
        lines.extend(f'| {r["target"]} | {r["arm"]} | {r["seed"]} | {r["status"]} |' for r in rows)
        lines+=['','| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |',
            '|---|---|---|---|---|---|',
            '| Preserve checked candidates and failures | Per-stage and final screens above | Failed screens reject their candidates | Finite replications and unexplored repairs | Review stage evidence before further work | General superiority, reliable training on all targets, q20 readiness |',
            '', '| Inference status | Finding |','|---|---|',
            '| Hard veto screen | See per-phase numerical and statistical screens |',
            '| Statistically supported ranking | None |',
            '| Descriptive differences | Loss, runtime and finite-campaign success counts |',
            '| Default readiness | Not established |',
            '| Next evidence needed | Independent replication of any viable complete procedure |','']
        (self.root/'repair-master-result.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','status','configure'))
    parser.add_argument('--targets');parser.add_argument('--seeds');parser.add_argument('--arms')
    args=parser.parse_args()
    if args.mode=='status':
        path=CAMPAIGN/'repair-master-state.json';state=load(path) if path.exists() else {'status':'not_started'}
        print(json.dumps({k:state.get(k) for k in ('status','active_job','remaining','outcomes')},indent=2));return
    with (CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller(CAMPAIGN)
        if args.mode=='configure':controller.refresh('configured',decision='focused review required before execution');return
        try:
            controller.run(args.targets.split(',') if args.targets else None,
                [int(s) for s in args.seeds.split(',')] if args.seeds else None,args.arms.split(',') if args.arms else None)
        except BaseException as exc:
            controller.refresh('stopped_with_preserved_evidence',decision=str(exc))
            controller.write_results();raise


if __name__=='__main__':main()
