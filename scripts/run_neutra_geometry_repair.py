#!/usr/bin/env python3
"""Bounded saved-map diagnosis and actual-consumer repair; no new training."""
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
PLAN='docs/plans/bayesfilter-neutra-geometry-selection-plan-2026-10-01.md'


def read(path):return json.loads(Path(path).read_text())


def master_at(root):
    spec=importlib.util.spec_from_file_location('geometry_master',root/'scripts/run_neutra_warm_start_master.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Controller:
    def __init__(self):
        self.root=CAMPAIGN;self.output=self.root/'geometry-selection-20261001-r1'
        self.output.mkdir(exist_ok=True)
        self.original=read(self.root/'config.json');self.state=read(self.root/'state.json')
        if self.state.get('active_job'):raise RuntimeError('shared campaign has active or unreconciled work')
        live=master_at(ROOT);self.snapshot=self.output/'source'
        manifest=self.snapshot/'frozen-source.json'
        if not manifest.exists():
            self.snapshot.mkdir(exist_ok=False)
            paths=set(live.SOURCES)|{PLAN,'docs/reference/hmc-tuning-interface.md',
                'tests/test_neutra_geometry_repair.py'}
            paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'bayesfilter').rglob('*.py'))
            hashes={}
            for name in sorted(paths):
                saved=self.snapshot/name;saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,saved);hashes[name]=hashlib.sha256(saved.read_bytes()).hexdigest()
            live.write(manifest,{'git_commit':live.git_head(),'source_sha256':hashes,
                'purpose':'exact executed source; unrelated dirty work preserved','created_unix':time.time()})
        for name,digest in read(manifest)['source_sha256'].items():
            if hashlib.sha256((self.snapshot/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'changed frozen source: {name}')
        self.master=master_at(self.snapshot)
        self.record=read(self.output/'state.json') if (self.output/'state.json').exists() else {
            'plan':PLAN,'jobs':{},'outcomes':{},'budget_at_start':live.remaining(self.state,self.original),
            'subcap':{'gpu_process_seconds':14400.,'cpu_core_seconds':28800.},'started_unix':time.time()}
        self.base=copy.deepcopy(self.original)
        self.base['closure']={'plan':PLAN}
        # This completed cycle is frozen at v3. A new v4 numerical campaign
        # needs a fresh output/source snapshot; never relabel saved outcomes.
        self.base['hmc'].update(diagnostic_profile='moments_regions_shape_v3',posterior_member_limit=3,
            measurement_num_results=64,verification_num_results=64,shortlist_wall_seconds=3600.)
        self.base['job_wall_seconds'].update(closure_geometry=1200.,closure_qualify=3600.)

    def refresh(self,status,**kwargs):
        remaining=self.master.remaining(self.state,self.original)
        used={k:self.record['budget_at_start'][k]-remaining[k] for k in self.record['subcap']}
        self.record.update(status=status,used=used,remaining=remaining,updated_unix=time.time(),**kwargs)
        self.master.write(self.output/'state.json',self.record)
        self.master.write(self.root/'repair-next-phase.json',{'plan':PLAN,'status':status,
            'evidence':str(self.output/'state.json'),'active_job':self.state.get('active_job'),
            'next_phase':kwargs.get('next_phase'),'remaining':remaining,
            'resume_command':f'{sys.executable} scripts/run_neutra_geometry_repair.py campaign'})

    def phase(self,name,phase,target,seed,*,prepared,training=None,settings=None):
        if name in self.record['jobs']:
            row=self.record['jobs'][name]
            if row['result'].get('continuation_veto'):
                raise RuntimeError(f'preserved numerical veto still requires localization: {row["output"]}')
            return Path(row['output']),row['result']
        self.refresh('preparing',next_phase=name)
        if any(self.record['used'][k]>=cap for k,cap in self.record['subcap'].items()):
            raise RuntimeError('geometry sub-cap exhausted')
        if sum(r['phase']!='reference' for r in self.record['jobs'].values())>=16:
            raise RuntimeError('geometry phase cap exhausted')
        cfg=copy.deepcopy(self.base);cfg['closure'].update(settings or {})
        for k,cap in self.record['subcap'].items():
            cfg[k]=min(self.original[k],self.original[k]-self.record['budget_at_start'][k]+cap)
        extra=['--target',target,'--seed',str(seed),'--prepared',str(prepared)]
        if training:extra+=['--training',str(training)]
        job='geometry-20261001-'+name
        self.master.run_one(self.root,self.state,cfg,job,'closure_'+phase,extra)
        path=self.master.completed_path(self.root,self.state,job)
        if path is None:raise RuntimeError(f'worker failed; inspect preserved attempt: {job}')
        result=read(path/'phase.json')
        self.record['jobs'][name]={'phase':phase,'output':str(path),'result':result}
        self.refresh('phase_complete',next_phase=None)
        if result.get('continuation_veto'):raise RuntimeError(f'numerical localization required: {path}')
        return path,result

    def prepared(self,target):return self.root/'attempts'/f'closure-prepare-{target}-v0-r1'

    def maps(self):
        cases=[]
        for seed,width in ((11,16),(37,8)):
            stem=f'gap-20261001-init-w{width}-v0.2-s{seed}'
            warm=self.root/'attempts'/f'{stem}-r1';refine=self.root/'attempts'/f'{stem}-refine-r1'
            for stage in ('warm','rkl-256','rkl-1024','rkl-2048'):
                directory=warm if stage=='warm' else refine
                prefix='selected' if stage=='warm' else stage
                cases.append((f'mixture-s{seed}-{stage}','mixture',seed,stage,
                    directory/f'{prefix}-frozen.json',directory/f'{prefix}-assessment.json'))
        wiggle=self.root/'attempts/closure-refine-wiggle-gabrie-s23-r1'
        cases.append(('wiggle-s23-selected','wiggle',23,'selected',wiggle/'selected-frozen.json',wiggle/'selected-assessment.json'))
        return cases

    def diagnose(self):
        for name,target,seed,stage,frozen,assessment in self.maps():
            self.phase(name,'geometry',target,seed,prepared=self.prepared(target),
                settings={'geometry_map':str(frozen),'geometry_assessment':str(assessment)})
        self.refresh('geometry_diagnosed',next_phase='checkpoint shortlists and broader posterior checks')

    def shortlist(self,seed):
        destination=self.output/f'shortlist-s{seed}'
        if (destination/'checkpoint-candidates.json').exists():return destination
        destination.mkdir(exist_ok=True);rows=[]
        for stage in ('rkl-2048','rkl-1024','rkl-256','warm'):
            evidence=Path(self.record['jobs'][f'mixture-s{seed}-{stage}']['output'])
            frozen=read(evidence/'frozen.json');assessment=read(evidence/'assessment.json')
            probe=read(evidence/'post-training-1000.json')
            for source,suffix in (('frozen.json','frozen.json'),('assessment.json','assessment.json'),
                                  ('post-training-1000.json','post-training-1000.json')):
                shutil.copyfile(evidence/source,destination/f'{stage}-{suffix}')
            rows.append({'stage':stage,'filename':f'{stage}-frozen.json',
                'probe_file':f'{stage}-post-training-1000.json','assessment_file':f'{stage}-assessment.json',
                'transport_hash':frozen['transport_hash'],
                'eligible':assessment['passed'] and probe['finite'] and probe['valid_rows']==1000,
                'evidence':str(evidence)})
        self.master.write(destination/'checkpoint-candidates.json',{'schema':'neutra.checkpoint_candidates.v1',
            'selection_order':'latest_eligible_RKL_then_earlier_RKL_then_warm','candidates':rows})
        self.master.write(destination/'phase.json',{'passed':any(row['eligible'] for row in rows),
            'role':'reviewed preserved checkpoints; no new training'})
        return destination

    def campaign(self):
        self.diagnose()
        for seed in (11,37):
            candidate=self.shortlist(seed)
            # New predetermined reference/chain stream, chosen before its values.
            stream=37100+seed
            reference,_=self.phase(f's{seed}-reference','reference','mixture',stream,prepared=self.prepared('mixture'))
            path,result=self.phase(f's{seed}-qualify','qualify','mixture',stream,prepared=reference,training=candidate)
            self.record['outcomes'][str(seed)]={'evidence':str(path),'passed':result['passed'],
                'qualification':read(path/'qualification.json')}
        self.state['status']='neutra_geometry_selection_executed_pending_review'
        self.master.write(self.root/'state.json',self.state)
        self.refresh('executed_pending_terminal_review',next_phase='review geometry, selection, holdout isolation and costs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('diagnose','campaign'))
    args=parser.parse_args()
    with (CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller()
        if controller.record.get('status')=='reviewed_complete':return
        try:getattr(controller,'diagnose' if args.action=='diagnose' else 'campaign')()
        except BaseException as error:
            controller.refresh('requires_localization',error=str(error));raise


if __name__=='__main__':main()
