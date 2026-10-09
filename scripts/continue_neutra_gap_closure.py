#!/usr/bin/env python3
"""Resume the gap program and continue only supported nonlinear fit progress."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gap_base',ROOT/'scripts/run_neutra_gap_closure.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class Continuation(base.Controller):
    def refresh(self,status,**details):
        super().refresh(status,**details)
        path=self.root/'repair-next-phase.json';next_phase=base.read(path)
        next_phase['resume_command']=f'{sys.executable} scripts/continue_neutra_gap_closure.py'
        self.master.write(path,next_phase)

    def progressing_parent(self,seed):
        # A downstream failure must not be disguised as a fit-budget failure.
        rows=[r for r in self.record['outcomes'].values() if r.get('target')=='mixture' and r.get('seed')==seed]
        if any(r.get('passed') or r.get('status') not in ('finite_training_protocol_exhausted','finite_measure_unresolved')
               for r in rows):return None
        names=[f'control-s{seed}',*[f'init-w{w}-v{v:g}-s{seed}'
            for w,v in ((8,.02),(8,.2),(8,1.),(16,.2))]]
        if any(self.record['jobs'].get(name,{}).get('result',{}).get('passed') for name in names):
            return None
        for width,variance in ((8,.2),(8,1.),(16,.2)):
            name=f'init-w{width}-v{variance:g}-s{seed}'
            row=self.record['jobs'].get(name,{})
            result=row.get('result',{})
            parents=result.get('continuation_candidates',[])
            if result.get('repair')=='continue_checkpoint' and parents:
                return name,Path(parents[0])
        return None

    def continue_progress(self):
        files={'controller.py':Path(__file__),'plan.md':ROOT/base.PLAN}
        # Version ordinary control artifacts automatically on an edited plan
        # or launcher; preserve earlier copies without blocking a safe resume.
        revision=hashlib.sha256(b'\0'.join(path.read_bytes() for path in files.values())).hexdigest()[:12]
        archived=self.output/f'continuation-amendment-{revision}'
        archived.mkdir(exist_ok=True)
        hashes={}
        for name,path in files.items():
            destination=archived/name
            if not destination.exists():destination.write_bytes(path.read_bytes())
            elif destination.read_bytes()!=path.read_bytes():raise RuntimeError('Control archive checksum collision')
            hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
        self.master.write(archived/'manifest.json',{'command':sys.argv,'source_sha256':hashes,
            'numerical_source':str(self.snapshot),'budget_expansion':False})
        self.base['closure']['plan']=str(archived/'plan.md')
        self.record['continuation_disposition']={}
        for seed in (11,37):
            parent=self.progressing_parent(seed)
            if parent is None:
                qualified=any(r.get('target')=='mixture' and r.get('seed')==seed and r.get('passed')
                              for r in self.record['outcomes'].values())
                self.record['continuation_disposition'][str(seed)]=(
                    'not_needed_posterior_already_qualified' if qualified else 'not_a_supported_fit_continuation')
                continue
            self.record['continuation_disposition'][str(seed)]='supported_progress_continuation'
            arm,parent=parent
            for cap in (131072,262144):
                name=f'{arm}-continue-n{cap}'
                path,result=self.phase(name,'fit','mixture',seed,
                    prepared=self.prepared('mixture'),training=self.prepared('mixture'),parent=parent,
                    closure={'fit_rungs':[cap],'plateau_min_updates':cap,
                             'continuation_driver_sha256':hashes['controller.py']})
                self.record.setdefault('training',{})[name]={'evidence':str(path),'passed':result['passed']}
                if result['passed']:
                    self.finish_map(name,seed,path);break
                self.record.setdefault('continuation_results',{})[str(seed)]={
                    'status':result['reason'],'evidence':str(path),'updates':cap}
                self.refresh('continuation_assessed',active_job=None)
                candidates=result.get('continuation_candidates',[])
                if not candidates:break
                parent=Path(candidates[0])
        self.refresh('bounded_protocol_and_continuation_complete',active_job=None,
            next_phase='terminal review of all preserved candidate and posterior evidence')


def main():
    with (base.CAMPAIGN/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Continuation()
        try:
            controller.campaign()
            controller.continue_progress()
        except base.BudgetStop as exc:
            controller.refresh('bounded_budget_stop',active_job=None,error=str(exc))
        except BaseException as exc:
            controller.refresh('stopped_with_evidence',error=str(exc));raise


if __name__=='__main__':main()
