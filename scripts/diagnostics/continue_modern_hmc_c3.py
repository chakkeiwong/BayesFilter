"""Bounded pricing-to-posterior continuation under the existing C3 plan."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PLAN='docs/plans/bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md'
MARKER='<!-- C3-AUTOMATIC-CONTINUATION -->'

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+chr(10));tmp.replace(path)
def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def allocation(preparation,settlement,price,quote_function):
    if settlement.get('status')!='completed': raise ValueError('pricing did not complete')
    charge=settlement.get('inclusive_charge_seconds')
    if type(charge) not in (int,float) or not math.isfinite(charge) or charge<0:
        raise ValueError('pricing charge is invalid')
    balance=preparation['prior_upper_balance_seconds']-preparation['reference_charge_seconds']
    balance-=preparation['preparation_reserve_seconds']+charge
    if balance<=180: raise ValueError('campaign funds exhausted')
    quote=quote_function(price,balance-180)
    cap=math.ceil(quote['required_seconds'])+180
    if not quote['affordable'] or cap>balance:
        raise ValueError('all-member posterior design is not affordable')
    return dict(previous_upper_balance_seconds=preparation['prior_upper_balance_seconds'],
        reference_charge_seconds=preparation['reference_charge_seconds'],
        preparation_reserve_seconds=preparation['preparation_reserve_seconds'],
        actual_pricing_charge_seconds=charge,balance_before_posterior_seconds=balance,
        posterior_reservation_seconds=cap,unreserved_upper_balance_seconds=balance-cap,
        separate_sir_seconds=preparation['separate_sir_seconds'],quote=quote)

def refresh_program(repo,record):
    lines=[MARKER,'## C3 automatic continuation — '+record['updated_utc'],'',
        'Current continuation status: **'+record['status']+'**.',
        'Exact progress: N5/c3-posterior-20261029-01/continuation-01/status.json.']
    if record.get('allocation'):
        a=record['allocation']
        lines += ['Posterior reservation: %.0fs; actual pricing charge: %.3fs.' %
            (a['posterior_reservation_seconds'],a['actual_pricing_charge_seconds']),
            'Unreserved balance after reservation: %.3fs; SIR remains separate.' % a['unreserved_upper_balance_seconds']]
    if record.get('posterior_settlement'):
        s=record['posterior_settlement']
        lines += ['Posterior inclusive charge: %.3fs.' % s['inclusive_charge_seconds'],
            'Remaining upper balance after settlement: %.3fs, preparation still withheld.' % record['remaining_upper_balance_seconds'],
            'Members passing all declared checks: %d of 7.' % record.get('passed_members',0)]
    if record.get('error'):lines += ['Required follow-up: '+record['error']+'.']
    lines += ['C3 diagnostic success does not grant posterior/default/release promotion.',
        'C2 migration, C4 reliability, C5 independent SIR and C6/N8 reconciliation remain open.',MARKER,'']
    block=chr(10).join(lines)+chr(10)
    for name in ('bayesfilter-modern-hmc-implementation-plan-2026-10-07.md',
                 'bayesfilter-modern-hmc-active-checkpoint-2026-10-13.md'):
        path=repo/'docs/plans'/name;current=path.read_text()
        if MARKER in current:
            first=current.index(MARKER);last=current.index(MARKER,first+len(MARKER))+len(MARKER)
            updated=current[:first]+block+current[last:].lstrip(chr(10))
        else:
            position=current.index(chr(10))+1
            updated=current[:position]+chr(10)+block+current[position:].lstrip(chr(10))
        tmp=path.with_suffix('.c3.tmp');tmp.write_text(updated);tmp.replace(path)

def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cpu_start=time.process_time();process=None
    state=dict(status='waiting_for_prices',plan=PLAN,updated_utc=utc(),
        command=[sys.executable,*sys.argv],posterior_qualified=False,release_qualified=False)
    def save(refresh=False):
        state.update(updated_utc=utc(),observer_process_cpu_seconds=time.process_time()-cpu_start)
        write(args.output/'status.json',state)
        if refresh: refresh_program(args.repo,state)
    def stop(signum,frame): raise InterruptedError('continuation signal '+str(signum))
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,stop)
    try:
        launch=read(args.root/'posterior-preparation.json')
        def check_sources():
            for path,expected in launch['source_sha256'].items():
                if digest(path)!=expected:raise ValueError('prepared source changed: '+path)
        check_sources()
        spec=importlib.util.spec_from_file_location('c3_posterior_driver',launch['driver'])
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true':
            raise ValueError('GPU growth must be present before launch')
        save(True);price_root=args.root/'price-01'
        deadline=time.monotonic()+launch['price_wait_max_seconds']
        while not (price_root/'settlement.json').exists():
            if time.monotonic()>=deadline:raise TimeoutError('price settlement missing after bounded wait')
            p=price_root/'manifest.json'
            if p.exists():state['pricing_status']=read(p).get('status')
            save();time.sleep(30)
        settlement=read(price_root/'settlement.json');price_path=price_root/'evidence/result.json'
        price=read(price_path)
        driver.validate_price_scope(price,read(price_path.parent/'manifest.json'),
            read(Path(launch['campaign'])/'launch.json'),Path(launch['reference']))
        funding=allocation(read(args.root/'preparation.json'),settlement,price,driver.posterior_quote)
        state['allocation']=funding;write(args.output/'allocation.json',funding);check_sources()
        target=args.root/'posterior-01'
        if target.exists():raise ValueError('posterior directory already exists; inspect before resume')
        command=[sys.executable,launch['driver'],'--campaign',launch['campaign'],'--output',str(target),
            '--reference',launch['reference'],'--quantity-fixture',launch['quantity_fixture'],
            '--stage','posterior','--price-result',str(price_path),
            '--wall-seconds',str(funding['posterior_reservation_seconds'])]
        write(args.output/'launch.json',dict(command=command,plan=PLAN,allocation=funding,
            source_sha256=launch['source_sha256'],started_utc=utc()))
        with (args.output/'posterior-supervisor.log').open('x') as log:
            process=subprocess.Popen(command,cwd=args.repo,env=os.environ.copy(),stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='posterior_running',supervisor_pid=process.pid);save(True)
            while process.poll() is None:
                p=target/'manifest.json'
                if p.exists():state['posterior_progress']=read(p).get('progress')
                save();time.sleep(30)
        terminal=read(target/'settlement.json');state['posterior_settlement']=terminal
        state['remaining_upper_balance_seconds']=funding['balance_before_posterior_seconds']-terminal['inclusive_charge_seconds']
        result_path=target/'evidence/result.json';result=read(result_path) if result_path.exists() else {}
        state.update(status='posterior_assessment_completed' if terminal['status']=='completed' else 'posterior_followup_required',
            exit_code=process.returncode,passed_members=result.get('all_declared_checks_passed_count',0),
            result_file=str(result_path))
        if terminal['status']!='completed':state['error']=result.get('error','inspect supervisor and preserved chunks')
        save(True)
    except Exception as exc:
        if process is not None and process.poll() is None:
            state.update(active_posterior_supervisor_pid=process.pid,posterior_survives_observer_failure=True)
        state.update(status='continuation_followup_required',error=repr(exc));save(True);return 1
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('repo','root','output'):setattr(args,name,getattr(args,name).resolve())
    return run(args)

if __name__=='__main__':raise SystemExit(main())
