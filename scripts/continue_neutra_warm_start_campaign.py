#!/usr/bin/env python3
"""Durable ordered continuation around the existing budgeted NeuTra master.

No framework imports or numerical authority. Each command uses the same
campaign ledger and creates its own unique attempts through the master.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'scripts/run_neutra_warm_start_master.py'


def write(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temp.replace(path)


def steps():
    queue=[('qualify-aft-mixture',['--through','qualify','--targets','mixture','--arms','aft','--seeds','11']),
           ('qualify-seed11',['--through','qualify','--seeds','11'])]
    for seed in (23,37):
        for width in (8,16):
            queue.append((f'mixture-oracle-capacity-w{width}-s{seed}',[
                '--through','qualify','--targets','mixture','--arms','oracle','--seeds',str(seed),
                '--capacity-width',str(width),'--warm-total-updates','8192']))
    queue.append(('replications-seeds23-37',['--through','qualify','--seeds','23,37']))
    return queue


def run_queue(root,*,execute=subprocess.run):
    root=Path(root).resolve()
    if not (root/'config.json').is_file():raise ValueError('existing funded campaign config required')
    path=root/'continuation-queue-state.json'
    with (root/'continuation-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(path.read_text()) if path.exists() else {'schema':'neutra.warm_start_continuation_queue.v1','steps':{},'events':[]}
        state.update(status='waiting_for_active_master',pid=os.getpid());write(path,state)
        # The prior interactive command may still be finishing its queue.
        # Do not signal it or replace its active-job record.
        with (root/'master.lock').open('a') as master_lock:
            fcntl.flock(master_lock,fcntl.LOCK_EX)
        for name,arguments in steps():
            if state['steps'].get(name,{}).get('status')=='complete':continue
            command=[sys.executable,str(MASTER),'run','--output',str(root),*arguments]
            row={'status':'running','argv':command,'started_unix':time.time()}
            state['steps'][name]=row;state.update(status='running',active_step=name);write(path,state)
            print(json.dumps({'event':'queue_start','step':name}),flush=True)
            try:
                result=execute(command,cwd=ROOT,check=False)
            except BaseException:
                row.update(status='interrupted',finished_unix=time.time())
                state['status']='interrupted_master_recovery_required';write(path,state)
                raise
            row.update(status='complete' if result.returncode==0 else 'stopped',
                exit_code=result.returncode,finished_unix=time.time())
            state['events'].append({'step':name,**row})
            if result.returncode:
                state['status']='budget_exhausted' if result.returncode==3 else 'master_requires_repair'
                write(path,state);return result.returncode
            write(path,state)
        state.update(status='queue_attempted',active_step=None)
        state['interpretation']='Commands completed; scientific success is determined by individual results, not exit code.'
        write(path,state);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit(run_queue(args.output))
