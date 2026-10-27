"""Bounded local Stan-port audit/benchmark launcher; no framework import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'docs/benchmarks/artifacts/stan-full-port-20261027'
PLAN='docs/plans/bayesfilter-full-stan-nuts-port-execution-2026-10-27.md'
RESULT='docs/plans/bayesfilter-full-stan-nuts-port-result-2026-10-27.md'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=int,required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--reason',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    previous=[json.loads(x.read_text()) for x in CAMPAIGN.glob('*/attempt.json')]
    if any(x.get('returncode') is None for x in previous):
        raise RuntimeError('Unclosed attempt: reconcile before next launch')
    spent=sum(x['wall_seconds'] for x in previous)
    if len(previous)>=30 or spent+a.seconds>5400:
        raise RuntimeError('Declared campaign budget exhausted')
    folder=CAMPAIGN/a.name
    folder.mkdir(exist_ok=False)
    env=dict(os.environ,TF_FORCE_GPU_ALLOW_GROWTH='true',BAYESFILTER_PRELOAD_CUSTOM_OP='0',
             BAYESFILTER_REQUIRE_STAN_ORACLE='1',BAYESFILTER_STAN_AUDIT_OUTPUT=str(folder/'oracle'),
             BAYESFILTER_STAN_PORT_OUTPUT=str(folder),PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'docs/benchmarks'))
    env['CUDA_VISIBLE_DEVICES']='-1' if a.device=='cpu' else '0'
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    source=folder/'source';source.mkdir()
    paths=[Path(__file__),ROOT/'docs/benchmarks/stan_nuts.py',ROOT/'docs/benchmarks/stan_transition_oracle.cpp',
           ROOT/'docs/benchmarks/stan_nuts_warmup.py',ROOT/PLAN,ROOT/'AGENTS.md']
    paths += list((ROOT/'docs/benchmarks').glob('*full_stan*.py'))
    paths += list((ROOT/'docs/benchmarks').glob('stan_service_oracle.cpp'))
    paths += list((ROOT/'tests').glob('test_stan_nuts_port.py'))
    hashes={}
    for file in paths:
        key=str(file.relative_to(ROOT));hashes[key]=hashlib.sha256(file.read_bytes()).hexdigest()
        shutil.copyfile(file,source/(file.name+'.txt'))
    record=dict(command=command,plan=PLAN,result_file=RESULT,environment=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_source_snapshot=hashes,device=a.device,gpu_intentionally_hidden=a.device=='cpu',
        memory_growth_environment=True,reason=a.reason,seeds='per worker/test artifacts',
        data_version='pinned analytic fixtures and Stan 2.37.0',artifact_paths=[str(folder)],
        watchdog_seconds=a.seconds,previous_wall_seconds=spent,wall_seconds=0.,returncode=None)
    manifest=folder/'attempt.json';manifest.write_text(json.dumps(record,indent=2)+'\n')
    began=time.perf_counter()
    try:
        with (folder/'worker.log').open('w') as log:
            proc=subprocess.run(['timeout','--signal=TERM','--kill-after=15',str(a.seconds),*command],
                                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['returncode']=proc.returncode
    finally:
        record['wall_seconds']=time.perf_counter()-began
        record['remaining_wall_seconds']=5400-spent-record['wall_seconds']
        manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('returncode','wall_seconds','remaining_wall_seconds')}
                     | {'manifest':str(manifest)},indent=2))
    sys.exit(record['returncode'])

if __name__=='__main__':main()
