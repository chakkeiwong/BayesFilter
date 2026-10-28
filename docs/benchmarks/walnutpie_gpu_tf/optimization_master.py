"""Bounded optimization campaign; snapshots, budgets, subprocess isolation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).resolve().parent
PIN=ROOT/'.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953'
PLAN=ROOT/'docs/plans/bayesfilter-walnutpie-optimization-2026-10-29.md'
BUDGET=3600
ATTEMPTS=18

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['tests','build','qualify','layout','batch','native'],required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--gpu',default='0')
    p.add_argument('--layout',choices=['fieldwise','packed','flat'],default='packed')
    p.add_argument('--layouts')
    p.add_argument('--targets',default='gaussian,banana,funnel,regression,logistic')
    p.add_argument('--oracle',type=Path)
    p.add_argument('--native-oracle',type=Path)
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--draws',type=int,default=100)
    a=p.parse_args()
    if a.layouts is None:a.layouts='fieldwise,packed,flat' if a.phase=='layout' else 'packed'
    if a.phase in ('qualify','native') and a.oracle is None:p.error('--oracle is required')
    if a.phase=='native' and a.native_oracle is None:p.error('--native-oracle is required')
    if a.phase in ('batch','native') and a.layouts not in ('fieldwise','packed','flat'):
        p.error('--layouts must select one layout for batch/native')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=[]
    for path in out.parent.glob('*/manifest.json'):
        old=json.loads(path.read_text())
        if old.get('plan')!=str(PLAN):continue
        if old['status']=='running':raise RuntimeError('unsettled worker: '+str(path))
        prior.extend(old['attempts'])
    begin=time.monotonic()
    m=dict(plan=str(PLAN),result_note='docs/plans/bayesfilter-walnutpie-optimization-result-2026-10-29.md',
        command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=sys.executable,started_utc=datetime.now(timezone.utc).isoformat(),status='running',
        prior_attempts=len(prior),prior_worker_seconds=sum(x['wall_seconds'] for x in prior),
        budget_seconds=BUDGET,maximum_attempts=ATTEMPTS,attempts=[],source_hashes={},
        seeds=[20261029+1000*i for i in range(7)],data_version='deterministic trigonometric design v1',
        scope='standalone engineering optimization; no public default or posterior promotion')
    m['compiler']=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    m['cpu']=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                   if x.startswith('model name')),'unknown')
    m['oracle_binaries']={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest()
                         for x in (a.oracle,a.native_oracle) if x is not None}
    for path in [*SOURCE.glob('*.py'),*SOURCE.glob('*.cc'),*SOURCE.glob('*.md'),PLAN,
                 ROOT/'tests/test_walnutpie_gpu_tf.py',*PIN.glob('include/walnutpie/*.hpp')]:
        rel=path.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);m['source_hashes'][str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        m['wall_seconds']=time.monotonic()-begin
        (out/'manifest.json').write_text(json.dumps(m,indent=2))
    def run(name,cmd,device='cpu',limit=900):
        used=m['prior_worker_seconds']+sum(x['wall_seconds'] for x in m['attempts'])
        if used>=BUDGET or len(prior)+len(m['attempts'])>=ATTEMPTS:raise RuntimeError('campaign budget exhausted')
        limit=min(limit,BUDGET-used)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if device=='cpu' else a.gpu,
             'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
             'PYTHONPATH':str(ROOT),'TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2'}
        if device=='gpu':
            if 'xla_gpu_enable_command_buffer' in env.get('XLA_FLAGS',''):raise ValueError('conflicting flags')
            env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
        if device=='gpu':
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
            (out/(name+'-gpu-inventory.txt')).write_text(inventory+processes)
        e=dict(name=name,command=cmd,device=device,environment={k:env.get(k) for k in
            ['CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','XLA_FLAGS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']},
            timeout_seconds=limit,wall_seconds=0.,status='running',log=str(out/(name+'.log')))
        m['attempts'].append(e);save();t=time.monotonic();print('Running '+name,flush=True)
        try:
            with open(e['log'],'w') as f:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:rc=proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            e['returncode']=rc;e['status']='passed' if rc==0 else 'failed'
            if rc:raise RuntimeError(name+' failed; '+e['log'])
        except Exception as ex:e.update(status='failed',error=repr(ex),failure_class='inspect log: numerical, harness or resource');raise
        finally:e['wall_seconds']=time.monotonic()-t;save()
    try:
        if a.phase=='tests':run('tests',[sys.executable,'-m','pytest','-q','tests/test_walnutpie_gpu_tf.py'],limit=240)
        elif a.phase=='build':
            for name,flags in [('generic',[]),('native',['-march=native'])]:
                run('build-'+name,['g++','-O3','-std=c++20','-pthread',*flags,
                    '-I'+str(ROOT/'.localresources/cmdstan/cmdstan-2.37.0/stan/lib/stan_math/lib/eigen_3.4.0'),
                    '-I'+str(PIN/'include'),str(SOURCE/'oracle.cc'),'-o',str(out/('oracle-'+name))],limit=120)
        elif a.phase=='qualify':
            run(a.device+'-qualify',[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.worker',
                '--device',a.device,'--mode','validate','--execution-layout',a.layout,
                '--oracle',str(a.oracle.resolve()),'--output',str(out/'validation')],a.device,600)
        else:
            cmd=[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.optimization_worker',
                '--mode',a.phase,'--device',a.device,'--output',str(out/'results'),
                '--layouts',a.layouts,'--targets',a.targets,'--warmup',str(a.warmup),'--draws',str(a.draws)]
            if a.oracle:cmd+=['--oracle',str(a.oracle.resolve())]
            if a.native_oracle:cmd+=['--native-oracle',str(a.native_oracle.resolve())]
            run(a.device+'-'+a.phase,cmd,a.device,900)
        m['status']='completed'
    except Exception as ex:m.update(status='failed',error=repr(ex));raise
    finally:save()
    print(json.dumps(dict(output=str(out),worker_seconds=sum(x['wall_seconds'] for x in m['attempts']))))
if __name__=='__main__':main()
