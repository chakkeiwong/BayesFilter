"""Bounded, source-snapshotted standalone Walnutpie GPU/TF execution program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).resolve().parent
PIN=ROOT/'.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953'
PLAN=ROOT/'docs/plans/bayesfilter-walnutpie-gpu-tf-master-2026-10-28.md'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['build','tests','validate','benchmark','diagnose','profile','qualify','all'],default='all')
    p.add_argument('--diagnostic-runs',type=Path,nargs='*',default=[])
    p.add_argument('--device',choices=['cpu','gpu','both'],default='gpu')
    p.add_argument('--gpu',default='1');p.add_argument('--oracle',type=Path)
    p.add_argument('--command-buffer',action='store_true')
    p.add_argument('--targets',default='gaussian,banana,funnel,regression')
    p.add_argument('--warmup',type=int,default=500);p.add_argument('--draws',type=int,default=500)
    p.add_argument('--regression-size',type=int,default=4096)
    p.add_argument('--regression-dimension',type=int,default=128)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=[]
    for path in out.parent.glob('*/manifest.json'):
        m=json.loads(path.read_text())
        if m.get('plan')!=str(PLAN):continue
        if m.get('status')=='running':raise RuntimeError('unsettled running manifest: '+str(path))
        prior.extend(m['attempts'])
    begin=time.monotonic()
    m=dict(plan=str(PLAN),command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=sys.executable,started_utc=datetime.now(timezone.utc).isoformat(),status='running',attempts=[],
        prior_attempts=len(prior),prior_worker_seconds=sum(x.get('wall_seconds',x['timeout_seconds']) for x in prior),
        budget_seconds=3600,maximum_attempts=16,source_hashes={},seed=20261028,
        data_version='analytic targets; deterministic trigonometric regression design v1',
        result_note='docs/plans/bayesfilter-walnutpie-gpu-tf-result-2026-10-28.md',
        scope='standalone engineering validation; no public default or posterior promotion')
    paths=[*SOURCE.glob('*.py'),*SOURCE.glob('*.cc'),*SOURCE.glob('*.md'),SOURCE/'LICENSE',PLAN,ROOT/'tests/test_walnutpie_gpu_tf.py',*PIN.glob('include/walnutpie/*.hpp')]
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);m['source_hashes'][str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        m['wall_seconds']=time.monotonic()-begin
        (out/'manifest.json').write_text(json.dumps(m,indent=2))
    def run(name,cmd,device='cpu',limit=600):
        used=m['prior_worker_seconds']+sum(x.get('wall_seconds',0.) for x in m['attempts'])
        remaining=3600-used
        if remaining<=0 or m['prior_attempts']+len(m['attempts'])>=16:raise RuntimeError('campaign budget exhausted')
        limit=min(limit,remaining)
        env={**os.environ,'CUDA_VISIBLE_DEVICES': '-1' if device=='cpu' else a.gpu,
             'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
             'PYTHONPATH':str(ROOT),'TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2'}
        if device=='gpu' and a.command_buffer:
            if 'xla_gpu_enable_command_buffer' in env.get('XLA_FLAGS',''):raise ValueError('conflicting command-buffer flags')
            env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
        log=out/(name+'.log');e=dict(name=name,command=cmd,device=device,status='running',timeout_seconds=limit,
            log=str(log),xla_flags=env.get('XLA_FLAGS',''),gpu_hidden=device=='cpu',memory_growth=env['TF_FORCE_GPU_ALLOW_GROWTH'])
        m['attempts'].append(e);save();t=time.monotonic();print('Running '+name,flush=True)
        try:
            with log.open('w') as f:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:rc=proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            e['returncode']=rc;e['status']='passed' if rc==0 else 'failed'
            if rc:raise RuntimeError(name+' failed; '+str(log))
        except Exception as ex:e.update(status='failed',error=repr(ex));raise
        finally:e['wall_seconds']=time.monotonic()-t;save()
    oracle=(a.oracle.resolve() if a.oracle else out/'oracle')
    try:
        if a.phase in ('build','qualify','all'):
            run('native-build',['g++','-O3','-std=c++20','-pthread',
                '-I'+str(ROOT/'.localresources/cmdstan/cmdstan-2.37.0/stan/lib/stan_math/lib/eigen_3.4.0'),
                '-I'+str(PIN/'include'),str(SOURCE/'oracle.cc'),'-o',str(oracle)],limit=120)
        if a.phase in ('tests','qualify','all'):
            run('unit-tests',[sys.executable,'-m','pytest','-q','tests/test_walnutpie_gpu_tf.py'],limit=120)
        if a.phase=='diagnose':
            run('post-run-diagnostics',[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.diagnose',
                '--output',str(out/'diagnostics'),*[str(x.resolve()) for x in a.diagnostic_runs]],limit=120)
        if a.phase=='profile':
            run('gpu-profile',[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.profile',
                '--output',str(out/'profile')],'gpu',180)
        if a.phase not in ('build','tests','diagnose','profile'):
            if not oracle.is_file():raise FileNotFoundError(oracle)
            m['oracle_sha256']=hashlib.sha256(oracle.read_bytes()).hexdigest();m['oracle']=str(oracle)
            for device in (['cpu','gpu'] if a.device=='both' else [a.device]):
                for mode in (['validate','benchmark'] if a.phase=='all' else ['validate'] if a.phase=='qualify' else [a.phase]):
                    run(device+'-'+mode,[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.worker',
                        '--device',device,'--mode',mode,'--output',str(out/(device+'-'+mode)),
                        '--oracle',str(oracle),'--targets',a.targets,'--warmup',str(a.warmup),'--draws',str(a.draws),
                        '--regression-size',str(a.regression_size),'--regression-dimension',str(a.regression_dimension)],device,900)
        m['status']='completed'
    except Exception as ex:m.update(status='failed',error=repr(ex));raise
    finally:save()
    print(json.dumps(dict(output=str(out),worker_seconds=sum(x['wall_seconds'] for x in m['attempts']))))

if __name__=='__main__':main()
