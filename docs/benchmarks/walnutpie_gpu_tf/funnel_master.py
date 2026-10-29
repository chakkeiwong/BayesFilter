"""Budgeted launcher for the author-funnel reproduction plan."""
import argparse
from datetime import datetime,timezone
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
PLAN='docs/plans/bayesfilter-walnuts-author-funnel-2026-10-29.md'
RESULT='docs/plans/bayesfilter-walnuts-author-funnel-result-2026-10-29.md'
BUDGET=3600


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['tests','tune','sample','transient','extreme','analyze'],required=True)
    p.add_argument('--engine',choices=['tf','author-original','author-corrected'],default='tf')
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--gpu',default='1')
    p.add_argument('--dimension',type=int,choices=[10,11],default=11)
    p.add_argument('--draws',type=int,default=512);p.add_argument('--settle',type=int,default=512)
    p.add_argument('--tuning',type=Path);p.add_argument('--runs',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=[]
    for path in out.parent.glob('*/manifest.json'):
        m=json.loads(path.read_text())
        if m.get('plan')!=PLAN:continue
        if m['status']=='running':raise RuntimeError('unsettled prior launch '+str(path))
        previous.append(m)
    used=sum(m['wall_seconds'] for m in previous)
    if used>=BUDGET or len(previous)>=24:raise RuntimeError('campaign exhausted')
    if a.device=='gpu' and (a.engine!='tf' or a.phase not in ('sample','transient')):
        raise ValueError('only TF sampling uses GPU')
    if a.draws>8192 or a.settle>512 or a.draws<1 or a.settle<0:raise ValueError('outside plan draw bounds')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if a.device=='cpu' else a.gpu,
         'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
         'PYTHONPATH':str(ROOT),'TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2',
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    if a.device=='gpu':
        if 'xla_gpu_enable_command_buffer' in env.get('XLA_FLAGS',''):raise ValueError('conflicting flags')
        env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
    command=[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.funnel_worker',
        '--phase',a.phase,'--engine',a.engine,'--device',a.device,'--dimension',str(a.dimension),
        '--draws',str(a.draws),'--settle',str(a.settle),'--output',str(out)]
    if a.tuning:command+=['--tuning',str(a.tuning.resolve())]
    if a.runs:command+=['--runs',*[str(x.resolve()) for x in a.runs]]
    if a.phase=='tests':command=[sys.executable,'-m','pytest','-q','tests/test_walnuts_author_funnel.py']
    start=time.monotonic()
    m=dict(plan=PLAN,result_note=RESULT,command=command,git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),started_utc=datetime.now(timezone.utc).isoformat(),
        status='running',wall_seconds=0.,previous_seconds=used,previous_attempts=len(previous),
        budget_seconds=BUDGET,attempt_cap=24,environment={k:env.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','XLA_FLAGS','TF_NUM_INTRAOP_THREADS',
         'TF_NUM_INTEROP_THREADS','OPENBLAS_NUM_THREADS','BAYESFILTER_PRELOAD_CUSTOM_OP')},
        python=sys.executable,device=a.device,gpu_intentionally_hidden=a.device=='cpu',
        seeds=[0,1,2,3] if a.engine!='tf' or a.phase=='tune' else [104729,104730,104731,104732],
        data_version='Analytic centered funnel; total dimension '+str(a.dimension),source_hashes={},
        artifact_paths={'output':str(out),'manifest':str(out/'manifest.json'),'log':str(out/'worker.log'),
                        'result':str(out/'result.json')},cpu_workers=4 if a.engine!='tf' and a.phase in ('sample','transient') else 1)
    files=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.joinpath('author_reference').rglob('*.py'),
        Path(__file__).parent/'author_reference/provenance.json',ROOT/PLAN,ROOT/'tests/test_walnuts_author_funnel.py']
    for file in files:
        rel=file.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest);m['source_hashes'][str(rel)]=hashlib.sha256(file.read_bytes()).hexdigest()
    if a.tuning:m['tuning_sha256']=hashlib.sha256(a.tuning.read_bytes()).hexdigest()
    if a.runs:
        m['analysis_inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in a.runs for p in directory.iterdir()
            if p.name in ('result.json','draws.json','chains.json','manifest.json')}
    paper=ROOT/'.localresources/funnel-local-adaptation-20261028/walnuts-jmlr-2026.pdf'
    m['paper_sha256']=hashlib.sha256(paper.read_bytes()).hexdigest()
    def save():
        m['wall_seconds']=time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps(m,indent=2))
    save()
    try:
        if a.device=='gpu':m['gpu_inventory']=subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(60 if a.phase=='extreme' else 600,BUDGET-used))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                m.update(status='timeout',failure_class='bounded_cost_limit');raise
        m.update(status='passed' if code==0 else 'failed',returncode=code)
        if code:m['failure_class']='worker_failure';raise RuntimeError('worker failed; '+str(out/'worker.log'))
    except Exception as e:
        if m['status']=='running':m['status']='failed'
        m['error']=repr(e);raise
    finally:save()
    print(json.dumps(dict(output=str(out),status=m['status'],wall_seconds=m['wall_seconds'])),flush=True)

if __name__=='__main__':main()
