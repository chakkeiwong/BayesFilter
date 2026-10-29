"""Bounded engineering qualification for initialization/control/R2P gap closure."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import signal,subprocess,sys,time,shutil
ROOT=Path(__file__).resolve().parents[3]
PLAN='docs/plans/bayesfilter-walnutpie-gap-closure-2026-10-29.md'
BUDGET=3600;ATTEMPTS=18

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['tests','validate','deterministic'],required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=[]
    for file in out.parent.glob('*/manifest.json'):
        m=json.loads(file.read_text())
        if m.get('plan')!=PLAN:continue
        if m.get('status')=='running':raise RuntimeError('unsettled prior launch '+str(file))
        previous.append(m)
    used=sum(m['wall_seconds'] for m in previous)
    if used>=BUDGET or len(previous)>=ATTEMPTS:raise RuntimeError('campaign budget exhausted')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if a.device=='cpu' else a.gpu,
         'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
         'PYTHONPATH':str(ROOT),'TF_NUM_INTRAOP_THREADS':'4','TF_NUM_INTEROP_THREADS':'2'}
    if a.device=='gpu':
        if 'xla_gpu_enable_command_buffer' in env.get('XLA_FLAGS',''):raise ValueError('conflicting XLA flags')
        env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL'
    command=([sys.executable,'-m','pytest','-q','tests/test_walnutpie_gpu_tf.py','tests/test_walnutpie_gap_closure.py']
       if a.phase=='tests' else [sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.gap_validation','--device',a.device,'--output',str(out)])
    oracle=Path(__file__).parent/'prebuilt/linux-x86_64/20261029/oracle'
    if a.phase=='deterministic':
        command=[sys.executable,'-m','docs.benchmarks.walnutpie_gpu_tf.worker',
            '--mode','validate','--device',a.device,'--output',str(out/'deterministic'),'--oracle',str(oracle)]
    if a.phase=='tests' and a.device!='cpu':raise ValueError('pytest is CPU reference; GPU uses validation')
    begin=time.monotonic();m=dict(plan=PLAN,result_note='docs/plans/bayesfilter-walnutpie-gap-closure-result-2026-10-29.md',
        command=command,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),status='running',wall_seconds=0.,
        previous_seconds=used,previous_attempts=len(previous),budget_seconds=BUDGET,attempt_cap=ATTEMPTS,
        environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','XLA_FLAGS',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','BAYESFILTER_PRELOAD_CUSTOM_OP')},
        python=sys.executable,seeds=[17,29,43],
        seed_policy={'r2p_macro_tape_keys':list(range(1,49)),'r2p_tree_tape_keys':list(range(1,25)),
            'native_init_tape_keys':[17,18,19],
            'other_cases':'Exact keys/counters in snapshotted tests, worker.py and saved oracle records.'},
        data_version='analytic engineering fixtures v1',
        source_hashes={},device=a.device,gpu_intentionally_hidden=a.device=='cpu')
    if a.phase=='deterministic':m['oracle_sha256']=hashlib.sha256(oracle.read_bytes()).hexdigest()
    m['artifact_paths']={'worker_log':str(out/'worker.log'),
        'manifest':str(out/'manifest.json'),'source':str(out/'source'),
        'result':str(out/('deterministic/result.json' if a.phase=='deterministic' else 'validation.json')) if a.phase!='tests' else str(out/'worker.log')}
    for file in [*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cc'),ROOT/PLAN,
                 ROOT/'tests/test_walnutpie_gap_closure.py',ROOT/'tests/test_walnutpie_gpu_tf.py']:
        rel=file.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest);m['source_hashes'][str(rel)]=hashlib.sha256(file.read_bytes()).hexdigest()
    m['upstream_sources']={}
    for directory in (ROOT/'.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953/include/walnutpie',
                      ROOT/'.localresources/walnuts-r2p-895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a/WALNUTSpy'):
        for file in sorted(directory.iterdir()):
            if file.suffix not in ('.hpp','.py'):continue
            rel=file.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest);m['upstream_sources'][str(rel)]=hashlib.sha256(file.read_bytes()).hexdigest()
    paper=ROOT/'.localresources/funnel-local-adaptation-20261028/walnuts-jmlr-2026.pdf'
    m['paper_sha256']=hashlib.sha256(paper.read_bytes()).hexdigest()
    def save():
        m['wall_seconds']=time.monotonic()-begin;(out/'manifest.json').write_text(json.dumps(m,indent=2))
    save()
    try:
        if a.device=='gpu':
            m['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(900,BUDGET-used))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        m.update(status='passed' if code==0 else 'failed',returncode=code)
        if code:raise RuntimeError('worker failed; '+str(out/'worker.log'))
    except Exception as e:m.update(status='failed',error=repr(e));raise
    finally:save()
    print(json.dumps(dict(output=str(out),status=m['status'],wall_seconds=m['wall_seconds'])))
if __name__=='__main__':main()
