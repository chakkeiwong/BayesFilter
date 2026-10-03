"""Bounded independent-reference experiment for the score-regression extension."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from diagnose_iapf_adaptive_consumer import REPO, r_literal, save, setup, controls

ROOT=REPO/'docs/plans/artifacts/iapf-score-regression-20260922-01'
PLAN='docs/plans/iapf-score-regression-2026-09-22.md'
GPU='GPU-68251639-fe82-8f81-3ccc-2953c32e805b'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(value):
    if isinstance(value,dict): return {k:materialize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [materialize(v) for v in value]
    return value.numpy().tolist() if hasattr(value,'numpy') else value


def preflight(tf,directory):
    from diagnostic_iapf_score_regression_tf import make_score_fit,target_log_score
    cases=[];checks={};dtype=tf.float64
    for d in [2,5,10,40,80]:
        indices=tf.range(d);distance=tf.cast(tf.abs(indices[:,None]-indices[None,:]),dtype)
        covariance=tf.constant(.6,dtype)**distance+.3*tf.eye(d,dtype=dtype)
        center=.2*tf.sin(tf.cast(indices+1,dtype))
        x=tf.random.stateless_normal([256,d],[94121+d,d],dtype=dtype)
        scores=-tf.transpose(tf.linalg.solve(covariance,tf.transpose(x-center)))
        kernel=make_score_fit(d,256);c,v,info=kernel(x,scores)
        error=max(float(tf.reduce_max(tf.abs(c-center)).numpy()),float(tf.reduce_max(tf.abs(info['full_covariance']-covariance)).numpy()))
        checks['gaussian_'+str(d)]=bool(info['valid']) and error<=1e-8 and kernel.experimental_get_tracing_count()==1
        cases.append(materialize(dict(d=d,points=x,scores=scores,center=c,covariance=v,info=info,
                                     exact_center=center,exact_covariance=covariance,error=error)))
    d=5;kernel=make_score_fit(d,256)
    x=tf.random.stateless_normal([256,d],[94512,2],dtype=dtype)
    checks['reject_nonpositive_precision']=not bool(kernel(x,x)[-1]['valid'])
    checks['reject_rank_deficient_cloud']=not bool(kernel(tf.ones([256,d],dtype),-tf.ones([256,d],dtype))[-1]['valid'])
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    model=parameterized_model(tf.constant(controls()['fit_theta'],dtype),d,d)
    A,H,_,_,Q,R=model[::2]
    y=tf.linspace(tf.constant(-.7,dtype),tf.constant(.4,dtype),d)
    c=tf.linspace(tf.constant(.2,dtype),tf.constant(.9,dtype),d)
    V=.6*tf.eye(d,dtype=dtype)+tf.fill([d,d],tf.constant(.04,dtype));floor=tf.constant(-5.,dtype)
    @tf.function(input_signature=[tf.TensorSpec([8,d],dtype)],jit_compile=True)
    def score_check(points):
        value,score=target_log_score(points,y,A,H,Q,R,c,V,floor)
        h=tf.constant(1e-5,dtype);columns=[]
        for direction in tf.unstack(tf.eye(d,dtype=dtype)):
            plus=target_log_score(points+h*direction,y,A,H,Q,R,c,V,floor)[0]
            minus=target_log_score(points-h*direction,y,A,H,Q,R,c,V,floor)[0]
            columns.append((plus-minus)/(2*h))
        return value,score,tf.stack(columns,axis=1)
    value,score,fd=score_check(x[:8]);error=float(tf.reduce_max(tf.abs(score-fd)).numpy())
    checks['target_score_fd']=error<=1e-6
    target=materialize(dict(points=x[:8],y=y,A=A,H=H,Q=Q,R=R,next_center=c,next_cov=V,
                            next_floor=floor,value=value,score=score,fd=fd,error=error))
    payload=dict(cases=cases,target=target,checks=checks)
    save(directory/'tf.json',payload)
    (directory/'reference-input.R').write_text('input <- '+r_literal(payload)+'\n')
    return dict(checks=checks,passed=all(checks.values()))


def r_run(directory,script,input_path):
    command=['Rscript','--vanilla',script,str(input_path),str(directory)]
    started=time.monotonic()
    with (directory/'R.log').open('x') as log:
        code=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=500).returncode
    record=dict(command=command,exit_code=code,wall_seconds=time.monotonic()-started)
    save(directory/'R-run.json',record)
    if code: raise RuntimeError('Independent R check failed')
    return record


def worker(args):
    directory=ROOT/args.attempt;directory.mkdir(exist_ok=False)
    if args.mode.startswith('gpu'):
        tf,environment=setup('gpu');save(directory/'environment.json',environment)
        if args.mode=='gpu_preflight': result=preflight(tf,directory)
        else:
            from diagnose_iapf_score_regression_downstream import downstream
            result=downstream(tf,directory)
        result['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1': raise RuntimeError('CPU must hide GPU')
        save(directory/'environment.json',dict(device='CPU_reference_GPU_intentionally_hidden',R=subprocess.check_output(['Rscript','--version'],text=True,stderr=subprocess.STDOUT).strip()))
        if args.mode=='cpu_reference':
            prior=ROOT/'attempt01_gpu_preflight'
            if not read(prior/'summary.json')['passed']: raise RuntimeError('Preflight not passed')
            source=prior/'reference-input.R';copied=directory/'reference-input.R';shutil.copy2(source,copied)
            r_run(directory,'docs/benchmarks/check_iapf_score_regression.R',copied)
            result=dict(passed=True,input_sha256=sha(copied))
        else:
            from diagnose_iapf_score_regression_downstream import results
            result=results(directory)
    save(directory/'summary.json',result)
    if not result['passed']: raise RuntimeError('Verification failed')


def launch(args):
    if not args.attempt.isidentifier(): raise ValueError('Unique versioned attempt required')
    manifest=ROOT/(args.attempt+'-launch.json')
    if manifest.exists(): raise FileExistsError(manifest)
    resource='gpu' if args.mode.startswith('gpu') else 'cpu';seconds=600
    budget=read(ROOT/'budget.json')
    if len(list(ROOT.glob('*-launch.json')))>=6 or budget['phase_'+resource+'_seconds']+seconds>3600 or budget['remaining_'+resource+'_seconds']<seconds:
        raise RuntimeError('Attempt or budget cap')
    paths=[PLAN,'docs/benchmarks/diagnose_iapf_score_regression.py',
        'docs/benchmarks/diagnostic_iapf_score_regression_tf.py','docs/benchmarks/check_iapf_score_regression.R',
        'docs/benchmarks/diagnose_iapf_adaptive_consumer.py','docs/benchmarks/reference_iapf_paper.R',
        'bayesfilter/runtime/gpu_memory_policy.py']
    for optional in ['docs/benchmarks/diagnose_iapf_score_regression_downstream.py','docs/benchmarks/check_iapf_score_regression_downstream.R']:
        if (REPO/optional).exists(): paths.append(optional)
        elif args.mode in ('gpu_downstream','cpu_results'): raise FileNotFoundError(optional)
    paths += [str(p.relative_to(REPO)) for p in (REPO/'bayesfilter/score_study').glob('*.py')]
    hashes={}
    for relative in paths:
        source=REPO/relative;dest=ROOT/(args.attempt+'-source')/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[relative]=sha(source)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU if resource=='gpu' else '-1',
        'TF_FORCE_GPU_ALLOW_GROWTH':'true','BAYESFILTER_PRELOAD_CUSTOM_OP':'0','OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1'}
    command=[sys.executable,str(Path(__file__).resolve()),'--mode',args.mode,'--attempt',args.attempt,'--worker']
    record=dict(status='running',command=command,resource=resource,plan=PLAN,result=str(ROOT/'result.md'),
        python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','BAYESFILTER_PRELOAD_CUSTOM_OP',
            'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']},
        seeds='94121+d controls;92101:92104 fresh data; phase12 SHA256 namespaces for pilot/final',sources=hashes)
    save(manifest,record);started=time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try: code=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired: code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed);save(manifest,record)
    budget['phase_'+resource+'_seconds']+=elapsed;budget['remaining_'+resource+'_seconds']-=elapsed;save(ROOT/'budget.json',budget)
    print(json.dumps(dict(exit_code=code,wall_seconds=elapsed)));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--mode',choices=['gpu_preflight','cpu_reference','gpu_downstream','cpu_results'],required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker(args)
    else: sys.exit(launch(args))
