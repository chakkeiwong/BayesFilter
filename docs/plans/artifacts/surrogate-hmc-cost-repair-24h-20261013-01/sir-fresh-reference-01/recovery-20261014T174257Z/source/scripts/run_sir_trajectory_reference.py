#!/usr/bin/env python3
"""Bounded diagnostic exact-weight trajectory reference for the saved SIR point."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sir_fresh_reference as shared


def summarize(rows):
    from docs.benchmarks.run_nonlinear_bootstrap_reference import combine
    pooled=combine(rows)
    omitted=[combine(rows[:j]+rows[j+1:]) for j in range(len(rows))]
    n=len(rows)
    mcse=[math.sqrt((n-1)/n*sum((x[k]-statistics.mean(r[k] for r in omitted))**2
                              for x in omitted)) for k in range(len(pooled))]
    w=[math.exp(r['log_likelihood']-max(x['log_likelihood'] for x in rows)) for r in rows]
    return dict(log_likelihood=pooled[0],score=pooled[1:],
                jackknife_mcse_log_likelihood=mcse[0],jackknife_mcse_score=mcse[1:],
                replications=n,between_replication_likelihood_ess=sum(w)**2/sum(x*x for x in w),
                minimum_importance_ess=min(r['importance_ess'] for r in rows),
                maximum_weight=max(r['maximum_weight'] for r in rows))


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['TF_NUM_INTRAOP_THREADS']='2'
    os.environ['TF_NUM_INTEROP_THREADS']='2'
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.testing.nonlinear_trajectory_fisher_reference_tf import make_trajectory_fisher_kernel
    from docs.benchmarks.run_ledh_nonlinear_master import prepare_dataset
    args.output.mkdir(exist_ok=False)
    job=shared.read(shared.OUTPUT/'input/job.json')
    spec=NonlinearSQMCSpec('sir_d18')
    observations,data=prepare_dataset(tf,spec,dict(job,dataset_file=str(shared.OUTPUT/'input/dataset.json')),tf.float64)
    theta=tf.constant(job['theta_points'][0],tf.float64)
    files=[Path(__file__),ROOT/'bayesfilter/testing/nonlinear_trajectory_fisher_reference_tf.py',
           ROOT/'bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py',
           ROOT/'bayesfilter/highdim/sqmc_nonlinear_tf.py',ROOT/'bayesfilter/highdim/models.py',
           ROOT/'bayesfilter/highdim/ledh_canonical_models_tf.py']
    manifest=dict(command=sys.argv,plan=str(shared.PLAN),git_commit=subprocess.check_output(
                  ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
                  environment=sys.prefix,tensorflow=tf.__version__,cpu_only=False,jit_compile=True,
                  dtype='float64',tf32=False,memory_policy=memory,device='/GPU:0',
                  gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices('GPU')],
                  particles=args.particles,seeds=args.seeds,theta=theta.numpy().tolist(),
                  target_id=spec.target_id,observation_sha256=data['observation_sha256'],
                  source_sha256={str(p.relative_to(ROOT)):shared.digest(p) for p in files},
                  evidence_role='independent_exact_weight_trajectory_reference_not_exact_oracle',
                  score_method='analytical_complete_data_Fisher_score',status='RUNNING')
    shared.save(args.output/'manifest.json',manifest)
    start=time.monotonic();rows=[]
    try:
        kernel=make_trajectory_fisher_kernel(spec.model,spec.initial_mean(tf.float64),3,9,50,args.particles)
        for seed in args.seeds:
            tic=time.monotonic()
            raw=kernel(theta,observations,tf.constant(seed))
            row={k:(v.numpy().tolist()) for k,v in raw.items()}
            if not row['valid'] or not all(math.isfinite(x) for x in [row['log_likelihood'],*row['score']]):
                raise ValueError('invalid trajectory reference result')
            row.update(seed=seed,particles=args.particles,wall_seconds=time.monotonic()-tic)
            rows.append(row);shared.save(args.output/'replications.json',rows)
            print(json.dumps(row),flush=True)
        summary=dict(particles=args.particles,**summarize(rows))
        shared.save(args.output/'summary.json',[summary]);manifest['status']='COMPLETE'
    finally:
        manifest.update(wall_seconds=time.monotonic()-start,allocator=tf.config.experimental.get_memory_info('GPU:0'))
        shared.save(args.output/'manifest.json',manifest)


def execute(args):
    if shared.read(shared.OUTPUT/'trajectory-regression.json')['exit_code']!=0:
        raise ValueError('trajectory reference regression failed')
    state=shared.read(shared.OUTPUT/'state.json')
    if any(r['status']=='RUNNING' for r in state['runs']):
        raise ValueError('bootstrap reference still running')
    for n,count,seed0 in [(8192,8,261015101),(32768,16,261015201),(131072,16,261015301)]:
        if any(r.get('method')=='trajectory' and r['particles']==n and r['status']=='COMPLETE' for r in state['runs']):
            continue
        ledger=shared.read(shared.CAMPAIGN/'budget.json')
        if any(r.get('status')=='RUNNING' for r in ledger['attempts']):
            raise ValueError('another campaign worker active')
        left=min(shared.CAP-state['used_seconds'],ledger['authorized_gpu_wall_seconds']-ledger['used_seconds'])
        if left<60 or len(state['runs'])>=6 or len(ledger['attempts'])>=ledger['max_attempts']:
            state['status']='BUDGET_EXHAUSTED';break
        timeout=int(min(left,1800))
        output=shared.OUTPUT/('trajectory-%d-attempt-%02d'%(n,len(state['runs'])+1))
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--particles',str(n),
             '--seeds',*map(str,range(seed0,seed0+count)),'--output',str(output)]
        entry=dict(model='sir_d18_independent_reference',method='trajectory',particles=n,
                   output=str(output),command=cmd,status='RUNNING',started_utc=shared.now(),reserved_seconds=timeout)
        ledger['attempts'].append(entry.copy());ledger['updated_utc']=shared.now()
        shared.save(shared.CAMPAIGN/'budget.json',ledger)
        state['runs'].append(entry.copy());state['status']='RUNNING'
        shared.save(shared.OUTPUT/'state.json',state)
        tic=time.monotonic();code=None
        try:
            with (shared.OUTPUT/(output.name+'.log')).open('w') as log:
                code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code=124
        finally:
            seconds=time.monotonic()-tic
            complete=code==0 and (output/'summary.json').exists()
            patch=dict(status='COMPLETE' if complete else 'FAILED',exit_code=code,wall_seconds=seconds,ended_utc=shared.now())
            state['runs'][-1].update(patch);state['used_seconds']+=seconds
            state['status']=patch['status'];shared.save(shared.OUTPUT/'state.json',state)
            ledger=shared.read(shared.CAMPAIGN/'budget.json')
            matches=[r for r in ledger['attempts'] if r.get('output')==str(output)]
            if len(matches)!=1:raise ValueError('attempt accounting mismatch')
            matches[0].update(patch)
            ledger['used_seconds']=sum(float(r.get('wall_seconds',r.get('reserved_seconds',0))) for r in ledger['attempts'])
            ledger['remaining_seconds']=ledger['authorized_gpu_wall_seconds']-ledger['used_seconds']
            ledger['updated_utc']=shared.now();ledger['next_action']='Assess independent SIR references and exact saved OT arms.'
            shared.save(shared.CAMPAIGN/'budget.json',ledger)
        print(json.dumps(patch),flush=True)
        if not complete:break
    else:state['status']='REFERENCE_LADDERS_COMPLETE'
    shared.save(shared.OUTPUT/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['execute','worker'])
    parser.add_argument('--particles',type=int)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    globals()[args.action](args)
