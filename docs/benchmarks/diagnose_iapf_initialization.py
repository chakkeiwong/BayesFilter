"""Diagnostic factorial for actual shared iAPF initialization/scaling controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from diagnose_iapf_adaptive_consumer import REPO, setup, save

ROOT=REPO/'docs/plans/artifacts/iapf-initialization-isolation-20260922-01'
INPUT=REPO/'docs/plans/artifacts/iapf-density-scale-20260922-01/attempt01_cpu'
ARMS=[(i,s) for i in ('cloud_moments','log_quadratic') for s in ('native','initial_peak')]


def oracle(tf,directory):
    from bayesfilter.score_study.iapf_fit_tf import bounded_density_fit
    inputs=json.loads((INPUT/'inputs.json').read_text())
    old=json.loads((INPUT/'results.json').read_text())
    shutil.copy2(INPUT/'inputs.json',directory/'inputs.json')
    kernels={};results=[]
    for f,previous in zip(inputs,old):
        d=f['d'];x=tf.constant(f['points'],tf.float64);y=tf.constant(f['targets'],tf.float64)
        c0=tf.constant(f['center'],tf.float64);v0=tf.constant(f['variance'],tf.float64)
        case={k:f[k] for k in ('d','seed','regime')};case['arms']={}
        for initialization,scale in ARMS:
            key=(d,initialization,scale)
            if key not in kernels:
                # Bind categorical controls in the traced closure.
                def build(initialization,scale):
                    @tf.function(input_signature=[tf.TensorSpec([1000,d],tf.float64),tf.TensorSpec([1000],tf.float64)],jit_compile=True)
                    def kernel(x,y):
                        return bounded_density_fit(x,y,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                            max_steps=2000,max_backtracks=30,tolerance=1e-7,floor_ratio=.01,
                            initialization=initialization,objective_scale=scale)
                    return kernel
                kernels[key]=build(initialization,scale)
            start=time.monotonic();c,V,floor,info=kernels[key](x,y);v=tf.linalg.diag_part(V)
            rec={k:a.numpy().tolist() for k,a in info.items()}
            rec.update(center=c.numpy().tolist(),variance=v.numpy().tolist(),log_floor=float(floor.numpy()),
                center_error=float(tf.reduce_max(tf.abs(c-c0)).numpy()),
                variance_error=float(tf.reduce_max(tf.abs(v-v0)).numpy()),
                KL_target_to_guide=float((.5*tf.reduce_sum((v0+(c0-c)**2)/v-1+tf.math.log(v/v0))).numpy()),
                wall_seconds=time.monotonic()-start,trace_count=kernels[key].experimental_get_tracing_count())
            rec['exact_recovery_pass']=rec['valid'] and rec['converged'] and rec['center_error']<=1e-9 and rec['variance_error']<=1e-9 and rec['normalized_shape_residual']<=1e-12
            if initialization=='cloud_moments':
                oldarm=previous['arms']['baseline' if scale=='native' else 'fixed_initial_density_scale']
                errors=[abs(a-b) for k in ('center','variance') for a,b in zip(rec[k],oldarm[k])]
                errors += [abs(rec[k]-oldarm[k]) for k in ('normalized_shape_residual','projected_gradient')]
                rec['prior_max_error']=max(errors)
                rec['prior_parity_pass']=max(errors)<=1e-9 and rec['iterations']==oldarm['iterations']
            case['arms'][initialization+':'+scale]=rec
        results.append(case);save(directory/'results.json',results)
        print(json.dumps({k:case[k] for k in ('d','seed','regime')}),flush=True)
    return dict(cases=len(results),QR_recovery_pass=all(v['exact_recovery_pass'] for x in results for arm,v in x['arms'].items() if arm.startswith('log_quadratic')),
        prior_parity_pass=all(v['prior_parity_pass'] for x in results for arm,v in x['arms'].items() if arm.startswith('cloud_moments')),
        healthy_nonharm_pass=all(max(v['center_error'],v['variance_error'])<=1e-9 for x in results if x['regime']=='healthy' for v in x['arms'].values()))


def worker(args):
    directory=ROOT/args.attempt;directory.mkdir(exist_ok=False)
    tf,environment=setup(args.device)
    with tf.device('/CPU:0' if args.device=='cpu' else '/GPU:0'):
        if args.mode=='oracle': result=oracle(tf,directory)
        else:
            from diagnose_iapf_initialization_consumer import consumer
            result=consumer(tf,directory)
    if args.device=='gpu': environment['allocator']=tf.config.experimental.get_memory_info('GPU:0')
    save(directory/'summary.json',dict(environment=environment,**result))


def launch(args):
    if not args.attempt.isidentifier(): raise ValueError('versioned attempt name required')
    manifest=ROOT/(args.attempt+'-launch.json')
    if manifest.exists(): raise FileExistsError(manifest)
    budget=json.loads((ROOT/'budget.json').read_text())
    seconds=900 if args.device=='cpu' else 600
    if len(list(ROOT.glob('*-launch.json')))>=10: raise RuntimeError('launch cap')
    resource='cpu' if args.device=='cpu' else 'gpu'
    if budget['phase_'+resource+'_seconds']+seconds>(7200 if resource=='cpu' else 3600): raise RuntimeError('phase budget')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--attempt',args.attempt,'--device',args.device,'--mode',args.mode]
    sources=['docs/benchmarks/diagnose_iapf_initialization.py','docs/benchmarks/diagnose_iapf_adaptive_consumer.py',
        'bayesfilter/score_study/iapf_fit_tf.py','bayesfilter/score_study/iapf_adapter.py',
        'bayesfilter/score_study/iapf_scope.py','bayesfilter/score_study/fitted_twist_tf.py',
        'bayesfilter/score_study/gaussian_tf.py','docs/benchmarks/reference_iapf_paper.R',
        'docs/plans/iapf-initialization-isolation-2026-09-22.md']
    consumer_path='docs/benchmarks/diagnose_iapf_initialization_consumer.py'
    if (REPO/consumer_path).exists(): sources.append(consumer_path)
    hashes={}
    for name in sources:
        p=REPO/name;dest=ROOT/(args.attempt+'-source')/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(command=command,device=args.device,mode=args.mode,sources=hashes,status='running',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        plan='docs/plans/iapf-initialization-isolation-2026-09-22.md')
    save(manifest,record)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if args.device=='cpu' else 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b',
         'TF_FORCE_GPU_ALLOW_GROWTH':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
         'MPLCONFIGDIR':'/tmp/iapf-init-matplotlib'}
    started=time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try: code=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
        except subprocess.TimeoutExpired: code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed)
    save(manifest,record)
    budget=json.loads((ROOT/'budget.json').read_text())
    budget['phase_'+resource+'_seconds']+=elapsed;budget['remaining_'+resource+'_seconds']-=elapsed
    save(ROOT/'budget.json',budget)
    print(json.dumps({k:record[k] for k in ('status','exit_code','wall_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu');parser.add_argument('--mode',choices=['oracle','consumer'],default='oracle')
    parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    worker(a) if a.worker else launch(a)
