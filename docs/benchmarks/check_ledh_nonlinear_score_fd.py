#!/usr/bin/env python3
"""Explicit CPU/GPU finite-program derivative diagnostic; no filter-accuracy claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.update(TF_FORCE_GPU_ALLOW_GROWTH='true',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=float,default=1000)
    parser.add_argument('--models',nargs='+',choices=('predator_prey','sir_d18'),default=['predator_prey','sir_d18'])
    parser.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--steps',nargs='+',type=float,default=[1e-5,5e-6])
    args=parser.parse_args()
    if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=args.device=='gpu')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim import sqmc_campaign_tf as common
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec,trace_kernel
    from docs.benchmarks.run_ledh_nonlinear_master import BASE,arm_settings,json_safe
    from docs.benchmarks.run_ledh_zhao_horizons import OUT,PLAN,dump,read
    manifest=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan=PLAN,
        python=sys.executable,tensorflow=tf.__version__,cpu_only=args.device=='cpu',gpu_status='intentionally hidden' if args.device=='cpu' else 'trusted escalated GPU',memory_policy=memory,
        jit_compile=True,dtype='float64',particles=144,horizon=10,design_seed=261006101,
        steps=args.steps,normalized_tolerance=2e-4,claim='derivative of the same fixed finite program only')
    dump(args.output/'manifest.json',manifest)
    rows=[]
    for model in args.models:
        spec=NonlinearSQMCSpec(model)
        theta=spec.default_theta(tf.float64)
        obs_record=read(OUT/'inputs'/f'{model}-T10'/'dataset.json')
        observations=tf.constant(obs_record['observations'],tf.float64)
        digest=hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
        if digest!=obs_record['observation_sha256']: raise ValueError('input hash mismatch')
        inputs=common.random_inputs('iid_dual_cap',261006101,144,spec.dimension,10,tf.float64,jit_compile=True)
        for policy in ('ancestor','marginal_mixture'):
            controls,design=arm_settings('guarded_pairwise',dict(BASE,importance_weight_policy=policy))
            def evaluate(point):
                if time.monotonic()-started>args.timeout_seconds: raise TimeoutError('bounded score diagnostic')
                value,score,valid=common.value_and_score(spec,'iid_dual_cap',controls,point,observations,
                    261006101,144,jit_compile=True,inputs=inputs,reset_design_kind=design)
                if not bool(valid): raise ValueError('invalid finite program')
                return float(value),score
            value,score=evaluate(theta)
            scale=tf.maximum(tf.abs(theta),tf.ones_like(theta))
            differences=[]
            for h in manifest['steps']:
                fd=[]
                for k in range(spec.parameter_count):
                    shift=tf.one_hot(k,spec.parameter_count,dtype=tf.float64)*scale[k]*h
                    plus,_=evaluate(theta+shift); minus,_=evaluate(theta-shift)
                    fd.append((plus-minus)/(2*h*float(scale[k])))
                differences.append(tf.constant(fd,tf.float64))
            errors=[tf.abs(fd-score)/(1+tf.abs(score)) for fd in differences]
            traced=trace_kernel(spec,'iid_dual_cap',controls,144,10,tf.float64,jit_compile=True,reset_design_kind=design)
            trace_value,trace_score,trace=traced(theta,*inputs,observations)
            trace_parity=abs(float(trace_value)-value)<=1e-10*(1+abs(value)) and abs(float(trace_score[0]-score[0]))<=1e-10*(1+abs(float(score[0])))
            row=dict(model=model,policy=policy,observation_sha256=digest,log_likelihood=value,score=score.numpy().tolist(),
                finite_differences=[v.numpy().tolist() for v in differences],normalized_errors=[v.numpy().tolist() for v in errors],
                step_stability=(tf.abs(differences[0]-differences[1])/(1+tf.abs(score))).numpy().tolist(),
                trace_value=float(trace_value),trace_score_coordinate_zero=float(trace_score[0]),
                trace_value_difference=float(trace_value)-value,trace_score_difference=float(trace_score[0]-score[0]),
                trace_parity=trace_parity,passes=trace_parity and float(tf.reduce_max(errors[-1]))<=2e-4)
            h=manifest['steps'][-1]
            shift=tf.one_hot(0,spec.parameter_count,dtype=tf.float64)*scale[0]*h
            plus,_,plus_trace=traced(theta+shift,*inputs,observations)
            minus,_,minus_trace=traced(theta-shift,*inputs,observations)
            row['trace_finite_difference_coordinate_zero']=(float(plus)-float(minus))/(2*h*float(scale[0]))
            branch_keys=('higher_moment_moment_safety_trials','higher_moment_moment_safety_rejected_steps',
                'higher_moment_moment_safety_minimum_step','higher_moment_moment_safety_final_rejected')
            row['trace_branch_changes']=[dict(time=t+1,key=k,base=base[k].numpy().tolist(),
                plus=above[k].numpy().tolist(),minus=below[k].numpy().tolist())
                for t,(base,above,below) in enumerate(zip(trace,plus_trace,minus_trace)) for k in branch_keys
                if base[k].numpy().tolist()!=above[k].numpy().tolist() or base[k].numpy().tolist()!=below[k].numpy().tolist()]
            rows.append(row)
            dump(args.output/f'trace-{model}-{policy}.json',json_safe([{k:v.numpy().tolist() for k,v in r.items()} for r in trace]))
            dump(args.output/'rows.json',rows)
            print(json.dumps(dict(model=model,policy=policy,passes=row['passes'],max_normalized_error=float(tf.reduce_max(errors[-1])))),flush=True)
    dump(args.output/'result.json',dict(status='complete',passes=all(r['passes'] for r in rows),rows=rows,wall_seconds=time.monotonic()-started))
    return int(not all(r['passes'] for r in rows))

if __name__=='__main__': raise SystemExit(main())
