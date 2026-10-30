#!/usr/bin/env python3
"""Bounded M10 independent posterior qualification; no HMC execution."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from screen_surrogate_hmc_nonlinear_tails import ROOT,clean,save,sha

PLAN='docs/plans/surrogate-hmc-nonlinear-posterior-2026-09-21.md'
MARKER='-m10-posterior-'


def worker(out):
    cpu_start,wall_start=time.process_time(),time.monotonic()
    manifest=json.loads((out/'run-manifest.json').read_text())
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference
    from bayesfilter.testing.surrogate_hmc_nonlinear_posterior_reference import (
        make_summary,evidence_lower_bound,omission_allowance,PRIOR_SD,MARGINS,FUNCTION_NAMES)
    observations=tf.constant(json.loads(Path(manifest['data_file']).read_text())['raw_observations'],tf.float64)
    lower=evidence_lower_bound(observations)
    report=dict(status='RUNNING',role='independent_model_posterior_reference',
        model='M10',partition='claim-nominal',qualified=False,
        evidence_lower_bound=lower,claim_data_used_for='independent reference only',
        posterior_comparisons='NOT_RUN',heuristic_dominance='NOT_RUN',grids=[])
    save(out/'report.json',report)
    manifest.update(tensorflow=tf.__version__,dtype='float64',jit_compile=True)
    save(out/'run-manifest.json',manifest)
    levels=[(24,20,128),(40,20,128),(64,20,128),(64,32,128),(64,32,64)]
    timings={}
    forecast_cpu=forecast_wall=60.
    for state_order,normal in ((20,128),(32,128),(32,64)):
        kernel=make_reference(state_order,normal,value_only=True)
        clean(kernel(tf.zeros([2],tf.float64),observations))
        measured=[]
        for point in ((0.,0.),(-3.,3.),(3.,-3.)):
            cpu,wall=time.process_time(),time.monotonic()
            result=clean(kernel(tf.constant(point,tf.float64)*PRIOR_SD,observations))
            measured.append(dict(cpu=time.process_time()-cpu,wall=time.monotonic()-wall))
            if not result['valid']:
                report.update(status='PROFILE_NUMERICAL_FAILURE',failed_profile=dict(point=point,result=result))
                save(out/'report.json',report)
                return 0
        count=sum(n*n for n,s,g in levels if s==state_order and g==normal)
        forecast_cpu+=1.5*count*max(x['cpu'] for x in measured)
        forecast_wall+=1.5*count*max(x['wall'] for x in measured)
        timings[f'{state_order}-{normal}']=measured
    report.update(timings=timings,remaining_forecast=dict(cpu=forecast_cpu,wall=forecast_wall))
    cpu_left=manifest['cpu_budget']-(time.process_time()-cpu_start)-5
    wall_left=manifest['wall_budget']-(time.monotonic()-wall_start)-5
    if forecast_cpu>cpu_left or forecast_wall>wall_left:
        report.update(status='DEFERRED_BY_COST',remaining_at_forecast=dict(cpu=cpu_left,wall=wall_left))
        save(out/'report.json',report)
        return 0
    save(out/'report.json',report)
    results={}
    for n,state_order,normal in levels:
        axis,_=make_summary(n)
        axis_list=clean(axis)
        kernel=make_reference(state_order,normal,value_only=True)
        rows=[]
        grid_file=out/f'grid-{n}-{state_order}-{normal}.json'
        for index,point in enumerate(itertools.product(axis_list,repeat=2)):
            result=clean(kernel(tf.constant(point,tf.float64),observations))
            rows.append(result)
            if not result['valid']:
                save(grid_file,dict(axis=axis_list,rows=rows))
                report.update(status='GRID_NUMERICAL_FAILURE',failed_grid=str(grid_file),
                              failed_point=list(point),failed_index=index)
                save(out/'report.json',report)
                return 0
            if (index+1)%64==0: save(grid_file,dict(axis=axis_list,rows=rows))
        save(grid_file,dict(axis=axis_list,rows=rows))
        likelihood=tf.reshape(tf.constant([row['likelihood'] for row in rows],tf.float64),[n,n])
        predictive=tf.reshape(tf.constant([row['predictive_integrals'] for row in rows],tf.float64),[n,n,2])
        results[n,state_order,normal]=(likelihood,predictive)
        report['grids'].append(dict(order=n,state_order=state_order,normal_order=normal,
            path=str(grid_file),sha256=sha(grid_file),zero_likelihoods=sum(row['likelihood']==0 for row in rows),
            maximum_inverse_residual=max(row['inverse_relative_residual'] for row in rows)))
        save(out/'report.json',report)
        print('grid',n,state_order,normal,'complete',flush=True)
    _,summary=make_summary(64)
    fine=summary(*results[64,32,128],tf.zeros([2],tf.float64),
                 tf.constant([PRIOR_SD]*2,tf.float64),tf.zeros([2,3],tf.float64))
    means,scales,cuts=fine['means'],fine['scales'],fine['cuts']
    summaries=[]
    for level in levels:
        _,calculate=make_summary(level[0])
        summaries.append(calculate(*results[level],means,scales,cuts))
    expectations=tf.stack([item['expectations'] for item in summaries])
    parameter_changes=tf.abs(expectations[1:3]-expectations[:2])
    state_change=tf.abs(expectations[3]-expectations[2])
    convolution_change=tf.abs(expectations[4]-expectations[3])
    bounds=omission_allowance(clean(means),clean(scales),clean(expectations[3]),lower['lower_bound'])
    allowance=2*(tf.reduce_max(parameter_changes,axis=0)+state_change+convolution_change)+tf.constant(bounds['function_allowances'],tf.float64)
    evidence=tf.stack([item['evidence'] for item in summaries])
    relative_changes=tf.abs((evidence[1:]-evidence[:-1])/evidence[1:])
    precision=allowance<=tf.constant(MARGINS,tf.float64)/10
    contraction=parameter_changes[1]<=.5*parameter_changes[0]+1e-5
    checks=dict(finite_positive=bool(tf.reduce_all(tf.math.is_finite(expectations)) &
        tf.reduce_all(tf.math.is_finite(evidence)&(evidence>lower['lower_bound'])) &
        tf.reduce_all(tf.math.is_finite(scales)&(scales>0))),
        normalization_refinement=bool(tf.reduce_all(relative_changes[1:]<=1e-6)),
        function_precision=bool(tf.reduce_all(precision)),parameter_contraction=bool(tf.reduce_all(contraction)),
        marginal_interpolation=all(bool(row['interpolation_valid']) for row in summaries[2:]),
        predictive_probabilities=bool(tf.reduce_all((expectations[:,-2:]>=0)&(expectations[:,-2:]<=1+1e-12))))
    report.update(checks=checks,qualified=all(checks.values()),function_names=FUNCTION_NAMES,
        margins=MARGINS,posterior_mean=means,posterior_sd=scales,marginal_quantiles=cuts,
        grid_expectations=expectations,reference_expectations=expectations[3],
        normalization=evidence,normalization_relative_changes=relative_changes,
        parameter_function_changes=parameter_changes,state_function_change=state_change,
        convolution_function_change=convolution_change,function_allowance=allowance,
        precision_passed=precision,contraction_passed=contraction,omission_bounds=bounds,
        allowance_is_rigorous_bound=False,
        interpolation=[{k:row[k] for k in ('interpolation_valid','minimum_cdf_increment','minimum_interpolated_density')} for row in summaries])
    report['status']='QUALIFIED_MODEL_POSTERIOR_REFERENCE' if report['qualified'] else 'POSTERIOR_REFERENCE_REPAIR_REQUIRED'
    save(out/'report.json',report)
    return 0


def main():
    started,parent_cpu=time.monotonic(),time.process_time()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    out=args.output.resolve()
    if args.worker: return worker(out)
    if MARKER not in out.name or '-tests-' in out.name: parser.error('fresh posterior root required')
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state
    state=m10_reference_state(ROOT)
    if not state['qualified']: raise ValueError('qualified local reference and frozen data required')
    stage_cpu=stage_wall=total_cpu=0.
    attempts=0
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        payload=json.loads(path.read_text())
        charge=payload['charged_this_run'] if 'charged_this_run' in payload else payload['charged']
        if not all(math.isfinite(float(v)) and float(v)>=0 for v in charge.values()):
            raise ValueError('invalid accounting')
        total_cpu+=float(charge['cpu_worker_seconds'])
        if MARKER in path.parent.name and '-tests-' not in path.parent.name:
            stage_cpu+=float(charge['cpu_worker_seconds']);attempts+=1
            previous=json.loads((path.parent/'report.json').read_text())
            stage_wall+=float(previous['wall_seconds'])
            if previous['status'] not in ('EXECUTION_FAILED','RESOURCE_LIMIT'):
                raise ValueError('completed posterior attempt requires interpretation')
    cpu_budget=min(700-stage_cpu,54000-total_cpu,state['family_cpu_remaining'])
    wall_budget=600-stage_wall
    if attempts>=2 or min(cpu_budget,wall_budget)<5: raise ValueError('posterior stage budget exhausted')
    reference=Path(state['qualified']['report'])
    data=json.loads(reference.read_text())['partitions']['claim-nominal']
    names=[PLAN,'scripts/prepare_surrogate_hmc_nonlinear_posterior.py',
        'scripts/screen_surrogate_hmc_nonlinear_tails.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_posterior_reference.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
        'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py']
    sources={name:sha(ROOT/name) for name in names}
    out.mkdir(parents=True,exist_ok=False)
    for name in names:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    if sha(data['data_file'])!=data['data_sha256']: raise ValueError('claim data hash mismatch')
    manifest=dict(command=[sys.executable,*sys.argv],environment=sys.prefix,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan=PLAN,result_file=str(out/'report.json'),source_hashes=sources,
        data_file=data['data_file'],data_sha256=data['data_sha256'],
        seed=json.loads(Path(data['data_file']).read_text())['seed'],
        data_version='frozen nominal M10 FP32 claim observations',
        device='CPU-only independent reference; GPU intentionally hidden',cuda_visible_devices='-1',
        cpu_budget=cpu_budget,wall_budget=wall_budget)
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',qualified=False))
    child_cpu=cpu_budget-(time.process_time()-parent_cpu)-2
    def limit():resource.setrlimit(resource.RLIMIT_CPU,(math.floor(child_cpu),math.ceil(child_cpu)))
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit)
        try: code=process.wait(timeout=max(1.,wall_budget-(time.monotonic()-started)-1))
        except subprocess.TimeoutExpired:process.kill();process.wait();code=124
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    wall=time.monotonic()-started
    report=json.loads((out/'report.json').read_text())
    manifest=json.loads((out/'run-manifest.json').read_text())
    if code:report['status']='RESOURCE_LIMIT' if code in (-24,-9,124) else 'EXECUTION_FAILED'
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):report['status'],code='SOURCE_DRIFT',2
    if sha(data['data_file'])!=data['data_sha256']:report['status'],code='DATA_DRIFT',2
    report.update(exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    manifest.update(status=report['status'],exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    save(out/'report.json',report);save(out/'run-manifest.json',manifest)
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=0.,cpu_worker_seconds=cpu)))
    print(report['status'],'CPU',round(cpu,3),'wall',round(wall,3),flush=True)
    return code if code>=0 else 2


if __name__=='__main__':raise SystemExit(main())
