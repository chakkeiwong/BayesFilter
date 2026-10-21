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

gpu_selection=os.environ.get('CUDA_VISIBLE_DEVICES')
from screen_surrogate_hmc_nonlinear_tails import ROOT,clean,save,sha
if '--gpu' in sys.argv:
    if gpu_selection!='GPU-68251639-fe82-8f81-3ccc-2953c32e805b':
        raise ValueError('designated GPU1 UUID required')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_selection
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

PLAN='docs/plans/surrogate-hmc-nonlinear-posterior-2026-09-21.md'
MARKER='-m10-posterior-'


def worker(out):
    cpu_start,wall_start=time.process_time(),time.monotonic()
    manifest=json.loads((out/'run-manifest.json').read_text())
    import tensorflow as tf
    if manifest['gpu']:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest['memory_policy']=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        device='/GPU:0'
    else:
        device='/CPU:0'
    from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference,FP64_MIN_NORMAL
    from bayesfilter.testing.surrogate_hmc_nonlinear_posterior_reference import (
        make_summary,evidence_lower_bound,omission_allowance,PRIOR_SD,MARGINS,FUNCTION_NAMES)
    observations=tf.constant(json.loads(Path(manifest['data_file']).read_text())['raw_observations'],tf.float64)
    lower=evidence_lower_bound(observations)
    report=dict(status='RUNNING',role='independent_model_posterior_reference',
        model='M10',partition='claim-nominal',qualified=False,
        evidence_lower_bound=lower,claim_data_used_for='independent reference only',
        posterior_comparisons='NOT_RUN',heuristic_dominance='NOT_RUN',grids=[])
    save(out/'report.json',report)
    manifest.update(tensorflow=tf.__version__,dtype='float64',jit_compile=True,tf32=False,device=device)
    save(out/'run-manifest.json',manifest)
    # The original 24/40/64 candidate failed. New preparations start with the
    # checked 40/64/96 numerical hypothesis; all accuracy checks still run.
    levels=[(40,20,128),(64,20,128),(96,20,128),(96,32,128),(96,32,64)]
    final_order=levels[2][0]
    reused=manifest.get('reuse_grids',{})
    timings={}
    parity=[]
    forecast_cpu=forecast_wall=60.
    for state_order,normal in ((20,128),(32,128),(32,64)):
        kernel=make_reference(state_order,normal,value_only=True)
        with tf.device(device):clean(kernel(tf.zeros([2],tf.float64),observations))
        measured=[]
        for point in ((0.,0.),(-3.,3.),(3.,-3.)):
            cpu,wall=time.process_time(),time.monotonic()
            with tf.device(device):result=clean(kernel(tf.constant(point,tf.float64)*PRIOR_SD,observations))
            measured.append(dict(cpu=time.process_time()-cpu,wall=time.monotonic()-wall))
            if not result['valid']:
                report.update(status='PROFILE_NUMERICAL_FAILURE',failed_profile=dict(point=point,result=result))
                save(out/'report.json',report)
                return 0
            if manifest['gpu']:
                with tf.device('/CPU:0'):
                    cpu_result=clean(kernel(tf.constant(point,tf.float64)*PRIOR_SD,observations))
                relative=abs(result['likelihood']/cpu_result['likelihood']-1) if cpu_result['likelihood']>0 else math.inf
                prediction=max(abs(a/result['likelihood']-b/cpu_result['likelihood']) for a,b in
                    zip(result['predictive_integrals'],cpu_result['predictive_integrals'])) if min(result['likelihood'],cpu_result['likelihood'])>0 else math.inf
                passed=bool(cpu_result['valid'] and relative<=1e-10 and prediction<=1e-10)
                parity.append(dict(state_order=state_order,normal_order=normal,point=point,
                    relative_likelihood=relative,predictive_probability_difference=prediction,passed=passed))
                if not passed:
                    report.update(status='CPU_GPU_PARITY_FAILED',cpu_gpu_parity=parity)
                    save(out/'report.json',report)
                    return 0
        count=sum(n*n for n,s,g in levels if s==state_order and g==normal and f'{n}-{s}-{g}' not in reused)
        forecast_cpu+=1.5*count*max(x['cpu'] for x in measured)
        forecast_wall+=1.5*count*max(x['wall'] for x in measured)
        timings[f'{state_order}-{normal}']=measured
    if manifest.get('subnormal_guard_repair'):
        with tf.device(device):
            regression=clean(make_reference(32,128,value_only=True)(
                tf.constant([-3.3648781186133125,-3.192114385833288],tf.float64),observations))
        passed=bool(regression['valid'] and 0<=regression['likelihood']<FP64_MIN_NORMAL
                    and all(0<=x<FP64_MIN_NORMAL for x in regression['predictive_integrals']))
        report['subnormal_guard_regression']=dict(passed=passed,result=regression,
            absolute_rounding_floor=FP64_MIN_NORMAL*1e-12)
        if not passed:
            report['status']='SUBNORMAL_GUARD_REGRESSION_FAILED'
            save(out/'report.json',report)
            return 0
    report.update(timings=timings,cpu_gpu_parity=parity,remaining_forecast=dict(cpu=forecast_cpu,wall=forecast_wall))
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
        reuse=reused.get(f'{n}-{state_order}-{normal}')
        if reuse:
            if sha(reuse['path'])!=reuse['sha256']:raise ValueError('saved grid changed')
            saved=json.loads(Path(reuse['path']).read_text())
            if saved['axis']!=axis_list or len(saved['rows'])!=n*n or not all(row['valid'] for row in saved['rows']):
                raise ValueError('saved grid axes, shape or numerical validity changed')
            rows=saved['rows']
        for index,point in (() if reuse else enumerate(itertools.product(axis_list,repeat=2))):
            with tf.device(device):result=clean(kernel(tf.constant(point,tf.float64),observations))
            rows.append(result)
            if not result['valid']:
                save(grid_file,dict(axis=axis_list,rows=rows))
                report.update(status='GRID_NUMERICAL_FAILURE',failed_grid=str(grid_file),
                              failed_point=list(point),failed_index=index)
                save(out/'report.json',report)
                return 0
            if (index+1)%64==0:
                save(grid_file,dict(axis=axis_list,rows=rows))
                if manifest['gpu'] and tf.config.experimental.get_memory_info('GPU:0')['peak']>=4*1024**3:
                    raise RuntimeError('posterior reference exceeded 4 GiB allocator bound')
        save(grid_file,dict(axis=axis_list,rows=rows))
        likelihood=tf.reshape(tf.constant([row['likelihood'] for row in rows],tf.float64),[n,n])
        predictive=tf.reshape(tf.constant([row['predictive_integrals'] for row in rows],tf.float64),[n,n,2])
        results[n,state_order,normal]=(likelihood,predictive)
        report['grids'].append(dict(order=n,state_order=state_order,normal_order=normal,
            path=str(grid_file),sha256=sha(grid_file),zero_likelihoods=sum(row['likelihood']==0 for row in rows),
            maximum_inverse_residual=max(row['inverse_relative_residual'] for row in rows),
            reused_from=reuse['path'] if reuse else None))
        save(out/'report.json',report)
        print('grid',n,state_order,normal,'complete',flush=True)
    _,summary=make_summary(final_order)
    fine=summary(*results[final_order,32,128],tf.zeros([2],tf.float64),
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
    if manifest['gpu']:report['gpu_allocator']=tf.config.experimental.get_memory_info('GPU:0')
    save(out/'report.json',report)
    return 0


def main():
    started,parent_cpu=time.monotonic(),time.process_time()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--gpu',action='store_true',help='designated GPU FP64 reference with CPU parity')
    parser.add_argument('--refine-from',type=Path,help='preserved failed 24/40/64 reference for planned 40/64/96 refinement')
    parser.add_argument('--subnormal-guard-repair',action='store_true',
        help='planned guard repair: recompute all refined grids under remaining stage budget')
    parser.add_argument('--reference-root',type=Path,help='frozen reference/data prerequisite in a master run')
    parser.add_argument('--cpu-budget',type=float,default=850.)
    parser.add_argument('--wall-budget',type=float,default=800.)
    args=parser.parse_args()
    out=args.output.resolve()
    if args.worker: return worker(out)
    if args.subnormal_guard_repair and not args.refine_from:
        parser.error('subnormal repair requires the preserved parameter-refinement source')
    nested=(out.name=='P3-M10-posterior' and out.parent.name.startswith('surrogate-hmc-all-models-'))
    if (MARKER not in out.name and not nested) or '-tests-' in out.name: parser.error('fresh posterior root required')
    if not all(math.isfinite(v) and v>0 for v in (args.cpu_budget,args.wall_budget)):
        parser.error('finite positive resource ceilings required')
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state,m10_posterior_state
    state=m10_reference_state(ROOT,additional_reports=[args.reference_root/'report.json'] if args.reference_root else ())
    if not state['qualified']: raise ValueError('qualified local reference and frozen data required')
    posterior_state=m10_posterior_state(ROOT)
    if posterior_state['qualified']:raise ValueError('reuse the qualified posterior through the master; no automatic rerun')
    total_cpu=total_gpu=0.
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        payload=json.loads(path.read_text())
        charge=payload['charged_this_run'] if 'charged_this_run' in payload else payload['charged']
        if not all(math.isfinite(float(v)) and float(v)>=0 for v in charge.values()):
            raise ValueError('invalid accounting')
        total_cpu+=float(charge['cpu_worker_seconds'])
        total_gpu+=float(charge['gpu_seconds'])
    cpu_budget=min(args.cpu_budget,posterior_state['cpu_worker_seconds'],54000-total_cpu)
    wall_budget=min(args.wall_budget,posterior_state['wall_seconds'])
    if args.gpu:wall_budget=min(wall_budget,226800-total_gpu)
    if min(cpu_budget,wall_budget)<5:raise ValueError('posterior stage budget exhausted')
    reference=Path(state['qualified']['report'])
    data=json.loads(reference.read_text())['partitions']['claim-nominal']
    reuse={}
    if args.refine_from:
        old_root=args.refine_from.resolve()
        old=json.loads((old_root/'report.json').read_text())
        old_manifest=json.loads((old_root/'run-manifest.json').read_text())
        if (old['status']!='POSTERIOR_REFERENCE_REPAIR_REQUIRED' or old_manifest.get('refine_from')
                or old_manifest['data_sha256']!=data['data_sha256']
                or not all(old['checks'][key] is True for key in ('finite_positive','parameter_contraction','predictive_probabilities'))):
            raise ValueError('only the declared parameter-refinement repair is supported')
        if not args.subnormal_guard_repair:
            for name in ('bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py',
                'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
                'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py'):
                if sha(ROOT/name)!=old_manifest['source_hashes'][name]:raise ValueError('saved likelihood source changed')
            for row in old['grids']:
                key=f"{row['order']}-{row['state_order']}-{row['normal_order']}"
                if key in ('40-20-128','64-20-128'):
                    if sha(row['path'])!=row['sha256']:raise ValueError('saved grid hash mismatch')
                    reuse[key]=row
            if set(reuse)!={'40-20-128','64-20-128'}:raise ValueError('missing reusable grids')
    names=[PLAN,'scripts/prepare_surrogate_hmc_nonlinear_posterior.py',
        'scripts/screen_surrogate_hmc_nonlinear_tails.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_posterior_reference.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
        'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py',
        'bayesfilter/runtime/gpu_memory_policy.py']
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
        device='GPU1 FP64 independent reference' if args.gpu else 'CPU-only independent reference; GPU intentionally hidden',
        gpu=args.gpu,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        refine_from=str(args.refine_from.resolve()) if args.refine_from else None,reuse_grids=reuse,
        subnormal_guard_repair=args.subnormal_guard_repair,
        cpu_budget=cpu_budget,wall_budget=wall_budget)
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',qualified=False))
    child_cpu=cpu_budget-(time.process_time()-parent_cpu)-2
    def limit():resource.setrlimit(resource.RLIMIT_CPU,(math.floor(child_cpu),math.ceil(child_cpu)))
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)]+(['--gpu'] if args.gpu else []),
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
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=wall if args.gpu else 0.,cpu_worker_seconds=cpu)))
    print(report['status'],'CPU',round(cpu,3),'wall',round(wall,3),flush=True)
    return code if code>=0 else 2


if __name__=='__main__':raise SystemExit(main())
