#!/usr/bin/env python3
"""Bounded independent quadrature-repair screen for M10; no inference."""
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

from screen_surrogate_hmc_nonlinear_tails import ROOT, clean, save, sha

PLAN = 'docs/plans/surrogate-hmc-nonlinear-quadrature-repair-2026-09-21.md'
MARKER = '-m10-quadrature-'


def worker(out):
    manifest = json.loads((out/'run-manifest.json').read_text())
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import grid_reference, PRIOR_SD, source_theta
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    probes = list(itertools.product((-3.,-2.,0.,2.,3.),repeat=2))+[(-6.,0.),(6.,0.),(0.,-6.),(0.,6.)]
    local = list(itertools.product((-1.,0.,1.),repeat=2))
    probes += [point for point in local if point not in probes]
    levels = [(12,64),(20,64),(20,128),(32,128)]
    report = dict(status='RUNNING',role='quadrature_repair_screen_only',probes_in_prior_sd=probes,
        levels=levels,rows=[],checks={},comparisons=[],posterior_qualified=False,
        surrogate_hmc_tested=False,claim_data_used=False)
    manifest.update(tensorflow=tf.__version__,dtype='float64',jit_compile=True)
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',report)
    outputs = {}
    for order,normal in levels:
        kernel = make_reference(order,normal)
        for name,row in manifest['datasets'].items():
            observations = tf.constant(json.loads(Path(row['path']).read_text())['raw_observations'],tf.float64)
            for index,point in enumerate(probes):
                start = time.monotonic()
                result = clean(kernel(tf.constant(point,tf.float64)*PRIOR_SD,observations))
                result.update(partition=name,probe=index,level=[order,normal],wall_seconds=time.monotonic()-start)
                report['rows'].append(result)
                outputs[name,index,order,normal] = result
                save(out/'report.json',report)
            print(name,order,'complete',flush=True)
    reference = grid_reference(513,9.)
    for name,row in manifest['datasets'].items():
        observations = tf.constant(json.loads(Path(row['path']).read_text())['raw_observations'],tf.float64)
        for index,point in enumerate(probes):
            a,b,c = (outputs[name,index,20,128],outputs[name,index,32,128],
                     outputs[name,index,20,64])
            valid = bool(a['valid'] and b['valid'] and c['valid'])
            differences = dict(value=None,score=None)
            normal_differences = dict(value=None,score=None)
            if valid:
                differences = dict(value=abs(a['log_likelihood']-b['log_likelihood']),
                    score=max(abs(x-y) for x,y in zip(a['score'],b['score'])))
                normal_differences = dict(value=abs(a['log_likelihood']-c['log_likelihood']),
                    score=max(abs(x-y) for x,y in zip(a['score'],c['score'])))
            passed = (valid and differences['value']<=1e-6 and differences['score']<=1e-5
                      and normal_differences['value']<=1e-6 and normal_differences['score']<=1e-5)
            overlap = None
            if point in local:
                exact = clean(reference(tf.constant(point,tf.float64)*PRIOR_SD,observations))
                overlap = dict(value=abs(b['log_likelihood']-exact['log_likelihood']),
                    score=max(abs(x-y) for x,y in zip(b['score'],exact['score']))) if valid else None
                passed = passed and overlap['value']<=1e-6 and overlap['score']<=1e-5
            report['comparisons'].append(dict(partition=name,probe=index,passed=passed,
                refinement=differences,normal_refinement=normal_differences,overlap=overlap))
    observations = tf.constant(json.loads(Path(manifest['datasets']['calibration-nominal']['path']).read_text())['raw_observations'],tf.float64)
    a,_,_,_,_,_,p,_,q,_,_,_ = parameterized_model(source_theta(tf.zeros([2],tf.float64)),1,1)
    switch = float(tf.math.log((a[0,0]+tf.cast(.2,tf.float64))*tf.sqrt(p[0,0]/q[0,0])))
    switch_rows = []
    for delta in (-1e-6,0.,1e-6):
        point = tf.constant([switch+delta,0.],tf.float64)
        first = clean(make_reference(20,128,'initial')(point,observations))
        second = clean(make_reference(20,128,'inverse')(point,observations))
        passed = bool(first['valid'] and second['valid'])
        difference = dict(value=abs(first['log_likelihood']-second['log_likelihood']),
            score=max(abs(x-y) for x,y in zip(first['score'],second['score']))) if passed else None
        passed = passed and difference['value']<=1e-6 and difference['score']<=1e-5
        switch_rows.append(dict(point=clean(point),initial=first,inverse=second,difference=difference,passed=passed))
    report['switch_rows'] = switch_rows
    report['checks'] = dict(all_finite_valid=all(row['valid'] for row in report['rows']),
        refinement_and_overlap=all(row['passed'] for row in report['comparisons']),
        convolution_switch=all(row['passed'] for row in switch_rows))
    report['status'] = 'QUADRATURE_REPAIR_SCREEN_PASSED' if all(report['checks'].values()) else 'QUADRATURE_REPAIR_REQUIRED'
    report['failed_probes'] = sum(not row['passed'] for row in report['comparisons'])
    save(out/'report.json',report)
    return 0


def main():
    started,parent_cpu = time.monotonic(),time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    if MARKER not in out.name or '-tests-' in out.name:
        parser.error('versioned M10 quadrature execution root required')
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state
    state = m10_reference_state(ROOT)
    if not state['qualified']:
        raise ValueError('unchanged local reference required')
    stage_cpu=stage_wall=total_cpu=0.
    attempts=0
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        payload=json.loads(path.read_text())
        charge=payload['charged_this_run'] if 'charged_this_run' in payload else payload['charged']
        if not all(math.isfinite(float(value)) and float(value)>=0 for value in charge.values()):
            raise ValueError('invalid prior accounting')
        total_cpu+=float(charge['cpu_worker_seconds'])
        if MARKER in path.parent.name:
            stage_cpu+=float(charge['cpu_worker_seconds'])
            if '-tests-' not in path.parent.name:
                attempts+=1
                old=json.loads((path.parent/'report.json').read_text())
                stage_wall+=float(old['wall_seconds'])
                if old['status'] in ('QUADRATURE_REPAIR_SCREEN_PASSED','QUADRATURE_REPAIR_REQUIRED'):
                    raise ValueError('completed repair screen requires interpretation, not an automatic retry')
    cpu_budget=min(200-stage_cpu,54000-total_cpu,state['family_cpu_remaining'])
    wall_budget=180-stage_wall
    if attempts>=2 or min(cpu_budget,wall_budget)<5:
        raise ValueError('quadrature repair allocation exhausted')
    reference=Path(state['qualified']['report'])
    datasets={name:dict(path=row['data_file'],sha256=row['data_sha256'])
        for name,row in json.loads(reference.read_text())['partitions'].items()
        if name.startswith(('calibration-','validation-'))}
    if set(datasets)!={f'{p}-{r}' for p in ('calibration','validation') for r in ('nominal','stress')}:
        raise ValueError('four frozen non-claim partitions required')
    names=[PLAN,'scripts/qualify_surrogate_hmc_nonlinear_quadrature.py',
        'scripts/screen_surrogate_hmc_nonlinear_tails.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_quadrature.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
        'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py']
    sources={name:sha(ROOT/name) for name in names}
    out.mkdir(parents=True,exist_ok=False)
    for name in names:
        target=out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    manifest=dict(command=[sys.executable,*sys.argv],environment=sys.prefix,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan=PLAN,result_file=str(out/'report.json'),source_hashes=sources,datasets=datasets,
        seeds={name:json.loads(Path(row['path']).read_text())['seed'] for name,row in datasets.items()},
        data_version='frozen M10 FP32 calibration/validation; claim untouched',
        device='CPU-only independent reference; GPU intentionally hidden',cuda_visible_devices='-1',
        cpu_budget=cpu_budget,wall_budget=wall_budget,reference_report_sha256=sha(reference))
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',posterior_qualified=False,rows=[]))
    child_cpu=cpu_budget-(time.process_time()-parent_cpu)-2.
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(math.floor(child_cpu),math.ceil(child_cpu)))
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit)
        try: code=process.wait(timeout=max(1.,wall_budget-(time.monotonic()-started)-1.))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code=124
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    wall=time.monotonic()-started
    report=json.loads((out/'report.json').read_text())
    manifest=json.loads((out/'run-manifest.json').read_text())
    if code:
        report['status']='RESOURCE_LIMIT' if code in (-24,-9,124) else 'EXECUTION_FAILED'
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):
        report['status'],code='SOURCE_DRIFT',2
    if any(sha(row['path'])!=row['sha256'] for row in datasets.values()):
        report['status'],code='DATA_DRIFT',2
    report.update(exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    manifest.update(status=report['status'],exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    save(out/'report.json',report)
    save(out/'run-manifest.json',manifest)
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=0.,cpu_worker_seconds=cpu)))
    print(report['status'],'CPU',round(cpu,3),'wall',round(wall,3),flush=True)
    return code if code>=0 else 2


if __name__=='__main__':
    raise SystemExit(main())
