#!/usr/bin/env python3
"""CPU-only diagnostic of M10 reference tails; cannot authorize inference."""
import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/plans/surrogate-hmc-nonlinear-tail-screen-2026-09-21.md'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'numpy'):
        return clean(value.numpy().tolist())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def worker(out):
    manifest = json.loads((out/'run-manifest.json').read_text())
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import grid_reference, PRIOR_SD, BASE_THETA
    manifest.update(tensorflow=tf.__version__, jit_compile=True, dtype='float64')
    save(out/'run-manifest.json', manifest)
    probes = list(itertools.product((-3.,-2.,0.,2.,3.), repeat=2)) + [(-6.,0.),(6.,0.),(0.,-6.),(0.,6.)]
    grids = [(257,9.),(513,9.),(1025,9.),(1537,13.5)]
    constant = math.exp(-2*BASE_THETA[2])/(2*math.pi)
    scale = constant*math.exp(2*PRIOR_SD**2)
    envelopes = []
    for k in range(1,8):
        q_tail = math.erfc(k/math.sqrt(2.))
        r_tail = .5*math.erfc((k-2*PRIOR_SD)/math.sqrt(2.)) + .5*math.erfc((k+2*PRIOR_SD)/math.sqrt(2.))
        envelopes.append(dict(k=k, bound=scale*(q_tail+r_tail-q_tail*r_tail)))
    report = dict(status='RUNNING', classification='reference_tail_screen_only',
        prior_sd=PRIOR_SD, grids=grids, probes_in_prior_sd=probes,
        envelope_constant=constant, envelope_total_integral=scale,
        outside_box_unnormalized_mass_bounds=envelopes, rows=[], comparisons=[],
        posterior_qualified=False, surrogate_hmc_tested=False, claim_data_used=False)
    save(out/'report.json', report)
    results = {}
    for points,radius in grids:
        kernel = grid_reference(points,radius)
        for partition, data_info in manifest['datasets'].items():
            if sha(data_info['path']) != data_info['sha256']:
                raise ValueError('frozen reference data changed')
            dataset = json.loads(Path(data_info['path']).read_text())
            observations = tf.constant(dataset['raw_observations'],tf.float64)
            for index, probe in enumerate(probes):
                begin = time.monotonic()
                result = kernel(tf.constant(probe,tf.float64)*PRIOR_SD,observations)
                value = float(result['log_likelihood'])
                score = result['score'].numpy().tolist()
                row = dict(partition=partition, probe=index, grid=[points,radius],
                    log_likelihood=value, score=score, domain_loss=float(result['domain_loss']),
                    posterior_edge=float(result['posterior_edge']), wall_seconds=time.monotonic()-begin)
                row['finite'] = all(math.isfinite(v) for v in (value,*score,row['domain_loss'],row['posterior_edge']))
                report['rows'].append(clean(row))
                results[partition,index,points] = row
                save(out/'report.json', report)
            print(partition,points,'complete',flush=True)
    for partition in manifest['datasets']:
        for index,probe in enumerate(probes):
            a,b,c,d = [results[partition,index,p] for p,_ in grids]
            finite = all(v['finite'] for v in (a,b,c,d))
            diffs = {}
            if finite:
                for label,left,right in [('old_mesh',a,b),('mesh',b,c),('domain',c,d)]:
                    diffs[label+'_log_likelihood'] = abs(left['log_likelihood']-right['log_likelihood'])
                    diffs[label+'_score'] = max(abs(x-y) for x,y in zip(left['score'],right['score']))
            passed = finite and all(v <= (1e-5 if key.endswith('score') else 1e-6)
                for key,v in diffs.items() if not key.startswith('old_mesh'))
            passed = passed and all(v['domain_loss']<=1e-8 and v['posterior_edge']<=1e-8 for v in (b,c,d))
            report['comparisons'].append(dict(partition=partition,probe=index,coordinates_in_prior_sd=probe,
                passed=passed,finite=finite,differences=diffs,
                finest_domain_loss=d['domain_loss'],finest_posterior_edge=d['posterior_edge']))
    report['status'] = ('TAIL_PROBES_PASSED_NOT_GLOBAL_QUALIFICATION' if all(v['passed'] for v in report['comparisons'])
                        else 'REFERENCE_REPAIR_REQUIRED')
    report['failed_probes'] = sum(not v['passed'] for v in report['comparisons'])
    save(out/'report.json', report)
    return 0  # A faithfully measured numerical failure is a completed diagnostic.


def main():
    started, parent_cpu = time.monotonic(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    if '-m10-tail-screen-' not in out.name:
        parser.error('versioned M10 tail-screen output required')
    from bayesfilter.inference.surrogate_hmc_program import m10_reference_state
    state = m10_reference_state(ROOT)
    if not state['qualified']:
        raise ValueError('unchanged local reference required')
    total = stage_cpu = stage_wall = 0.
    attempts = 0
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        entry = json.loads(path.read_text())
        charge = entry['charged_this_run'] if 'charged_this_run' in entry else entry['charged']
        total += float(charge['cpu_worker_seconds'])
        if '-m10-tail-screen-' in path.parent.name:
            attempts += 1
            stage_cpu += float(charge['cpu_worker_seconds'])
            old = json.loads((path.parent/'report.json').read_text())
            stage_wall += float(old['wall_seconds'])
            if old['status'] in ('REFERENCE_REPAIR_REQUIRED','TAIL_PROBES_PASSED_NOT_GLOBAL_QUALIFICATION'):
                raise ValueError('completed numerical screen requires interpretation, not automatic retry')
    cpu_budget = min(150-stage_cpu,54000-total,state['family_cpu_remaining'])
    wall_budget = 150-stage_wall
    if attempts >= 2 or min(cpu_budget,wall_budget)<5:
        raise ValueError('tail screen allocation exhausted')
    reference = Path(state['qualified']['report'])
    data = json.loads(reference.read_text())['partitions']
    datasets = {name:dict(path=row['data_file'],sha256=row['data_sha256'])
                for name,row in data.items() if name.startswith(('calibration-','validation-'))}
    if set(datasets) != {f'{p}-{r}' for p in ('calibration','validation') for r in ('nominal','stress')}:
        raise ValueError('four frozen non-claim partitions required')
    names = [PLAN,'scripts/screen_surrogate_hmc_nonlinear_tails.py',
             'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
             'bayesfilter/score_study/nonlinear_tf.py','bayesfilter/score_study/gaussian_tf.py']
    sources = {name:sha(ROOT/name) for name in names}
    out.mkdir(parents=True,exist_ok=False)
    for name in names:
        target = out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    manifest = dict(command=[sys.executable,*sys.argv],environment=sys.prefix,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan=PLAN,result_file=str(out/'report.json'),source_hashes=sources,datasets=datasets,
        seeds={name:json.loads(Path(row['path']).read_text())['seed'] for name,row in datasets.items()},
        data_version='frozen M10 FP32 calibration and validation observations; claim untouched',
        device='CPU-only reference diagnostic; GPU intentionally hidden',cuda_visible_devices='-1',
        cpu_budget=cpu_budget,wall_budget=wall_budget,reference_report_sha256=sha(reference))
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',posterior_qualified=False,rows=[]))
    child_cpu = cpu_budget-(time.process_time()-parent_cpu)-2.
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(math.floor(child_cpu),math.ceil(child_cpu)))
    with (out/'worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit)
        try:
            code = process.wait(timeout=max(1.,wall_budget-(time.monotonic()-started)-1.))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = 124
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-parent_cpu
    wall = time.monotonic()-started
    report = json.loads((out/'report.json').read_text())
    manifest = json.loads((out/'run-manifest.json').read_text())
    if code:
        report['status'] = 'RESOURCE_LIMIT' if code in (-24,-9,124) else 'EXECUTION_FAILED'
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):
        report['status'],code = 'SOURCE_DRIFT',2
    if any(sha(row['path'])!=row['sha256'] for row in datasets.values()):
        report['status'],code = 'DATA_DRIFT',2
    report.update(exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    manifest.update(status=report['status'],exit_code=code,cpu_worker_seconds=cpu,wall_seconds=wall)
    save(out/'report.json',report)
    save(out/'run-manifest.json',manifest)
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=0.,cpu_worker_seconds=cpu)))
    print(report['status'],'CPU',round(cpu,3),'wall',round(wall,3),flush=True)
    return code if code >= 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
