#!/usr/bin/env python3
"""Diagnostic supervisor: frozen filter validation on independent T50 datasets.

Numerical work calls existing shared TensorFlow and author-code reference paths.
This module contains orchestration/provenance only and cannot promote defaults.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import run_ledh_matched_comparison as matched

OUT = ROOT / 'docs/plans/artifacts/ledh-independent-validation-20261008-01'
FROZEN = ROOT / 'docs/plans/artifacts/ledh-matched-comparison-20261008-01'
PLAN = ROOT / 'docs/plans/ledh-independent-validation-20261008.md'
RESULT = ROOT / 'docs/benchmarks/ledh-independent-validation-results-20261008.md'
MODELS = matched.MODELS
DATA_SEEDS = (26100831, 26100832)
FIT_SEEDS = (41, 53)
BUDGET = 86400
MAX_ATTEMPTS = 40
WALL_CAP = 43200
LOCK = threading.Lock()
read = matched.read
sha = matched.sha


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def design_seeds(data_seed):
    index = DATA_SEEDS.index(data_seed)
    return list(range(261008301 + 100*index, 261008309 + 100*index))


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES='1' if gpu else '-1',
                TF_FORCE_GPU_ALLOW_GROWTH='true', TF_CPP_MIN_LOG_LEVEL='2',
                TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')


def settings(model):
    return SimpleNamespace(model=model, device='gpu', dtype='float64', no_tf32=True,
        lgssm_family='p44', particles=1008, horizon=50, flow_steps=16,
        reset_steps=40, reset_epsilon=1., correction_steps=0)


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ.update(environment(True))
    tf, memory = matched.candidate.prepare(settings('lgssm'))
    manifest = dict(plan=str(PLAN), result=str(RESULT), git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), command=[sys.executable,*sys.argv],
        environment=sys.executable, data_seeds=DATA_SEEDS, designs={str(s):design_seeds(s) for s in DATA_SEEDS},
        models=MODELS, particles=1008, horizon=50, dtype='float64', jit_compile=True, tf32=False,
        cpu_only=False, cuda_visible_devices='1', memory_policy=memory, tensorflow=tf.__version__,
        budget_seconds=BUDGET, max_attempts=MAX_ATTEMPTS, wall_cap_seconds=WALL_CAP,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PLAN,*matched.source_paths()]},
        frozen_tuning={}, data=[])
    for model in MODELS:
        args=settings(model)
        spec=matched.specs(model)
        source=FROZEN/(model+'-tuning')/'tuning.json'
        tune=read(source)
        if not tune['valid'] or tune['scope']!=matched.candidate.scope(args,spec):
            raise ValueError('frozen tuning scope mismatch: '+model)
        if set(DATA_SEEDS)&{tune['calibration_data_seed'],tune['validation_data_seed'],matched.DATA}:
            raise ValueError('independent-data leakage')
        snapshot=OUT/'frozen'/(model+'-tuning.json')
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        manifest['frozen_tuning'][model]=dict(source=str(source),sha256=sha(source),snapshot=str(snapshot),beta=tune['beta'])
        for seed in DATA_SEEDS:
            spec=matched.specs(model)
            obs=matched.candidate.observations(spec,50,seed,tf,tf.float64)
            folder=OUT/'inputs'/f'{model}-d{seed}'
            dataset=dict(observations=obs.numpy().tolist(),observation_sha256=matched.tensor_sha(tf,obs),
                data_seed=seed,target_id=spec.target_id,dtype='float64',
                generator='same_model_simulator_independent_T50_GPU_XLA',cpu_only=False)
            write(folder/'dataset.json',dataset)
            write(folder/'job.json',dict(model=model,horizon=50,data_seed=seed,
                theta_points=[spec.default_theta(tf.float64).numpy().tolist()],
                dataset_file=str(folder/'dataset.json'),jit_compile=True))
            manifest['data'].append(dict(model=model,data_seed=seed,path=str(folder),sha256=sha(folder/'dataset.json'),
                observation_sha256=dataset['observation_sha256'],target_id=spec.target_id))
    manifest['wall_seconds']=time.monotonic()-started
    write(OUT/'campaign.json',manifest)
    print(json.dumps(dict(prepared=str(OUT),datasets=len(manifest['data']),wall_seconds=manifest['wall_seconds'])),flush=True)


def validate_frozen():
    campaign=read(OUT/'campaign.json')
    for model,record in campaign['frozen_tuning'].items():
        if sha(record['source'])!=record['sha256'] or sha(record['snapshot'])!=record['sha256']:
            raise ValueError('frozen tuning changed: '+model)
    for record in campaign['data']:
        if sha(Path(record['path'])/'dataset.json')!=record['sha256']:
            raise ValueError('prepared dataset changed')
    return campaign


def attempt(label, script, arguments, cap, gpu=False, **scope):
    with LOCK:
        rows=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
        used=sum(r.get('wall_seconds',0) if r['status']!='running' else r['limit_seconds'] for r in rows)
        wall_left=WALL_CAP-(time.time()-read(OUT/'run.json')['started_epoch'])
        limit=min(cap,BUDGET-used,wall_left)
        if len(rows)>=MAX_ATTEMPTS or limit<60:
            raise RuntimeError('campaign budget exhausted')
        index=len(rows)+1
        destination=OUT/f'{index:03d}-{label}'
        command=[sys.executable,'-B',str(ROOT/script),*[str(a).replace('{output}',str(destination)) for a in arguments]]
        row=dict(index=index,label=label,command=command,output=str(destination),log=str(destination)+'.log',
            limit_seconds=limit,status='running',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            environment=sys.executable,cpu_only=not gpu,gpu_intentionally_hidden=not gpu,
            cuda_visible_devices='1' if gpu else '-1',plan=str(PLAN),result=str(RESULT),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256=sha(ROOT/script),**scope)
        rows.append(row);write(OUT/'attempts.json',rows)
    started=time.monotonic()
    with open(row['log'],'w') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:
            rows=read(OUT/'attempts.json');rows[index-1]['pid']=proc.pid;write(OUT/'attempts.json',rows)
        try: code=proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    with LOCK:
        rows=read(OUT/'attempts.json')
        rows[index-1].update(exit_code=code,status='complete' if code==0 else 'failed',wall_seconds=time.monotonic()-started)
        write(OUT/'attempts.json',rows)
    print(json.dumps(dict(label=label,exit_code=code,wall_seconds=time.monotonic()-started,output=str(destination))),flush=True)
    return code,destination


def gpu_queue():
    for seed in DATA_SEEDS:
        for model in MODELS:
            validate_frozen()
            attempt(f'filter-{model}-d{seed}','docs/benchmarks/run_ledh_matched_comparison.py',[
                'worker','--model',model,'--output','{output}','--tuning',OUT/'frozen'/(model+'-tuning.json'),
                '--device','gpu','--dtype','float64','--no-tf32','--particles','1008','--horizon','50',
                '--lgssm-family','p44','--independent','--data-seed',seed,
                '--dataset-file',OUT/'inputs'/f'{model}-d{seed}'/'dataset.json',
                '--design-seeds',*design_seeds(seed),'--plan-file',PLAN,'--result-file',RESULT],1200,True,
                kind='filter',model=model,data_seed=seed)
    attempt('bootstrap','docs/benchmarks/run_nonlinear_bootstrap_reference.py',[
        '--worker-dirs',*[OUT/'inputs'/f'{model}-d{seed}' for seed in DATA_SEEDS for model in ('predator_prey','sir_d18')],
        '--particles',1008,32768,131072,'--seeds',261008501,261008502,261008503,261008504,
        '--budget-seconds',3300,'--output','{output}','--plan-file',PLAN,'--result-file',RESULT],3600,True,kind='bootstrap')


def reference_queue(seed):
    for model in ('predator_prey','sir_d18'):
        for fit in FIT_SEEDS:
            validate_frozen()
            limit=1800 if model=='predator_prey' else 14400
            code,proposal=attempt(f'zhao-{model}-d{seed}-fit{fit}','docs/benchmarks/run_zhao_cui_publication_replication.py',[
                '--output-root','{output}','--plan-file',PLAN,'--model','pp' if model=='predator_prey' else 'sir_austria',
                '--profile','current_target','--route','linear','--rank',20,
                '--dataset',OUT/'inputs'/f'{model}-d{seed}'/'dataset.json','--horizon',50,'--save-times',50,
                '--smooth-samples',100000,'--fit-seed',fit,'--smooth-seed',3000+fit,'--timeout-seconds',limit-120,
                *(['--pp-tail-precision-repair'] if model=='predator_prey' else [])],limit,
                kind='zhao_fit',model=model,data_seed=seed,fit_seed=fit)
            if code: continue
            attempt(f'score-{model}-d{seed}-fit{fit}','docs/benchmarks/run_zhao_cui_quadratic_score_reference.py',[
                '--proposal',proposal,'--model',model,'--output-root','{output}','--plan-file',PLAN,
                '--horizon',50,'--points',128,'--heldout',64,'--design-seed',43017,
                '--radii',.000625,.0003125 if model=='predator_prey' else .00015625,'--timeout-seconds',5300],5400,
                kind='zhao_score',model=model,data_seed=seed,fit_seed=fit)


def run():
    validate_frozen()
    if (OUT/'run.json').exists(): raise ValueError('run already exists; preserve attempts and use a reviewed targeted retry')
    write(OUT/'run.json',dict(started_epoch=time.time(),pid=os.getpid(),status='running',command=[sys.executable,*sys.argv]))
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(gpu_queue),*[pool.submit(reference_queue,s) for s in DATA_SEEDS]]
        for future in concurrent.futures.as_completed(jobs):
            try: future.result()
            except Exception as exc: errors.append(type(exc).__name__+': '+str(exc))
    state=read(OUT/'run.json');state.update(status='finished',wall_seconds=time.time()-state['started_epoch'],errors=errors)
    write(OUT/'run.json',state)
    status()
    return int(bool(errors) or any(r['status']!='complete' for r in read(OUT/'attempts.json')))


def status():
    rows=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
    print(json.dumps(dict(attempts=len(rows),complete=sum(r['status']=='complete' for r in rows),
        failed=[r['label'] for r in rows if r['status']=='failed'],
        running=[dict(label=r['label'],pid=r.get('pid')) for r in rows if r['status']=='running'],
        used_worker_seconds=sum(r.get('wall_seconds',0) for r in rows)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report'])
    args=parser.parse_args()
    if args.command=='report':
        from summarize_ledh_independent_validation import report
        return report() or 0
    return globals()[args.command]() or 0


if __name__=='__main__': raise SystemExit(main())
