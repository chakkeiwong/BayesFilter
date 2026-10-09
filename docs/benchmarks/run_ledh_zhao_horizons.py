#!/usr/bin/env python3
"""Bounded diagnostic supervisor for the owner's eight-scope comparison.

No numerical implementation lives here. CPU references and GPU LEDH call their
existing shared authorities. The run plan fixes the target and total budget.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PYTHON = '/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
OUT = ROOT / 'docs/plans/artifacts/ledh-zhao-horizons-20261006-01'
PLAN = 'docs/plans/ledh-zhao-horizon-comparison-20261006.md'
HORIZONS = (10,20,40,50)
MODELS = ('predator_prey','sir_d18')
SEEDS = (261006101,261006102,261006103,261006104)
TOTAL_BUDGET = 172800
LOCK = threading.Lock()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def read(path):
    return json.loads(path.read_text())

def base_environment(gpu=False):
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH='true',
               TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TF_CPP_MIN_LOG_LEVEL='2')
    if not gpu: env['CUDA_VISIBLE_DEVICES'] = '-1'
    return env

def attempt(label, script, arguments, limit, gpu=False):
    with LOCK:
        ledger = read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
        charged = budget_accounting()['reserved_including_running_caps']
        if charged + limit > TOTAL_BUDGET:
            raise RuntimeError('aggregate campaign budget cannot cover the next attempt')
        index = len(ledger) + 1
        destination = OUT/f'{index:03d}-{label}'
        log = OUT/f'{index:03d}-{label}.log'
        command = [PYTHON, '-B', str(ROOT/script), *[str(a).replace('{output}',str(destination)) for a in arguments]]
        row = dict(index=index, label=label, command=command, output=str(destination),
                   log=str(log), limit_seconds=limit, status='running',
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   hardware='GPU FP64/XLA reference' if gpu else 'CPU reference, GPU intentionally hidden',
                   plan=PLAN, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   source_sha256=hashlib.sha256((ROOT/script).read_bytes()).hexdigest())
        ledger.append(row); dump(OUT/'attempts.json',ledger)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command,cwd=ROOT,env=base_environment(gpu),stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            code = 124
    with LOCK:
        ledger=read(OUT/'attempts.json')
        ledger[index-1].update(exit_code=code,wall_seconds=time.monotonic()-started,
                               status='complete' if code == 0 else 'failed')
        dump(OUT/'attempts.json',ledger)
    print(json.dumps(dict(label=label,exit_code=code,output=str(destination),wall_seconds=time.monotonic()-started)),flush=True)
    return code,destination

def prepare():
    os.environ.update(base_environment(False))
    import tensorflow as tf
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    for model in MODELS:
        spec=NonlinearSQMCSpec(model)
        observations=spec.observations(50,26100611,dtype=tf.float64,jit_compile=True)
        theta=spec.default_theta(tf.float64).numpy().tolist()
        for horizon in HORIZONS:
            directory=OUT/'inputs'/f'{model}-T{horizon}'
            prefix=observations[:horizon]
            dataset=dict(observations=prefix.numpy().tolist(),
                         observation_sha256=hashlib.sha256(bytes(tf.io.serialize_tensor(prefix).numpy())).hexdigest(),
                         data_seed=26100611,target_id=spec.target_id,dtype='float64',
                         generator='canonical_T50_simulation_exact_prefix',cpu_only=True)
            if (directory/'dataset.json').exists() and read(directory/'dataset.json') != dataset:
                raise ValueError('saved dataset differs; use a fresh campaign root')
            dump(directory/'dataset.json',dataset)
            dump(directory/'job.json',dict(model=model,horizon=horizon,data_seed=26100611,
                                          theta_points=[theta],dataset_file=str(directory/'dataset.json'),jit_compile=True))
    print(json.dumps({'prepared':str(OUT/'inputs'),'cpu_only':True}))

def preflight():
    os.environ.update(base_environment(True))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    @tf.function(input_signature=[tf.TensorSpec([2,2],tf.float64)],jit_compile=True)
    def probe(x): return tf.linalg.matmul(x,x)
    with tf.device('/GPU:0'): value=probe(tf.eye(2,dtype=tf.float64))
    report=dict(tensorflow=tf.__version__,memory_policy=memory,device=value.device,
                xla_result=value.numpy().tolist(),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                plan=PLAN,cpu_only=False)
    dump(OUT/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='nvidia_smi'}))

def master_arguments(model,horizon,n,seeds,policies,arms,budget):
    return ['run','--output','{output}','--models',model,'--horizons',horizon,
            '--particles',n,'--data-seeds',26100611,'--design-seeds',*seeds,
            '--routes','iid_dual_cap','--arms',*arms,'--importance-weight-policies',*policies,
            '--dtype','float64','--device','gpu','--plan-file',PLAN,
            '--dataset-file',OUT/'inputs'/f'{model}-T{horizon}'/'dataset.json',
            '--budget-seconds',budget,'--worker-seconds',max(300,budget/len(policies)-30)]

def pilot():
    statuses=[]
    for model in MODELS:
        code,path=attempt(f'pilot-{model}','docs/benchmarks/run_ledh_nonlinear_master.py',
             master_arguments(model,10,144,SEEDS[:1],['ancestor','marginal_mixture'],['guarded_pairwise'],1700),1800,True)
        statuses.append(dict(model=model,exit_code=code,output=str(path)))
    dump(OUT/'pilot.json',statuses)
    return int(any(r['exit_code'] for r in statuses))

def gpu_queue():
    for model in MODELS:
        for horizon in HORIZONS:
            attempt(f'ledh-{model}-T{horizon}','docs/benchmarks/run_ledh_nonlinear_master.py',
                master_arguments(model,horizon,1008,SEEDS,['ancestor','marginal_mixture'],['guarded_pairwise'],2100),2200,True)
            attempt(f'covariance-{model}-T{horizon}','docs/benchmarks/run_ledh_nonlinear_master.py',
                master_arguments(model,horizon,1008,SEEDS,['ancestor'],['covariance_only'],440),500,True)
    attempt('bootstrap-references','docs/benchmarks/run_nonlinear_bootstrap_reference.py',
        ['--worker-dirs',*[OUT/'inputs'/f'{model}-T{horizon}' for model in MODELS for horizon in HORIZONS],
         '--particles',1008,32768,131072,'--seeds',261006201,261006202,261006203,261006204,
         '--budget-seconds',14300,'--output','{output}','--plan-file',PLAN,
         '--result-file','docs/benchmarks/ledh-zhao-horizon-results-20261006.md'],14400,True)

def zhao_queue(model):
    source_model='pp' if model=='predator_prey' else 'sir_austria'
    for seed in (2,17,29):
        code,path=attempt(f'zhao-{model}-fit{seed}','docs/benchmarks/run_zhao_cui_publication_replication.py',
            ['--output-root','{output}','--plan-file',PLAN,'--model',source_model,'--profile','current_target','--route','linear',
             '--rank',20 if model=='predator_prey' else 40,'--dataset',OUT/'inputs'/f'{model}-T50'/'dataset.json',
             '--horizon',50,'--save-times',*HORIZONS,'--smooth-samples',100000,'--fit-seed',seed,
             '--smooth-seed',3000+seed,'--timeout-seconds',14200,
             *(['--pp-tail-precision-repair'] if model=='predator_prey' else [])],14400)
        if code:
            print(json.dumps({'repair_required':str(path),'model':model}),flush=True)
            return
        for horizon in HORIZONS:
            attempt(f'quadratic-{model}-fit{seed}-T{horizon}','docs/benchmarks/run_zhao_cui_quadratic_score_reference.py',
                ['--proposal',path,'--model',model,'--output-root','{output}','--plan-file',PLAN,'--horizon',horizon,
                 '--points',128,'--heldout',64,'--radii',.000625,.0003125 if model=='predator_prey' else .00015625,
                 '--timeout-seconds',1100],1200)

def run():
    if not (OUT/'preflight.json').exists() or not (OUT/'pilot.json').exists():
        raise ValueError('preflight and pilot evidence are required')
    if any(row['exit_code'] for row in read(OUT/'pilot.json')):
        raise ValueError('resolve pilot failures before the main comparison')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(gpu_queue),*[executor.submit(zhao_queue,model) for model in MODELS]]
        for future in concurrent.futures.as_completed(futures): future.result()
    status()

def all_attempts():
    rows=read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
    return rows+(read(OUT/'score-repair-attempts.json') if (OUT/'score-repair-attempts.json').exists() else [])


def budget_accounting(rows=None):
    rows=all_attempts() if rows is None else rows
    auxiliary=[read(p) for p in OUT.glob('score-check-*/supervisor.json')]
    now=dt.datetime.now(dt.timezone.utc)
    completed=sum(r.get('wall_seconds',0) for r in rows)+sum(r['wall_seconds'] for r in auxiliary)
    running=sum(max(0.,(now-dt.datetime.fromisoformat(r['started_utc'])).total_seconds()) for r in rows if r['status']=='running')
    reserved=completed+sum(r['limit_seconds'] for r in rows if r['status']=='running')
    return dict(completed_job_seconds=completed,running_elapsed_job_seconds=running,
        used_including_running_job_seconds=completed+running,reserved_including_running_caps=reserved,
        budget_seconds=TOTAL_BUDGET,remaining_after_reservations=TOTAL_BUDGET-reserved,
        includes_failed_attempts=True,includes_auxiliary_derivative_checks=True)

def status():
    rows=all_attempts()
    print(json.dumps(dict(budget=budget_accounting(rows),
        completed=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='failed' for r in rows),
        running=[{k:r[k] for k in ('label','output','log','started_utc')} for r in rows if r['status']=='running']),indent=2))


def tests():
    command=[PYTHON,'-B','-m','pytest','-q','tests/highdim/test_ledh_marginal_weights.py',
             'tests/test_quadratic_score_reference.py','tests/test_zhao_cui_path_likelihood_reference.py',
             'tests/test_zhao_cui_pp_tail_reference.py','tests/test_ledh_zhao_horizon_harness.py']
    log=OUT/'focused-tests.log'
    with log.open('w') as stream:
        result=subprocess.run(command,cwd=ROOT,env=base_environment(False),stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps({'exit_code':result.returncode,'log':str(log)}))
    return result.returncode

def diagnostics_fine():
    return diagnostics(fine=True)

def diagnostics(fine=False):
    name='score-check-002' if fine else 'score-check-001'
    destination=OUT/name
    if destination.exists(): raise ValueError('diagnostic already exists; preserve it')
    command=[PYTHON,'-B',str(ROOT/'docs/benchmarks/check_ledh_nonlinear_score_fd.py'),
             '--output',str(destination),'--timeout-seconds','1000']
    if fine: command += ['--models','sir_d18','--device','gpu','--steps','1e-6','5e-7','1e-7','5e-8']
    started=time.monotonic()
    with (OUT/(name+'.log')).open('w') as log:
        try:
            code=subprocess.run(command,cwd=ROOT,env=base_environment(fine),stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
        except subprocess.TimeoutExpired: code=124
    dump(destination/'supervisor.json',dict(command=command,exit_code=code,wall_seconds=time.monotonic()-started,
        budget='charged in addition to attempts.json; no concurrent write to that ledger'))
    print(json.dumps(dict(exit_code=code,output=str(destination))))
    return code

def repair_sir():
    if any(r['status']=='running' for r in read(OUT/'attempts.json')):
        raise ValueError('wait for the first supervisor to finish; single ledger writer')
    def reference(rank,seed,limit):
        code,path=attempt(f'zhao-sir_d18-rank{rank}-fit{seed}-runtime-repair',
            'docs/benchmarks/run_zhao_cui_publication_replication.py',
            ['--output-root','{output}','--plan-file',PLAN,'--model','sir_austria','--profile','current_target','--route','linear',
             '--rank',rank,'--dataset',OUT/'inputs'/'sir_d18-T50'/'dataset.json','--horizon',50,
             '--save-times',*HORIZONS,'--smooth-samples',100000,'--fit-seed',seed,'--smooth-seed',3000+seed,
             '--timeout-seconds',limit-200],limit)
        if code:
            print(json.dumps(dict(repair_required=str(path),rank=rank)),flush=True)
            return
        for horizon in HORIZONS:
            attempt(f'quadratic-sir_d18-rank{rank}-fit{seed}-T{horizon}',
                'docs/benchmarks/run_zhao_cui_quadratic_score_reference.py',
                ['--proposal',path,'--model','sir_d18','--output-root','{output}','--plan-file',PLAN,'--horizon',horizon,
                 '--points',128,'--heldout',64,'--radii',.000625,.00015625,'--timeout-seconds',2300],2400)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(reference,40,2,115200),executor.submit(reference,20,17,14400)]
        for future in concurrent.futures.as_completed(futures): future.result()
    status()

def report():
    from docs.benchmarks.summarize_ledh_zhao_horizons import report as summarize
    summarize()

def repair_scores():
    """Watch bounded parent jobs and retry only their missing fixed radii."""
    auxiliary_path=OUT/'score-repair-attempts.json'
    if auxiliary_path.exists() and any(r['status']=='running' for r in read(auxiliary_path)):
        raise ValueError('a score repair is already active; inspect it before restarting')
    started=time.monotonic()
    while time.monotonic()-started < TOTAL_BUDGET:
        primary=read(OUT/'attempts.json')
        # The rank20 parent queue must finish before taking its freed CPU slot.
        rank20_end=[r for r in primary if r['label']=='quadratic-sir_d18-rank20-fit17-T50']
        if not rank20_end or rank20_end[0]['status']=='running':
            if not any(r['status']=='running' for r in primary): break
            time.sleep(30)
            continue
        auxiliary=read(auxiliary_path) if auxiliary_path.exists() else []
        repaired={r['parent_label'] for r in auxiliary}
        pending=[r for r in primary if r['label'].startswith('quadratic-') and r['status']=='failed' and r['label'] not in repaired]
        if not pending:
            if not any(r['status']=='running' for r in primary): break
            time.sleep(30)
            continue
        parent=pending[0]
        result_file=Path(parent['output'])/'result.json'
        if not result_file.exists(): raise RuntimeError('missing failed-job metadata; inspect before retrying')
        previous=read(result_file)
        if previous.get('failure_type')!='TimeoutError': raise RuntimeError('non-timeout score failure requires investigation')
        command=list(parent['command'])
        left=command.index('--radii')+1
        right=next((i for i in range(left,len(command)) if command[i].startswith('--')),len(command))
        original=[float(value) for value in command[left:right]]
        completed={estimate['radius'] for estimate in previous['estimates']}
        missing=[radius for radius in original if radius not in completed]
        if not missing: raise RuntimeError('failed job has all radii; inspect its terminal failure')
        limit=7200
        accounting=budget_accounting()
        # Reserve the parent's remaining four rank40 score slots as well.
        remaining_parent_score_slots=max(0,4-sum(r['label'].startswith('quadratic-sir_d18-rank40-') for r in primary))
        if accounting['reserved_including_running_caps']+limit+remaining_parent_score_slots*2400>TOTAL_BUDGET:
            raise RuntimeError('remaining total campaign budget cannot cover this repair')
        index=len(auxiliary)+1
        destination=OUT/f'score-repair-{index:03d}'
        command[command.index('--output-root')+1]=str(destination)
        command[left:right]=[str(radius) for radius in missing]
        command[command.index('--timeout-seconds')+1]=str(limit-100)
        row=dict(label=parent['label']+'-runtime-repair',parent_label=parent['label'],
            command=command,output=str(destination),log=str(OUT/f'score-repair-{index:03d}.log'),
            status='running',limit_seconds=limit,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            hardware='CPU reference, GPU intentionally hidden',plan=PLAN,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256=hashlib.sha256((ROOT/'docs/benchmarks/run_zhao_cui_quadratic_score_reference.py').read_bytes()).hexdigest(),
            failure_class='runtime_budget_underestimate',preserved_radii=sorted(completed),missing_radii=missing)
        auxiliary.append(row);dump(auxiliary_path,auxiliary)
        job_started=time.monotonic()
        with Path(row['log']).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=base_environment(False),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
        auxiliary[-1].update(exit_code=code,wall_seconds=time.monotonic()-job_started,status='complete' if code==0 else 'failed')
        dump(auxiliary_path,auxiliary)
        print(json.dumps(dict(repaired=parent['label'],exit_code=code,output=str(destination))),flush=True)
        if code: raise RuntimeError('bounded missing-radius repair failed; inspect before another attempt')
    status()

def document():
    import re
    directory=OUT/'documentation-01'
    directory.mkdir(exist_ok=True)
    started=time.monotonic()
    commands=[]
    converged=False
    for index in range(2,14):
        command=['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']
        with (directory/f'pdflatex-{index:02d}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT/'docs',stdout=log,stderr=subprocess.STDOUT,timeout=180)
        commands.append(command)
        if result.returncode: raise RuntimeError(f'monograph build failed, pass {index}')
        log=(ROOT/'docs/main.log').read_text(errors='replace')
        if not any(text in log for text in ('Rerun to get cross-references right','Rerun to get outlines right','There were undefined references','There were undefined citations')):
            converged=True
            break
    if not converged: raise RuntimeError('monograph references did not settle within build budget')
    subprocess.run(['pdftotext','-layout',str(ROOT/'docs/main.pdf'),str(directory/'main-text.txt')],check=True,timeout=90)
    pages=(directory/'main-text.txt').read_text().split('\f')
    selected=[i+1 for i,page in enumerate(pages) if 'A particle moved by LEDH may lie far' in page]
    if len(selected)!=1: raise RuntimeError('new explanation was not located exactly once')
    first=selected[0]
    rendered=[]
    for page in range(first,first+3):
        prefix=directory/f'page-{page}'
        subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-r','100','-png','-singlefile',str(ROOT/'docs/main.pdf'),str(prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
        rendered.append(str(prefix.with_suffix('.png')))
    chapter=ROOT/'docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex'
    dump(directory/'build-result.json',dict(status='complete',command=commands,wall_seconds=time.monotonic()-started,
        latexmk='unavailable; installed pdflatex used',references_converged=converged,
        pdf_pages=len(pages)-1,pdf_sha256=hashlib.sha256((ROOT/'docs/main.pdf').read_bytes()).hexdigest(),
        chapter_sha256=hashlib.sha256(chapter.read_bytes()).hexdigest(),rendered_pages=rendered,
        visual_review='pending',human_readability_review='pending',
        undefined_reference_warning='There were undefined references' in log,
        undefined_citation_warning='There were undefined citations' in log))
    print(json.dumps(dict(build=str(directory/'build-result.json'),rendered_pages=rendered)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','tests','pilot','run','status','report','diagnostics','diagnostics_fine','repair_sir','document','repair_scores'))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    return globals()[args.action]() or 0

if __name__=='__main__': raise SystemExit(main())
