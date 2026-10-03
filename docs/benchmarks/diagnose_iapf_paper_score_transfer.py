"""Bounded CPU-only independent R comparison at the paper's LG settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/plans/artifacts/iapf-paper-score-transfer-20260922-01'
PRIOR=REPO/'docs/plans/artifacts/iapf-covariance-floor-factorial-20260922-01'
PLAN='docs/plans/iapf-paper-score-transfer-2026-09-22.md'
SCRIPT='docs/benchmarks/run_iapf_paper_score_transfer.R'
REFERENCES={
 'docs/benchmarks/reference_iapf_paper.R':'979c84f9dbe906742a8101b94ab5391c3268b2b46c81a903b1d5529022916887',
 'docs/benchmarks/reference_iapf_author_choices.R':'420455b782efa415783eba9d0ae3fbd3483a7994c24bb76d4e126900b081d2b4',
 'docs/benchmarks/reference_iapf_constrained_diagnostic.R':'7ef3fd67dd198c6443634494ef4f2d85cf1abdfbafd1c2477ca294509a404da6'}


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_rows(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def verify_inputs():
    for path,digest in REFERENCES.items():
        if sha(REPO/path)!=digest: raise RuntimeError('Frozen source mismatch: '+path)
    manifest=read(PRIOR/'manifest.json')
    for path,digest in manifest['outputs'].items():
        if sha(PRIOR/path)!=digest: raise RuntimeError('Prior artifact mismatch: '+path)
    if not read(PRIOR/'verification.json')['passed']: raise RuntimeError('Prior failed')


def reserve(count):
    budget=read(ROOT/'budget.json')
    if len(list(ROOT.glob('*-launch.json')))+count>14 or budget['phase_cpu_seconds']+600*count>10800 or budget['remaining_cpu_seconds']<600*count:
        raise RuntimeError('Phase attempt or compute budget exhausted')


def charge(seconds):
    budget=read(ROOT/'budget.json');budget['phase_cpu_seconds']+=seconds;budget['remaining_cpu_seconds']-=seconds
    save(ROOT/'budget.json',budget)


def launch(name,mode,extra):
    directory=ROOT/name;directory.mkdir(exist_ok=False)
    paths=[PLAN,SCRIPT,'docs/benchmarks/diagnostic_iapf_paper_score.R',
        'docs/benchmarks/diagnose_iapf_paper_score_transfer.py',*REFERENCES]
    hashes={}
    for relative in paths:
        source=REPO/relative;dest=ROOT/(name+'-source')/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[relative]=sha(source)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=['Rscript','--vanilla',SCRIPT,mode,str(directory),*map(str,extra)]
    record=dict(status='running',command=command,plan=PLAN,result=str(ROOT/'result.md'),sources=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        cpu_only=True,gpu_intentionally_hidden=True,R=subprocess.check_output(['Rscript','--version'],text=True,stderr=subprocess.STDOUT).strip(),
        data_version='GJL2017 section5.2 matrices; independent generated observations',
        seeds='data951000+10*d+j,pilot952000+10*d+j,final953000+100*d+10*j+r',
        prior_manifest_sha256=sha(PRIOR/'manifest.json'))
    manifest=ROOT/(name+'-launch.json');save(manifest,record);started=time.monotonic()
    with (directory/'R.log').open('x') as stream:
        try:code=subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed)
    save(manifest,record)
    print(json.dumps(dict(attempt=name,exit_code=code,wall_seconds=elapsed)),flush=True)
    return record


def preflight():
    verify_inputs();reserve(1)
    result=launch('attempt01_preflight','preflight',[PRIOR/'attempt01_gpu_factorial/reference-input.R'])
    charge(result['wall_seconds'])
    if result['exit_code']:raise RuntimeError('Preflight rejected')


def cases():
    verify_inputs()
    if read(ROOT/'attempt01_preflight-launch.json')['exit_code']!=0:raise RuntimeError('Preflight required')
    if len(csv_rows(ROOT/'attempt01_preflight/checks.csv'))!=48:raise RuntimeError('Incomplete preflight')
    reserve(10)
    tasks=[(d,j) for d in [5,10,20,40,80] for j in [1,2]]
    failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(launch,f'case_d{d}_s{j}','case',[d,j]):(d,j) for d,j in tasks}
        for future in as_completed(futures):
            if future.cancelled(): continue
            result=future.result();charge(result['wall_seconds'])
            if result['exit_code']:
                failures.append(dict(case=futures[future],exit_code=result['exit_code']))
                for pending in futures: pending.cancel()
    save(ROOT/'case-launch-summary.json',dict(passed=not failures,failures=failures,cases=len(tasks)))
    if failures:raise RuntimeError('Case attempts require classification; preserved in case-launch-summary.json')


def results():
    started=time.monotonic();verify_inputs()
    if not read(ROOT/'case-launch-summary.json')['passed']:raise RuntimeError('Unresolved case failure')
    records=[];case_summaries=[];checks={};fit_status=[];perturbations=[]
    methods=None
    for d in [5,10,20,40,80]:
        for j in [1,2]:
            name=f'case_d{d}_s{j}';directory=ROOT/name;rows=csv_rows(directory/'records.csv')
            for row in rows:
                for key in ['d','data','replicate','N','resampling_count']:
                    row[key]=int(row[key]) if row[key]!='NA' else None
                for key in ['log_error','ratio','relative_error','seconds','floor_max','floor_median_mean','minimum_ess']:
                    row[key]=float(row[key]) if row[key]!='NA' else None
            records.extend(rows);methods=sorted(set(r['method'] for r in rows))
            checks[name+'/rows']=len(rows)==4*len(methods)==40
            control=csv_rows(directory/'checks.csv')[0]
            checks[name+'/oracle_coefficients']=float(control['oracle_coefficient_error'])<=1e-8
            checks[name+'/oracle_likelihood']=all(abs(r['log_error'])<=1e-8 for r in rows if r['method'] in ('full_oracle','full_negligible'))
            fit_status.extend([dict(d=d,data=j,**r) for r in csv_rows(directory/'fit-status.csv')])
            perturbations.extend([dict(d=d,data=j,**r) for r in csv_rows(directory/'floor-perturbation.csv')])
            for method in methods:
                selected=[r for r in rows if r['method']==method];complete=all(r['status']=='complete' for r in selected)
                ratios=[r['ratio'] for r in selected] if complete else []
                mean=sum(ratios)/4 if complete else None
                case_summaries.append(dict(d=d,data=j,method=method,N=selected[0]['N'],complete=complete,
                    ratio_mean=mean,ratio_sd=math.sqrt(sum((v-mean)**2 for v in ratios)/3) if complete else None,
                    relative_mse=sum(r['relative_error']**2 for r in selected)/4 if complete else None,
                    log_mse=sum(r['log_error']**2 for r in selected)/4 if complete else None,
                    floor_max=max(r['floor_max'] for r in selected) if complete else None))
    conditional=[];heuristic=[]
    learned=['diagonal_positive','diagonal_tail8','full_positive','full_tail8','diagonal_QR_tail8']
    for d in [5,10,20,40,80]:
        means={}
        for method in methods:
            selected=[r for r in case_summaries if r['d']==d and r['method']==method]
            complete=all(r['complete'] for r in selected)
            means[method]=sum(r['relative_mse'] for r in selected)/2 if complete else None
            conditional.append(dict(d=d,method=method,complete=complete,
                relative_rmse=math.sqrt(means[method]) if complete else None,
                log_mse=sum(r['log_mse'] for r in selected)/2 if complete else None,
                ratio_mean=sum(r['ratio_mean'] for r in selected)/2 if complete else None,
                floor_max=max(r['floor_max'] for r in selected) if complete else None))
        for method in learned:
            losses=[h for h in ['bootstrap','fully_adapted','current_observation'] if means[method] is not None and means[method]>means[h]]
            heuristic.append(dict(d=d,method=method,losses=losses,promotion_veto=means[method] is None or bool(losses),
                oracle_gap=means[method],ranking_statistically_supported=False))
    for launch_path in ROOT.glob('*-launch.json'):
        record=read(launch_path);checks[launch_path.stem+'/exit']=record['exit_code']==0
        for path,digest in record['sources'].items():
            checks[launch_path.stem+'/'+path]=sha(ROOT/(launch_path.stem.replace('-launch','')+'-source')/path)==digest
    checks['all_finite_or_explicit_rejection']=all(r['status']=='fit_rejected' or (r['status']=='complete' and math.isfinite(r['log_error'])) for r in records)
    for path,value in [('records.json',records),('per-data-summary.json',case_summaries),('conditional-summary.json',conditional),
                       ('heuristic-dominance.json',heuristic),('fit-status.json',fit_status),('floor-perturbation.json',perturbations)]:
        save(ROOT/path,value)
    save(ROOT/'verification.json',dict(passed=all(checks.values()),checks=checks))
    charge(time.monotonic()-started)
    print(json.dumps(dict(passed=all(checks.values()),checks=len(checks),records=len(records),
        rejected=sum(r['status']=='fit_rejected' for r in records))))
    if not all(checks.values()):raise RuntimeError('Terminal checks failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['preflight','cases','results'],required=True)
    args=parser.parse_args()
    {'preflight':preflight,'cases':cases,'results':results}[args.mode]()
