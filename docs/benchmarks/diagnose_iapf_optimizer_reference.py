"""Independent CPU/R bounded optimizer diagnosis on preserved fitting inputs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from diagnose_iapf_adaptive_consumer import REPO, r_literal, save

ROOT=REPO/'docs/plans/artifacts/iapf-optimizer-isolation-20260922-01'
PRIOR=REPO/'docs/plans/artifacts/iapf-fit-input-isolation-20260922-01'
GEOMETRY=REPO/'docs/plans/artifacts/iapf-guide-geometry-20260922-01'
PLAN='docs/plans/iapf-optimizer-isolation-2026-09-22.md'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cast(x):return struct.unpack('f',struct.pack('f',x))[0]


def preserve(directory,source_root,path):
    p=source_root/path
    if sha(p)!=read(source_root/'manifest.json')['outputs'][path]:raise RuntimeError('Changed input '+path)
    target=directory/'inputs'/source_root.name/path;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target);return target


def launch(attempt):
    if not attempt.isidentifier():raise ValueError('Unique attempt name required')
    record_path=ROOT/(attempt+'-launch.json')
    if record_path.exists():raise FileExistsError(record_path)
    budget=read(ROOT/'budget.json')
    if len(list(ROOT.glob('*-launch.json')))>=4 or budget['phase_cpu_seconds']+300>3600 or budget['remaining_cpu_seconds']<300:raise RuntimeError('Budget/attempt cap')
    directory=ROOT/attempt;directory.mkdir(exist_ok=False)
    started=time.monotonic()
    input_path=preserve(directory,PRIOR,'attempt02_gpu_isolate/reference-input.R')
    rows=read(preserve(directory,PRIOR,'attempt02_gpu_isolate/results.json'))
    exact=read(preserve(directory,GEOMETRY,'attempt01_cpu_reference/exact-guides.json'))
    controls=[];captures={}
    for row in rows:
        case=row['case']
        if case not in captures:captures[case]=read(preserve(directory,PRIOR,'attempt02_gpu_isolate/captures/'+case+'.json'))
        cfg=captures[case]['row']['config'];d=row['d'];seed=captures[case]['row']['seed']
        oracle=next(v for k,v in exact.items() if k.startswith(f'd{d}-s{seed}-') and k.endswith('/fitted'))
        controls.append(dict(key=f"{case}/t{row['time']}/{row['target']}/{row['objective']}",case=case,
            time=row['time'],target=row['target'],objective=row['objective'],dimension=d,N=row['N'],
            lower=[cast(-cfg['mean_bound'])]*d+[cast(math.log(cfg['sd_lower']))]*d,
            upper=[cast(cfg['mean_bound'])]*d+[cast(math.log(cfg['sd_upper']))]*d,
            tolerance=cast(cfg['fit_tolerance']),maxit=cfg['max_fit_steps'],lmm=5,factr=0,
            objective_log_density_scale=row['initial_info']['objective_log_density_scale'],
            exact_center=oracle['centers'][row['time']],exact_covariance=oracle['covariances'][row['time']],
            baseline_converged=row['final_info']['converged'],baseline_projected_gradient=row['final_info']['projected_gradient'],
            baseline_KL=row['states']['final']['KL'],baseline_shape=row['states']['final']['shape']))
    (directory/'controls.R').write_text('controls <- '+r_literal(controls)+'\n')
    sources=[str(Path(__file__).resolve().relative_to(REPO)), 'docs/benchmarks/diagnose_iapf_optimizer_reference.R',
             'docs/benchmarks/diagnose_iapf_adaptive_consumer.py',PLAN]
    hashes={}
    for name in sources:
        p=REPO/name;q=ROOT/(attempt+'-source')/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);hashes[name]=sha(p)
    shutil.copy2('/tmp/iapf-optim-help.txt',directory/'installed-optim-documentation.txt')
    command=['Rscript','--vanilla','docs/benchmarks/diagnose_iapf_optimizer_reference.R',str(input_path),str(directory/'controls.R'),str(directory)]
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    record=dict(status='running',command=command,environment={k:environment[k] for k in ['CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        device='CPU independent R; GPU intentionally hidden',sources=hashes,plan=PLAN,prior_manifest=sha(PRIOR/'manifest.json'),
        geometry_manifest=sha(GEOMETRY/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        seeds='No new randomness; exact preceding captured clouds and fitting problems',data='Preserved/hash-checked inputs',
        inputs={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()})
    save(record_path,record)
    with (ROOT/(attempt+'.log')).open('x') as log:
        try:code=subprocess.run(command,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.monotonic()-started;record.update(status='finished',exit_code=code,wall_seconds=elapsed);save(record_path,record)
    budget=read(ROOT/'budget.json');budget['phase_cpu_seconds']+=elapsed;budget['remaining_cpu_seconds']-=elapsed;save(ROOT/'budget.json',budget)
    if code==0:
        with (directory/'results.csv').open() as stream:rows=list(csv.DictReader(stream))
        summary=dict(rows=len(rows),initial_identity_error=max(float(r['initial_identity_error']) for r in rows),
            finite=all(r['finite']=='TRUE' for r in rows),R_version=(directory/'R-version.txt').read_text().strip())
        summary['passed']=len(rows)==64 and summary['initial_identity_error']<=1e-8 and summary['finite']
        save(directory/'summary.json',summary)
        if not summary['passed']:code=1
    print(json.dumps(dict(exit_code=code,wall_seconds=elapsed,artifact=str(directory))));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    raise SystemExit(launch(parser.parse_args().attempt))
