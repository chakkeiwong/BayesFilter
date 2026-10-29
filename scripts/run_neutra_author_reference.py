#!/usr/bin/env python3
"""Bounded original-code CPU reference; never imported by a runtime algorithm."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'.localresources/flonaco-author-20260929/upstream'
OUTPUT = ROOT/'docs/plans/artifacts/neutra-source-fit-remedy-2026-10-04/author-reference'
PYTHON = '/home/ubuntu/anaconda3/envs/mathdevmcp-backends/bin/python'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def worker(output,iterations,seed):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.path.insert(0,str(UPSTREAM))
    from flonaco.real_nvp_mlp import RealNVP_MLP
    from flonaco.gaussian_utils import MoG
    from flonaco.training import train
    means=[torch.tensor([-5.,0.]),torch.tensor([5.,0.])]
    target=MoG(means,[torch.eye(2),torch.eye(2)],weights=[1.,2.],device='cpu')
    model=RealNVP_MLP(2,6,1,init_weight_scale=1e-6,hidden_dim=100,hidden_depth=3,device='cpu')
    settings=dict(n_iter=iterations,lr=.005,bs=400,
        args_loss=dict(type='fwd',samp='mhlangevin',dt=1e-4,beta=1.,n_tot=40,
                       ratio_pos_init=None,n_steps_burnin=100,x_init_samp=None),
        args_stop={'acc':None},estimate_tau=False,return_all_xs=False,
        jump_tol=100.,save_splits=10,grad_clip=1e4)
    write(output/'effective-settings.json',dict(settings=settings,seed=seed,
        architecture=dict(coupling_pairs=6,hidden_depth=3,width=100,init_scale=1e-6),
        original_source_unmodified=True,paper_figure_exact_reproduction=False,
        source_default_hypotheses=['dt','initialization','burnin','jump_tol','grad_clip']))
    started=time.monotonic()
    result=train(model,target,**settings)
    elapsed=time.monotonic()-started
    if not isinstance(result,dict):
        raise RuntimeError('original controller returned early non-dictionary failure')
    completed=len(result['losses'])
    torch.save(model.state_dict(),output/'model-state.pt')
    plt.savefig(output/'training-densities.png')
    plt.close('all')
    # Independent descriptive endpoint integration; no sampling convergence claim.
    torch.manual_seed(seed+10000)
    with torch.no_grad():
        reference=target.sample(32768)
        proposal=model.sample(32768)
        fkl=model.nll(reference)-target.U(reference)
        rkl=target.U(proposal)-model.nll(proposal)
        logs=torch.stack([-.5*((proposal-m)**2).sum(1)+math.log(w)
                          for m,w in zip(means,[1/3,2/3])],1)
        responsibilities=torch.softmax(logs,1).mean(0)
        radii=torch.stack([((proposal-m)**2).sum(1) for m in means],1)
        off=(radii.min(1).values>8).float().mean()
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    report=dict(status='complete' if completed==iterations and finite else 'incomplete',
        iterations_requested=iterations,iterations_completed=completed,
        training_seconds=elapsed,loss_first=float(result['losses'][0]),loss_last=float(result['losses'][-1]),
        forward_kl=float(fkl.mean()),forward_kl_integration_se=float(fkl.std()/math.sqrt(len(fkl))),
        reverse_kl=float(rkl.mean()),reverse_kl_integration_se=float(rkl.std()/math.sqrt(len(rkl))),
        responsibility_mass=responsibilities.tolist(),target_mass=[1/3,2/3],
        off_component_probability=float(off),target_off_component_upper_bound=math.exp(-4),
        final_mh_acceptance=float(result['acc_rates'][-1]),
        finite=finite,torch_version=torch.__version__,scientific_promotion=False,
        role='pricing_only' if iterations==100 else 'single_original_code_reference',
        exact_published_figure_reproduction=False,posterior_confirmation='not_run')
    write(output/'result.json',report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['price','run','worker'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='worker':
        worker(args.output,args.iterations,args.seed)
        return 0
    if args.action=='price':
        iterations,seed,cap=100,1705,180.
    else:
        pricing=json.loads((OUTPUT/'price-r1/manifest.json').read_text())
        if pricing['exit_code']!=0:
            raise RuntimeError('original-code compatibility/pricing did not complete')
        iterations,seed=1500,1706
        cap=math.ceil(15*pricing['wall_seconds'])
        if cap>1800:
            raise RuntimeError('full source reference exceeds initial 1800-second reserved CPU allocation')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    index=1
    while (OUTPUT/f'{args.action}-r{index}').exists():
        index+=1
    output=OUTPUT/f'{args.action}-r{index}'
    output.mkdir()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg',
        'MPLCONFIGDIR':str(output/'matplotlib-cache'),'PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONUNBUFFERED':'1'}
    command=[PYTHON,str(Path(__file__).resolve()),'worker','--output',str(output),
             '--iterations',str(iterations),'--seed',str(seed)]
    manifest=dict(command=command,environment=PYTHON,plan='docs/plans/bayesfilter-neutra-author-reference-2026-10-05.md',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((UPSTREAM/'flonaco').glob('*.py'))},
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_threads=1,gpu_devices_intentionally_hidden=True,seed=seed,
        wall_limit=cap,cpu_limit=math.ceil(cap),output=str(output),
        result_file=str(output/'result.json'),data_version='paper_separation10_unitcov_weights1to2_symmetric_realization')
    write(output/'manifest.json',manifest)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.monotonic()
    with (output/'stdout.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CPU,(math.ceil(cap),math.ceil(cap))))
        try:
            code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    manifest.update(exit_code=code,wall_seconds=time.monotonic()-started,gpu_process_seconds=0.,
        cpu_core_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        status='complete' if code==0 else 'failed')
    write(output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','exit_code','wall_seconds','cpu_core_seconds','output')}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
