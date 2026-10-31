"""Independent CPU/reference diagnostics only; NumPy is not a runtime backend."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
ROOT = Path('/home/chakwong/BayesFilter')
BASE = ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PREVIOUS = BASE/'gap-closure-20261010'

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def main(out):
    import numpy as np
    from scipy.io import loadmat
    from scipy.linalg import lstsq
    from scipy.linalg.lapack import dposv, dgesv
    import torch
    torch.set_num_threads(1)
    out.mkdir(exist_ok=False)
    mat = loadmat(PREVIOUS/'c0-source-fit-20261010/fit-trace.mat', simplify_cells=True)
    assert mat['source_result_exact'] and mat['source_rng_exact']
    rows = []
    for row in mat['FIT_TRACE']:
        d, y, g, rhs, expected = [row[k] for k in ('design','target','gram','rhs','beta')]
        assert np.array_equal(expected,row['repeated_beta'])
        _, chol, info = dposv(g, rhs, lower=0)
        _, _, dense, dense_info = dgesv(g,rhs)
        truncated, _, rank, singular = lstsq(g,rhs,cond=np.finfo(float).eps,lapack_driver='gelsd')
        lapack_default = lstsq(g,rhs,cond=-1.,lapack_driver="gelsd")[0]
        direct = lstsq(d,y,cond=np.finfo(float).eps,lapack_driver='gelsd')[0]
        candidates = {'cholesky':chol,'dense':dense,'gram_gelsd_epsilon':truncated,'gram_gelsd_lapack_default':lapack_default,'design_gelsd':direct}
        checks = {}
        for name,beta in candidates.items():
            finite = np.isfinite(beta).all()
            checks[name] = dict(finite=bool(finite), coefficient_max_error=float(np.max(np.abs(beta-expected))) if finite else None,
                prediction_max_error=float(np.max(np.abs(d@(beta-expected)))) if finite else None,
                residual_l2=float(np.linalg.norm(d@beta-y)) if finite else None)
        checks['source'] = dict(residual_l2=float(np.linalg.norm(d@expected-y)))
        rows.append(dict(time=int(row['time']),branch=row['branch'],source_rcond=float(row['rcond']),
            gram_rank=int(rank),dimension=g.shape[0],smallest_singular_ratio=float(singular[-1]/singular[0]),
            cholesky_info=int(info),dense_info=int(dense_info),design_condition=float(np.linalg.cond(d)),
            independently_recomputed_gram_max_error=float(np.max(np.abs(d.T@d-g))),
            independently_recomputed_rhs_max_error=float(np.max(np.abs(d.T@y-rhs))),checks=checks))
    write(out/'csmc-solvers.json',dict(cpu_only=True,gpu_intentionally_hidden=True,
        numpy_role='independent reference diagnosis only',rows=rows))

    # Replay the saved rounded Gram/RHS in the same Octave executable. This
    # checks the branch independently of SciPy/LAPACK implementation choices.
    oq=lambda p:"'"+str(p).replace("'","''")+"'"
    driver="load("+oq(PREVIOUS/'c0-source-fit-20261010/fit-trace.mat')+"); rows=struct([]);\n"
    driver+="for k=1:numel(FIT_TRACE); r=FIT_TRACE{k}; b=r.gram\\r.rhs; rows(k).time=r.time; rows(k).branch=matrix_type(r.gram); rows(k).coefficient_max_error=max(abs(b-r.beta)); rows(k).prediction_max_error=max(abs(r.design*(b-r.beta))); end;\n"
    driver+="fid=fopen("+oq(out/'octave-replay.tsv')+",'w'); for k=1:numel(rows); r=rows(k); fprintf(fid,'%d\\t%s\\t%.17g\\t%.17g\\n',r.time,r.branch,r.coefficient_max_error,r.prediction_max_error); end; fclose(fid);\n"
    (out/'octave-replay.m').write_text(driver)
    with (out/'octave-replay.log').open('w') as log:
        subprocess.run(['/usr/bin/octave-cli','--no-gui','--quiet',str(out/'octave-replay.m')],stdout=log,stderr=log,check=True,timeout=60)
    replay=[]
    for line in (out/'octave-replay.tsv').read_text().splitlines():
        t,branch,coefficient,prediction=line.split('\t')
        replay.append(dict(time=int(t),branch=branch,coefficient_max_error=float(coefficient),prediction_max_error=float(prediction)))
    assert len(replay)==len(rows)
    write(out/'octave-replay.json',replay)

    ref=json.loads((PREVIOUS/'localization-01/author.json').read_text())
    assert len(ref['replay_controls'])==64 and all(x['pass_check'] for x in ref['replay_controls'])
    source=ROOT/'.localresources/code/younis-mdpf-neurips-2023/src/models/particle_transformer.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==ref['author_hashes']['src/models/particle_transformer.py']
    spec=importlib.util.spec_from_file_location('author_transformer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    transformer=module.BearingsOnlyTransformer()
    angle_cases=[]
    for case in ref['cases']:
        dtype=getattr(torch,case['dtype'])
        for j,update in enumerate(case['updates']):
            raw=torch.tensor(update['captured']['proposal_before_normalization'],dtype=dtype).reshape(-1,4)
            radius=torch.linalg.vector_norm(raw[:,2:],dim=-1)
            indices=torch.cat((torch.arange(32),torch.argsort(radius)[:32]))
            x=raw[indices].clone().requires_grad_(True)
            angles=transformer.forward_tranform(transformer.apply_norm(x))[:,2]
            gradient=torch.autograd.grad(angles.sum(),x)[0][:,2:]
            uv=x.detach()[:,2:].double();r2=(uv*uv).sum(-1)
            # Here u=sin(theta), v=cos(theta), so theta=atan2(u,v).
            exact=torch.stack((uv[:,1],-uv[:,0]),-1)/r2[:,None]
            perturbed=x.detach().clone();perturbed[:,2]+=1e-8
            angle2=transformer.forward_tranform(transformer.apply_norm(perturbed))[:,2]
            delta=torch.atan2(torch.sin(angle2-angles.detach()),torch.cos(angle2-angles.detach()))
            angle_cases.append(dict(variant=case['variant'],dtype=case['dtype'],update=j,
                inputs=x.detach().tolist(),angles=angles.detach().tolist(),gradients=gradient.tolist(),
                exact_gradients=exact.tolist(),minimum_radius=float(radius.min()),
                source_gradient_max_relative_error=float(((gradient.double()-exact).abs()/(1+exact.abs())).max()),
                perturbation_requested=1e-8,maximum_realized_perturbation=float((perturbed-x.detach()).abs().max()),
                maximum_wrapped_angle_change=float(delta.abs().max())))
    write(out/'angles.json',dict(cpu_only=True,gpu_intentionally_hidden=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=angle_cases))
    summary=json.loads((BASE/'csmc-qualification-20261009-final-013640/summary.json').read_text())
    forecast=[]
    for row in summary['strata']:
        inf=row['inference'];mean_width=(inf['upper'][0]-inf['lower'][0])/2
        # Descriptive precision scenarios, not prospective power calculations.
        variance_width=(inf['upper'][1]-inf['lower'][1])/2
        mean_margin=np.log(1.01)-abs(inf['estimate'][0])
        variance_margin=np.log(1.05)-inf['estimate'][1]
        mean_upper_width=max(inf['upper'][0]-inf['estimate'][0],inf['estimate'][0]-inf['lower'][0])
        variance_upper_width=inf['upper'][1]-inf['estimate'][1]
        forecast.append(dict(cell=row['cell'],forcing=row['forcing'],pairs=row['valid_pairs'],
            likelihood_adequate=inf['likelihood_adequate'],variance_adequate=inf['variance_adequate'],
            likelihood_pass=inf['likelihood_pass'],variance_pass=inf['variance_pass'],
            supported_mean_discrepancy=inf['supported_mean_discrepancy'],
            supported_variance_degradation=inf['supported_variance_degradation'],
            additional_failures=len(row['additional_port_failure_ids']),
            plug_in_mean_pairs=None if mean_margin<=0 else int(np.ceil(row['valid_pairs']*(mean_upper_width/mean_margin)**2)),
            plug_in_variance_pairs=None if variance_margin<=0 else int(np.ceil(row['valid_pairs']*(variance_upper_width/variance_margin)**2)),
            estimate=inf['estimate'],lower=inf['lower'],upper=inf['upper'],
            residual_effective_count=inf['residual_effective_count'],
            influence_maximum_share=inf['influence_maximum_share'],
            zero_effect_mean_pairs=int(np.ceil(row['valid_pairs']*(mean_width/np.log(1.01))**2)),
            zero_effect_variance_pairs=int(np.ceil(row['valid_pairs']*(variance_width/np.log(1.05))**2))))
    write(out/'precision-feasibility.json',dict(scope='descriptive forecast, not a power guarantee',
        assumptions='zero-effect and observed-effect scenarios, constant asymptotic variance and sqrt(n) scaling; neither validates tail adequacy, and heavy tails can invalidate the forecasts',
        rows=forecast))
    print(json.dumps(dict(status='COMPLETED',solver_rows=len(rows),angle_cases=len(angle_cases),qualification_cells=len(forecast))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
