"""Independent CPU diagnostic: observe author regressions without altering them."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oq(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    start = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import numpy as np
    from scipy.io import loadmat

    campaign = root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    capture = campaign/'attempt147_p0_forward_capture'
    fixture = json.loads((capture/'fixture.json').read_text())
    source = root/'.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz'
    baseline = Path(fixture['reference_root'])
    archive = baseline/'author-F4.8801-seed20260930.mat'
    assert sha(source/'cSMC_Resample.m') == fixture['source_sha256']
    assert sha(archive) == next(c for c in fixture['cases'] if c['case'] == 3)['reference_archive_sha256']
    observed = args.output/'observed'
    observed.mkdir()
    original = (source/'cSMC_Resample.m').read_text()
    anchor = "            BetaHat = (DesignMatrix' * DesignMatrix) \\ (DesignMatrix' * Value); % (d^2/2 + 3d/2 + 1) x 1"
    assert original.count(anchor) == 1
    observer = '''
            global FIT_TRACE;
            row=struct('time',t-1,'design',DesignMatrix,'target',Value,...
                'correction',LogNormalisingFunc,'beta',BetaHat);
            row.gram=DesignMatrix'*DesignMatrix;row.rhs=DesignMatrix'*Value;
            row.repeated_beta=row.gram\\row.rhs;
            assert(isequal(row.repeated_beta,BetaHat));
            row.branch=matrix_type(row.gram);row.rcond=rcond(row.gram);
            FIT_TRACE{end+1}=row;
'''
    (observed/'cSMC_Resample.m').write_text(original.replace(anchor, anchor+observer))
    driver = f'''root={oq(root)};source={oq(source)};output={oq(args.output)};
stats=fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst');
addpath(stats);addpath(fullfile(stats,'distributions'));addpath(source);cd(source);
driver=fileread(fullfile(source,'cSMC_VaryF.m'));
parts=strsplit(driver,'% Vary F');assert(numel(parts)==2);eval(parts{{1}});
cd(output);addpath(fullfile(output,'observed'),'-begin');
assert(strcmp(which('cSMC_Resample'),fullfile(output,'observed/cSMC_Resample.m')));
F=4.8801;seed=20260930;
Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
global FIT_TRACE;FIT_TRACE={{}};
rand('state',seed);randn('state',seed);
result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
baseline=load({oq(archive)});assert(isequal(result,baseline.result));
tape=load({oq(capture/'author-F4.8801-seed20260930.mat')});
assert(isequal(rand('state'),tape.final_rand));
assert(isequal(randn('state'),tape.final_randn));
assert(numel(FIT_TRACE)==101);
source_result_exact=true;source_rng_exact=true;
save('-mat7-binary',fullfile(output,'fit-trace.mat'),'FIT_TRACE','source_result_exact','source_rng_exact');
'''
    (args.output/'driver.m').write_text(driver)
    command=['/usr/bin/octave-cli','--no-gui','--quiet',str(args.output/'driver.m')]
    with (args.output/'octave.log').open('w') as log:
        subprocess.run(command,cwd=args.output,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    trace=loadmat(args.output/'fit-trace.mat',simplify_cells=True)
    rows=sorted(trace['FIT_TRACE'],key=lambda r:r['time'])
    assert len(rows)==101 and trace['source_result_exact'] and trace['source_rng_exact']
    diagnostics=[]
    for row in rows:
        design=row['design'];y=row['target'];beta=row['beta']
        x=design[:,1:9];mu=x.mean(0);scale=np.sqrt(np.mean((x-mu)**2,0));z=(x-mu)/scale
        pairs=[(i,j) for i in range(8) for j in range(i+1,8)]
        centered=np.column_stack([np.ones(len(x)),z,*[z[:,i]*z[:,j] for i,j in pairs],z*z])
        q,r=np.linalg.qr(centered,mode='reduced')
        bq=np.linalg.solve(r,q.T@(y-y.mean()));bq[0]+=y.mean()
        fitted_q=centered@bq;fitted_source=design@beta
        residual=fitted_source-y;qr_residual=fitted_q-y
        # Normal-equation residual in a well-scaled basis tests whether source
        # coefficients solve the intended least-squares objective accurately.
        gradient=centered.T@residual
        singular=np.linalg.svd(design,compute_uv=False)
        diagnostics.append(dict(time=int(row['time']),branch=row['branch'],rcond=float(row['rcond']),
            design_condition=float(singular[0]/singular[-1]),centered_design_condition=float(np.linalg.cond(centered)),
            source_residual_rms=float(np.sqrt(np.mean(residual**2))),
            qr_residual_rms=float(np.sqrt(np.mean(qr_residual**2))),
            source_vs_qr_prediction_max=float(np.max(np.abs(fitted_source-fitted_q))),
            source_vs_qr_prediction_rms=float(np.sqrt(np.mean((fitted_source-fitted_q)**2))),
            source_centered_gradient_max=float(np.max(np.abs(gradient))),
            qr_centered_gradient_max=float(np.max(np.abs(centered.T@qr_residual)))))
    arrays={key:np.stack([row[key] for row in rows]) for key in ('design','target','correction','gram','rhs','beta')}
    np.savez(args.output/'regressions.npz',**arrays)
    summary=dict(status='PASS',replication='INCOMPLETE',scope='AUTHOR_BACKWARD_REGRESSION_TRACE',
        source_result_exact=True,source_rng_exact=True,rows=diagnostics,
        branch_counts={b:sum(r['branch']==b for r in diagnostics) for b in sorted(set(r['branch'] for r in diagnostics))},
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='independent reference diagnostic',
        input_sha256={str(source/'cSMC_Resample.m'):sha(source/'cSMC_Resample.m'),str(archive):sha(archive)},
        source_instrumentation='Original solve unchanged; observer adds copies and a checked identical repeated solve.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    manifest=dict(status='PASS',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        command=[sys.executable,*sys.argv],subprocess_command=command,cwd=str(root),environment=sys.executable,
        cpu_only=True,gpu_intentionally_hidden=True,seed=20260930,forcing=4.8801,
        plan='docs/plans/controlled-smc-offset-localization-20261006.md',
        result_file='docs/plans/controlled-smc-offset-localization-result-20261006.md',
        wall_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=summary['input_sha256'],source_sha256={str(Path(__file__)):sha(Path(__file__))},
        output_sha256={name:sha(args.output/name) for name in ('fit-trace.mat','regressions.npz','summary.json','driver.m','observed/cSMC_Resample.m','octave.log')})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',wall_seconds=manifest['wall_seconds'],branch_counts=summary['branch_counts'],
        largest_prediction_discrepancies=sorted(diagnostics,key=lambda r:-r['source_vs_qr_prediction_max'])[:5])))


if __name__=='__main__':
    main()
