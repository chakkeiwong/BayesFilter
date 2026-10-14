"""Independent CPU reference for the single-case Lorenz offset diagnosis."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from reference_octave_system_lapack import SystemLapack
    manifest=json.loads((args.trace/'manifest.json').read_text())
    assert sha(args.trace/'regressions.npz')==manifest['output_sha256']['regressions.npz']
    original=json.loads((args.trace/'summary.json').read_text())
    assert original['source_result_exact'] and original['source_rng_exact']
    library=Path('/usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3')
    assert sha(library)=='426b4bf536065cc63ef879444dad76dbc9f85e36e5cde906269acd6f177c5151'
    solver=SystemLapack(library)
    values=np.load(args.trace/'regressions.npz'); rows=[];qr_prediction=[]
    for t in range(101):
        beta,details=solver.solve(values['gram'][t],values['rhs'][t])
        assert np.array_equal(beta,values['beta'][t]),t
        assert details['branch']==original['rows'][t]['branch']
        x=values['design'][t,:,1:9];y=values['target'][t]
        mu=x.mean(0);z=(x-mu)/np.sqrt(np.mean((x-mu)**2,0))
        pairs=[(i,j) for i in range(8) for j in range(i+1,8)]
        design=np.column_stack([np.ones(len(x)),z,*[z[:,i]*z[:,j] for i,j in pairs],z*z])
        q,r=np.linalg.qr(design,mode='reduced')
        bq=np.linalg.solve(r,q.T@(y-y.mean()));bq[0]+=y.mean()
        qr_prediction.append(design@bq)
        rows.append(dict(time=t,coefficient_exact=True,**details))
    arrays={}
    for name,value in dict(x=values['design'][:,:,1:9],target=values['target'],beta=values['beta'],
                           qr_prediction=np.asarray(qr_prediction),
                           svd_mask=np.asarray([r['branch']=='Singular' for r in rows],dtype='float64')).items():
        path=args.output/(name+'.bin');np.asarray(value,dtype='<f8').tofile(path)
        arrays[name]=dict(file=path.name,shape=list(value.shape),sha256=sha(path))
    result=dict(status='PASS',scope='SINGLE_CASE_SOURCE_SOLVER_AND_QR_REFERENCE',replication='INCOMPLETE',
        source_coefficients_exact=101,source_solver_rows=rows,arrays=arrays,
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='independent reference only',
        case=3,forcing=4.8801,seed=20260930,
        input_trace_sha256=sha(args.trace/'regressions.npz'),library_sha256=sha(library))
    (args.output/'fixture.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    metadata=dict(status='PASS',command=[sys.executable,*sys.argv],environment=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        plan='docs/plans/controlled-smc-offset-localization-20261006.md',
        result_file='docs/plans/controlled-smc-offset-localization-result-20261006.md',
        cpu_only=True,gpu_intentionally_hidden=True,wall_seconds=time.monotonic()-start,
        ended_utc=datetime.now(timezone.utc).isoformat(),seeds=[20260930],
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name('reference_octave_system_lapack.py')):sha(Path(__file__).with_name('reference_octave_system_lapack.py'))},
        input_artifacts_sha256={str(args.trace/'regressions.npz'):sha(args.trace/'regressions.npz')},
        output_sha256={item['file']:item['sha256'] for item in arrays.values()})
    metadata['output_sha256']['fixture.json']=sha(args.output/'fixture.json')
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',source_coefficients_exact=101,svd_rows=sum(r['branch']=='Singular' for r in rows),
                          retained_ranks=sorted(set(r['rank'] for r in rows if r['rank'] is not None)),wall_seconds=metadata['wall_seconds'])))


if __name__=='__main__':main()
