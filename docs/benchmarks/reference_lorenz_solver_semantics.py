"""Independent CPU diagnostic of Octave solve semantics; never a runtime backend."""
import argparse
import contextlib
import hashlib
import io
import importlib.util
import re
import subprocess
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','reference','fallback'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--system-lapack',type=Path)
    args=parser.parse_args();started=time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    import numpy as np
    import scipy
    from scipy.linalg import lapack, lstsq
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    system=None;backend=dict(kind='scipy_lapack')
    if args.system_lapack:
        linkage=subprocess.run(['ldd','/usr/bin/octave-cli'],capture_output=True,text=True,check=True,timeout=10).stdout
        linked=re.search(r'liblapack\.so\.3 => (\S+)',linkage).group(1)
        assert args.system_lapack.resolve()==Path(linked).resolve(),'Not the Octave-linked LAPACK'
        helper=args.output/'sources/docs/benchmarks/reference_octave_system_lapack.py'
        spec=importlib.util.spec_from_file_location('system_lapack_reference',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        system=module.SystemLapack(args.system_lapack)
        check,details=system.solve(np.diag([2.,3.]),np.array([4.,9.]))
        assert np.allclose(check,[2.,3.],rtol=1e-14,atol=0) and details['branch']=='Positive Definite'
        backend=dict(kind='octave_linked_system_lapack',path=str(args.system_lapack.resolve()),sha256=sha(args.system_lapack),adapter_sha256=sha(helper))
        (args.output/'octave-linkage.txt').write_text(linkage)
    fixtures={p:json.loads((p/'fixture.json').read_text()) for p in (args.reference,args.fallback)}
    previous=json.loads((args.fallback/'summary.json').read_text())
    assert previous['status']=='PASS'
    def load(directory,name):
        item=fixtures[directory]['arrays'][name];p=directory/item['file']
        assert sha(p)==item['sha256'],'Corrupt input '+name
        return np.frombuffer(p.read_bytes(),dtype='<f8').reshape(item['shape'])
    native_g,native_r,tf_g,tf_r=[load(args.reference,k) for k in ('native_gram','native_rhs','tf_gram','tf_rhs')]
    expected=load(args.fallback,'ordinary_beta')
    rows=[];betas=[]
    for arm,(grams,rhs) in enumerate(((native_g,native_r),(native_g,tf_r),(tf_g,native_r),(tf_g,tf_r))):
        arm_beta=[]
        for k,(g,r) in enumerate(zip(grams,rhs)):
            norm=float(np.linalg.norm(g,1));ch,info=lapack.dpotrf(g,lower=1,clean=0)
            assert info==0,'Prior saved Gram Cholesky was positive definite'
            chol_rcond,info=lapack.dpocon(ch,norm,uplo='L');assert info==0
            lu_rcond=None;rank=None
            if float(chol_rcond)+1.0!=1.0:
                beta,info=lapack.dpotrs(ch,r,lower=1);assert info==0
                branch='Positive Definite'
            else:
                lu,piv,info=lapack.dgetrf(g);assert info>=0
                lu_rcond,cond_info=lapack.dgecon(lu,norm);assert cond_info==0
                if info==0 and float(lu_rcond)+1.0!=1.0:
                    beta,info=lapack.dgetrs(lu,piv,r);assert info==0
                    branch='Full'
                else:
                    beta,_,rank,spectrum=lstsq(g,r,cond=-1,lapack_driver='gelsd')
                    branch='Singular'
            if system is not None:
                beta,details=system.solve(g,r)
                branch=details['branch'];chol_rcond=details['chol_rcond']
                lu_rcond=details['lu_rcond'];rank=details['rank']
            assert np.all(np.isfinite(beta))
            delta=np.abs(beta-expected[arm,k]);scaled=delta/(1e-7+1e-9*np.abs(expected[arm,k]))
            inv,info=lapack.dpotrs(ch,np.eye(g.shape[0]),lower=1);assert info==0
            inverse_rcond=1.0/(norm*float(np.linalg.norm(inv,1)))
            row=dict(arm=arm,row=k,branch=branch,source_branch=previous['diagnostics']['cached_types'][arm][k],
                     branch_matches=branch==previous['diagnostics']['cached_types'][arm][k],
                     chol_rcond=float(chol_rcond),lu_rcond=None if lu_rcond is None else float(lu_rcond),
                     rank=None if rank is None else int(rank),coefficient_screen_passed=bool(np.all(scaled<=1)),
                     max_absolute_error=float(np.max(delta)),max_scaled_error=float(np.max(scaled)),
                     residual_norm=float(np.linalg.norm(g@beta-r)),
                     exact_inverse_rcond=inverse_rcond,exact_inverse_rounded_singular=inverse_rcond+1.0==1.0)
            rows.append(row);arm_beta.append(beta)
        betas.append(arm_beta)
    np.save(args.output/'explicit_branch_beta.npy',np.asarray(betas))
    configuration=io.StringIO()
    with contextlib.redirect_stdout(configuration):
        scipy.show_config()
    (args.output/'scipy-config.txt').write_text(configuration.getvalue())
    assert len(rows)==128
    summary=dict(status='PASS',scope='INDEPENDENT_SOURCE_SOLVER_SEMANTICS',replication='INCOMPLETE',
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='Independent reference only',
        backend=backend,numpy_version=np.__version__,scipy_version=scipy.__version__,lapack_epsilon=float(lapack.dlamch('E')),
        branches_agree=sum(r['branch_matches'] for r in rows),coefficients_agree=sum(r['coefficient_screen_passed'] for r in rows),
        count=len(rows),rows=rows,input_fixture_sha256={str(p):sha(p/'fixture.json') for p in fixtures},
        algorithm_seconds=time.monotonic()-started)
    summary['scientific_verdict']='CONDITIONAL_SOLVER_REFERENCE_AGREES' if summary['branches_agree']==128 and summary['coefficients_agree']==128 else 'CONDITIONAL_SOLVER_REFERENCE_DISAGREES'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','scientific_verdict','branches_agree','coefficients_agree','algorithm_seconds')}))


if __name__=='__main__':
    main()
