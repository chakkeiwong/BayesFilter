"""Independent CPU reference: alternative solvers on a frozen NNLS failure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import lsq_linear, nnls
from diagnose_rbf_reference_feasibility import boundary_projection
from reference_rbf_svd_nnls import svd_nnls


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--svd-only',action='store_true');args=parser.parse_args();start=time.monotonic();data=np.load(args.fixture);G=data['design'];y=data['target'];n=len(y)
    tol=10*np.finfo(float).eps*np.linalg.norm(G,1)*n;trials=[];selected=None
    controls=[]
    if args.svd_only:
        for matrix,signal,expected in [
            (np.eye(3),[2.,-1.,0.],[2.,0.,0.]),
            ([[10.,1.,0.],[0.,1.,0.],[0.,0.,1.]],[1.,2.,0.],[0.,1.5,0.]),
            ([[1.,1.,0.],[0.,0.,1.],[0.,0.,0.]],[2.,3.,0.],[2.,0.,3.]),
            (np.eye(3),[0.,0.,0.],[0.,0.,0.])]:
            solution,info=svd_nnls(np.asarray(matrix),np.asarray(signal))
            controls.append(bool(np.allclose(solution,expected,atol=1e-12,rtol=1e-12)))
    methods=(('svd_active_set',.01),) if args.svd_only else (('trf',1e-14),('trf',1e-15),('scaled_nnls',None))
    for method,accuracy in methods:
        begun=time.monotonic();trial=dict(method=method,tolerance=accuracy);w=None
        try:
            if method=='svd_active_set':
                w,info=svd_nnls(G,y,stopping_factor=accuracy);converged=True;trial['solver_diagnostics']=info
            elif method=='trf':
                result=lsq_linear(G,y,bounds=(0.,np.inf),method='trf',lsq_solver='exact',tol=accuracy,max_iter=5*n)
                w=result.x;converged=bool(result.success);trial['iterations']=int(result.nit)
            else:
                scale=np.sqrt(np.sum(G*G,axis=0));z,_=nnls(G/scale[None,:],y,maxiter=5*n)
                w=z/scale;converged=True
            w,correction=boundary_projection(G,w);r=G@w-y;g=G.T@r
            dual=max(0.,float(-np.min(g)));comp=float(np.max(np.abs(w*g)))
            okay=bool(converged and correction['eligible'] and np.all(np.isfinite(w)) and np.min(w)>=0 and dual<=20*tol and comp<=20*tol*max(1.,float(np.max(w))))
            # Gaussian entries are positive and each column sum is >=1.
            # Shift the residual into dual feasibility; this provides an
            # independently recomputed convex primal-dual gap diagnostic.
            shift=max(0.,float(np.max(-g/np.sum(G,axis=0))))+8*np.finfo(float).eps
            dual_point=r+shift;dual_gradient=G.T@dual_point
            gap=float(w@dual_gradient+.5*np.sum((dual_point-r)**2))
            trial.update(valid=okay,converged=converged,dual_error=dual,complementarity=comp,rss=float(r@r),
                boundary_correction=correction,positive_coefficients=int(np.count_nonzero(w)),primal_dual_gap=gap,
                dual_certificate_feasible=bool(np.min(dual_gradient)>=0),prediction_error_bound=float(np.sqrt(2*max(0.,gap))))
            np.save(args.output/(method+'_'+str(accuracy)+'_coefficients.npy'),w)
            if okay and selected is None:selected=dict(method=method,tolerance=accuracy,coefficients=w.tolist())
        except Exception as exc:trial.update(valid=False,failure=repr(exc))
        trial['seconds']=time.monotonic()-begun;trials.append(trial)
        (args.output/'progress.json').write_text(json.dumps(dict(trials=trials),allow_nan=False,indent=2)+'\n')
    summary=dict(status='PASS' if selected and all(controls) else 'FAIL',scope='RBF_REFERENCE_SOLVER_DIAGNOSIS',replication='INCOMPLETE',trials=trials,controls=controls,
        selected={k:v for k,v in selected.items() if k!='coefficients'} if selected else None,
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='independent_reference_only',gradient_tolerance=float(tol),wall_seconds=time.monotonic()-start)
    if selected:
        fixture=dict(scope='FROZEN_RBF_NNLS_FAILURE',particles=n,x=data['x'].tolist(),beta=float(data['beta']),target=y.tolist(),reference_coefficients=selected['coefficients'])
        (args.output/'fixture.json').write_text(json.dumps(fixture,allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(summary))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
