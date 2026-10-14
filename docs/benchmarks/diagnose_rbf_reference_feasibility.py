"""CPU-only independent-reference diagnosis of NNLS boundary roundoff."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
import time
import numpy as np
from scipy.optimize import lsq_linear


def boundary_projection(design, coefficients):
    """Reference-only candidate; explicitly reject corrections beyond roundoff."""
    band=16*np.finfo(float).eps*max(1.,float(np.max(np.abs(coefficients))))
    projected=np.maximum(coefficients,0.)
    before=design@coefficients;displacement=float(np.max(np.abs(design@(projected-coefficients))))
    error_bound=float(np.max(np.sum(np.abs(design),axis=1))*np.max(np.abs(projected-coefficients)))
    okay=bool(np.min(coefficients)>=-band and displacement<=1e-12*(1+float(np.max(np.abs(before)))))
    return projected,dict(eligible=okay,minimum_before=float(np.min(coefficients)),negative_count=int(np.sum(coefficients<0)),
        roundoff_band=band,prediction_displacement=displacement,prediction_error_bound=error_bound,
        healthy_unchanged=bool(np.array_equal(projected,coefficients)) if np.min(coefficients)>=0 else None)


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();data=np.load(args.fixture);design=data['design'];target=data['target'];n=len(target)
    tol=10*np.finfo(float).eps*np.linalg.norm(design,1)*n
    solved=lsq_linear(design,target,bounds=(0.,np.inf),method='bvls',lsq_solver='exact',tol=1e-14,max_iter=5*n)
    def metrics(coef):
        residual=design@coef-target;gradient=design.T@residual
        return dict(finite=bool(np.all(np.isfinite(coef))),nonnegative=bool(np.min(coef)>=0),minimum=float(np.min(coef)),
            dual_error=max(0.,float(-np.min(gradient))),dual_ok=bool(np.min(gradient)>=-20*tol),
            complementarity=float(np.max(np.abs(coef*gradient))),complementarity_ok=bool(np.max(np.abs(coef*gradient))<=20*tol*max(1.,float(np.max(coef)))),rss=float(residual@residual))
    projected,correction=boundary_projection(design,solved.x);before=metrics(solved.x);after=metrics(projected)
    healthy=np.array([0.,.1,2.]);same,healthy_info=boundary_projection(np.eye(3),healthy)
    _,bad=boundary_projection(np.eye(3),np.array([-1e-5,1.,2.]))
    checks=dict(converged=bool(solved.success),roundoff_correction_eligible=correction['eligible'],
        projected_valid=all(after[k] for k in ('finite','nonnegative','dual_ok','complementarity_ok')),
        healthy_unchanged=bool(np.array_equal(same,healthy)),material_negative_rejected=not bad['eligible'])
    summary=dict(status='PASS' if all(checks.values()) else 'FAIL',scope='RBF_REFERENCE_FEASIBILITY_DIAGNOSIS',replication='INCOMPLETE',
        checks=checks,before=before,after=after,correction=correction,gradient_tolerance=float(tol),
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='independent_reference_only',wall_seconds=time.monotonic()-start)
    np.save(args.output/'reference_coefficients.npy',projected)
    (args.output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(summary))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
