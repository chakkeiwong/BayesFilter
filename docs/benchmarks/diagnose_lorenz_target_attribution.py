"""CPU-only independent attribution of saved ADP targets; no runtime candidate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as reference
from diagnose_lorenz_policy_geometry import ratio_on_fixed_inputs


def polynomial(x,a,b,c):
    z=x.astype(np.longdouble)
    return (np.einsum('ni,ij,nj->n',z,a.astype(np.longdouble),z)+z@b.astype(np.longdouble)+np.longdouble(c)).astype(float)


def absolute_integral(x,a,b,c):
    base=reference.rk4(x,2.5); precision=1000*np.eye(8)+2*a
    covariance=np.linalg.inv(precision); mean=(1000*base-b)@covariance
    distance=mean-base
    value=.5*(np.linalg.slogdet(covariance)[1]-8*np.log(.001))
    return value-polynomial(mean,a,b,c)-500*np.sum(distance*distance,axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','root','design'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True)
    args=p.parse_args();start=time.monotonic()
    summary=dict(status='FAIL',scope='frozen_ADP_target_attribution',replication='INCOMPLETE',
        device='CPU',gpu_intentionally_hidden=True,role='independent_diagnostic_only',checks=[],rows=[],input_sha256={})
    def load(path):
        raw=path.read_bytes();summary['input_sha256'][str(path)]=hashlib.sha256(raw).hexdigest()
        return tf.io.parse_tensor(raw,tf.float64).numpy()
    def check(name,actual,expected,atol=1e-6,rtol=1e-8):
        error=float(np.max(np.abs(actual-expected)))
        summary['checks'].append(dict(name=name,passed=bool(np.all(np.abs(actual-expected)<=atol+rtol*np.abs(expected))),max_error=error))
    try:
        # Both candidates have the same pre-projection time0 regression (attempt128).
        # Attribute the feasibility-trigger candidate; this does not rerun the filter.
        candidate=args.candidate[0];meta=json.loads((candidate/'summary.json').read_text())
        folder=candidate/meta['candidates'][0]['folder'];summary['candidate']=candidate.name
        h={key:load(folder/(key+'.tensor')) for key in ('A','b','c','x','log_incremental_weight','y')}
        for stage,t in ((3,12),(4,13),(4,12),(4,11),(4,8),(4,1),(4,0)):
            old=[h[k][stage-1,:,0] for k in ('A','b','c')]
            new=[h[k][stage,:,0] for k in ('A','b','c')]
            x=h['x'][stage-1,t,0];weight=h['log_incremental_weight'][stage-1,t,0]
            delta,_=ratio_on_fixed_inputs(x,tuple(v[t+1] for v in old),tuple(v[t+1] for v in new))
            target=-weight-delta
            parts=[reference.regress(x,value) for value in (-weight,-delta,target)]
            for k,name in enumerate(('A','b','c')):check('target_linearity_%d_%d_%s'%(stage,t,name),parts[0][k]+parts[1][k],parts[2][k])
            y=h['y'][t,0] if h['y'].ndim==3 else h['y'][t]
            obs=5000*np.sum((x[:,:6]-y)**2,axis=-1)
            future=-absolute_integral(x,*(v[t+1] for v in new))
            oldq=polynomial(x,*(v[t] for v in old))
            check('absolute_target_identity_%d_%d'%(stage,t),target+oldq,obs+future)
            obs_a=np.diag([5000.]*6+[0.,0.]);obs_b=np.concatenate([-10000*y,[0.,0.]])
            obs_c=5000*float(y@y)
            fa,fb,fc,rss,sratio,center=reference.regress(x,future)
            da,db,dc=(old[k][t]+parts[2][k] for k in range(3))
            for name,actual,expected in zip(('A','b','c'),(da,db,dc),(obs_a+fa,obs_b+fb,obs_c+fc)):
                check('absolute_coefficient_identity_%d_%d_%s'%(stage,t,name),actual,expected)
            base_scale=100. if t==0 else 1000.
            precision=base_scale*np.eye(8)+2*da;ev,v=np.linalg.eigh(precision);weak=v[:,0]
            row=dict(stage=stage,time=t,min_precision_eigenvalue=float(ev[0]),singular_ratio=sratio,
                future_fit_rms=float(np.sqrt(rss/len(x))),future_target_std=float(np.std(future)),
                delta_std=float(np.std(delta)),weight_std=float(np.std(weight)),
                weakest_direction=weak.tolist(),weak_observed_squared_mass=float(np.sum(weak[:6]**2)),
                weak_precision_terms=dict(base=base_scale,observation=float(2*weak@obs_a@weak),future=float(2*weak@fa@weak)),
                weak_linear_terms=dict(observation=float(weak@obs_b),future=float(weak@fb)))
            if t==0:
                mean=np.linalg.solve(precision,-db);oldprecision=100*np.eye(8)+2*old[0][0]
                oldmean=np.linalg.solve(oldprecision,-old[1][0]);components=[]
                for label,fit in zip(('current_weight','future_correction'),parts[:2]):
                    shift=-np.linalg.solve(precision,fit[1]+2*fit[0]@oldmean)
                    components.append(dict(name=label,mean_shift=shift.tolist(),norm=float(np.linalg.norm(shift))))
                check('mean_shift_target_decomposition',sum(np.asarray(k['mean_shift']) for k in components),mean-oldmean)
                row.update(mean=mean.tolist(),oldmean=oldmean.tolist(),mean_shift_components=components,
                    weak_mean_component=float(weak@mean),
                    empirical_coordinate_extrapolation=((mean-center['center'])/center['scale']).tolist())
            summary['rows'].append(row)
        summary['status']='PASS' if all(c['passed'] for c in summary['checks']) else 'FAIL'
        summary['scientific_verdict']='EXPLANATORY_TARGET_DECOMPOSITION_NO_REPAIR_ADMITTED'
    except Exception as exc:summary['error']=repr(exc)
    summary['wall_seconds']=time.monotonic()-start
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'],checks=len(summary['checks']),error=summary.get('error'),wall_seconds=summary['wall_seconds'])))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
