"""CPU-only post-run diagnosis of full-basis fit and top-M RBF recursion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();fixture=json.loads(args.fixture.read_text())
    def tensor(name):return tf.io.parse_tensor(tf.io.read_file(str(args.candidate/('fit_'+name+'.tensor'))),tf.float64).numpy()
    actual={k:tensor(k) for k in ('coefficients','beta','weights','knots','constant','target')}
    n=int(fixture['particles']);h=int(fixture['observations']);rows=[];first=[];fixture_error=0.
    def logpolicy(x,beta,weights,knots,constant):
        with np.errstate(divide='ignore'):return constant+logsumexp(np.log(weights)[None,:]-beta*(x[:,None]-knots[None,:])**2,axis=1)
    def integral(x,q,beta,weights,knots,constant):
        with np.errstate(divide='ignore'):return constant-.5*np.log(1+2*beta*q)+logsumexp(np.log(weights)[None,:]-beta/(1+2*beta*q)*(x[:,None]-knots[None,:])**2,axis=1)
    for b,case in enumerate(fixture['cases']):
        cloud=np.asarray(case['rounds'][0]['x']);refcoef=np.asarray(case['full_coefficients']);ref_target=np.asarray(case['fit_target']);policy=case['rounds'][1];first_case=None
        for t in range(h-1,-1,-1):
            x=cloud[t];beta=policy['beta'][t];G=np.exp(-beta*(x[:,None]-x[None,:])**2)
            a=actual['coefficients'][t,b];r=refcoef[t];ar=G@a;rr=G@r
            ar_target=actual['target'][t,b];rt=ref_target[t]
            same_target=bool(np.all(np.abs(ar_target-rt)<=1e-7+1e-6*np.abs(rt)))
            prediction_same=bool(np.all(np.abs(ar-rr)<=1e-7+1e-6*np.abs(rr)))
            candidate_policy=(actual['beta'][t,b],actual['weights'][t,b],actual['knots'][t,b],actual['constant'][t,b])
            reference_policy=(beta,np.asarray(policy['weights'][t]),np.asarray(policy['knots'][t]),policy['constant'][t])
            alog=logpolicy(x,*candidate_policy);rlog=logpolicy(x,*reference_policy)
            retained_same=bool(np.all(np.abs(alog-rlog)<=1e-7+1e-6*np.abs(rlog)))
            ix=np.argsort(-r,kind='stable')[:16]
            reconstructed=logpolicy(x,beta,r[ix],x[ix],policy['constant'][t]);fixture_error=max(fixture_error,float(np.max(np.abs(reconstructed-rlog))))
            parents=(.5*cloud[t-1]+25*cloud[t-1]/(1+cloud[t-1]**2)+8*np.cos(1.2*t)) if t>0 else np.zeros(n)
            ai=integral(parents,10. if t>0 else 5.,*candidate_policy);ri=integral(parents,10. if t>0 else 5.,*reference_policy)
            row=dict(seed=case['seed'],time=t,beta_error=float(abs(actual['beta'][t,b]-beta)),constant_error=float(abs(actual['constant'][t,b]-policy['constant'][t])),
                target_error=float(np.max(np.abs(ar_target-rt))),same_target=same_target,prediction_error=float(np.max(np.abs(ar-rr))),same_prediction=prediction_same,
                coefficient_error=float(np.max(np.abs(a-r))),retained_log_policy_error=float(np.max(np.abs(alog-rlog))),same_retained_policy=retained_same,
                retained_integral_error=float(np.max(np.abs(ai-ri))),top16_overlap=len(set(np.argsort(-a,kind='stable')[:16])&set(ix)),
                candidate_own_rss=float(np.sum((ar-ar_target)**2)),reference_rss=float(np.sum((rr-rt)**2)))
            rows.append(row)
            if first_case is None and not(same_target and prediction_same and retained_same):
                first_case=row
                frozen=dict(scope='FROZEN_RBF_NNLS_FAILURE',particles=n,x=x.tolist(),beta=float(beta),target=rt.tolist(),reference_coefficients=r.tolist(),seed=case['seed'],time=t)
                (args.output/('frozen_seed%d.json'%case['seed'])).write_text(json.dumps(frozen,allow_nan=False)+'\n')
        first.append(first_case)
    summary=dict(status='PASS' if fixture_error<=1e-10 else 'FAIL',scope='RBF_FIRST_DIVERGENCE_DIAGNOSIS',replication='INCOMPLETE',
        first_divergence=first,reference_top16_reconstruction_error=fixture_error,rows=rows,
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='postrun_diagnostic_only',wall_seconds=time.monotonic()-start)
    (args.output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
