#!/usr/bin/env python3
"""Independent saved-path diagnostic; NumPy/mpmath never feed runtime decisions.

Replays the exact PP RK4 equations in the generated author callbacks. This
classifies a failed saved path; it is not a replacement transition kernel.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import mpmath as mp
from scipy.io import loadmat
from scipy.special import ndtr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args(); run=args.run.resolve(); out=args.output_root.resolve()
    out.mkdir(parents=True,exist_ok=False)
    badpath=next(run.glob('invalid-smoothing-t*.mat'))
    d=loadmat(badpath); T=d['sams'].shape[2]-1
    preamble=(run/'run.m').read_text().split("setenv('BAYESFILTER_REFERENCE_PROGRESS'")[0]
    q=lambda s:"'"+str(s).replace("'","''")+"'"
    script=preamble+f"""
cd({q(out)}); name='pp'; d=6; m=2; n=2; T={T};
rng(1); myModel=setup(ssmodel(name,d,m,n,T));
myModel.pre.ncons=[.1;.1;0;0;1;0];
load({q(run/'data.mat')}); myModel.Y=observations;
load({q(badpath)}); N=size(sams,2);
prior_log=reference_logprior(myModel,fulldata(1:d+m,:,1));
transition_logs=zeros(N,T); observation_logs=zeros(N,T); means=zeros(m,N,T);
for k=1:T
 transition_logs(:,k)=reference_logtransition(myModel,fulldata(:,:,k+1),k)';
 observation_logs(:,k)=reference_loglike(myModel,fulldata(:,:,k+1),k)';
 theta=bsxfun(@plus,myModel.pre.ncons,normcdf(thetas));
 means(:,:,k)=predator_step(myModel,sams(:,:,k),theta,'RK4');
end
save('-mat7-binary','density-replay.mat','prior_log','transition_logs','observation_logs','means','theta');
"""
    (out/'run.m').write_text(script)
    start=time.monotonic()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        subprocess.run(['octave-cli','--quiet','--no-gui',str(out/'run.m')],
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
                       stdout=stdout,stderr=stderr,timeout=60,check=True)
    replay=loadmat(out/'density-replay.mat'); logs=replay['transition_logs']
    invalid=np.argwhere(~np.isfinite(logs)); records=[]
    mp.mp.dps=100
    for row,k in invalid:
        theta=[mp.mpf(float(v)) for v in replay['theta'][:,row]]
        r,s,u,v,K0,a0=theta; K=90+20*K0; a=20+10*a0; h=mp.mpf('.1')
        x=[mp.mpf(float(v)) for v in d['sams'][:,row,k]]
        observed=[mp.mpf(float(v)) for v in d['sams'][:,row,k+1]]
        def f(z):
            transfer=z[0]*z[1]/(a+z[0])
            return [r*z[0]*(1-z[0]/K)-s*transfer,u*transfer-v*z[1]]
        trace=[]
        for step in range(20):
            f1=f(x); f2=f([x[j]+h*f1[j]/2 for j in range(2)])
            f3=f([x[j]+h*f2[j]/2 for j in range(2)])
            f4=f([x[j]+h*f3[j] for j in range(2)])
            x=[x[j]+h*(f1[j]+2*f2[j]+2*f3[j]+f4[j])/6 for j in range(2)]
            trace.append(dict(substep=step+1,signs=[int(mp.sign(v)) for v in x],
                              log10_abs=[float(mp.log10(abs(v))) for v in x]))
        logp=-mp.log(2*mp.pi)-2*mp.log(2)-sum((observed[j]-x[j])**2 for j in range(2))/8
        records.append(dict(path_index_zero_based=int(row),transition_time=int(k+1),
            initial_state=d['sams'][:,row,k].tolist(),next_state=d['sams'][:,row,k+1].tolist(),
            physical_theta=[float(r),float(K),float(a),float(s),float(u),float(v)],
            log10_abs_negative_log_transition=float(mp.log10(-logp)),
            high_precision_finite=bool(mp.isfinite(logp)),trace=trace))
    result=dict(schema='zhao_cui_saved_path_overflow_diagnostic_v1',
        source_transition=str(run/'derived/models/pp/predator_step.m'),
        source_equations=str(run/'derived/models/pp/odefun.m'),
        cpu_only=True,gpu_intentionally_hidden=True,mpmath_decimal_digits=mp.mp.dps,
        nonfinite_transition_count=int((~np.isfinite(logs)).sum()),
        nonfinite_prior_count=int((~np.isfinite(replay['prior_log'])).sum()),
        nonfinite_observation_count=int((~np.isfinite(replay['observation_logs'])).sum()),
        records=records,wall_seconds=time.monotonic()-start,
        limitation='High precision replay diagnoses overflow; no failed path was dropped and no inference run was repaired here.')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    brief={k:v for k,v in result.items() if k!='records'}
    brief['failures']=[{k:v for k,v in row.items() if k!='trace'} for row in records]
    print(json.dumps(brief,indent=2))


if __name__=='__main__':main()
