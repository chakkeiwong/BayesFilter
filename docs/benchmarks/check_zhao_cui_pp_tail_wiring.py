#!/usr/bin/env python3
"""Independent saved-path parity check of the isolated Octave tail adapter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.io import loadmat
from run_zhao_cui_publication_replication import prepare,quote


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
    run=args.run.resolve();replay=args.replay.resolve();derived=prepare(out,'paper',pp_tail_precision_repair=True)
    prefix=(run/'run.m').read_text().split("setenv('BAYESFILTER_REFERENCE_PROGRESS'")[0]
    script=prefix+f"""
cd({quote(out)}); name='pp';d=6;m=2;n=2;T=5;
rng(1);myModel=setup(ssmodel(name,d,m,n,T));myModel.pre.ncons=[.1;.1;0;0;1;0];
load({quote(run/'data.mat')});myModel.Y=observations;
load({quote(run/'invalid-smoothing-t05.mat')});
load({quote(replay/'density-replay.mat')});baseline=transition_logs;
addpath({quote(derived/'models')});clear reference_logtransition;
repaired=zeros(size(baseline));
for k=1:T
 repaired(:,k)=reference_logtransition(myModel,fulldata(:,:,k+1),k)';
end
healthy=isfinite(baseline);
if ~isequal(repaired(healthy),baseline(healthy)),error('healthy density changed');end
if any(isnan(repaired(:)) | repaired(:)==Inf),error('invalid repaired density');end
raw_log_weight=prior_log+sum(repaired+observation_logs,2)'-logpdf_e;
w=exp(raw_log_weight-max(raw_log_weight));w=w/sum(w);
if any(~isfinite(w)) || abs(sum(w)-1)>1e-12,error('invalid repaired weights');end
save('-mat7-binary','repaired-paths.mat','sams','thetas','w','raw_log_weight','repaired','baseline');
"""
    (out/'run.m').write_text(script);start=time.monotonic()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        subprocess.run(['octave-cli','--quiet','--no-gui',str(out/'run.m')],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
                 'BAYESFILTER_REFERENCE_PYTHON':'/home/chakwong/anaconda3/envs/tftwogpu/bin/python'},
            stdout=stdout,stderr=stderr,timeout=60,check=True)
    d=loadmat(out/'repaired-paths.mat');w=d['w'].ravel();baseline=d['baseline'];repaired=d['repaired']
    healthy=np.isfinite(baseline)
    result=dict(schema='zhao_cui_pp_tail_wiring_v1',healthy_exact_equality=bool(np.array_equal(repaired[healthy],baseline[healthy])),
                healthy_transition_count=int(healthy.sum()),repaired_transition_count=int((~healthy).sum()),
                retained_sample_count=int(w.size),zero_weight_count=int((w==0).sum()),
                ess=float(1/np.sum(w*w)),ess_fraction=float(1/np.sum(w*w)/w.size),max_weight=float(w.max()),
                original_failure_sample_weight=float(w[5383]),finite_normalized_weights=bool(np.isfinite(w).all()),
                wall_seconds=time.monotonic()-start,cpu_only=True,gpu_intentionally_hidden=True,
                scope='saved t5 paths only; not a full T20 replication')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
