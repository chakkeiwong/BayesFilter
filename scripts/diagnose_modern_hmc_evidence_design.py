"""CPU diagnostic of acceptance resolution; never issues tuning evidence.

Constant scores are favorable benchmarks, not random replications or a power
estimate. All classifications come from the existing TensorFlow statistic.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import hashlib
import subprocess
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def resolution_screen(policy, *, evidence_rungs=(1,)):
    import tensorflow as tf
    from bayesfilter.inference.hmc_acceptance_statistics import (
        replicated_acceptance_statistics, replicated_qualification_attainability)
    count=policy.repetition_target(evidence_rungs[-1])
    center=sum(policy.preferred_region)/2.
    stages={}
    for stage in ('measurement','verification'):
        row=replicated_acceptance_statistics(tf.fill([count,4],tf.constant(center,tf.float64)),
            policy=policy,stage=stage,evidence_rungs=evidence_rungs)
        stages[stage]={k:row[k] for k in ('acceptance_decision','one_sided_alpha','start_intervals','pooled_mean')}
    return dict(policy_id=policy.identity,policy=policy.payload(),
        attainable=replicated_qualification_attainability(policy=policy,evidence_rungs=evidence_rungs),
        preferred_center=center,trial_count=count,stages=stages,
        preferred_center_resolvable=all(r['acceptance_decision']=='passed' for r in stages.values()),
        role='necessary_planning_screen_only',statistical_power='not_established',
        qualification_authority=False,scope='constant_score_sensitivity_not_target_simulation')


def sensitivity_report(policy, *, counts=(32,64,128), evidence_rungs=(1,)):
    import tensorflow as tf
    from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics
    rows=[]
    center=sum(policy.preferred_region)/2.
    for count in counts:
        proposal=replace(policy,base_repetitions=count,max_repetitions=count)
        scenarios={'preferred_center':[center]*4,'high_acceptance':[1.]*4,
            'qualification_boundary':[policy.qualification_region[0]]*4,
            'heterogeneous_starts':[1.,1.,1.,.2]}
        for name,values in scenarios.items():
            scores=tf.repeat(tf.constant([values],tf.float64),count,axis=0)
            row=replicated_acceptance_statistics(scores,policy=proposal,stage='measurement',evidence_rungs=(1,))
            rows.append(dict(scenario=name,repetitions=count,policy_id=proposal.identity,
                **{k:row[k] for k in ('acceptance_decision','pooled_mean','start_means','start_intervals','one_sided_alpha')}))
        if count%2==0:
            scores=tf.tile(tf.constant([[.5]*4,[.9]*4],tf.float64),[count//2,1])
            row=replicated_acceptance_statistics(scores,policy=proposal,stage='measurement',evidence_rungs=(1,))
            rows.append(dict(scenario='alternating_trial_scores',repetitions=count,
                **{k:row[k] for k in ('acceptance_decision','pooled_mean','start_intervals','one_sided_alpha')}))
    return dict(current_design=resolution_screen(policy, evidence_rungs=evidence_rungs),sensitivity=rows,
        future_counts_role='illustrative_fresh_designs_not_approved_continuations',
        statistical_power='not_established',no_ranking=True,posterior_qualification=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--design',type=Path)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise ValueError('design diagnostic requires GPUs intentionally hidden before import')
    if args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic()
    if args.design:
        from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
        raw=json.loads(args.design.read_text())
        policy=HMCReplicatedAcceptancePolicy.from_payload(raw['policy'])
        evidence_rungs=tuple(raw.get('evidence_rungs',(1,)))
    else:
        from scripts.continue_modern_hmc_macrofinance_broad import policy as configured
        policy=configured()
        evidence_rungs=(1,)
    report=sensitivity_report(policy,evidence_rungs=evidence_rungs)
    report.update(command=[sys.executable,*sys.argv],elapsed_seconds=time.monotonic()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(),device='CPU',gpu_intentionally_hidden=True,
        environment=os.environ.get('CONDA_PREFIX',str(Path(sys.executable).parents[1])),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan='docs/plans/bayesfilter-modern-hmc-broad-repair-2026-10-25.md',
        output=str(args.output.resolve()),data_version='deterministic_synthetic_score_design',
        seed='N/A: deterministic diagnostic',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(),ROOT/'bayesfilter/inference/hmc_acceptance_protocol.py',
            ROOT/'bayesfilter/inference/hmc_acceptance_statistics.py')})
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['current_design'],indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
