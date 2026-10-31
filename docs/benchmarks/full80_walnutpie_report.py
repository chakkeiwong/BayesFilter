"""Read-only posterior diagnostics for the matched frozen full80 comparison.

Uses the original pinned estimators for both streams, preserves estimator
differences, and permits no compatibility verdict for unqualified chains.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

def assess(tf, samples):
    from bayesfilter.inference.hmc_posterior_diagnostics import (rank_normalized_split_rhat,
        rank_normalized_bulk_tail_ess,posterior_mean_diagnostics)
    from bayesfilter.inference.hmc_precision import mean_precision
    samples=tf.ensure_shape(tf.convert_to_tensor(samples,tf.float64),[None,4,80])
    tf.debugging.assert_all_finite(samples,'physical posterior draws')
    if int(samples.shape[0]) < 4:raise ValueError('at least four draws required')
    chains=tf.transpose(samples,[1,0,2])
    rh=rank_normalized_split_rhat(chains)['maximum']
    ess=rank_normalized_bulk_tail_ess(chains)
    means=posterior_mean_diagnostics(chains)
    precision=mean_precision(samples)
    passed=tf.reduce_all(tf.math.is_finite(rh) & (rh<=1.01) &
        tf.math.is_finite(ess['bulk']) & (ess['bulk']>=400) &
        tf.math.is_finite(ess['tail']) & (ess['tail']>=400) &
        precision['valid'] & (precision['mcse_sd_ratio']<=.05))
    return dict(passed=bool(passed),draws_per_chain=int(samples.shape[0]),
        rhat=rh,bulk_ess=ess['bulk'],tail_ess=ess['tail'],mean=means['pooled_mean'],
        sd=means['posterior_sd'],reporting_mcse=means['mean_mcse'],
        reporting_mcse_sd=means['mcse_sd_ratio'],controller_mcse=precision['mcse'],
        controller_mcse_sd=precision['mcse_sd_ratio'],controller_estimator=precision['estimator'],
        max_rhat=tf.reduce_max(rh),min_bulk_ess=tf.reduce_min(ess['bulk']),
        min_tail_ess=tf.reduce_min(ess['tail']),max_controller_mcse_sd=tf.reduce_max(precision['mcse_sd_ratio']))

def compare(tf, reference, candidate, names, *, warmup_ready):
    # Two independent chains of estimation: Var(mean difference)=sum variances.
    delta=candidate['mean']-reference['mean']
    se=tf.sqrt(tf.square(candidate['reporting_mcse'])+tf.square(reference['reporting_mcse']))
    valid_mcse=bool(tf.reduce_all(tf.math.is_finite(se) & (se>0.) &
        tf.math.is_finite(delta) & tf.math.is_finite(reference['reporting_mcse']) &
        tf.math.is_finite(candidate['reporting_mcse']) &
        (reference['reporting_mcse']>0.) & (candidate['reporting_mcse']>0.)))
    qualified=reference['passed'] and candidate['passed'] and warmup_ready and valid_mcse
    threshold=statistics.NormalDist().inv_cdf(1-.05/(2*len(names)))
    z=delta/se
    flags=tf.abs(z)>threshold
    from docs.benchmarks.full80_walnutpie import plain
    rows=[]
    for i,name in enumerate(names):
        rows.append(plain(dict(parameter=name,hmc_mean=reference['mean'][i],
            walnutpie_mean=candidate['mean'][i],difference=delta[i],combined_mcse=se[i],
            standardized_difference=z[i],flag=flags[i],hmc_sd=reference['sd'][i],
            walnutpie_sd=candidate['sd'][i])))
    return dict(qualified=qualified,adaptation_complete=warmup_ready,valid_mean_mcse=valid_mcse,
        adaptation_interpretation='source adaptation handoff; posterior screens are separate',
        verdict=('incompatible_at_declared_screen' if bool(tf.reduce_any(flags)) else 'compatible_at_available_precision')
                if qualified else 'not_assessable_until_posterior_and_precision_screens_pass',
        familywise_alpha=.05,parameters=len(names),normal_threshold=threshold,
        flagged_parameters=[r['parameter'] for r in rows if r['flag']],rows=rows,
        uncertainty='Approximate normal MCSE comparison; finite chains can share missed modes. No equivalence or speed ranking.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,help='Completed sampling directory with assessment and physical tensors')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('use fresh diagnostic output')
    args.output.mkdir(parents=True)
    started=time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2')
    repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
    from docs.benchmarks.full80_walnutpie import write,plain
    from docs.benchmarks.full80_walnutpie_support import (verify_binding,activate_pinned_runtime,
        verify_loaded_runtime,load_arrays,BASELINE,sha)
    binding,source=verify_binding();activate_pinned_runtime(binding)
    import tensorflow as tf
    from dz5_financial_contract import NAMES
    with tf.device('/CPU:0'):
        path=BASELINE/'sequential-samples.npz'
        samples=load_arrays(tf,path,['private_retained_raw'])['private_retained_raw']
        reference=assess(tf,samples)
        saved=json.loads((BASELINE/'parameters.json').read_text())
        if [r['parameter'] for r in saved] != list(NAMES):raise ValueError('parameter order mismatch')
        for field,key in [('mean','mean'),('sd','sd'),('rhat','rhat'),('bulk_ess','bulk_ess'),
                          ('tail_ess','tail_ess'),('reporting_mcse','mean_mcse')]:
            tf.debugging.assert_near(reference[field],tf.constant([r[key] for r in saved],tf.float64),
                atol=2e-5,rtol=2e-7,message='baseline reconstruction '+field)
        if not reference['passed']:raise ValueError('saved reference fails matched posterior screens')
        write(args.output/'baseline.json',reference)
        if args.candidate:
            assessment=json.loads((args.candidate/'assessment.json').read_text())
            records=json.loads((args.candidate/'physical-samples.json').read_text())
            arrays=[]
            for record in records:
                p=args.candidate/record['path']
                if sha(p)!=record['sha256']:raise ValueError('candidate samples changed')
                arrays.append(tf.io.parse_tensor(p.read_bytes(),tf.float64))
            candidate=assess(tf,tf.concat(arrays,0))
            write(args.output/'candidate.json',candidate)
            write(args.output/'comparison.json',compare(tf,reference,candidate,NAMES,
                warmup_ready=assessment['warmup_ready']))
    manifest=dict(command=[sys.executable,*sys.argv],baseline=str(BASELINE),source=source,
        saved_samples_sha256=sha(path),diagnostic_source_hashes=verify_loaded_runtime(binding),
        elapsed_seconds=time.monotonic()-started,tensorflow=tf.__version__,
        gpu='intentionally hidden for read-only CPU diagnostics',plan='docs/plans/bayesfilter-full80-walnutpie-comparison-2026-10-30.md')
    write(args.output/'manifest.json',manifest)
    print(json.dumps(plain({k:reference[k] for k in ('passed','draws_per_chain','max_rhat','min_bulk_ess','min_tail_ess','max_controller_mcse_sd')})))

if __name__=='__main__':main()
