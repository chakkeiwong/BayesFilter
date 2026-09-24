"""Fresh-stream diagnostic confirmation of the optional input-precision cutoff.

Acceptance is numerical non-harm, not MSE optimization. Fixed datasets/twists
are inherited explicitly; calibration and final particle streams are fresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import diagnose_younis_iapf_resampling_control as prior


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    source = output.parent/'attempt02'
    old_manifest = json.loads((source/'manifest.json').read_text())
    old = json.loads((source/'results.json').read_text())
    if old_manifest['status'] != 'complete':
        raise RuntimeError('source campaign incomplete')
    if len(list(output.parent.glob('*/manifest.json'))) >= 4:
        raise RuntimeError('campaign launch budget exhausted')
    output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.score_study.runtime import configure_runtime, memory_usage
    runtime = configure_runtime(device='GPU', tf32=True, jit_compile=True)
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.score_study.combinations_tf import make_combination_kernels
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    prior.tf = prior.base.tf = tf
    prior.base.THETA = old_manifest['theta']
    prior.base.REGIMES['affine'] = (0., 0.)
    prior_seconds = old_manifest['prior_driver_seconds'] + old_manifest['wall_seconds']
    budget = prior.base.Budget(min(600., 1800.-prior_seconds))
    budget.started = started
    budget.limits = old_manifest['budget']['limits']
    budget.counts.update(old_manifest['budget']['counts'])
    files = [prior.REPO/path for path in old_manifest['source_sha256']] + [Path(__file__)]
    hashes = {str(path.relative_to(prior.REPO)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    changed = [path for path, digest in old_manifest['source_sha256'].items() if hashes[path] != digest]
    allowed = {prior.PLAN, 'bayesfilter/score_study/combinations_tf.py'}
    if set(changed) - allowed:
        raise RuntimeError('unexpected numerical source changes: '+str(changed))
    manifest = dict(schema='iapf_control_conditioning_confirmation_v1', plan=prior.PLAN,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=prior.REPO,text=True).strip(),
        command=sys.argv,cwd=str(prior.REPO),python=sys.executable,python_version=sys.version,
        source_sha256=hashes,source_changes_from_prior=changed,runtime=runtime,
        source=str(source),source_results_sha256=hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),master_seed=9182026,
        datasets=prior.DATASETS,theta=prior.base.THETA,particles=4096,horizon=2,
        data_version='same fixed data and frozen fits as resampling_control_fresh_20260918',
        seeds='conditioning_confirmation_20260918_{calibration|final}_N4096',
        calibration_replicates=192,final_replicates=128,uniform_bits=23,
        filter_dtype='float32',control_regression_dtype='float64',control_input_dtype='float32',
        reference_exception='same CPU FP64 references; FP64 diagnostic control centering and regression',
        prior_driver_seconds=prior_seconds,status='running',default_adoption=False)
    results = {'datasets':{}}
    prior.base.write(output/'manifest.json',manifest)

    def save(stage):
        results.update(stage=stage,budget=budget.record())
        prior.base.write(output/'results.json',results)

    try:
        # Check all new streams, including cross-partition collisions, against
        # every calibration/final stream of the earlier experiment.
        used = set()
        for dataset,regime in prior.DATASETS.items():
            for n in ((4096,) if regime=='affine' else (1024,4096)):
                for partition,count in [('calibration',192),('final',128)]:
                    group=f'resampling_control_20260918_{partition}_N{n}'
                    for rep in range(count):
                        for label in ('initial','process','ancestors','mixture'):
                            pair=tuple(prior.base.seed(dataset,label,rep,group))
                            if pair in used:raise RuntimeError('old seed collision')
                            used.add(pair)
        old_seed_count=len(used)
        for dataset in prior.DATASETS:
            for partition,count in [('calibration',192),('final',128)]:
                group=f'conditioning_confirmation_20260918_{partition}_N4096'
                for rep in range(count):
                    for label in ('initial','process','ancestors','mixture'):
                        pair=tuple(prior.base.seed(dataset,label,rep,group))
                        if pair in used:raise RuntimeError('fresh seed collision')
                        used.add(pair)
        manifest.update(old_seed_pairs=old_seed_count,new_seed_pairs=len(used)-old_seed_count,
                        all_particle_seed_pairs_disjoint=True)

        def streams(dataset,rep,partition):
            group=f'conditioning_confirmation_20260918_{partition}_N4096'
            return prior.stream_kernel(4096)(tf.constant([prior.base.seed(dataset,label,rep,group)
                for label in ('initial','process','ancestors','mixture')],tf.int32))

        with tf.device('/CPU:0'):
            critical=float(tfp.distributions.StudentT(tf.constant(127.,tf.float64),0.,1.).quantile(
                tf.constant(1.-.01/12,tf.float64)))
        manifest['affine_bias_critical']=critical
        with tf.device('/GPU:0'):
            original,apply,_,_=make_combination_kernels(6,12,'float64')
            protected,_,_,_=make_combination_kernels(6,12,'float64',control_input_dtype_name='float32')
            for dataset,regime in prior.DATASETS.items():
                key=str(dataset)
                if old['fits'][key]['status'] != 'valid':
                    raise RuntimeError('frozen fit unavailable')
                c,b=prior.base.REGIMES[regime]
                kernel=make_fitted_twist_kernel(1,1,4096,2,'float32',transition_curve=c,
                    observation_curve=b,include_fisher_score=True,include_resampling_controls=True,
                    resampling_uniform_bits=23)
                theta=tf.constant(prior.base.THETA,tf.float32)
                observations=tf.constant(old['references'][key]['executed_observations'],tf.float32)
                coefficients=[tf.constant(old['fits'][key]['fit'][name],tf.float32)
                    for name in ('centers','covariances','log_floors')]
                record={'regime':regime,'calibration':[],'final':[]}
                results['datasets'][key]=record
                for rep in range(192):
                    budget.charge('filter_calls')
                    out=kernel(theta,observations,*streams(dataset,rep,'calibration'),*coefficients)
                    prior.finite(*out)
                    record['calibration'].append(dict(replicate=rep,score=prior.base.materialize(out[3]),
                        control=prior.base.materialize(tf.reshape(out[4],[12]))))
                scores=tf.constant([row['score'] for row in record['calibration']],tf.float64)
                controls=tf.constant([row['control'] for row in record['calibration']],tf.float64)
                old_coefficient,old_rank,valid_old=original(scores,controls)
                coefficient,rank,valid=protected(scores,controls)
                if not bool(valid and valid_old):raise RuntimeError('invalid regression')
                prior.finite(old_coefficient,coefficient)
                record.update(original_coefficient=prior.base.materialize(old_coefficient),
                    protected_coefficient=prior.base.materialize(coefficient),original_rank=int(old_rank),
                    protected_rank=int(rank),coefficient_identical=bool(tf.reduce_all(old_coefficient==coefficient)),
                    original_max_coefficient=float(tf.reduce_max(tf.abs(old_coefficient))),
                    protected_max_coefficient=float(tf.reduce_max(tf.abs(coefficient))),
                    singular_values=prior.base.materialize(tf.linalg.svd(controls-tf.reduce_mean(controls,0),compute_uv=False)),
                    frozen_before_final=True)
                save(f'coefficients_frozen_{dataset}')
                for rep in range(128):
                    budget.charge('filter_calls')
                    out=kernel(theta,observations,*streams(dataset,rep,'final'),*coefficients)
                    prior.finite(*out)
                    record['final'].append(dict(replicate=rep,value=prior.base.materialize(out[0]),
                        score=prior.base.materialize(out[3]),fixed_score=prior.base.materialize(out[1]),
                        control=prior.base.materialize(tf.reshape(out[4],[12]))))
                    if (rep+1)%32==0:save(f'final_progress_{dataset}')
                scores=tf.constant([row['score'] for row in record['final']],tf.float64)
                controls=tf.constant([row['control'] for row in record['final']],tf.float64)
                old_corrected=apply(scores,controls,old_coefficient)
                corrected=apply(scores,controls,coefficient)
                prior.finite(old_corrected,corrected)
                record.update(final_correction_identical=bool(tf.reduce_all(old_corrected==corrected)),
                    original_scores=prior.base.materialize(old_corrected),protected_scores=prior.base.materialize(corrected),
                    kernel_trace_count=kernel.experimental_get_tracing_count())
                summaries={}
                for name,values in [('raw',scores),('original',old_corrected),('protected',corrected)]:
                    rows=[dict(value=row['value'],score=score) for row,score in zip(record['final'],prior.base.materialize(values))]
                    summaries[name]=prior.summarize(rows,old['references'][key],critical)
                    summaries[name]['bias_screen_scope']='six components, approximate Bonferroni 99%; affine safety criterion only'
                record['summary']=summaries
                record['criterion_pass']=(record['coefficient_identical'] and record['final_correction_identical']) if regime!='affine' else (
                    record['protected_rank']==7 and record['protected_max_coefficient'] < record['original_max_coefficient']
                    and summaries['protected']['bias_screen_pass'])
                save(f'completed_{dataset}')
                print(dataset,'safety pass',record['criterion_pass'],'ranks',int(old_rank),int(rank),flush=True)
        decision=dict(all_safety_checks_pass=all(record['criterion_pass'] for record in results['datasets'].values()),
            optional_safeguard_validated_scopes=list(results['datasets']),default_adoption=False,
            statistical_superiority=False,ledger='numerical safety, not scientific promotion',
            fresh_calibration=True,fresh_final=True)
        results['decision']=decision
        save('complete')
        manifest.update(status='complete',wall_seconds=time.monotonic()-started,budget=budget.record(),
            memory_usage=memory_usage('GPU'),regression_trace_counts=[original.experimental_get_tracing_count(),protected.experimental_get_tracing_count(),apply.experimental_get_tracing_count()])
        prior.base.write(output/'manifest.json',manifest)
        prior.base.write(output/'decision.json',decision)
    except Exception as error:
        save('stopped')
        manifest.update(status='stopped',exception=repr(error),wall_seconds=time.monotonic()-started,budget=budget.record())
        prior.base.write(output/'manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
