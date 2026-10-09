#!/usr/bin/env python3
"""Bounded independent-reference diagnostic; same saved data/parameter target.

Gaussian bootstrap PF, analytical complete-data Fisher scores, FP64/XLA.
Finite-particle estimates and replication uncertainty are not an exact oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docs.benchmarks.run_ledh_nonlinear_master import dump, git, prepare_dataset, read


def combine(rows):
    logs = [r['log_likelihood'] for r in rows]
    maximum = max(logs)
    weights = [math.exp(value-maximum) for value in logs]
    total = sum(weights)
    return [maximum+math.log(total/len(rows)), *[
        sum(weight*row['score'][k] for weight, row in zip(weights, rows))/total
        for k in range(len(rows[0]['score']))]]


def summarize(rows):
    combined = combine(rows)
    omitted = [combine(rows[:k]+rows[k+1:]) for k in range(len(rows))]
    n = len(rows)
    se = [math.sqrt((n-1)/n*sum((r[k]-statistics.mean(x[k] for x in omitted))**2
                               for r in omitted)) for k in range(len(combined))]
    logs = [r['log_likelihood'] for r in rows]
    maximum = max(logs)
    weights = [math.exp(value-maximum) for value in logs]
    return dict(log_likelihood=combined[0], score=combined[1:],
                jackknife_mcse_log_likelihood=se[0], jackknife_mcse_score=se[1:],
                mean_log_likelihood=statistics.mean(logs),
                sd_log_likelihood=statistics.stdev(logs), replications=n,
                between_replication_likelihood_ess=sum(weights)**2/sum(w*w for w in weights),
                minimum_particle_ess=min(r['minimum_particle_ess'] for r in rows),
                maximum_particle_weight=max(r['maximum_particle_weight'] for r in rows),
                minimum_final_distinct_initial_ancestors=min(
                    r['final_distinct_initial_ancestors'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-file',default='docs/plans/ledh-nonlinear-execution-20261002.md')
    parser.add_argument('--result-file',default='docs/benchmarks/ledh-nonlinear-execution-results-20261002.md')
    parser.add_argument('--worker-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--particles', nargs='+', type=int, default=[8192,32768])
    parser.add_argument('--seeds', nargs='+', type=int, default=[260601,260602,260603,260604])
    parser.add_argument('--budget-seconds', type=int, default=900)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('at least two distinct replication seeds are required')
    if min(args.particles) < 2 or args.budget_seconds < 1:
        parser.error('positive budget and at least two particles are required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.testing.nonlinear_bootstrap_fisher_reference_tf import make_bootstrap_fisher_kernel

    manifest = dict(schema='bayesfilter.bootstrap_fisher_reference.v1',
                    command=sys.argv, git_commit=git('rev-parse','HEAD'),
                    git_dirty=bool(git('status','--porcelain')), python=sys.executable,
                    tensorflow=tf.__version__, dtype='float64', jit_compile=True, tf32=False,
                    device='/GPU:0', visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    memory_policy=memory, GPU_trust='escalated_GPU_access',
                    plan=args.plan_file,
                    result=args.result_file,
                    particles=args.particles, seeds=args.seeds, budget_seconds=args.budget_seconds,
                    evidence_role='approximate_independent_reference_not_oracle',
                    score_method='posterior_average_analytical_complete_data_Fisher_score',
                    aggregation='log_mean_unbiased_likelihood_and_likelihood_weighted_score',
                    uncertainty='delete_one_replication_jackknife_MCSE_not_particle_bias',
                    validation='tests/highdim/test_nonlinear_bootstrap_fisher_reference.py',
                    source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [Path(__file__), ROOT/'bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py', ROOT/'bayesfilter/highdim/ledh_canonical_models_tf.py']})
    dump(args.output/'manifest.json', manifest)
    rows, summaries, scope_data = [], [], []
    def timeout(signum, frame):
        raise TimeoutError('reference campaign wall budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1,args.budget_seconds-int(time.monotonic()-started)))
    status = 'running'
    try:
        with tf.device('/GPU:0'):
            for worker in args.worker_dirs:
                job = read(worker/'job.json')
                spec = NonlinearSQMCSpec(job['model'])
                points = ({tuple(theta) for theta in job['theta_points']} if job.get('theta_points')
                          else {tuple(row['theta']) for row in read(worker/'rows.json')})
                if len(points) != 1:
                    raise ValueError('reference runner currently requires exactly one saved parameter point')
                theta = tf.constant(list(next(iter(points))),tf.float64)
                observations, dataset = prepare_dataset(tf, spec, dict(job,dataset_file=str(worker/'dataset.json')),tf.float64)
                scope = dict(model=spec.name, target_id=spec.target_id, horizon=job['horizon'],
                             data_seed=job['data_seed'], observation_sha256=dataset['observation_sha256'],
                             theta=theta.numpy().tolist(), parameter_names=list(spec.parameter_names),
                             source_worker=str(worker), source_dataset_sha256=hashlib.sha256((worker/'dataset.json').read_bytes()).hexdigest())
                scope_data.append(scope)
                dump(args.output/'scopes.json',scope_data)
                for n in args.particles:
                    kernel = make_bootstrap_fisher_kernel(spec.model,spec.initial_mean(tf.float64),
                                spec.parameter_count,spec.observation_dimension,job['horizon'],n)
                    group = []
                    for seed in args.seeds:
                        tic = time.monotonic()
                        ell, score, trace = kernel(theta,observations,tf.constant(seed))
                        ell, score, trace = float(ell),score.numpy().tolist(),trace.numpy().tolist()
                        if not all(math.isfinite(value) for value in [ell,*score,*[v for r in trace for v in r]]):
                            raise ValueError('nonfinite bootstrap reference result')
                        row = dict(scope, particles=n, seed=seed,log_likelihood=ell,score=score,
                                   minimum_particle_ess=min(r[1] for r in trace),
                                   maximum_particle_weight=max(r[2] for r in trace),
                                   final_distinct_initial_ancestors=trace[-1][3],
                                   trace_columns=['log_likelihood_increment','particle_ess','maximum_weight','distinct_initial_ancestors_before_resampling'],
                                   trace=trace,wall_seconds=time.monotonic()-tic)
                        rows.append(row); group.append(row)
                        dump(args.output/'replications.json',rows)
                        print(json.dumps({k:row[k] for k in ('model','data_seed','particles','seed','log_likelihood','score','wall_seconds')}),flush=True)
                    summary = dict(scope,particles=n,**summarize(group))
                    summaries.append(summary)
                    dump(args.output/'summary.json',summaries)
            status = 'complete'
    except Exception as exc:
        status = type(exc).__name__+': '+str(exc)
        raise
    finally:
        signal.alarm(0)
        manifest.update(wall_seconds=time.monotonic()-started,status=status,
                        allocator=tf.config.experimental.get_memory_info('GPU:0'),
                        artifact_paths=[str(p) for p in sorted(args.output.glob('*.json'))])
        dump(args.output/'manifest.json',manifest)
        print(json.dumps({'status':status,'wall_seconds':manifest['wall_seconds']}),flush=True)


if __name__ == '__main__':
    main()
