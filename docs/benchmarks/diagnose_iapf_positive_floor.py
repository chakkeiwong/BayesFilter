"""Diagnostic-only floor mechanism study using the actual fitted TF consumer.

Frozen-guide counterfactuals are not refitting, tuning, paper replication or
method-ranking evidence. Every execution uses a versioned output directory.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from diagnose_iapf_adaptive_consumer import REPO, save, setup

ROOT = REPO / 'docs/plans/artifacts/iapf-positive-floor-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-initialization-isolation-20260922-01'
PLAN = 'docs/plans/iapf-positive-floor-diagnosis-2026-09-22.md'
ARMS = ('baseline', 'ratio_1e_6', 'R_tail_N_minus2')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles():
    with (ROOT / 'floor-quantiles.csv').open() as stream:
        return {(int(r['d']), int(r['N'])): float(r['quantile'])
                for r in csv.DictReader(stream)}


def log_ratios(d, n):
    return dict(baseline=math.log(.01), ratio_1e_6=math.log(1e-6),
                R_tail_N_minus2=-.5 * quantiles()[d, n])


def oracle(tf, directory):
    from bayesfilter.score_study.fitted_twist_tf import normalizer, twisted_transition
    dtype = tf.float64
    records = []
    errors = dict(log_probability=0., log_normalizer=0., tangent=0.)
    trace_counts = {}
    for d in (2, 5, 10, 20, 40, 80):
        @tf.function(input_signature=[tf.TensorSpec([3, d], dtype),
            tf.TensorSpec([d, d], dtype), tf.TensorSpec([d, d], dtype),
            tf.TensorSpec([], dtype)], jit_compile=True)
        def kernel(mean, Q, V, floor):
            return normalizer(mean, tf.zeros([6, 3, d], dtype), Q,
                tf.concat([V[None], tf.zeros([5, d, d], dtype)], 0),
                tf.zeros([d], dtype), V, floor)

        # V is deliberately non-spherical; Q=qV retains an independent formula.
        diagonal = [1. + j / d for j in range(d)]
        V = tf.linalg.diag(tf.constant(diagonal, dtype))
        peak = -.5 * (d * math.log(2 * math.pi) + sum(map(math.log, diagonal)))
        for q in (.1, 1., 10.):
            radii = [0., float(d), 4. * d]
            means = [[math.sqrt((1 + q) * diagonal[0] * r)] + [0.] * (d - 1)
                     for r in radii]
            for arm, lrho in log_ratios(d, 1024).items():
                lf = peak + lrho
                values = kernel(tf.constant(means, dtype), tf.constant(q, dtype) * V,
                                V, tf.constant(lf, dtype))
                totals, tangents, probabilities = [v.numpy().tolist() for v in values]
                for i, radius in enumerate(radii):
                    lg = peak - .5 * d * math.log1p(q) - .5 * radius
                    high = max(lg, lf)
                    total = high + math.log(math.exp(lg - high) + math.exp(lf - high))
                    logp = lg - total
                    expected_tangent = math.exp(logp) * (radius - d) / (2 * (1 + q))
                    actual_logp = math.log(probabilities[i])
                    local_errors = dict(log_probability=abs(actual_logp - logp),
                        log_normalizer=abs(totals[i] - total),
                        tangent=max(abs(tangents[0][i] - expected_tangent),
                                    *[abs(tangents[j][i]) for j in range(1, 6)]))
                    for k, v in local_errors.items():
                        errors[k] = max(errors[k], v)
                    records.append(dict(d=d, N=1024, q=q, squared_radius=radius,
                        arm=arm, log_floor_ratio=lrho, log_probability=actual_logp,
                        probability=probabilities[i], log_normalizer=totals[i],
                        tangent=tangents[0][i], expected_log_probability=logp,
                        errors=local_errors))
        trace_counts[str(d)] = kernel.experimental_get_tracing_count()
    # Non-harm where both positive floors are negligible: unchanged mixture labels.
    @tf.function(input_signature=[tf.TensorSpec([], dtype)], jit_compile=True)
    def healthy(lrho):
        mean = tf.constant([[0., 0.], [1., 0.], [0., 1.]], dtype)
        zero = tf.zeros([6, 3, 2], dtype)
        cov = tf.eye(2, dtype=dtype)
        dcov = tf.zeros([6, 2, 2], dtype)
        floor = -tf.constant(math.log(2 * math.pi), dtype) + lrho
        norm = normalizer(mean, zero, cov, dcov, tf.zeros([2], dtype), cov, floor)
        moved = twisted_transition(mean, zero, cov, dcov, tf.zeros([2], dtype),
            cov, norm[2], tf.ones([3, 2], dtype), tf.fill([3], tf.constant(.5, dtype)))
        return norm[0], norm[1], norm[2], moved[0], moved[1]
    low = healthy(tf.constant(math.log(1e-24), dtype))
    lower = healthy(tf.constant(math.log(1e-30), dtype))
    nonharm = max(float(tf.reduce_max(tf.abs(a-b)).numpy()) for a, b in zip(low, lower))
    save(directory / 'results.json', records)
    passed = all(x <= 1e-9 for x in errors.values()) and nonharm <= 1e-12
    passed = passed and all(v == 1 for v in trace_counts.values())
    return dict(cases=len(records), errors=errors, trace_counts=trace_counts,
                negligible_floor_nonharm_error=nonharm, passed=passed)


def probability_stats(tf, probabilities):
    rows = []
    for p in tf.unstack(probabilities):
        n = int(p.shape[0])
        order = tf.sort(p)
        rows.append(dict(minimum=float(order[0].numpy()), maximum=float(order[-1].numpy()),
            mean=float(tf.reduce_mean(p).numpy()),
            q10=float(order[round(.1 * (n-1))].numpy()),
            median=float(order[round(.5 * (n-1))].numpy()),
            q90=float(order[round(.9 * (n-1))].numpy()),
            fraction_below_01=float(tf.reduce_mean(tf.cast(p < .01, tf.float64)).numpy())))
    return rows


def make_frozen_ancestor_kernel(tf, d, n, T):
    from bayesfilter.score_study.fitted_twist_tf import normalizer
    from bayesfilter.score_study.gaussian_tf import parameterized_model
    dtype = tf.float64
    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([n, d], dtype),
        tf.TensorSpec([T, d], dtype), tf.TensorSpec([T, d, d], dtype),
        tf.TensorSpec([T], dtype), tf.TensorSpec([T, n, d], dtype),
        tf.TensorSpec([T+1, n], tf.int32)], jit_compile=True)
    def kernel(theta, initial_noise, centers, covariances, floors, clouds, ancestors):
        A, _, _, _, m, _, P, _, Q, _, _, _ = parameterized_model(theta, d, d)
        initial = m + tf.einsum('ij,nj->ni', tf.linalg.cholesky(P), initial_noise)
        probabilities = tf.TensorArray(dtype, size=T, element_shape=[n])
        def step(t, array):
            raw = tf.cond(t == 0, lambda: initial, lambda: clouds[t-1])
            previous = tf.gather(raw, ancestors[t])
            mean = tf.einsum('ij,nj->ni', A, previous)
            p = normalizer(mean, tf.zeros([6, n, d], dtype), Q,
                tf.zeros([6, d, d], dtype), centers[t], covariances[t], floors[t])[2]
            return t+1, array.write(t, p)
        return tf.while_loop(lambda t, *_: t < T, step,
                             (0, probabilities), parallel_iterations=1)[1].stack()
    return kernel


def replay(tf, directory):
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    old_manifest = read(PRIOR / 'manifest.json')
    def preserved(relative):
        path = PRIOR / relative
        if sha(path) != old_manifest['outputs'][relative]:
            raise RuntimeError('Prior input hash mismatch: ' + relative)
        target = directory / 'inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return read(target)
    previous = preserved('attempt03_gpu_consumer/results.json')
    rows = []
    kernels = {}
    frozen_kernels = {}
    checks = dict(value_score_replay=True, frozen_ancestor_reconstruction=True,
                  finite_outputs=True, valid_cdf=True)
    max_replay_error = 0.
    max_reconstruction_error = 0.
    for prior in previous:
        if prior['status'] != 'complete':
            continue
        d = prior['d']; n = prior['diagnostics']['actual_particle_count']; T = 4
        name = f"d{d}-s{prior['seed']}-{prior['initialization']}-{prior['objective_scale']}"
        inputs = preserved('attempt03_gpu_consumer/' + name + '/final-inputs.json')
        theta = tf.constant(inputs['theta'], tf.float64)
        draws = [tf.constant(v, tf.float64) for v in inputs['draws']]
        centers, covariances, old_floors = [tf.constant(v, tf.float64) for v in inputs['coefficients']]
        peak = -.5 * (tf.constant(d * math.log(2 * math.pi), tf.float64)
                      + tf.linalg.logdet(covariances))
        floors = {arm: old_floors if arm == 'baseline' else peak + tf.constant(lrho, tf.float64)
                  for arm, lrho in log_ratios(d, n).items()}
        key = (d, n)
        if key not in kernels:
            kernels[key] = make_fitted_twist_kernel(d, d, n, T, include_numerical_trace=True)
            frozen_kernels[key] = make_frozen_ancestor_kernel(tf, d, n, T)
        kernel = kernels[key]; frozen = frozen_kernels[key]
        for regime, observations in prior['observations'].items():
            y = tf.constant(observations, tf.float64)
            saved = prior['regimes'][regime]
            baseline = kernel(theta, y, *draws, centers, covariances, old_floors)
            error = max(abs(float(baseline[0].numpy()) - saved['value']),
                float(tf.reduce_max(tf.abs(baseline[1] - tf.constant(saved['score'], tf.float64))).numpy()))
            max_replay_error = max(max_replay_error, error)
            checks['value_score_replay'] &= error <= 1e-9
            base_trace = baseline[-1]
            base_labels = draws[3] < base_trace['gaussian_probability']
            for arm in ARMS:
                out = baseline if arm == 'baseline' else kernel(theta, y, *draws, centers, covariances, floors[arm])
                trace = out[-1]; probabilities = trace['gaussian_probability']
                fixed = frozen(theta, draws[0], centers, covariances, floors[arm],
                               baseline[2], base_trace['ancestor_indices'])
                if arm == 'baseline':
                    delta = float(tf.reduce_max(tf.abs(fixed - probabilities)).numpy())
                    max_reconstruction_error = max(max_reconstruction_error, delta)
                    checks['frozen_ancestor_reconstruction'] &= delta <= 1e-9
                cdf = trace['ancestor_cdf']
                weights = cdf - tf.concat([tf.zeros([T+1, 1], tf.float64), cdf[:, :-1]], 1)
                checks['valid_cdf'] &= bool(tf.reduce_all(weights >= -1e-12).numpy())
                checks['valid_cdf'] &= bool(tf.reduce_all(tf.abs(cdf[:, -1] - 1) <= 1e-10).numpy())
                checks['finite_outputs'] &= all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy())
                    for v in (out[0], out[1], out[2], probabilities, fixed, cdf))
                labels = draws[3] < probabilities
                value = float(out[0].numpy()); exact = saved['exact_value']
                rec = dict(case=name, d=d, N=n, seed=prior['seed'],
                    initialization=prior['initialization'], objective_scale=prior['objective_scale'],
                    regime=regime, arm=arm, value=value, score=out[1].numpy().tolist(),
                    exact_value=exact, log_value_error=value-exact,
                    heuristic_absolute_errors=saved['heuristic_absolute_errors'],
                    heuristic_dominance_verdict='descriptive_veto' if any(abs(value-exact) > e
                        for e in saved['heuristic_absolute_errors'].values()) else 'no_observed_loss_this_draw',
                    actual_log_floor_ratios=(floors[arm]-peak).numpy().tolist(),
                    probability_by_time=probability_stats(tf, probabilities),
                    fixed_baseline_ancestor_probability_by_time=probability_stats(tf, fixed),
                    actual_Gaussian_fraction=tf.reduce_mean(tf.cast(labels, tf.float64), 1).numpy().tolist(),
                    mixture_label_change_fraction=tf.reduce_mean(tf.cast(labels != base_labels, tf.float64), 1).numpy().tolist(),
                    ancestor_change_fraction=tf.reduce_mean(tf.cast(trace['ancestor_indices'] != base_trace['ancestor_indices'], tf.float64), 1).numpy().tolist(),
                    ess_by_time=(1 / tf.reduce_sum(weights**2, 1)).numpy().tolist(),
                    baseline_value_score_replay_error=error)
                rows.append(rec)
                # Keep full probabilities/indices/clouds for discriminating followups.
                target = directory / 'cases' / name / regime / arm
                target.mkdir(parents=True)
                save(target / 'result.json', rec)
                save(target / 'trace.json', dict(clouds=out[2].numpy().tolist(),
                    gaussian_probability=probabilities.numpy().tolist(),
                    fixed_ancestor_gaussian_probability=fixed.numpy().tolist(),
                    ancestor_indices=trace['ancestor_indices'].numpy().tolist(),
                    ancestor_cdf=cdf.numpy().tolist(), mixture_labels=labels.numpy().tolist()))
        save(directory / 'results.json', rows)
        print(json.dumps(dict(case=name, status='complete', N=n)), flush=True)
    counts = {str(k): [v.experimental_get_tracing_count(), frozen_kernels[k].experimental_get_tracing_count()]
              for k, v in kernels.items()}
    checks['stable_traces'] = all(c == [1, 1] for c in counts.values())
    checks['all_cases'] = len(rows) == 19 * 2 * 3
    return dict(cases=19, evaluations=len(rows), prior_rejected_excluded=5,
        checks=checks, passed=all(checks.values()), max_replay_error=max_replay_error,
        max_reconstruction_error=max_reconstruction_error, trace_counts=counts,
        statistical_ranking='none; paired conditional mechanism diagnostics', default_changed=False)


def worker(args):
    directory = ROOT / args.attempt
    directory.mkdir(exist_ok=False)
    tf, environment = setup(args.device)
    with tf.device('/CPU:0' if args.device == 'cpu' else '/GPU:0'):
        result = oracle(tf, directory) if args.mode == 'oracle' else replay(tf, directory)
    if args.device == 'gpu':
        environment['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    save(directory / 'summary.json', dict(environment=environment, **result))
    if not result['passed']:
        raise RuntimeError('Mechanism diagnostic failed required checks; inspect summary')


def launch(args):
    if not args.attempt.isidentifier():
        raise ValueError('Versioned attempt name required')
    manifest = ROOT / (args.attempt + '-launch.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    budget = read(ROOT / 'budget.json')
    resource = args.device
    seconds = 300
    if len(list(ROOT.glob('*-launch.json'))) >= 6:
        raise RuntimeError('Launch cap')
    if (budget['phase_' + resource + '_seconds'] + seconds > (3600 if resource == 'cpu' else 900)
            or budget['remaining_' + resource + '_seconds'] < seconds):
        raise RuntimeError('Insufficient phase/campaign budget')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--attempt', args.attempt,
               '--device', args.device, '--mode', args.mode]
    sources = ['docs/benchmarks/diagnose_iapf_positive_floor.py',
        'docs/benchmarks/diagnose_iapf_adaptive_consumer.py',
        'bayesfilter/score_study/fitted_twist_tf.py', 'bayesfilter/score_study/gaussian_tf.py',
        'bayesfilter/score_study/conditional_means_tf.py',
        'bayesfilter/score_study/complete_data_score_tf.py',
        'bayesfilter/runtime/gpu_memory_policy.py', 'docs/benchmarks/reference_iapf_paper.R', PLAN]
    hashes = {}
    for name in sources:
        path = REPO / name
        target = ROOT / (args.attempt + '-source') / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = sha(path)
    record = dict(command=command, device=args.device, mode=args.mode, sources=hashes,
        quantile_hash=sha(ROOT / 'floor-quantiles.csv'), status='running',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        plan=PLAN, data='Saved previous consumer inputs / deterministic oracle fixtures',
        seed_policy='No new random draws; saved actual arrays or deterministic inputs')
    save(manifest, record)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1' if args.device == 'cpu' else
        'GPU-68251639-fe82-8f81-3ccc-2953c32e805b', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
        'BAYESFILTER_PRELOAD_CUSTOM_OP': '0', 'MPLCONFIGDIR': '/tmp/iapf-floor-matplotlib'}
    started = time.monotonic()
    with (ROOT / (args.attempt + '.log')).open('x') as log:
        try:
            code = subprocess.run(command, cwd=REPO, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    elapsed = time.monotonic() - started
    record.update(status='finished', exit_code=code, wall_seconds=elapsed)
    save(manifest, record)
    budget = read(ROOT / 'budget.json')
    budget['phase_' + resource + '_seconds'] += elapsed
    budget['remaining_' + resource + '_seconds'] -= elapsed
    save(ROOT / 'budget.json', budget)
    print(json.dumps({k: record[k] for k in ('status', 'exit_code', 'wall_seconds')}))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--mode', choices=['oracle', 'replay'], default='oracle')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        sys.exit(launch(args))
