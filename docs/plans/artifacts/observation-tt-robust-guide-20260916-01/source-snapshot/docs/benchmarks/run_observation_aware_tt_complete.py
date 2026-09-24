"""Staged executable SGQF/TT repair, with preserved failures and bounded compute.

Algorithm kernels are TensorFlow; independent grid/reference reporting and
frozen-score checks are explicitly diagnostic. See the associated plan.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'docs/plans/observation-aware-tt-repair-complete-program-20260913.md'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--fixture-json', required=True, help='Frozen model/observation data; no runtime NumPy generation')
    p.add_argument('--dimensions', type=int, nargs='+', default=[1, 4])
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--particles', type=int, default=512)
    p.add_argument('--reference-particles', type=int, default=32768)
    p.add_argument('--seeds', type=int, nargs='+', default=[1101, 1102, 1103, 1104])
    p.add_argument('--rows', type=int, default=1024)
    p.add_argument('--rank', type=int, default=3)
    p.add_argument('--degree', type=int, default=3)
    p.add_argument('--sweeps', type=int, default=4)
    p.add_argument('--wall-budget-seconds', type=int, default=2400)
    p.add_argument('--cpu', action='store_true', help='Explicit diagnostic exception; hide GPUs before TF import')
    p.add_argument('--no-jit', action='store_true', help='Explicit diagnostic exception; default is XLA')
    p.add_argument('--training-seed', type=int, default=9100)
    p.add_argument('--reference-seed', type=int, default=70100)
    p.add_argument('--pair-sweeps', type=int, default=4)
    p.add_argument('--pair-rank', type=int, default=None, help='Pair rank; scalar comparator retains --rank')
    p.add_argument('--pair-proximal-steps', type=int, default=128)
    p.add_argument('--pair-block', action='store_true',
                   help='Run the paired current/previous TT extension in addition to baseline arms')
    p.add_argument('--reuse-fits-from', help='Diagnostic repair retry: reload all saved fits with exactly matching scope')
    return p


def plain(value):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(payload), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stationary_covariance(A, sigma):
    d = int(A.shape[0])
    kron = tf.reshape(tf.einsum('ij,kl->ikjl', A, A), [d*d, d*d])
    return tf.reshape(tf.linalg.solve(tf.eye(d*d, dtype=D)-kron,
                         tf.reshape(sigma*sigma*tf.eye(d, dtype=D), [-1, 1])), [d, d])


_FILTER_KERNELS = {}


def filter_kernel(model, jit):
    key = (tuple(float(x) for x in tf.reshape(model.transition, [-1])), model.beta, model.sigma, jit)
    if key in _FILTER_KERNELS:
        return _FILTER_KERNELS[key]
    d = model.dimension
    @tf.function(input_signature=[tf.TensorSpec([None, d], D), tf.TensorSpec([None], D),
        tf.TensorSpec([None, d], D), tf.TensorSpec([None], D), tf.TensorSpec([d], D),
        tf.TensorSpec([], tf.bool), tf.TensorSpec([], D)], jit_compile=jit, autograph=False)
    def update(x, logq, previous, logweights, y, initial, offset):
        logfactor = tf.cond(initial, lambda: model.prior_log_prob(x),
                           lambda: model.transition_log_prob(x, previous)) + model.observation_log_prob(x, y)
        logits = logweights + logfactor - logq
        increment = tf.reduce_logsumexp(logits)
        normalized = logits - increment
        weights = tf.exp(normalized)
        mean, covariance = lib.gaussian_moments(x, weights)
        ess = 1/tf.reduce_sum(weights*weights)
        maximum_weight = tf.reduce_max(weights)
        n = tf.shape(x)[0]
        resample = ess < .5*tf.cast(n, D)
        cumulative = tf.concat([tf.cumsum(weights)[:-1], tf.ones([1], D)], axis=0)
        positions = (tf.cast(tf.range(n), D)+offset)/tf.cast(n, D)
        indices = tf.minimum(tf.searchsorted(cumulative, positions, side='right'), n-1)
        indices = tf.where(resample, indices, tf.range(n))
        ancestor_counts = tf.math.unsorted_segment_sum(tf.ones_like(indices, dtype=tf.int32), indices, n)
        unique_ancestors = tf.reduce_sum(tf.cast(ancestor_counts > 0, tf.int32))
        carried = tf.gather(x, indices)
        carried_logweights = tf.where(resample, -tf.math.log(tf.cast(n, D))*tf.ones([n], D), normalized)
        return carried, carried_logweights, increment, mean, covariance, ess, maximum_weight, unique_ancestors, indices, resample, logfactor-logq
    _FILTER_KERNELS[key] = update
    return update


def particle_filter(model, observations, guide, tt_path, method, count, seed, jit, *, keep_history=False):
    d = model.dimension
    particles = tf.zeros([count, d], D)
    logweights = tf.fill([count], -tf.math.log(tf.cast(count, D)))
    update = filter_kernel(model, jit)
    records, history, total = [], [], tf.constant(0., D)
    for t, y in enumerate(tf.unstack(observations)):
        previous = particles
        diag = {}
        noise = tf.random.stateless_normal([count, d], [seed+t, 2], dtype=D)
        if method == 'tt_physical_defense':
            from bayesfilter.highdim.observation_robust_guide_tf import sample_physical_defense
            draws, logq, diag = sample_physical_defense(tt_path[t], previous, seed+t, jit)
        elif method in ('sgqf_joint', 'tt_sgqf_safeguard', 'tt_sgqf_initialized'):
            if tt_path is None:
                raise ValueError('A joint consumer requires its frozen proposal path')
            from bayesfilter.highdim.sgqf_joint_consumer_tf import sample_joint_step
            draws, logq, diag = sample_joint_step(tt_path[t], previous, seed+t, jit)
        elif tt_path is not None:
            if method == 'tt_pair_block' and t == 0:
                chart = guide[t][1]
                draws = chart.forward(noise)
                logq = chart.log_prob(draws)
                diag = {'pair_initial_gaussian': True}
            elif method == 'tt_pair_block':
                draws, logq, diag = lib.sample_pair_tt_step(tt_path[t], previous, seed+t, jit)
            else:
                draws, logq, diag = lib.sample_tt_step(tt_path[t], previous, seed+t, jit)
        elif method == 'sgqf_gaussian':
            chart = guide[t][1]
            draws = chart.forward(noise)
            logq = chart.log_prob(draws)
        elif method == 'stationary_prior' or t == 0:
            chart = lib.Chart(tf.zeros([d], D), tf.linalg.cholesky(model.covariance0))
            draws = chart.forward(noise)
            logq = model.prior_log_prob(draws)
        else:
            draws = tf.linalg.matmul(previous, model.transition, transpose_b=True)+model.sigma*noise
            logq = model.transition_log_prob(draws, previous)
        offset = tf.random.stateless_uniform([], [seed+t, 3], dtype=D)
        particles, logweights, inc, mean, cov, ess, maximum_weight, unique_ancestors, ancestors, resampled, correction = update(
            draws, logq, previous, logweights, y, tf.constant(t == 0), offset)
        lib.finite(inc, 'particle evidence')
        lib.finite(mean, 'particle moments')
        total += inc
        if keep_history:
            history.append(dict(x=draws, previous=previous, logq=logq, resampled=bool(resampled.numpy())))
        records.append(dict(time=t, mean=mean, covariance=cov, log_increment=inc, cumulative_log_evidence=total,
            ess=ess, maximum_weight=maximum_weight, unique_ancestors=unique_ancestors,
            resampled=resampled, minimum_log_correction=tf.reduce_min(correction),
            maximum_log_correction=tf.reduce_max(correction), **diag))
    return {'method': method, 'count': count, 'seed': seed, 'steps': records, 'log_evidence': total}, history


def scalar_grid_reference(model, observations, count):
    # Independent physical-space quadrature, not the fitted TT or its coordinates.
    bound = 12*math.sqrt(float(model.covariance0[0, 0]))
    x = tf.linspace(tf.constant(-bound, D), tf.constant(bound, D), count)
    dx = 2*bound/(count-1)
    weight = tf.concat([tf.constant([.5], D), tf.ones([count-2], D), tf.constant([.5], D)], 0)*dx
    density = tf.exp(model.prior_log_prob(x[:, None]))
    transition = tf.exp(-.5*((x[:, None]-model.transition[0, 0]*x[None, :])/model.sigma)**2)/(model.sigma*math.sqrt(2*math.pi))
    records, total = [], tf.constant(0., D)
    for t, y in enumerate(tf.unstack(observations)):
        if t:
            density = tf.linalg.matvec(transition, weight*density)
        logg = model.observation_log_prob(x[:, None], y)
        shift = tf.reduce_max(logg)
        density *= tf.exp(logg-shift)
        z = tf.reduce_sum(weight*density)
        total += tf.math.log(z)+shift
        density /= z
        mean = tf.reduce_sum(weight*density*x)
        variance = tf.reduce_sum(weight*density*(x-mean)**2)
        records.append(dict(mean=tf.reshape(mean, [1]), covariance=tf.reshape(variance, [1, 1])))
    return dict(method='independent_scalar_grid', points=count, bound=bound, log_evidence=total, steps=records)


def frozen_score_check(model, observations, history):
    # Analytical derivative of the SAME finite scalar with proposal values,
    # sampled states and resampling decisions frozen. All three physical controls.
    A, P = model.transition, model.covariance0
    d = model.dimension
    kron = tf.reshape(tf.einsum('ij,kl->ikjl', A, A), [d*d, d*d])
    rhs = P @ tf.transpose(A) + A @ P
    dP = tf.reshape(tf.linalg.solve(tf.eye(d*d, dtype=D)-kron, tf.reshape(rhs, [-1, 1])), [d, d])
    invP = tf.linalg.inv(P)
    count = int(history[0]['x'].shape[0])
    logweights = tf.fill([count], -tf.math.log(tf.cast(count, D)))
    dlogweights = tf.zeros([count, 3], D)
    value, score = tf.constant(0., D), tf.zeros([3], D)
    for t, (record, y) in enumerate(zip(history, tf.unstack(observations))):
        x, previous = record['x'], record['previous']
        beta_score = tf.reduce_sum(tf.square(y)/model.beta**2*tf.exp(-x)-1, axis=1)
        if t == 0:
            z = tf.linalg.matmul(x, invP)
            gamma_score = .5*(tf.einsum('ni,ij,nj->n', z, dP, z)-tf.linalg.trace(invP @ dP))
            sigma_score = -d+tf.einsum('ni,ni->n', x, z)
            logfactor = model.prior_log_prob(x)
        else:
            residual = x-tf.linalg.matmul(previous, A, transpose_b=True)
            gamma_score = tf.reduce_sum(residual*previous, axis=1)/model.sigma**2
            sigma_score = -d+tf.reduce_sum(residual**2, axis=1)/model.sigma**2
            logfactor = model.transition_log_prob(x, previous)
        logfactor += model.observation_log_prob(x, y)
        logits = logweights+logfactor-record['logq']
        inc = tf.reduce_logsumexp(logits)
        logweights = logits-inc
        directions = dlogweights+tf.stack([gamma_score, beta_score, sigma_score], axis=1)
        inc_score = tf.einsum('n,nk->k', tf.exp(logweights), directions)
        score += inc_score
        value += inc
        dlogweights = directions-inc_score
        if record['resampled']:
            logweights = tf.fill([count], -tf.math.log(tf.cast(count, D)))
            dlogweights = tf.zeros([count, 3], D)
    def replay(delta):
        matrix = A+delta[0]*tf.eye(d, dtype=D)
        sigma = model.sigma*tf.exp(delta[2])
        altered = lib.SVModel(matrix, stationary_covariance(matrix, sigma), model.beta, sigma)
        lw = tf.fill([count], -tf.math.log(tf.cast(count, D)))
        val = tf.constant(0., D)
        for t, (record, y) in enumerate(zip(history, tf.unstack(observations))):
            x, prev = record['x'], record['previous']
            factor = altered.prior_log_prob(x) if t == 0 else altered.transition_log_prob(x, prev)
            factor += model.observation_log_prob(x, y, tf.math.log(tf.constant(model.beta, D))+delta[1])
            raw = lw+factor-record['logq']
            inc = tf.reduce_logsumexp(raw)
            val += inc
            lw = raw-inc
            if record['resampled']:
                lw = tf.fill([count], -tf.math.log(tf.cast(count, D)))
        return val
    h = 1e-5
    fd = tf.stack([(replay(h*tf.one_hot(j, 3, dtype=D))-replay(-h*tf.one_hot(j, 3, dtype=D)))/(2*h) for j in range(3)])
    error = tf.reduce_max(tf.abs(fd-score)/(1+tf.abs(score)))
    return dict(parameters=['gamma_additive_diagonal', 'log_beta', 'log_sigma'], value=value,
                analytic_score=score, finite_difference=fd, relative_error=error,
                status='PASS' if float(error) < 2e-6 else 'FAIL',
                target='frozen_proposal_and_realized_resampling_finite_particle_log_evidence',
                retraining_derivative_claimed=False, posterior_score_accuracy_claimed=False)


def mean_and_se(values):
    x = tf.stack(values)
    mean = tf.reduce_mean(x, axis=0)
    n = int(x.shape[0])
    se = tf.sqrt(tf.reduce_sum((x-mean)**2, axis=0)/(n*(n-1))) if n > 1 else tf.ones_like(mean)*float('inf')
    return mean, se


def summarize(model, observations, methods, reference, grid):
    ref_value, ref_se = mean_and_se([x['log_evidence'] for x in reference])
    ref_means, ref_mean_se = mean_and_se([tf.stack([s['mean'] for s in x['steps']]) for x in reference])
    if grid is not None:
        ref_value, ref_se = grid['log_evidence'], tf.constant(0., D)
        ref_means = tf.stack([s['mean'] for s in grid['steps']])
        ref_mean_se = tf.zeros_like(ref_means)
    regimes = []
    for y in tf.unstack(observations):
        # Classify before examining method outcomes, using paper-scale beta.
        magnitude = float(tf.reduce_max(tf.abs(y)/model.beta))
        regimes.append('near_zero' if magnitude <= .5 else ('large' if magnitude >= 2 else 'ordinary'))
    table = {}
    for name, runs in methods.items():
        logz, se = mean_and_se([x['log_evidence'] for x in runs])
        means, means_se = mean_and_se([tf.stack([s['mean'] for s in x['steps']]) for x in runs])
        evidence_pass = bool(tf.abs(logz-ref_value) <= 3.182446*tf.sqrt(se*se+ref_se*ref_se)+.15)
        mean_pass = bool(tf.reduce_all(tf.abs(means-ref_means) <= 3.182446*tf.sqrt(means_se**2+ref_mean_se**2)+.15*model.sigma))
        conditional = {}
        for regime in ('near_zero', 'ordinary', 'large'):
            indices = [t for t, r in enumerate(regimes) if r == regime]
            conditional[regime] = {'count': len(indices), 'rmse_to_reference': (
                tf.sqrt(tf.reduce_mean(tf.gather(means-ref_means, indices)**2)) if indices else None)}
        table[name] = dict(log_evidence_mean=logz, log_evidence_mcse=se, reference_gap=logz-ref_value,
                          evidence_agreement_screen=evidence_pass, moment_agreement_screen=mean_pass,
                          conditional_mean_errors=conditional)
    adversaries = ('transition', 'stationary_prior', 'sgqf_gaussian')
    for name in ('tt_predictive', 'tt_guided', 'tt_pair_block'):
        if name not in table:
            continue
        losses = []
        for regime in ('near_zero', 'ordinary', 'large'):
            value = table[name]['conditional_mean_errors'][regime]['rmse_to_reference']
            if value is None:
                continue
            for adversary in adversaries:
                baseline = table[adversary]['conditional_mean_errors'][regime]['rmse_to_reference']
                if float(value) > float(baseline):
                    losses.append(dict(regime=regime, adversary=adversary))
        table[name]['heuristic_dominance'] = 'DESCRIPTIVE_UNDERPERFORMANCE_PROMOTION_VETO' if losses else 'NO_OBSERVED_LOSS_NOT_PROOF'
        table[name]['descriptive_losses'] = losses
        table[name]['statistically_supported_ranking'] = False
    return dict(reference_log_evidence=ref_value, reference_mcse=ref_se, methods=table,
                regimes=regimes, ranking='not established; four seeds support screening, not superiority',
                primary_criterion='reference agreement of filtering moments and evidence',
                default_ready=False)


def load_saved_fit(args, method, dimension, guide, on_step):
    """Reload the original frozen regression, checking charts and fit provenance."""
    source = Path(args.reuse_fits_from) / f'd{dimension}'
    saved = json.loads((source/f'{method}_proposals.json').read_text())
    if len(saved) != args.horizon:
        raise ValueError('saved fit horizon mismatch')
    steps = []
    for t, record in enumerate(saved):
        if record['time'] != t:
            raise ValueError('saved fit timestep mismatch')
        current = lib.Chart(tf.constant(record['mean'], D), tf.constant(record['factor'], D))
        condition = (lib.Chart(tf.constant(record['condition_mean'], D), tf.constant(record['condition_factor'], D))
                     if record['condition_mean'] is not None else None)
        expected = guide[t][0 if method == 'tt_predictive' else 1]
        tf.debugging.assert_near(current.mean, expected.mean, rtol=1e-12, atol=1e-12)
        tf.debugging.assert_near(current.factor, expected.factor, rtol=1e-12, atol=1e-12)
        if t and condition is None:
            raise ValueError('saved fit missing previous-state chart')
        if not t and condition is not None:
            raise ValueError('saved initial fit has unexpected conditioning chart')
        if condition is not None:
            tf.debugging.assert_near(condition.mean, guide[t-1][1].mean, rtol=1e-12, atol=1e-12)
            tf.debugging.assert_near(condition.factor, guide[t-1][1].factor, rtol=1e-12, atol=1e-12)
        cores = tuple(tf.constant(c, D) for c in record['cores'])
        for core in cores:
            lib.finite(core, 'saved fit core')
        tau = tf.constant(record['tau'], D)
        tf.debugging.assert_positive(tau)
        info = json.loads((source/f'{method}_fit_t{t:02d}.json').read_text())
        on_step(t, info)
        if method == 'tt_pair_block':
            step = lib.PairTTStep(cores, current, condition, tau, info, t, None)
        else:
            step = lib.TTStep(cores, current, condition, tau, info, t)
        steps.append(step)
    return steps


def execute(args, out, manifest, start):
    fixture = json.loads(Path(args.fixture_json).read_text())
    final = {'status': 'RUNNING', 'dimensions': {}, 'scientific_success': False}
    def budget():
        if time.monotonic()-start > args.wall_budget_seconds:
            raise TimeoutError('predeclared wall budget exhausted')
    for dimension in args.dimensions:
        budget()
        dest = out / f'd{dimension}'
        dest.mkdir()
        data = fixture['dimensions'][str(dimension)]
        model = lib.SVModel(tf.constant(data['A'], D), tf.constant(data['P0'], D), fixture['beta'], fixture['sigma'])
        observations = tf.constant(data['observations'][:args.horizon], D)
        residual = tf.reduce_max(tf.abs(model.covariance0-model.transition @ model.covariance0 @ tf.transpose(model.transition)-model.sigma**2*tf.eye(dimension, dtype=D)))
        if float(residual) > 1e-10:
            raise ValueError('fixture stationary covariance mismatch')
        print(f'd={dimension} stage=SGQF horizon={args.horizon}', flush=True)
        guide, guide_records = lib.build_guide_path(model, observations)
        # Magnitude perturbation and history dependence at the same later y.
        altered = tf.concat([observations[:1]*2+tf.constant(.03, D), observations[1:]], 0)
        alternate, _ = lib.build_guide_path(model, altered, levels=(4, 5))
        response = tf.linalg.norm(guide[0][1].mean-alternate[0][1].mean)
        history_response = tf.linalg.norm(guide[1][1].mean-alternate[1][1].mean) if len(guide)>1 else response
        if float(response) < 1e-8 or float(history_response) < 1e-8:
            raise ValueError('SGQF observation/history response missing')
        write(dest/'step0_sgqf.json', dict(records=guide_records, observation_response=response, history_response=history_response,
                                         recursive_observation_history=True, gaussian_closure=True, exact_filter_claimed=False))
        reference = []
        print(f'd={dimension} stage=independent_reference', flush=True)
        for index in range(len(args.seeds)):
            budget()
            ref, _ = particle_filter(model, observations, guide, None, 'transition', args.reference_particles, args.reference_seed+100*index, not args.no_jit)
            reference.append(ref)
        write(dest/'reference.json', reference)
        grid = None
        if dimension == 1:
            coarse = scalar_grid_reference(model, observations, 801)
            grid = scalar_grid_reference(model, observations, 1201)
            discrepancy = abs(float(coarse['log_evidence']-grid['log_evidence']))
            if discrepancy > 1e-6:
                raise ValueError('scalar grid resolution check failed')
            write(dest/'grid_reference.json', dict(coarse=coarse, fine=grid, log_evidence_resolution_gap=discrepancy))
        # Check the actual SGQF moments against an independent filtering reference.
        reference_mean, _ = mean_and_se([tf.stack([s['mean'] for s in x['steps']]) for x in reference])
        guide_gap = float(tf.reduce_max(tf.abs(tf.stack([g[1].mean for g in guide])-reference_mean)))
        write(dest/'step0_reference_check.json', dict(maximum_mean_gap=guide_gap,
            threshold=1.0*model.sigma, status='PASS' if guide_gap <= model.sigma else 'FAIL',
            interpretation='rough proposal guide, not exact filtering'))
        if guide_gap > model.sigma:
            raise ValueError('SGQF failed predeclared rough-guide reference screen')
        methods, histories = {}, {}
        methods_to_run = ('transition', 'stationary_prior', 'sgqf_gaussian', 'tt_predictive', 'tt_guided')
        if args.pair_block:
            methods_to_run = methods_to_run + ('tt_pair_block',)
        for method in methods_to_run:
            budget()
            path = None
            if method.startswith('tt_'):
                print(f'd={dimension} stage=fit method={method}', flush=True)
                def on_step(t, info):
                    budget()
                    write(dest/f'{method}_fit_t{t:02d}.json', info)
                    print(f'd={dimension} method={method} t={t} audit_rms={float(info["audit_relative_rms"]):.4g}', flush=True)
                if args.reuse_fits_from:
                    path = load_saved_fit(args, method, dimension, guide, on_step)
                elif method == 'tt_pair_block':
                    path = lib.build_pair_tt_path(model, observations, guide, guided=True,
                        seed=args.training_seed+dimension*100, degree=args.degree, rank=args.pair_rank or args.rank, rows=args.rows,
                        sweeps=args.pair_sweeps, proximal_steps=args.pair_proximal_steps, importance_rows=True,
                        jit_compile=not args.no_jit, on_step=on_step)
                else:
                    path = lib.build_tt_path(model, observations, guide, guided=method=='tt_guided',
                        seed=args.training_seed+dimension*100, degree=args.degree, rank=args.rank, rows=args.rows,
                        sweeps=args.sweeps, jit_compile=not args.no_jit, on_step=on_step)
                write(dest/f'{method}_proposals.json', [dict(time=s.time_index, cores=s.cores, tau=s.tau,
                    mean=s.current_chart.mean, factor=s.current_chart.factor,
                    condition_mean=s.conditioning_chart.mean if s.conditioning_chart else None,
                    condition_factor=s.conditioning_chart.factor if s.conditioning_chart else None) for s in path])
            runs = []
            for index, seed in enumerate(args.seeds):
                budget()
                run, history = particle_filter(model, observations, guide, path, method, args.particles,
                                                seed*100, not args.no_jit, keep_history=index==0 and method=='tt_guided')
                runs.append(run)
                if history:
                    histories[method] = history
            methods[method] = runs
            write(dest/f'{method}_particles.json', runs)
        score = frozen_score_check(model, observations, histories['tt_guided'])
        write(dest/'frozen_score_check.json', score)
        if score['status'] != 'PASS':
            raise ValueError('same-scalar analytical score check failed')
        write(dest/'frozen_score_history.json', histories['tt_guided'])
        result = summarize(model, observations, methods, reference, grid)
        result['sgqf_reference_gap'] = guide_gap
        result['score_status'] = score['status']
        write(dest/'decision.json', result)
        final['dimensions'][str(dimension)] = result
        write(out/'result.json', final)
    final['status'] = 'EXECUTED'
    selected_method = 'tt_pair_block' if args.pair_block else 'tt_guided'
    final['scientific_success'] = all(
        entry['methods'][selected_method]['evidence_agreement_screen'] and
        entry['methods'][selected_method]['moment_agreement_screen'] and
        not entry['methods'][selected_method]['descriptive_losses'] for entry in final['dimensions'].values())
    final['classification'] = 'extension_or_invention'
    final['default_ready'] = False
    final['remaining_budget_seconds'] = args.wall_budget_seconds-(time.monotonic()-start)
    return final


def main():
    global tf, lib, D, PLAN
    args = parser().parse_args()
    if args.pair_block:
        PLAN = ROOT / "docs/plans/observation-tt-pair-block-remedy-20260914.md"
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-tt-complete')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        plan=str(PLAN), result=str(out/'result.json'), config=vars(args),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(),
            ROOT/'bayesfilter/highdim/observation_guided_tt_tf.py', ROOT/'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py', ROOT/'bayesfilter/highdim/pair_block_tt_tf.py', PLAN]},
        fixture_sha256=sha(Path(args.fixture_json)), cpu_only=args.cpu, gpu_intentionally_hidden=args.cpu,
        environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS']},
        jit_compile=not args.no_jit, status='STARTING',
        setup_exception='SGQF rule construction, chart setup, target preparation and independent references run outside compiled hot kernels')
    write(out/'run_manifest.json', manifest)
    try:
        if args.reuse_fits_from:
            source = Path(args.reuse_fits_from)
            original = json.loads((source/'run_manifest.json').read_text())
            scope = ['dimensions', 'horizon', 'particles', 'reference_particles', 'rows',
                     'rank', 'degree', 'sweeps', 'seeds', 'training_seed', 'reference_seed',
                     'pair_block', 'pair_rank', 'pair_sweeps', 'pair_proximal_steps', 'cpu', 'no_jit']
            for name in scope:
                if vars(args)[name] != original['config'][name]:
                    raise ValueError(f'saved fit scope mismatch: {name}')
            if sha(Path(args.fixture_json)) != original['fixture_sha256']:
                raise ValueError('saved fit fixture mismatch')
            dependencies = [source/'run_manifest.json']
            for dimension in args.dimensions:
                for method in ['tt_predictive', 'tt_guided'] + (['tt_pair_block'] if args.pair_block else []):
                    dependencies.append(source/f'd{dimension}/{method}_proposals.json')
                    dependencies.extend(source/f'd{dimension}/{method}_fit_t{t:02d}.json' for t in range(args.horizon))
            manifest['reused_fit_hashes'] = {str(f): sha(f) for f in dependencies}
            manifest['refitting_performed'] = False
            manifest['recovery_plan'] = 'docs/plans/observation-tt-pair-sampler-recovery-20260914.md'
        import tensorflow as tf
        D = tf.float64
        if not args.cpu:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                raise RuntimeError('GPU execution requested but no physical GPU is visible')
            devices = []
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                growth = tf.config.experimental.get_memory_growth(gpu)
                if not growth:
                    raise RuntimeError('memory growth not verified before initialization')
                devices.append(dict(name=gpu.name, memory_growth=growth, details=tf.config.experimental.get_device_details(gpu)))
            manifest['gpu_memory_policy'] = dict(mode='memory_growth', devices=devices)
        from bayesfilter.highdim import observation_guided_tt_tf as lib
        manifest.update(tensorflow=tf.__version__, python=sys.version, conda_prefix=sys.prefix, status='RUNNING',
                        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(), numerical_dtype='float64')
        write(out/'run_manifest.json', manifest)
        final = execute(args, out, manifest, start)
        write(out/'result.json', final)
        manifest['status'] = final['status']
        print(json.dumps({'status': final['status'], 'scientific_success': final['scientific_success'], 'output': str(out)}), flush=True)
    except Exception as exc:
        manifest['status'] = 'FAILED'
        (out/'traceback.log').write_text(traceback.format_exc())
        write(out/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), wall_seconds=time.monotonic()-start))
        print(f'FAILED: {type(exc).__name__}: {exc}; see {out}/traceback.log', flush=True)
        raise
    finally:
        if not args.cpu and 'tf' in globals():
            try:
                manifest['gpu_allocator'] = tf.config.experimental.get_memory_info('GPU:0')
            except Exception as allocator_error:
                manifest['gpu_allocator_report_error'] = str(allocator_error)
        manifest.update(wall_seconds=time.monotonic()-start, completed_utc=datetime.now(timezone.utc).isoformat())
        write(out/'run_manifest.json', manifest)


if __name__ == '__main__':
    main()
