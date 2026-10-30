#!/usr/bin/env python3
"""Independent CPU diagnostic: quadratic regression of TT importance likelihoods.

NumPy/SciPy are reference/reporting only. TensorFlow canonical callbacks supply
log joint values. The result is a numerical estimate, never an automatic oracle.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROCESS_START = time.monotonic()
PROCESS_START_UTC = dt.datetime.now(dt.timezone.utc).isoformat()
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from scipy.stats import qmc, t as student_t
import tensorflow as tf
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.testing.quadratic_score_reference import (
    fit_quadratic_score, log_weight_likelihood_ratios, quadratic_design,
    delete_group_log_ratios, equal_group_jackknife_se,
)
from bayesfilter.testing.zhao_cui_path_likelihood_reference import make_path_log_joint

PLAN = 'docs/plans/zhao-cui-publication-replication-and-score-20261004.md'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def antithetic_design(count, dimension, seed):
    if count < 2 or count % 2 or (count // 2) & (count // 2 - 1):
        raise ValueError('design count must be twice a power of two')
    half = 2 * qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(int(np.log2(count // 2))) - 1
    return np.concatenate([half, -half])


def evaluate_paths(kernel, theta, paths, observations, chunk):
    result = []
    for start in range(0, len(paths), chunk):
        result.append(kernel(tf.constant(theta, tf.float64),
                             tf.constant(paths[start:start+chunk], tf.float64),
                             tf.constant(observations, tf.float64)).numpy())
    value = np.concatenate(result)
    if np.any(np.isnan(value)) or np.any(np.isposinf(value)):
        raise ValueError('NaN or positive infinite path joint density')
    return value


def load_proposal(path, spec, horizon):
    from bayesfilter.testing.zhao_cui_checkpoint_reference import checkpoint_metadata
    manifest, result = checkpoint_metadata(path, horizon)
    settings = manifest['settings']
    source_model = 'pp' if spec.name == 'predator_prey' else 'sir_austria'
    if (settings['profile'] != 'current_target' or settings['route'] != 'linear'
        or settings['model'] != source_model or not 1 <= horizon <= settings['horizon']
    ):
        raise ValueError('complete same-target linear conditional proposal required')
    parity = np.loadtxt(path / 'target-parity-errors.csv', delimiter=',')
    if not np.all(np.isfinite(parity)) or np.max(parity) > 1e-8:
        raise ValueError('proposal target parity failed')
    dataset = json.loads((path / (source_model + '-input-dataset.json')).read_text())
    observed_tensor = tf.constant(dataset['observations'], tf.as_dtype(dataset['dtype']))
    observed_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(observed_tensor).numpy())).hexdigest()
    if observed_hash != dataset['observation_sha256'] or dataset['target_id'] != spec.target_id:
        raise ValueError('wrong observation hash or target identity')
    if len(dataset['observations']) != settings['horizon']:
        raise ValueError('dataset horizon does not match source fit')
    prefix_tensor = observed_tensor[:horizon]
    prefix_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(prefix_tensor).numpy())).hexdigest()
    mat_path = path / f'smoothing-t{horizon:02d}.mat'
    mat = loadmat(mat_path)
    if mat['thetas'].size:
        raise ValueError('joint-parameter samples cannot represent a conditional likelihood')
    paths = np.transpose(mat['sams'], (1, 2, 0))
    raw = mat['raw_log_weight'].reshape(-1)
    history = mat['proposal_history']
    if paths.shape != (len(raw), horizon + 1, spec.dimension) or history.shape != (len(raw), horizon):
        raise ValueError('path/proposal dimensions do not match')
    if not all(np.all(np.isfinite(a)) for a in (paths, history)):
        raise ValueError('nonfinite proposal paths or density')
    if np.any(np.isnan(raw)) or np.any(np.isposinf(raw)) or not np.any(np.isfinite(raw)):
        raise ValueError('invalid baseline importance weights')
    return paths, raw, history, np.asarray(dataset['observations'][:horizon], dtype=np.float64), dict(
        directory=str(path), parent_status=result['parent_status'], manifest_sha256=sha(path / 'manifest.json'),
        smoothing_sha256=sha(mat_path), observation_sha256=prefix_hash,
        source_observation_sha256=observed_hash, source_horizon=settings['horizon'], horizon=horizon,
        fit_seed=settings['fit_seed'], smooth_seed=settings['smooth_seed'], rank=settings['rank'],
        samples=len(raw), target_parity_max=float(np.max(parity)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, action='append', required=True)
    parser.add_argument('--model', choices=['predator_prey', 'sir_d18'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--plan-file',default=PLAN)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--heldout', type=int, default=256)
    parser.add_argument('--radii', type=float, nargs='+', default=[.04, .02, .01])
    parser.add_argument('--design-seed', type=int, default=43003)
    parser.add_argument('--path-chunk', type=int, default=2048)
    parser.add_argument('--diagnostic-max-paths', type=int, help='Mechanics-only subset; prevents scientific use')
    parser.add_argument('--jackknife-groups', type=int, default=20, help='Equal deleted path groups; 0 disables this conditional diagnostic')
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    if args.jackknife_groups < 0 or args.jackknife_groups == 1:
        parser.error('jackknife groups must be zero or at least two')
    if args.path_chunk < 1 or not 1 <= args.timeout_seconds <= 28800:
        parser.error('invalid chunk or out-of-budget timeout')
    if not args.radii or any(not np.isfinite(h) or h <= 0 for h in args.radii):
        parser.error('positive finite radii required')
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runner-source.py').write_bytes(Path(__file__).read_bytes())
    start = PROCESS_START
    spec = NonlinearSQMCSpec(args.model)
    theta0 = spec.default_theta(tf.float64).numpy()
    scales = theta0.copy() if args.model == 'predator_prey' else np.ones_like(theta0)
    train = antithetic_design(args.points, len(theta0), args.design_seed)
    heldout = antithetic_design(args.heldout, len(theta0), args.design_seed + 1)
    axes = np.concatenate([np.eye(len(theta0)), -np.eye(len(theta0))])
    design = np.concatenate([train, heldout, axes])
    kernel = make_path_log_joint(args.model, args.horizon)
    manifest = dict(schema='zhao_cui_quadratic_reference.v1', plan=args.plan_file, result=str(out / 'result.json'),
        command=[sys.executable, *sys.argv], git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started_utc=PROCESS_START_UTC, environment=sys.executable,
        tensorflow_version=tf.__version__, cpu_only=True, gpu_status='intentionally hidden with CUDA_VISIBLE_DEVICES=-1',
        jit_compile=True, parameter_names=list(spec.parameter_names), theta0=theta0.tolist(), scales=scales.tolist(),
        design_seed=args.design_seed, path_chunk=args.path_chunk, radii=args.radii,
        design_points=args.points, heldout_points=args.heldout, status='running',
        jackknife_groups=args.jackknife_groups, sample_prefix_fractions=[.25,.5],
        diagnostic_max_paths=args.diagnostic_max_paths, threads=2,
        likelihood_source='canonical TensorFlow model value callbacks on author-TT conditional paths',
        derivative_method='full quadratic local regression, no analytic scores or autodiff')
    dump(out / 'manifest.json', manifest)
    np.savez_compressed(out / 'design.npz', train=train, heldout=heldout, axes=axes, theta0=theta0, scales=scales)
    records, estimates, observation_hashes = [], [], set()
    try:
        for proposal_index, proposal in enumerate(args.proposal):
            paths, raw, history, observations, record = load_proposal(proposal.resolve(), spec, args.horizon)
            observation_hashes.add(record['observation_sha256'])
            if len(observation_hashes) != 1:
                raise ValueError('proposal observations differ')
            if args.diagnostic_max_paths:
                n = args.diagnostic_max_paths
                paths, raw, history = paths[:n], raw[:n], history[:n]
            base_joint = evaluate_paths(kernel, theta0, paths, observations, args.path_chunk)
            source_joint = raw + history[:, 0]
            if not np.array_equal(np.isneginf(source_joint), np.isneginf(base_joint)):
                raise ValueError('Octave/TF zero-density masks differ')
            finite = np.isfinite(base_joint)
            parity_error = float(np.max(np.abs(base_joint[finite] - source_joint[finite])))
            if parity_error > 1e-7:
                raise ValueError(f'Octave/TF joint density mismatch {parity_error}')
            raw = base_joint - history[:, 0]
            base_w = np.exp(raw - logsumexp(raw))
            record.update(evaluated_paths=len(paths), joint_density_parity_max=parity_error,
                          log_likelihood=float(logsumexp(raw) - np.log(len(raw))),
                          ess=float(1 / np.sum(base_w**2)), max_weight=float(np.max(base_w)),
                          zero_weight_count=int(np.count_nonzero(base_w == 0)),
                          likelihood_evaluator='canonical TensorFlow joint minus fixed finite Octave log proposal')
            groups = args.jackknife_groups
            if groups and (len(raw) % groups or groups > len(raw)):
                raise ValueError('path count must be divisible by declared jackknife groups')
            prefix_counts = sorted({max(1, len(raw)//4), max(1, len(raw)//2)})
            record['prefix_baselines'] = [
                dict(paths=n,log_likelihood=float(logsumexp(raw[:n])-np.log(n)),
                     ess=float(np.exp(2*logsumexp(raw[:n])-logsumexp(2*raw[:n]))))
                if np.any(np.isfinite(raw[:n])) else dict(paths=n,status='unavailable',reason='all-zero prefix weights')
                for n in prefix_counts]
            if groups:
                group_mass = base_w.reshape(groups,-1).sum(axis=1)
                record['maximum_baseline_group_mass'] = float(group_mass.max())
                try:
                    # Baseline zeros make the ratio exactly the remaining log mean weight.
                    baseline_deleted = delete_group_log_ratios(raw[None,:],np.zeros_like(raw),groups)[0]
                    record['conditional_log_likelihood_jackknife_se'] = float(equal_group_jackknife_se(baseline_deleted[:,None])[0])
                except ValueError as exc:
                    record['conditional_log_likelihood_jackknife_unavailable'] = str(exc)
            records.append(record)
            for radius in args.radii:
                likelihoods, ess_values, weight_maxima = [], [], []
                deleted_values, jackknife_failure = [], None
                prefixes = {n: [] for n in prefix_counts}
                prefix_failures = {}
                progress_path = out / f'progress-proposal{proposal_index:02d}-h{radius:g}.csv'
                with progress_path.open('w') as progress:
                    writer = csv.writer(progress)
                    writer.writerow(['point','log_likelihood_ratio','ess','max_weight','wall_seconds'])
                    for point_index, z in enumerate(design):
                        if time.monotonic() - start > args.timeout_seconds:
                            raise TimeoutError('regression runtime limit reached')
                        theta = theta0 + radius * scales * z
                        if args.model == 'predator_prey' and np.any(theta <= 0):
                            raise ValueError('local parameter neighborhood left positive domain')
                        log_weights = evaluate_paths(kernel, theta, paths, observations, args.path_chunk) - history[:, 0]
                        ratio, ess = log_weight_likelihood_ratios(log_weights[None, :], raw)
                        max_weight = float(np.exp(np.max(log_weights) - logsumexp(log_weights)))
                        likelihoods.append(float(ratio[0])); ess_values.append(float(ess[0])); weight_maxima.append(max_weight)
                        if point_index < len(train):
                            for n in prefix_counts:
                                if n in prefix_failures:
                                    continue
                                try:
                                    prefix_ratio,_ = log_weight_likelihood_ratios(log_weights[None,:n],raw[:n])
                                    prefixes[n].append(float(prefix_ratio[0]))
                                except ValueError as exc:
                                    prefix_failures[n] = str(exc)
                            if groups and jackknife_failure is None:
                                try:
                                    deleted_values.append(delete_group_log_ratios(log_weights[None,:],raw,groups)[0])
                                except ValueError as exc:
                                    jackknife_failure = str(exc)
                        writer.writerow([point_index, ratio[0], ess[0], max_weight, time.monotonic()-start])
                        progress.flush()
                values = np.asarray(likelihoods)
                fit = fit_quadratic_score(train, values[:len(train)], scales, radius)
                predicted = quadratic_design(heldout) @ fit.coefficients
                residual = values[len(train):len(train)+len(heldout)] - predicted
                axis_values = values[-len(axes):]
                central = (axis_values[:len(theta0)] - axis_values[len(theta0):]) / (2 * radius * scales)
                linear_design = np.column_stack([np.ones(len(train)), train])
                linear = np.linalg.lstsq(linear_design, values[:len(train)], rcond=None)[0][1:] / (radius * scales)
                estimate = dict(proposal=proposal_index, radius=radius, score=fit.score.tolist(), hessian=fit.hessian.tolist(),
                    central_difference=central.tolist(), linear_regression=linear.tolist(),
                    central_minus_quadratic=(central-fit.score).tolist(), linear_minus_quadratic=(linear-fit.score).tolist(),
                    condition_number=fit.condition_number, design_rank=fit.design_rank,
                    training_residual_rms=fit.residual_rms, training_residual_max=fit.residual_max,
                    heldout_residual_rms=float(np.sqrt(np.mean(residual**2))), heldout_residual_max=float(np.max(np.abs(residual))),
                    ess_min=float(min(ess_values)), ess_median=float(np.median(ess_values)), max_weight_max=float(max(weight_maxima)))
                estimate['prefix_scores'] = [
                    dict(paths=n,status='complete',score=fit_quadratic_score(train,np.asarray(v),scales,radius).score.tolist())
                    if n not in prefix_failures else dict(paths=n,status='unavailable',reason=prefix_failures[n])
                    for n,v in prefixes.items()]
                uncertainty = dict(status='disabled' if not groups else 'unavailable' if jackknife_failure else 'complete',
                    groups=groups, scope='conditional iid path sampling; excludes between-fit variation, support error and radius bias')
                if groups and jackknife_failure is None:
                    deleted_matrix = np.asarray(deleted_values)
                    deleted_scores = np.asarray([fit_quadratic_score(train,deleted_matrix[:,g],scales,radius).score for g in range(groups)])
                    uncertainty.update(standard_error=equal_group_jackknife_se(deleted_scores).tolist(),
                        deleted_group_scores=deleted_scores.tolist())
                elif jackknife_failure:
                    uncertainty['reason'] = jackknife_failure
                estimate['conditional_path_jackknife'] = uncertainty
                estimates.append(estimate)
                dump(out / f'estimate-proposal{proposal_index:02d}-h{radius:g}.json', estimate)
                np.savez_compressed(out / f'values-proposal{proposal_index:02d}-h{radius:g}.npz',
                    log_likelihood_ratio=values, ess=np.asarray(ess_values), max_weight=np.asarray(weight_maxima))
        independent_fits = (len({r['fit_seed'] for r in records}) == len(records)
                            and len({r['rank'] for r in records}) == 1
                            and len({r['evaluated_paths'] for r in records}) == 1)
        intervals = []
        if len(records) >= 3 and independent_fits:
            for radius in args.radii:
                scores = np.asarray([e['score'] for e in estimates if e['radius'] == radius])
                mean = np.mean(scores, axis=0)
                half = student_t.ppf(.975, len(scores)-1) * np.std(scores, axis=0, ddof=1) / np.sqrt(len(scores))
                intervals.append(dict(radius=radius, mean=mean.tolist(), lower=(mean-half).tolist(), upper=(mean+half).tolist(),
                                      interpretation='between-fit Monte Carlo t interval; excludes shared TT/support bias and radius bias'))
        result = dict(status='complete', proposals=records, estimates=estimates, independent_fit_intervals=intervals,
            oracle_status='not_certified', diagnostic_only=bool(args.diagnostic_max_paths),
            hard_veto_screen='finite values, same target and density parity passed; inspect ESS and stability quantitatively',
            statistically_supported_ranking='none; this estimates a score and does not rank filters',
            descriptive_only='ESS, heldout residuals and coordinate differences without independent-fit uncertainty',
            default_readiness='not evaluated; independent reference only',
            next_evidence='same-target independent likelihood agreement, radius/rank/path stability, independent fits and numerical-bias uncertainty')
    except Exception as exc:
        result = dict(status='failed', failure_type=type(exc).__name__, reason=str(exc), proposals=records, estimates=estimates,
                      oracle_status='not_certified')
    result['wall_seconds'] = time.monotonic()-start
    dump(out / 'result.json', result)
    manifest.update(status=result['status'], wall_seconds=result['wall_seconds'], proposals=records)
    dump(out / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(out), status=result['status'], proposals=len(records), estimates=len(estimates), wall_seconds=result['wall_seconds'])))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
