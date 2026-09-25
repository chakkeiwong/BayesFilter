#!/usr/bin/env python3
"""A04 bounded SGQF-initialization diagnostic on exposed recursive targets.

This is not a production filter, scalable coefficient implementation, or
independent-sequence validation. TensorFlow numerical reference setup and
reporting surround compiled recurrence and existing GPU/XLA fitting kernels.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT/'docs/plans/observation-aware-tt-master-amendment-04-sgqf-initialization-20260915.md'
SOURCE = ROOT/'docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914'


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--wall-budget-seconds', type=int, default=3600)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import tensorflow as tf
    physical = tf.config.list_physical_devices('GPU')
    if not physical:
        raise RuntimeError('A04 requires trusted GPU access')
    growth = []
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
        actual = tf.config.experimental.get_memory_growth(gpu)
        if not actual:
            raise RuntimeError('GPU growth policy failed')
        growth.append(dict(device=gpu.name, memory_growth=actual,
                           details=tf.config.experimental.get_device_details(gpu)))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal
    from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as proj
    D = tf.float64
    fixture_path = SOURCE/'diagnostic-02/downstream_fixture.json'
    fixture = json.loads(fixture_path.read_text())
    old_manifest = json.loads((SOURCE/'campaign-02/run_manifest.json').read_text())
    dependencies = [Path(__file__).resolve(),
        ROOT/'docs/benchmarks/observation_tt_sgqf_projection_diagnostic.py',
        ROOT/'bayesfilter/highdim/observation_guided_tt_tf.py',
        ROOT/'bayesfilter/highdim/pair_block_tt_tf.py',
        ROOT/'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py', PLAN]
    manifest = dict(command=[sys.executable, *sys.argv], plan=str(PLAN),
        result=str(out/'result.json'), started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        inputs={str(fixture_path.relative_to(ROOT)): sha(fixture_path)},
        environment=sys.prefix, python=sys.version, tensorflow=tf.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'),
        cpu_only=False, gpu_intentionally_hidden=False, jit_compile=True,
        numerical_dtype='float64', tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        gpu_memory_policy=dict(schema='a04_verified_growth_v1', mode='memory_growth', devices=growth),
        trust_basis='escalated_gpu_access', status='RUNNING',
        setup_exception='Gaussian setup, TT-SVD, panel/reference evaluation and reporting use TF outside XLA; recurrence and fitter use stable GPU/XLA graphs.',
        seeds=[815100, 815200, 815300], split_offsets=[0, 100000, 200000],
        row_counts=[1024, 4096, 8192], degree=3, rank=3, defensive_mass=.05,
        row_mixture_rho_fraction=.2, l1_grid=[0., 1e-5, 1e-3],
        base_schedule=[4, 128], extension_schedule=[8, 256],
        classification='smoke_only' if args.smoke else 'exposed_target_initialization_diagnostic',
        data_version='campaign-02 frozen downstream fixtures and retained TT marginals')
    write(out/'run_manifest.json', manifest)
    result = dict(status='RUNNING', cases=[], filter_promotion=False,
        ranking='No population or independent-sequence ranking is supported.',
        heuristic_dominance_verdict='NOT_EVALUATED')
    write(out/'result.json', result)

    def budget():
        if time.monotonic()-start > args.wall_budget_seconds:
            raise TimeoutError('A04 per-launch wall budget exhausted')

    def h2(amplitude, target, weights):
        tf.debugging.assert_all_finite(amplitude, 'candidate amplitude')
        tf.debugging.assert_greater_equal(amplitude, tf.constant(0., D))
        numerator = tf.reduce_sum(weights*amplitude*target)
        denominator = tf.sqrt(tf.reduce_sum(weights*amplitude**2)*tf.reduce_sum(weights*target**2))
        tf.debugging.assert_positive(denominator)
        value = 1.-numerator/denominator
        tf.debugging.assert_greater_equal(value, tf.constant(-1e-12, D))
        return tf.maximum(value, 0.)

    def rms(cores, panel):
        value = pair.evaluate_pair_cores(cores, panel['rows'])
        return tf.sqrt(tf.reduce_sum(panel['weights']*(value-panel['target'])**2)/
                       tf.reduce_sum(panel['weights']*panel['target']**2))

    def density_amplitude(cores, panel, defensive=.05):
        value = pair.evaluate_pair_cores(cores, panel['rows'])
        mass = pair.pair_total_mass(cores)
        tf.debugging.assert_positive(mass)
        return tf.sqrt((1.-defensive)*value**2/mass+defensive)

    def score(cores, panel, defensive=.05):
        return h2(density_amplitude(cores, panel, defensive), panel['target'], panel['weights'])

    try:
        # The target evaluator must still agree with the historical input provenance.
        for name in ('bayesfilter/highdim/observation_guided_tt_tf.py',
                     'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py'):
            if sha(ROOT/name) != old_manifest['source_hashes'][name]:
                raise RuntimeError(f'Saved target source changed: {name}')
        for d in ([1] if args.smoke else [1, 4]):
            data = fixture['dimensions'][str(d)]
            model = lib.SVModel(tf.constant(data['A'], D), tf.constant(data['P0'], D),
                                beta=fixture['beta'], sigma=fixture['sigma'])
            saved_path = SOURCE/f'campaign-02/d{d}/tt_pair_block_proposals.json'
            manifest['inputs'][str(saved_path.relative_to(ROOT))] = sha(saved_path)
            saved = json.loads(saved_path.read_text())
            cases = [(1, 'gaussian')] if args.smoke else [(1, 'gaussian'), (1, 'recursive_tt'), (17, 'recursive_tt'), (18, 'recursive_tt')]
            for t, incoming in cases:
                budget()
                record, prior = saved[t], saved[t-1]
                if record['time'] != t or prior['time'] != t-1:
                    raise ValueError('Saved timestep mismatch')
                current = lib.Chart(tf.constant(record['mean'], D), tf.constant(record['factor'], D))
                condition = lib.Chart(tf.constant(record['condition_mean'], D), tf.constant(record['condition_factor'], D))
                tf.debugging.assert_near(condition.mean, tf.constant(prior['mean'], D), atol=1e-12, rtol=1e-12)
                tf.debugging.assert_near(condition.factor, tf.constant(prior['factor'], D), atol=1e-12, rtol=1e-12)
                oldcores = tuple(tf.constant(c, D) for c in record['cores'])
                prevcores = tuple(tf.constant(c, D) for c in prior['cores'])
                retained = lib.PairRetainedProposal(prevcores, condition, pair.pair_total_mass(prevcores),
                    tf.constant(prior['tau'], D), t-1)
                observation = tf.constant(data['observations'][t], D)
                def logtarget(rows):
                    u, v = rows[:, :, 0], rows[:, :, 1]
                    x, z = current.forward(u), condition.forward(v)
                    previous = condition.log_prob(z) if incoming == 'gaussian' else retained.physical_log_density(z)
                    return (model.observation_log_prob(x, observation)+model.transition_log_prob(x, z)+previous
                        +current.logdet+condition.logdet-_log_standard_normal(u)-_log_standard_normal(v))
                mean, covariance = proj.paired_gaussian(model, current, condition)
                gaussian = lib.Chart.from_moments(mean, covariance)
                predmean, predcov = proj.paired_gaussian(model, current, condition, predictive=True)
                predictive = lib.Chart.from_moments(predmean, predcov)
                construct_start = time.monotonic()
                with tf.device('/GPU:0'):
                    coefficients, logM = proj.coefficient_kernel(2*d)(mean, covariance)
                reference_coefficients, _ = proj.coefficient_kernel(2*d, 3, False)(mean, covariance)
                tf.debugging.assert_near(coefficients, reference_coefficients, rtol=1e-10, atol=1e-11)
                projection_error = 1.-tf.reduce_sum(coefficients**2)
                tolerance = 256.*2.220446049250313e-16*(4**(2*d))
                tf.debugging.assert_greater_equal(projection_error, tf.constant(-tolerance, D))
                with tf.device('/CPU:0'):
                    sgcores, compression_error, spectra = proj.pair_svd(coefficients)
                for spectrum in spectra:
                    tf.debugging.assert_greater(tf.reduce_min(spectrum),
                        tf.reduce_max(spectrum)*tf.constant(256.*2.220446049250313e-16, D),
                        'Deficient initializer rank; no silent perturbation permitted')
                conversion = dict(projection_squared_l2=tf.maximum(projection_error, 0.),
                    compression_squared_l2=compression_error,
                    total_signed_amplitude_squared_l2=tf.maximum(projection_error, 0.)+compression_error,
                    exact_gaussian_affinity_to_rho=tf.exp(logM), mass=pair.pair_total_mass(sgcores),
                    retained_singular_values=spectra, wall_seconds=time.monotonic()-construct_start,
                    cores=sgcores, covariance=covariance, mean=mean)
                write(out/f'd{d}-t{t}-{incoming}-conversion.json', conversion)
                for replicate, base_seed in enumerate([815100] if args.smoke else manifest['seeds']):
                    budget()
                    case_start = time.monotonic()
                    seed = base_seed+1000*d+10*t
                    identifier = f'd{d}-t{t}-{incoming}-r{replicate}'
                    case = dict(id=identifier, dimension=d, time=t, incoming=incoming,
                        seed=seed, observation=observation, conversion={k:v for k,v in conversion.items() if k not in ('cores', 'covariance', 'mean')},
                        candidates={}, fits={}, status='RUNNING', exposed_observation=True)
                    def panel(count, offset, scale=None):
                        coordinates, logweights, rowinfo = lib.joint_sgqf_row_sampler(model, current, condition, count, seed+offset)
                        rows = tf.stack([coordinates[:, :d], coordinates[:, d:]], axis=-1)
                        logs = logtarget(rows)
                        if scale is None:
                            scale = tf.reduce_logsumexp(logs+logweights)-tf.math.log(tf.cast(count, D))
                        target = tf.exp(.5*(logs-scale))
                        tf.debugging.assert_all_finite(target, 'target amplitude')
                        return dict(rows=rows, target=target, weights=tf.exp(logweights), row_info=rowinfo, scale=scale)
                    train = panel(1024, 0)
                    validation = panel(4096, 100000, train['scale'])
                    features = pair._pair_features(train['rows'], 3)
                    case['training_log_scale'] = train['scale']
                    case['row_diagnostics'] = dict(train=train['row_info'], validation=validation['row_info'])
                    candidate_cores = dict(sgqf_conversion=sgcores, saved_tt=oldcores)
                    for family, rawinitial in [('generic', pair.initial_pair_cores(d)), ('sgqf', sgcores)]:
                        initvalue = pair.evaluate_pair_cores(rawinitial, train['rows'])
                        scalar = tf.reduce_sum(train['weights']*initvalue*train['target'])/tf.reduce_sum(train['weights']*initvalue**2)
                        tf.debugging.assert_positive(scalar, 'nonpositive training-only initialization scale')
                        initial = (rawinitial[0]*scalar, *rawinitial[1:])
                        initial_loss = .5*tf.reduce_mean(train['weights']*(scalar*initvalue-train['target'])**2)
                        fits = []
                        for penalty in [0., 1e-5, 1e-3]:
                            budget()
                            clock = time.monotonic()
                            with tf.device('/GPU:0'):
                                fitted, info = pair.fit_pair_features(features, train['target'], train['weights'],
                                    penalty=penalty, initial=initial)
                            fit = dict(l1=penalty, cores=fitted, info=info,
                                initial_objective=initial_loss+penalty*sum(tf.reduce_sum(tf.abs(c)) for c in initial),
                                initial_training_rms=rms(initial, train), initialization_scale=scalar,
                                train_rms=rms(fitted, train), validation_rms=rms(fitted, validation),
                                validation_h2=score(fitted, validation), wall_seconds=time.monotonic()-clock)
                            fits.append(fit)
                        selected = min(fits, key=lambda f: float(f['validation_rms']))
                        candidate_cores[family+'_fit'] = selected['cores']
                        case['fits'][family] = dict(selected_l1=selected['l1'], grid=fits)
                        clock = time.monotonic()
                        with tf.device('/GPU:0'):
                            extended, info = pair.fit_pair_features(features, train['target'], train['weights'],
                                sweeps=8, proximal_steps=256, penalty=selected['l1'], initial=initial)
                        candidate_cores[family+'_extended'] = extended
                        case['fits'][family]['extended'] = dict(cores=extended, info=info,
                            initialization_scale=scalar, train_rms=rms(extended, train),
                            validation_rms=rms(extended, validation), wall_seconds=time.monotonic()-clock)
                    def all_scores(p):
                        rows = tf.reshape(p['rows'], [-1, 2*d])
                        logrho = _log_standard_normal(rows)
                        values = dict(product_sgqf=h2(tf.ones_like(p['target']), p['target'], p['weights']),
                            predictive_joint=h2(tf.exp(.5*(predictive.log_prob(rows)-logrho)), p['target'], p['weights']),
                            exact_sgqf_joint=h2(tf.exp(.5*(gaussian.log_prob(rows)-logrho)), p['target'], p['weights']))
                        values.update({name: score(cores, p) for name, cores in candidate_cores.items()})
                        values['sgqf_conversion_no_defense'] = score(sgcores, p, 0.)
                        return values
                    validation_scores = all_scores(validation)
                    eligible = ['exact_sgqf_joint', 'sgqf_conversion', 'generic_fit', 'sgqf_fit']
                    selected_name = min(eligible, key=lambda name: float(validation_scores[name]))
                    case['safeguard'] = dict(selected=selected_name, eligible=eligible,
                        selection_metric='empirical_validation_h2', audit_used_for_selection=False,
                        validation_no_worse_than_sgqf=bool(validation_scores[selected_name] <= validation_scores['exact_sgqf_joint']))
                    write(out/(identifier+'-selection.json'), case['safeguard'])
                    # Audit is generated and evaluated only after the choice is written.
                    audit = panel(8192, 200000, train['scale'])
                    audit_scores = all_scores(audit)
                    case['row_diagnostics']['audit'] = audit['row_info']
                    for name in validation_scores:
                        case['candidates'][name] = dict(validation_h2=validation_scores[name], audit_h2=audit_scores[name])
                    for name, cores in candidate_cores.items():
                        case['candidates'][name]['audit_signed_rms'] = rms(cores, audit)
                    case['safeguard']['audit_delta_vs_sgqf'] = audit_scores[selected_name]-audit_scores['exact_sgqf_joint']
                    case['safeguard']['audit_no_worse_than_sgqf'] = bool(audit_scores[selected_name] <= audit_scores['exact_sgqf_joint'])
                    cheapest = min(['product_sgqf', 'predictive_joint', 'exact_sgqf_joint'], key=lambda name: float(audit_scores[name]))
                    case['heuristic_dominance'] = dict(best_heuristic=cheapest,
                        sgqf_fit_no_worse=bool(audit_scores['sgqf_fit'] <= audit_scores[cheapest]),
                        safeguard_no_worse=bool(audit_scores[selected_name] <= audit_scores[cheapest]),
                        role='empirical promotion veto, not a tuning target or statistical rank')
                    case.update(status='COMPLETE', wall_seconds=time.monotonic()-case_start)
                    write(out/(identifier+'.json'), case)
                    compact = {k:v for k,v in case.items() if k not in ('fits', 'conversion')}
                    result['cases'].append(compact)
                    write(out/'result.json', result)
                    print(json.dumps(plain(dict(case=identifier, selected=selected_name,
                        sgqf_h2=audit_scores['exact_sgqf_joint'], warm_h2=audit_scores['sgqf_fit'],
                        generic_h2=audit_scores['generic_fit'], wall_seconds=case['wall_seconds']))), flush=True)
        result['status'] = 'COMPLETE'
        result['heuristic_dominance_verdict'] = ('PROMOTION_VETO' if any(not c['heuristic_dominance']['sgqf_fit_no_worse'] for c in result['cases'])
                                                else 'SCREEN_PASSED_NO_PROMOTION')
        result['decision'] = 'Diagnostic complete; no filter promotion or unseen-data lower bound.'
        manifest['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED', failure_type=type(error).__name__, failure=str(error))
        manifest.update(status='FAILED', failure_type=type(error).__name__, failure=str(error))
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-start, completed_utc=datetime.now(timezone.utc).isoformat())
        manifest['gpu_allocator'] = {g.name: tf.config.experimental.get_memory_info('GPU:'+g.name.split(':')[-1]) for g in physical}
        write(out/'run_manifest.json', manifest)
        write(out/'result.json', result)


if __name__ == '__main__':
    main()
