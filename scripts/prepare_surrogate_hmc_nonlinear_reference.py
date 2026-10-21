#!/usr/bin/env python3
"""Bounded CPU independent-reference preparation for the actual M10 law."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('BAYESFILTER_PRELOAD_CUSTOM_OP', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = 'docs/plans/surrogate-hmc-nonlinear-reference-2026-09-20.md'
MARKER = '-m10-reference-'


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def execute(output, report, remaining_wall, remaining_cpu):
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import (
        BASE_THETA, TRANSITION_CURVE, OBSERVATION_CURVE, PRIOR_SD, PARTITION_SEEDS,
        GRID_LADDER, PROBES, generate_dataset, grid_reference,
        linear_kalman_log_likelihood, direct_innovation_reference)
    report.update(tensorflow=tf.__version__, checks={}, grids=[], partitions={},
        model=dict(theta0=BASE_THETA, transition_curve=TRANSITION_CURVE,
            observation_curve=OBSERVATION_CURVE, horizon=2, prior_sd=PRIOR_SD,
            coordinate='additive log standard deviations, identity chart'),
        role='local likelihood reference only', posterior_reference_qualified=False,
        retained_inference=False, full_prior_support_checked=False)
    datasets = {}
    for role, component in PARTITION_SEEDS.items():
        for regime in ('nominal', 'stress'):
            seed = [20260920, component+(1000 if regime == 'stress' else 0)]
            values = generate_dataset(seed, stress=regime == 'stress')
            tf.debugging.assert_all_finite(values, 'generated observations')
            name = role+'-'+regime
            path = output/name
            path.mkdir()
            payload = dict(seed=seed, raw_observations=values.numpy().tolist(), dtype='float32',
                horizon=2, role=role, regime=regime, claim_used_for_preparation=False)
            save(path/'data.json', payload)
            report['partitions'][name] = dict(data_file=str(path/'data.json'),
                data_sha256=hashlib.sha256((path/'data.json').read_bytes()).hexdigest())
            if role != 'claim':
                datasets[name] = tf.cast(values, tf.float64)
    probes = tf.constant(PROBES, tf.float64)
    histories = []
    for points, radius in GRID_LADDER:
        started = time.monotonic()
        kernel = grid_reference(points, radius)
        rows = {}
        for name, observations in datasets.items():
            rows[name] = [kernel(probes[index], observations) for index in range(9)]
            for result in rows[name]:
                for value in result.values():
                    tf.debugging.assert_all_finite(value, 'grid reference')
                tf.debugging.assert_positive(result['final_variance'])
        serialized = tf.nest.map_structure(lambda value: value.numpy().tolist(), rows)
        filename = f'grid-{points}-{radius:g}.json'
        save(output/filename, serialized)
        report['grids'].append(dict(points=points, radius=radius, values_file=filename,
            sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest(),
            wall_seconds=time.monotonic()-started))
        histories.append(rows)
        save(output/'report.json', report)
        print('completed grid', points, radius, 'CPU remaining', round(remaining_cpu(), 2), flush=True)
    for name, coarse, fine in (('mesh', histories[-3], histories[-2]),
                               ('domain', histories[-2], histories[-1])):
        errors = []
        score_errors = []
        for dataset in datasets:
            for a, b in zip(coarse[dataset], fine[dataset]):
                errors.append(float(tf.abs(a['log_likelihood']-b['log_likelihood'])))
                score_errors.append(float(tf.reduce_max(tf.abs(a['score']-b['score'])/(1+tf.abs(b['score'])))))
        report[name+'_refinement'] = dict(maximum_log_likelihood_change=max(errors),
            maximum_relative_score_change=max(score_errors))
        report['checks'][name+'_refinement'] = max(errors) <= 1e-6 and max(score_errors) <= 1e-5
    final = histories[-1]
    for key in ('domain_loss', 'posterior_edge'):
        maximum = max(float(row[key]) for rows in final.values() for row in rows)
        report[key] = maximum
        report['checks'][key] = maximum <= 1e-8

    # The scalar Kalman recursion shares only physical parameter interpretation,
    # not the state quadrature or its value/derivative recursion.
    observations = datasets['calibration-nominal']
    linear = grid_reference(513, 9., 0., 0.)
    linear_errors = [float(tf.abs(linear(probes[i], observations)['log_likelihood']
        -linear_kalman_log_likelihood(probes[i], observations))) for i in range(9)]
    report['linear_identity_maximum_error'] = max(linear_errors)
    report['checks']['linear_identity'] = max(linear_errors) <= 1e-6
    score_errors = []
    kernel = grid_reference(513, 9.)
    for observations in datasets.values():
        for index in (0, 4, 8):
            score = kernel(probes[index], observations)['score']
            for j in (0, 1):
                delta = tf.one_hot(j, 2, dtype=tf.float64)*1e-5
                difference = (kernel(probes[index]+delta, observations)['log_likelihood']
                    -kernel(probes[index]-delta, observations)['log_likelihood'])/(2e-5)
                score_errors.append(float(tf.abs(difference-score[j])/(1+tf.abs(score[j]))))
    report['finite_difference_relative_error'] = max(score_errors)
    report['checks']['analytical_noise_scores'] = max(score_errors) <= 1e-5
    save(output/'report.json', report)
    if not all(report['checks'].values()):
        report['status'] = 'REFERENCE_CHECK_FAILED'
        return

    independent_rows = []
    lower, higher = direct_innovation_reference(64), direct_innovation_reference(128)
    for name in ('calibration-nominal', 'calibration-stress'):
        for index in (0, 4, 8):
            begun, before = time.monotonic(), cpu_seconds()
            low = lower(probes[index], datasets[name])
            high = higher(probes[index], datasets[name])
            tf.debugging.assert_all_finite(tf.stack([low, high]), 'independent integration')
            grid = final[name][index]['log_likelihood']
            row = dict(dataset=name, probe=PROBES[index], low_order=64, high_order=128,
                log_value_low=float(low), log_value_high=float(high), log_value_grid=float(grid),
                refinement_error=float(tf.abs(low-high)), agreement_error=float(tf.abs(high-grid)))
            independent_rows.append(row)
            report['independent_integration'] = independent_rows
            if len(independent_rows) == 1:
                # Includes compilation and overestimates steady cost deliberately.
                forecast = dict(wall_seconds=1.5*(time.monotonic()-begun)*5,
                    cpu_worker_seconds=1.5*(cpu_seconds()-before)*5)
                report['independent_integration_remaining_forecast'] = forecast
                if forecast['wall_seconds'] > remaining_wall() or forecast['cpu_worker_seconds'] > remaining_cpu():
                    report['status'] = 'UNDER_BUDGETED'
                    return
            save(output/'report.json', report)
    report['checks']['independent_nonlinear_refinement'] = all(row['refinement_error'] <= 1e-6 for row in independent_rows)
    report['checks']['independent_nonlinear_agreement'] = all(row['agreement_error'] <= 1e-6 for row in independent_rows)
    report['status'] = ('QUALIFIED_LOCAL_LIKELIHOOD_REFERENCE' if all(report['checks'].values())
                        else 'REFERENCE_CHECK_FAILED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if ((MARKER not in args.output.name and args.output.name != 'P3-M10-reference')
            or '-tests-' in args.output.name):
        parser.error('fresh output must use the M10 reference accounting marker')
    from bayesfilter.inference.surrogate_hmc_program import (
        validated_budget, charge_previous_runs, GPU_CAP, CPU_CAP, m10_reference_state)
    accounting = list((ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'))
    ledger = validated_budget(GPU_CAP, CPU_CAP)
    charge_previous_runs(ledger, accounting)
    state = m10_reference_state(ROOT)
    if state['qualified']:
        raise ValueError('M10 local likelihood reference already qualified; reuse its saved evidence')
    cpu_budget = min(state['cpu_worker_seconds'], CPU_CAP-ledger['charged']['cpu_worker_seconds'])
    wall_budget = state['wall_seconds']
    if min(cpu_budget, wall_budget) <= 0:
        raise ValueError('M10 preparation attempt/budget limit')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [PLAN, 'scripts/prepare_surrogate_hmc_nonlinear_reference.py',
        'bayesfilter/testing/surrogate_hmc_nonlinear_reference.py',
        'tests/test_surrogate_hmc_nonlinear_reference.py',
        'bayesfilter/score_study/nonlinear_tf.py', 'bayesfilter/score_study/gaussian_tf.py']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    report = dict(status='RUNNING', plan=PLAN)
    manifest = dict(command=[sys.executable, *sys.argv], environment=sys.prefix,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        plan=PLAN, result_file=str(output/'report.json'), source_hashes=hashes,
        cpu_only=True, gpu_intentionally_hidden=True, jit_compile=True,
        cpu_budget=cpu_budget, wall_budget=wall_budget,
        seeds=dict(date_key=20260920, nominal=[14100,14200,14300], stress=[15100,15200,15300]))
    start, before = time.monotonic(), cpu_seconds()
    save(output/'run-manifest.json', manifest)
    save(output/'report.json', report)

    def timeout(signum, frame):
        raise TimeoutError('M10 reference allocation exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_REAL, wall_budget)
    signal.setitimer(signal.ITIMER_PROF, cpu_budget)
    try:
        execute(output, report, lambda: wall_budget-(time.monotonic()-start),
                lambda: cpu_budget-(cpu_seconds()-before))
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
            raise ValueError('reference source drift')
    except Exception as exc:
        report.update(status='STOPPED_RESOURCE_LIMIT' if isinstance(exc, TimeoutError) else 'EXECUTION_FAILED',
            error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.setitimer(signal.ITIMER_PROF, 0.)
        wall, cpu = time.monotonic()-start, cpu_seconds()-before
        report.update(wall_seconds=wall, cpu_worker_seconds=cpu)
        manifest.update(status=report['status'], wall_seconds=wall, cpu_worker_seconds=cpu)
        save(output/'report.json', report)
        save(output/'run-manifest.json', manifest)
        save(output/'accounting.json', dict(charged_this_run=dict(gpu_seconds=0., cpu_worker_seconds=cpu)))
    print(report['status'], 'CPU', round(cpu, 3), 'wall', round(wall, 3), flush=True)
    return 0 if report['status'] == 'QUALIFIED_LOCAL_LIKELIHOOD_REFERENCE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
