#!/usr/bin/env python3
"""CPU-only prospective diagnostic calibration; never rescore closed HMC runs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_surrogate_hmc_mcse import cpu_seconds, save, wilson

PLAN = "docs/plans/surrogate-hmc-mcse-randomization-2026-09-20.md"
GROUPS = (("iid", 0., 0, 512, 400000), ("ar_positive", .5, 0, 256, 410000),
          ("ar_negative", -.5, 0, 256, 420000), ("ar_borderline", .9, 0, 256, 430000),
          ("ar_slow", .99, 0, 256, 440000), ("chain_offset", 0., 1, 256, 450000),
          ("linear_drift", 0., 2, 256, 460000))


def protocol_definition(name):
    if name not in ('randomization-v1', 'binary-fold-v2'):
        raise ValueError('unknown frozen diagnostic protocol')
    repaired = name == 'binary-fold-v2'
    return dict(name=name, binary_repair=repaired,
        plan='docs/plans/surrogate-hmc-mcse-binary-calibration-2026-09-20.md' if repaired else PLAN,
        artifact_marker='-mcse-binary-calibration-' if repaired else '-mcse-randomization-',
        worker_budget=900. if repaired else 600.,
        data_key=20260923 if repaired else 20260921,
        permutation_key=20260924 if repaired else 20260922,
        groups=tuple((n, rho, regime, count, base+(100000 if repaired else 0))
                     for n, rho, regime, count, base in GROUPS),
        looks={n: (4000, 8000) if repaired and n == 'ar_borderline' else (1000, 2000)
               for n, *_ in GROUPS})


def nonfinite_paths(value, path=''):
    """Undefined statistics remain explicit veto evidence, never zero-filled."""
    if isinstance(value, dict):
        return [name for key, child in value.items() for name in nonfinite_paths(child, f'{path}.{key}')]
    if isinstance(value, list):
        return [name for index, child in enumerate(value) for name in nonfinite_paths(child, f'{path}[{index}]')]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def check_undefined_diagnostics(values):
    """Only the declared raw binary fold may be undefined in an admitted row."""
    paths = nonfinite_paths(values)
    permitted = {f'.{key}[{i}]' for i, flag in enumerate(values.get('folded_binary_uninformative', []))
                 if flag for key in ('function_rhat_original', 'function_rhat_folded')}
    if paths and values['original_valid']:
        raise ValueError('nonfinite diagnostic admitted by original rule')
    if values['candidate_valid'] and set(paths)-permitted:
        raise ValueError('unexpected nonfinite diagnostic admitted by candidate')
    return paths


def event_summary(events):
    count, events_count = len(events), sum(events)
    return dict(count=count, events=events_count, rate=events_count/count if count else None,
                wilson95=wilson(events_count, count) if count else [None, None])


def group_summary(rows, stationary):
    admitted = [row for row in rows if any(look['candidate_valid'] for look in row['looks'].values())]
    result = dict(completed_replications=len(rows),
        candidate_any_look_veto=event_summary([any(not x['candidate_valid'] for x in r['looks'].values()) for r in rows]),
        candidate_both_looks_veto=event_summary([all(not x['candidate_valid'] for x in r['looks'].values()) for r in rows]),
        original_any_look_veto=event_summary([any(not x['original_valid'] for x in r['looks'].values()) for r in rows]))
    if stationary:
        def miss(row):
            return any(x['candidate_valid'] and not x['all_functions_covered'] for x in row['looks'].values())
        result.update(admitted_interval_failure=event_summary([miss(r) for r in rows]),
                      conditional_interval_failure=event_summary([miss(r) for r in admitted]))
        result['mean_mcse_oracle_ratio'] = {
            look: [sum(r['looks'][look]['mean_mcse_oracle_ratio'][j] for r in rows)/len(rows) for j in range(3)]
            for look in rows[0]['looks']}
        result['naive_mcse_oracle_ratio'] = {
            look: [sum(r['looks'][look]['naive_mcse_oracle_ratio'][j] for r in rows)/len(rows) for j in range(3)]
            for look in rows[0]['looks']}
    return result


def decision(groups):
    checks = {"iid_false_veto": groups['iid']['candidate_any_look_veto']['wilson95'][1] <= .05}
    for name in ('iid', 'ar_positive', 'ar_negative', 'ar_borderline'):
        row = groups[name]
        conditional = row['conditional_interval_failure']
        checks[name+'_coverage'] = (row['admitted_interval_failure']['wilson95'][1] <= .05
            and conditional['count'] >= 64 and conditional['wilson95'][1] <= .05)
    for name in ('ar_slow', 'chain_offset', 'linear_drift'):
        checks[name+'_veto'] = groups[name]['candidate_both_looks_veto']['wilson95'][0] >= .95
    slow = groups['ar_slow']['conditional_interval_failure']
    if slow['count'] >= 64:
        checks['ar_slow_admitted_coverage'] = (slow['wilson95'][1] <= .05
            and groups['ar_slow']['admitted_interval_failure']['wilson95'][1] <= .05)
    return dict(status='PASSED_GAUSSIAN_CALIBRATION_SCREENS' if all(checks.values()) else 'CANDIDATE_NOT_CALIBRATED',
                checks=checks, default_promoted=False, closed_results_reinterpreted=False,
                general_mcmc_validity=False)


def run(output, report, remaining_wall, remaining_cpu, protocol):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.surrogate_hmc_equivalence import _diagnostics, _batch_statistics, finite_report
    from bayesfilter.testing.surrogate_hmc_gaussian_control import PRECISION, functions_for_chunk, reference_contract
    from bayesfilter.testing.surrogate_hmc_mcse_randomization import (
        batch_randomization, binary_function_rhat, diagnostic_batch_statistics)
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        rank_normalized_split_rhat, rank_normalized_bulk_tail_ess)

    reference = reference_contract()
    truth = tf.constant(reference['expectations'], tf.float64)
    critical = tfp.distributions.StudentT(tf.constant(36., tf.float64), tf.constant(0., tf.float64),
        tf.constant(1., tf.float64)).quantile(tf.constant(1-.05/(2*15*2), tf.float64))

    def make_generator(draws):
        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.float64),
                                     tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
        def generate(seed, rho, regime):
            innovations = tf.random.stateless_normal([draws, 4, 3], seed, dtype=tf.float64)
            values = tf.cond(rho == 0., lambda: innovations,
                lambda: tf.concat([innovations[:1], tf.scan(
                    lambda prior, noise: rho*prior + tf.sqrt(1-rho*rho)*noise,
                    innovations[1:], initializer=innovations[0])], axis=0))
            coordinate = tf.constant([1., 0., 0.], tf.float64)[None, None, :]
            values += tf.cast(regime == 1, tf.float64)*tf.constant([-.6, -.2, .2, .6], tf.float64)[None, :, None]*coordinate
            values += tf.cast(regime == 2, tf.float64)*tf.linspace(tf.constant(-.75, tf.float64),
                        tf.constant(.75, tf.float64), draws)[:, None, None]*coordinate
            functions = tf.concat([functions_for_chunk(chunk/tf.sqrt(tf.constant(PRECISION, tf.float64)))[1]
                                   for chunk in tf.split(values, draws//250, axis=0)], axis=0)
            return tf.transpose(values, [1, 0, 2]), tf.transpose(functions, [1, 0, 2])
        return generate

    generators = {n: make_generator(n) for n in {looks[-1] for looks in protocol['looks'].values()}}

    calculators = {}
    for draws in sorted({n for looks in protocol['looks'].values() for n in looks}):
        def build(count):
            # Rank ties require the established non-XLA diagnostic graph. Hot
            # generation, batch and permutation kernels remain compiled.
            @tf.function(input_signature=[tf.TensorSpec([4, count, 3], tf.float64),
                                          tf.TensorSpec([4, count, 15], tf.float64),
                                          tf.TensorSpec([2], tf.int32)], jit_compile=False, autograph=False)
            def calculate(parameters, functions, seed):
                if count in (1000, 2000):
                    rhat, ess, function_rhat = _diagnostics(count, 3, 15)(parameters, functions)
                    mcse, lag = _batch_statistics(count, 15)(functions)
                else:
                    rhat, ess = rank_normalized_split_rhat(parameters), rank_normalized_bulk_tail_ess(parameters)
                    function_rhat = rank_normalized_split_rhat(functions)
                    mcse, lag = diagnostic_batch_statistics(count)(functions)
                effective_rhat, binary_fold = (binary_function_rhat(functions, function_rhat)
                    if protocol['binary_repair'] else (function_rhat['maximum'], tf.zeros([15], tf.bool)))
                randomized = batch_randomization(count)(functions, seed)
                conservative = tf.reduce_max(mcse, axis=0)
                sensitivity = conservative/tf.reduce_min(mcse, axis=0)
                finite = tf.reduce_all(tf.math.is_finite(parameters)) & tf.reduce_all(tf.math.is_finite(functions))
                mixing = (finite & tf.reduce_all(tf.math.is_finite(rhat['maximum']) & (rhat['maximum'] <= 1.01))
                          & tf.reduce_all(tf.math.is_finite(ess['bulk']) & (ess['bulk'] >= 400.))
                          & tf.reduce_all(tf.math.is_finite(ess['tail']) & (ess['tail'] >= 400.)))
                variance_valid = tf.reduce_all(tf.math.is_finite(sensitivity) & (sensitivity <= 2.)
                    & tf.math.is_finite(conservative) & (conservative > 0.))
                original = (mixing & variance_valid & tf.reduce_all(tf.math.is_finite(function_rhat['maximum'])
                    & (function_rhat['maximum'] <= 1.01))
                    & tf.reduce_all(tf.math.is_finite(lag[-1]) & (tf.abs(lag[-1]) <= .3)))
                candidate = (mixing & variance_valid & tf.reduce_all(tf.math.is_finite(effective_rhat)
                    & (effective_rhat <= 1.01)) & randomized['valid'] & (randomized['p_value'] > .01))
                covered = tf.reduce_all(tf.abs(tf.reduce_mean(functions, axis=[0, 1])-truth) <= critical*conservative)
                naive = tf.sqrt(tf.math.reduce_variance(functions[:, :, :3], axis=[0, 1])/(4*count-1))
                return dict(original_valid=original, candidate_valid=candidate, all_functions_covered=covered,
                    parameter_rhat=rhat['maximum'], bulk_ess=ess['bulk'], tail_ess=ess['tail'],
                    function_rhat=effective_rhat, function_rhat_original=function_rhat['maximum'],
                    function_rhat_bulk=function_rhat['bulk'], function_rhat_folded=function_rhat['folded'],
                    folded_binary_uninformative=binary_fold, mcse=conservative, naive_mean_mcse=naive,
                    batch_lag=lag, sensitivity=sensitivity, randomization=randomized,
                    all_inputs_finite=finite)
            return calculate
        calculators[draws] = build(draws)

    report.update(reference=reference, critical_value=float(critical), groups={},
                  tensorflow=tf.__version__, planned_replications=sum(g[3] for g in protocol['groups']))
    completed, costs = 0, []
    normalized_costs = []
    for group_index, (name, rho, regime, count, seed_base) in enumerate(protocol['groups']):
        rows = []
        looks = protocol['looks'][name]
        # Sorting grows slightly faster than n; conservatively weight the long
        # scope by n*log2(n) for the initial forecast, then measure its own cost.
        work_scale = looks[-1]*math.log2(looks[-1])/(2000*math.log2(2000))
        generate = generators[looks[-1]]
        with (output/(name+'-replications.jsonl')).open('x') as stream:
            for replication in range(count):
                start, before = time.monotonic(), cpu_seconds()
                parameters, functions = generate(tf.constant([protocol['data_key'], seed_base+replication], tf.int32),
                    tf.constant(rho, tf.float64), tf.constant(regime, tf.int32))
                row = dict(replication=replication, seed=[protocol['data_key'], seed_base+replication],
                    tensor_sha256=hashlib.sha256(tf.io.serialize_tensor(functions).numpy()).hexdigest(), looks={})
                for look_index, draws in enumerate(looks):
                    permutation_seed = [protocol['permutation_key'], seed_base+2*replication+look_index]
                    result = calculators[draws](parameters[:, :draws], functions[:, :draws], tf.constant(permutation_seed, tf.int32))
                    values = tf.nest.map_structure(lambda value: value.numpy().tolist(), result)
                    if not values['all_inputs_finite']:
                        raise ValueError('nonfinite generated data')
                    values['undefined_diagnostics'] = check_undefined_diagnostics(values)
                    values['permutation_seed'] = permutation_seed
                    if regime == 0:
                        oracle = math.sqrt((draws+2*sum((draws-k)*rho**k for k in range(1, draws)))/(4*draws**2))
                        values['mean_mcse_oracle_ratio'] = [x/oracle for x in values['mcse'][:3]]
                        values['naive_mcse_oracle_ratio'] = [x/oracle for x in values['naive_mean_mcse']]
                    row['looks'][str(draws)] = values
                stream.write(json.dumps(finite_report(row), allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                completed += 1
                costs.append((time.monotonic()-start, cpu_seconds()-before))
                normalized_costs.append(tuple(cost/work_scale for cost in costs[-1]))
                if replication == 7:
                    remaining_work = (count-8)*work_scale + sum(
                        c*protocol['looks'][n][-1]*math.log2(protocol['looks'][n][-1])/(2000*math.log2(2000))
                        for n, _, _, c, _ in protocol['groups'][group_index+1:])
                    forecast = {key: 1.5*max(sum(cost[i] for cost in normalized_costs[-4:])/4,
                                sum(cost[i] for cost in normalized_costs[4:8])/4)*remaining_work
                                for i, key in enumerate(('wall_seconds', 'cpu_worker_seconds'))}
                    report.setdefault('cost_forecasts', {})[name] = forecast
                    save(output/'report.json', report)
                    if forecast['wall_seconds'] > remaining_wall() or forecast['cpu_worker_seconds'] > remaining_cpu():
                        report.update(status='UNDER_BUDGETED', completed_replications=completed)
                        return
                if completed % 64 == 0:
                    report.update(completed_replications=completed)
                    save(output/'report.json', report)
                    print(name, replication+1, 'of', count, 'CPU remaining', round(remaining_cpu(), 1), flush=True)
        report['groups'][name] = group_summary(rows, regime == 0)
        save(output/'report.json', report)
    report.update(decision(report['groups']), completed_replications=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=('randomization-v1', 'binary-fold-v2'), default='randomization-v1')
    args = parser.parse_args()
    protocol = protocol_definition(args.protocol)
    if protocol['artifact_marker'] not in args.output.name or '-tests-' in args.output.name:
        raise ValueError('output must use the protocol accounting marker and cannot be a test root')
    from bayesfilter.inference.surrogate_hmc_program import validated_budget, charge_previous_runs, GPU_CAP, CPU_CAP
    ledger = validated_budget(GPU_CAP, CPU_CAP)
    accounting_paths = list((ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'))
    charge_previous_runs(ledger, accounting_paths)
    prior_paths = [path for path in accounting_paths if protocol['artifact_marker'] in path.parent.name
                   and '-tests-' not in path.parent.name]
    spent_cpu, spent_wall = 0., 0.
    for path in prior_paths:
        payload = json.loads(path.read_text())
        spent_cpu += payload['charged_this_run']['cpu_worker_seconds']
        spent_wall += json.loads((path.parent/'run-manifest.json').read_text())['wall_seconds']
    cpu_budget = min(protocol['worker_budget']-spent_cpu, CPU_CAP-ledger['charged']['cpu_worker_seconds'])
    wall_budget = protocol['worker_budget']-spent_wall
    if protocol['binary_repair']:
        repair_spent = sum(json.loads(path.read_text())['charged_this_run']['cpu_worker_seconds']
                           for path in accounting_paths if '-mcse-' in path.parent.name)
        cpu_budget = min(cpu_budget, 1800.-repair_spent)
    if len(prior_paths) >= 2 or min(cpu_budget, wall_budget) <= 0:
        raise ValueError('MCSE diagnostic cumulative attempt/budget limit')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [protocol['plan'], 'scripts/calibrate_surrogate_hmc_mcse_randomization.py',
        'scripts/diagnose_surrogate_hmc_mcse.py', 'bayesfilter/testing/surrogate_hmc_mcse_randomization.py',
        'bayesfilter/inference/surrogate_hmc_equivalence.py', 'bayesfilter/inference/hmc_posterior_diagnostics.py',
        'bayesfilter/testing/surrogate_hmc_gaussian_control.py', 'tests/test_surrogate_hmc_mcse_randomization.py']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        target = args.output/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    start, before = time.monotonic(), cpu_seconds()
    report = dict(status='RUNNING', plan=protocol['plan'], protocol=protocol, cpu_only=True, default_promoted=False,
                  closed_results_reinterpreted=False, source_hashes=hashes)
    manifest = dict(command=[sys.executable, *sys.argv], cwd=str(ROOT), python=sys.executable,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, plan=protocol['plan'], result=str(args.output/'report.json'),
        cpu_only=True, gpu_intentionally_hidden=True, cpu_budget=cpu_budget, wall_budget=wall_budget,
        seeds=dict(data_date_key=protocol['data_key'], permutation_date_key=protocol['permutation_key'],
                   groups=protocol['groups']), protocol=protocol,
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')})
    save(args.output/'run-manifest.json', manifest)
    save(args.output/'report.json', report)
    def timeout(signum, frame):
        raise TimeoutError('prospective MCSE diagnostic resource cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_REAL, wall_budget)
    signal.setitimer(signal.ITIMER_PROF, cpu_budget)
    try:
        run(args.output, report, lambda: wall_budget-(time.monotonic()-start),
            lambda: cpu_budget-(cpu_seconds()-before), protocol)
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError('executed source drift')
    except Exception as exc:
        report.update(status='STOPPED_RESOURCE_LIMIT' if isinstance(exc, TimeoutError) else 'EXECUTION_FAILED',
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.setitimer(signal.ITIMER_PROF, 0.)
        wall, cpu = time.monotonic()-start, cpu_seconds()-before
        report.update(wall_seconds=wall, cpu_worker_seconds=cpu)
        manifest.update(wall_seconds=wall, cpu_worker_seconds=cpu, status=report['status'])
        save(args.output/'report.json', report)
        save(args.output/'run-manifest.json', manifest)
        save(args.output/'accounting.json', dict(charged_this_run=dict(gpu_seconds=0., cpu_worker_seconds=cpu)))
    print(report['status'], 'CPU', round(cpu, 3), 'wall', round(wall, 3), flush=True)
    return 0 if report['status'] == 'PASSED_GAUSSIAN_CALIBRATION_SCREENS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
