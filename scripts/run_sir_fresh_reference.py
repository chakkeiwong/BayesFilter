#!/usr/bin/env python3
"""Independent diagnostic reference for the saved SIR OT discrepancy.

This is a finite-particle Fisher-identity reference, not an exact oracle.
Production LEDH defaults and its finite-program derivative are unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01'
OUTPUT = CAMPAIGN / 'sir-fresh-reference-01'
SOURCE = CAMPAIGN / 'ot-epsilon1-steps5-01/sir_d18/result.json'
PLAN = ROOT / 'docs/plans/sir-fresh-point-reference-2026-10-14.md'
CAP = 14400
LADDER = [(131072, 8, 261014101, 1800),
          (524288, 16, 261014201, 4200),
          (1048576, 32, 261014301, 8400)]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10))
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_input(source):
    datasets = [d for d in source['datasets'] if d['label'] == '26101401']
    rows = [r for r in source['rows']
            if r['dataset'] == '26101401' and r['position'] == 2]
    if len(datasets) != 1 or len(rows) != 1:
        raise ValueError('exact saved dataset and parameter row must be unique')
    theta = source['physical_parameters'][2]
    data, row = datasets[0], rows[0]
    if len(theta) != 3 or len(data['observations']) != 50:
        raise ValueError('unexpected SIR scope')
    if any(len(y) != 9 for y in data['observations']):
        raise ValueError('expected nine observation coordinates')
    if not all(math.isfinite(x) for y in data['observations'] for x in y):
        raise ValueError('nonfinite saved observations')
    return data, row, theta


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('preserve existing versioned reference directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    import tensorflow as tf
    sys.path.insert(0, str(ROOT))
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    source = read(SOURCE)
    data, row, theta = selected_input(source)
    observations = tf.constant(data['observations'], tf.float64)
    observed_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
    if observed_hash != data['input_sha256'][1]:
        raise ValueError('saved observations do not reproduce original tensor hash')
    spec = NonlinearSQMCSpec('sir_d18')
    model_path = 'bayesfilter/highdim/ledh_canonical_models_tf.py'
    original_hashes = source['source_sha256']
    model_hashes = [v for k, v in original_hashes.items() if k.endswith(model_path)]
    if len(model_hashes) != 1 or digest(ROOT / model_path) != model_hashes[0]:
        raise ValueError('SIR model source changed since the compared evaluations')
    OUTPUT.mkdir()
    worker = OUTPUT / 'input'
    worker.mkdir()
    save(worker / 'job.json', dict(model='sir_d18', horizon=50,
         data_seed=data['data_seed'], theta_points=[theta], jit_compile=True))
    save(worker / 'dataset.json', dict(target_id=spec.target_id,
         data_seed=data['data_seed'], dtype='float64', observations=data['observations'],
         observation_sha256=observed_hash, provenance=str(SOURCE),
         source_dataset_label=data['label'], cpu_only_preparation=True))
    save(OUTPUT / 'comparison_inputs.json', dict(source=str(SOURCE), source_sha256=digest(SOURCE),
         baseline=row['baseline'], candidate=row['candidate'], theta=theta,
         observation_sha256=observed_hash, parameter_names=list(spec.parameter_names),
         target_id=spec.target_id, baseline_controls=source['baseline_controls'],
         candidate_controls=source['candidate_controls'], source_scope=source['scope']))
    paths = [Path(__file__), ROOT / model_path,
             ROOT / 'bayesfilter/highdim/sqmc_nonlinear_tf.py',
             ROOT / 'bayesfilter/highdim/models.py',
             ROOT / 'bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py',
             ROOT / 'docs/benchmarks/run_nonlinear_bootstrap_reference.py',
             ROOT / 'docs/benchmarks/run_ledh_nonlinear_master.py']
    save(OUTPUT / 'state.json', dict(status='PREPARED', plan=str(PLAN), created_utc=now(),
         git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
         worker_wall_cap_seconds=CAP, max_launches=6, used_seconds=0., runs=[]))
    print(json.dumps(dict(status='PREPARED', output=str(OUTPUT), observation_sha256=observed_hash)))


def assess(summaries, comparison):
    if not summaries:
        return dict(status='NO_REFERENCE', uncertainty_resolved=False)
    latest = summaries[-1]
    values = [latest['log_likelihood'], *latest['score']]
    errors = [latest['jackknife_mcse_log_likelihood'], *latest['jackknife_mcse_score']]
    if len(summaries) < 2:
        return dict(status='PARTICLE_LADDER_PENDING', uncertainty_resolved=False)
    previous = summaries[-2]
    shift = [abs(a-b) for a,b in zip(values, [previous['log_likelihood'], *previous['score']])]
    uncertainty = [3*s + b for s,b in zip(errors, shift)]
    before = [comparison['baseline']['value'], *comparison['baseline']['score']]
    after = [comparison['candidate']['value'], *comparison['candidate']['score']]
    # A small before/after gap in log Z can require more precision than deciding
    # whether both methods disagree materially with the independent reference.
    precision = [u < abs(a-b)/4 for u,a,b in zip(uncertainty, before, after)]
    discrepancy = {name: [abs(v-r) > .01*max(1.,abs(r))+u
                         for v,r,u in zip(arm,values,uncertainty)]
                   for name,arm in [('baseline',before),('candidate',after)]}
    return dict(status='REFERENCE_LADDER_COMPLETE', uncertainty_resolved=all(precision),
                uncertainty_resolution_by_coordinate=precision,
                latest_particle_shift=shift, diagnostic_three_mcse_plus_shift=uncertainty,
                material_discrepancy_by_coordinate=discrepancy,
                note='Particle shift is a diagnostic, not a rigorous residual-bias bound.')


def report():
    state = read(OUTPUT / 'state.json')
    comparison = read(OUTPUT / 'comparison_inputs.json')
    summaries = []
    for run in state['runs']:
        path = Path(run['output']) / 'summary.json'
        if path.exists() and run['status'] == 'COMPLETE':
            summaries.extend(read(path))
    assessment = assess(summaries, comparison)
    result = dict(comparison=comparison, summaries=summaries, assessment=assessment,
                  campaign_status=state['status'], used_seconds=state['used_seconds'])
    save(OUTPUT / 'result.json', result)
    print(json.dumps(dict(status=state['status'], completed_rungs=len(summaries),
                         used_seconds=state['used_seconds'], assessment=assessment)))
    return result


def execute():
    state = read(OUTPUT / 'state.json')
    for path, expected in state['source_hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('reference dependency changed: ' + path)
    regression = OUTPUT / 'cpu-regression.json'
    if not regression.exists() or read(regression).get('exit_code') != 0:
        raise ValueError('CPU reference regressions must pass before GPU execution')
    if any(r['status'] == 'RUNNING' for r in state['runs']):
        raise ValueError('reconcile interrupted run before restarting')
    for n, count, first_seed, rung_cap in LADDER:
        if any(r['particles'] == n and r['status'] == 'COMPLETE' for r in state['runs']):
            continue
        if len(state['runs']) >= state['max_launches']:
            state['status'] = 'ATTEMPT_LIMIT'; break
        ledger_path = CAMPAIGN / 'budget.json'
        ledger = read(ledger_path)
        if any(r.get('status') == 'RUNNING' for r in ledger['attempts']):
            raise ValueError('another campaign GPU worker is active')
        used = sum(float(r.get('wall_seconds', r.get('reserved_seconds', 0))) for r in ledger['attempts'])
        available = min(CAP-state['used_seconds'], ledger['authorized_gpu_wall_seconds']-used)
        if available < 60 or len(ledger['attempts']) >= ledger['max_attempts']:
            state['status'] = 'BUDGET_EXHAUSTED'; break
        timeout = int(min(rung_cap, available))
        destination = OUTPUT / ('rung-%d-attempt-%02d' % (n, len(state['runs'])+1))
        command = [sys.executable, str(ROOT / 'docs/benchmarks/run_nonlinear_bootstrap_reference.py'),
                   '--plan-file', str(PLAN), '--result-file', str(OUTPUT / 'result.json'),
                   '--worker-dirs', str(OUTPUT / 'input'), '--particles', str(n),
                   '--seeds', *map(str, range(first_seed, first_seed+count)),
                   '--budget-seconds', str(max(1, timeout-15)), '--output', str(destination)]
        entry = dict(model='sir_d18_independent_reference', particles=n, output=str(destination),
                     command=command, status='RUNNING', started_utc=now(), reserved_seconds=timeout)
        ledger['attempts'].append(entry.copy())
        ledger['next_action'] = 'Independent SIR oracle diagnostic executing.'
        ledger['updated_utc'] = now()
        save(ledger_path, ledger)
        state['runs'].append(entry.copy()); state['status'] = 'RUNNING'
        save(OUTPUT / 'state.json', state)
        start = time.monotonic(); code = None
        env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH='true', TF_NUM_INTRAOP_THREADS='2',
                   TF_NUM_INTEROP_THREADS='2')
        try:
            with (OUTPUT / (destination.name + '.log')).open('w') as log:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            elapsed = time.monotonic()-start
            complete = code == 0 and (destination / 'summary.json').exists()
            status = 'COMPLETE' if complete else 'FAILED'
            patch = dict(status=status, exit_code=code, wall_seconds=elapsed, ended_utc=now())
            state['runs'][-1].update(patch); state['used_seconds'] += elapsed
            state['status'] = status
            ledger = read(ledger_path)
            matches = [r for r in ledger['attempts'] if r.get('output') == str(destination)]
            if len(matches) != 1:
                raise ValueError('campaign attempt accounting mismatch')
            matches[0].update(patch)
            ledger['used_seconds'] = sum(float(r.get('wall_seconds',r.get('reserved_seconds',0))) for r in ledger['attempts'])
            ledger['remaining_seconds'] = ledger['authorized_gpu_wall_seconds']-ledger['used_seconds']
            ledger['updated_utc'] = now()
            ledger['next_action'] = 'Inspect independent SIR reference ladder and uncertainty.'
            save(ledger_path, ledger); save(OUTPUT / 'state.json', state)
        result = report()
        if not complete:
            break  # Inspect infrastructure or numerical failure before retrying.
        if result['assessment']['uncertainty_resolved']:
            state['status'] = 'PRECISION_TARGET_REACHED'; break
    else:
        state['status'] = 'PLANNED_LADDER_COMPLETE'
    save(OUTPUT / 'state.json', state)
    report()


def regression():
    if not OUTPUT.exists():
        raise ValueError('prepare first')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_NUM_INTRAOP_THREADS='2',
               TF_NUM_INTEROP_THREADS='2')
    command = [sys.executable, '-m', 'pytest', '-q',
               'tests/highdim/test_nonlinear_bootstrap_fisher_reference.py',
               'tests/test_sir_fresh_reference.py']
    start = time.monotonic()
    with (OUTPUT / 'cpu-regression.log').open('w') as log:
        result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    save(OUTPUT / 'cpu-regression.json',dict(command=command,exit_code=result.returncode,
         cpu_only=True,wall_seconds=time.monotonic()-start))
    print(json.dumps(read(OUTPUT / 'cpu-regression.json')))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','regression','execute','report'])
    globals()[parser.parse_args().action]()
