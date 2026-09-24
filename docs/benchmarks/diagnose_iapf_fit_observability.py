"""Diagnostic replay of appended fit telemetry; no fitting/default changes."""
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
from unittest.mock import patch

from diagnose_iapf_adaptive_consumer import REPO, save, setup

ROOT = REPO/'docs/plans/artifacts/iapf-fit-observability-20260922-01'
INPUTS = REPO/'docs/plans/artifacts/iapf-fit-input-isolation-20260922-01'
GEOMETRY = REPO/'docs/plans/artifacts/iapf-representation-support-20260922-01'
PLAN = 'docs/plans/iapf-fit-observability-2026-09-22.md'
GPU = 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b'
CASES = ['d2-s82-cloud_moments-initial_peak', 'd2-s82-log_quadratic-initial_peak',
         'd5-s82-log_quadratic-initial_peak', 'd10-s82-log_quadratic-native']
TESTS = ['tests/highdim/test_younis_iapf_fit_observability_tf.py',
         'tests/highdim/test_younis_score_master_iapf_tf.py',
         'tests/highdim/test_younis_iapf_initialization_tf.py',
         'tests/highdim/test_younis_iapf_multivariate_consumer.py']
NEW_COLUMNS = ['target_squared_effective_count', 'target_squared_max_weight',
               'initial_log_density_energy', 'log_density_energy', 'initial_optimization_loss']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(directory, source, relative):
    path = source/relative
    if sha(path) != read(source/'manifest.json')['outputs'][relative]:
        raise RuntimeError('Changed historical input: '+str(path))
    dest = directory/'inputs'/source.name/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def difference(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError('Replay keys differ: '+str(set(a)^set(b)))
        return max([difference(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('Replay lengths differ')
        return max([difference(x, y) for x, y in zip(a, b)] or [0.])
    if isinstance(a, (bool, str)) or a is None:
        if a != b:
            raise ValueError('Replay status differs: '+str((a, b)))
        return 0.
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('Nonfinite replay value')
    return abs(a-b)


def original_diagnostics(value):
    """Remove new fields and measured wall times only, retaining all decisions."""
    if isinstance(value, list):
        return [original_diagnostics(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key in ('offline_wall_seconds', 'final_wall_seconds', 'total_wall_seconds'):
            continue
        if key == 'density_fit_diagnostics':
            out[key] = [row[:18] for row in item]
        elif key == 'fit_diagnostic_columns':
            out[key] = item[:18]
        else:
            out[key] = original_diagnostics(item)
    return out


def replay(tf, directory):
    from bayesfilter.score_study import iapf_fit_tf
    from bayesfilter.score_study.iapf_adapter import execute_iapf, FIT_DIAGNOSTIC_COLUMNS
    from bayesfilter.score_study.contracts import DiagnosticFailure
    assert FIT_DIAGNOSTIC_COLUMNS[18:] == NEW_COLUMNS
    references = read(preserve(directory, INPUTS, 'attempt02_gpu_isolate/results.json'))
    reference = {(r['case'], r['time']):r for r in references
                 if r['target'] == 'actual' and r['objective'] == 'density_l2'}
    path = preserve(directory, GEOMETRY, 'attempt01_cpu_geometry/results.csv')
    with path.open() as stream:
        geometry = {(r['case'], int(r['time'])):r for r in csv.DictReader(stream) if r['target'] == 'actual'}
    original = iapf_fit_tf.make_density_recursive_fit_kernel
    results = []
    for case in CASES:
        old = read(preserve(directory, INPUTS, 'attempt01_gpu_capture/'+case+'.json'))
        row = old['row']; calls = []; traces = []
        def factory(*args, **kwargs):
            kernel = original(*args, **kwargs)
            class Captured:
                def __call__(self, *values):
                    result = kernel(*values)
                    calls.append(dict(input={k:v.numpy().tolist() for k,v in zip(('theta','observations','clouds'), values)},
                        result={k:v.numpy().tolist() for k,v in zip(('centers','covariances','floors','valid','converged','diagnostics'), result)}))
                    traces.append(kernel.experimental_get_tracing_count())
                    return result
                def experimental_get_tracing_count(self):
                    return kernel.experimental_get_tracing_count()
            return Captured()
        d = row['d']; case_seed = row['seed']
        def seed(stream, replicate=case_seed, group='final'):
            raw = hashlib.sha256(f'{case_seed}:{d}:{d}:{group}:{replicate}:{stream}'.encode()).digest()
            return [int.from_bytes(raw[:4],'little')%(2**31-1), int.from_bytes(raw[4:8],'little')%(2**31-1)]
        status = 'complete'; error = value = score = None
        with patch.object(iapf_fit_tf, 'make_density_recursive_fit_kernel', factory):
            try:
                _, final, diag, _ = execute_iapf(dict(model='linear_gaussian', method='iapf', role='diagnostic', iapf=row['config']),
                    row['settings'], tf.constant(row['config']['fit_theta'], tf.float64),
                    tf.constant(row['observations']['fitted'], tf.float64), seed)
                value = float(final[0].numpy()); score = final[1].numpy().tolist()
            except DiagnosticFailure as exc:
                status = 'candidate_rejected'; error = str(exc); diag = exc.diagnostics['details']
        checks = {'same_status':status == old['status'], 'same_error':error == old['error'],
                  'same_call_count':len(calls) == len(old['calls']), 'one_trace':all(t == 1 for t in traces),
                  'append_only_columns':diag['fit_diagnostic_columns'][:18] == old['diagnostics']['fit_diagnostic_columns']
                       and diag['fit_diagnostic_columns'][18:] == NEW_COLUMNS}
        errors = {'diagnostics':difference(original_diagnostics(diag), original_diagnostics(old['diagnostics'])),
                  'value':difference(value, old['value']), 'score':difference(score, old['score'])}
        if len(calls) != len(old['calls']):
            raise RuntimeError('Changed adaptive fit count')
        wired = [item['density_fit_diagnostics'] for item in diag['fit_iterations'] if 'density_fit_diagnostics' in item]
        checks['actual_consumer_wiring'] = len(wired) == len(calls)
        details = []
        for index, (call, previous) in enumerate(zip(calls, old['calls'])):
            errors['input_'+str(index)] = difference(call['input'], previous['input'])
            projected = dict(call['result'], diagnostics=[v[:18] for v in call['result']['diagnostics']])
            errors['call_'+str(index)] = difference(projected, previous['result'])
            checks['wiring_'+str(index)] = difference(wired[index], call['result']['diagnostics']) == 0.
            for t, values in enumerate(call['result']['diagnostics']):
                checks[f'finite_{index}_{t}'] = len(values) == 23 and all(math.isfinite(v) for v in values)
                checks[f'concentration_range_{index}_{t}'] = 1-1e-10 <= values[18] <= len(call['input']['clouds'][t])+1e-10 and 0 < values[19] <= 1+1e-10
                # L = E S, with the optimizer's fixed amplitude scaling explicit.
                predicted_initial = math.exp(values[20]-2*values[17])*values[13]
                predicted_final = math.exp(values[21])*values[1]
                checks[f'loss_identity_{index}_{t}'] = abs(predicted_initial-values[22]) <= 1e-8*(1+abs(values[22])) and abs(predicted_final-values[0]) <= 1e-8*(1+abs(values[0]))
                if index == len(calls)-1:
                    ref = reference[(case, t)]; geom = geometry[(case, t)]
                    expected = [float(geom['pi_ESS']), float(geom['pi_max']),
                        math.log(ref['states']['initial']['energy']), math.log(ref['states']['final']['energy']),
                        ref['initial_info']['optimization_loss']]
                    delta = [abs(a-b) for a,b in zip(values[18:], expected)]
                    checks[f'independent_reference_{t}'] = max(delta) <= 1e-8
                    details.append(dict(time=t, values=dict(zip(NEW_COLUMNS, values[18:])), reference_errors=delta))
        checks['original_fields_replay'] = max(errors.values()) <= 1e-9
        payload = dict(case=case, status=status, error=error, value=value, score=score, diagnostics=diag,
                       calls=calls, traces=traces, replay_errors=errors, independent_reference=details,
                       checks=checks, passed=all(checks.values()))
        save(directory/(case+'.json'), payload)
        results.append(dict(case=case, status=status, calls=len(calls), max_replay_error=max(errors.values()),
                            reference_times=len(details), checks=checks, passed=payload['passed']))
        print(json.dumps({k:v for k,v in results[-1].items() if k != 'checks'}), flush=True)
        if not payload['passed']:
            raise RuntimeError('Observability replay failed: '+str([k for k,v in checks.items() if not v]))
    return dict(cases=results, passed=len(results) == 4 and all(r['passed'] for r in results))


def worker(args):
    directory = ROOT/args.attempt; directory.mkdir(exist_ok=False)
    device = 'cpu' if args.mode == 'cpu_tests' else 'gpu'
    tf, environment = setup(device)
    save(directory/'environment.json', environment)
    if device == 'cpu':
        import pytest
        command = TESTS+['-q', '--junitxml='+str(directory/'tests.xml')]
        code = pytest.main(command)
        result = dict(pytest_arguments=command, exit_code=int(code), passed=code == 0)
    else:
        result = replay(tf, directory)
        result['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    save(directory/'summary.json', result)
    if not result['passed']:
        raise RuntimeError('Required verification failed')


def launch(args):
    if not args.attempt.isidentifier():
        raise ValueError('Unique versioned attempt required')
    manifest = ROOT/(args.attempt+'-launch.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    resource = 'cpu' if args.mode == 'cpu_tests' else 'gpu'; seconds = 600
    budget = read(ROOT/'budget.json')
    if len(list(ROOT.glob('*-launch.json'))) >= 4 or budget['phase_'+resource+'_seconds']+seconds > 3600 or budget['remaining_'+resource+'_seconds'] < seconds:
        raise RuntimeError('Attempt or time cap')
    paths = TESTS+['docs/benchmarks/diagnose_iapf_fit_observability.py',
        'docs/benchmarks/diagnose_iapf_adaptive_consumer.py', PLAN,
        'bayesfilter/runtime/gpu_memory_policy.py', 'tests/highdim/test_younis_score_master_provider_consumers_tf.py']
    paths += [str(p.relative_to(REPO)) for p in (REPO/'bayesfilter/score_study').glob('*.py')]
    hashes = {}
    for relative in paths:
        source = REPO/relative; destination = ROOT/(args.attempt+'-source')/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        hashes[relative] = sha(source)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'-1' if resource == 'cpu' else GPU,
        'TF_FORCE_GPU_ALLOW_GROWTH':'true', 'BAYESFILTER_PRELOAD_CUSTOM_OP':'0',
        'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
        'TF_NUM_INTRAOP_THREADS':'1', 'TF_NUM_INTEROP_THREADS':'1'}
    command = [sys.executable, str(Path(__file__).resolve()), '--mode', args.mode, '--attempt', args.attempt, '--worker']
    record = dict(command=command, cwd=str(REPO), git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        python=sys.executable, sources=hashes, status='running', resource=resource, plan=PLAN,
        result=str(ROOT/'result.md'), seeds='Preserved phase7 observation/configuration and SHA256 adaptive stream namespace',
        environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','BAYESFILTER_PRELOAD_CUSTOM_OP',
            'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')})
    save(manifest, record); started = time.monotonic()
    with (ROOT/(args.attempt+'.log')).open('x') as log:
        try:
            code = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    elapsed = time.monotonic()-started
    record.update(status='finished', exit_code=code, wall_seconds=elapsed)
    save(manifest, record)
    budget['phase_'+resource+'_seconds'] += elapsed; budget['remaining_'+resource+'_seconds'] -= elapsed
    save(ROOT/'budget.json', budget)
    print(json.dumps({k:record[k] for k in ('status','exit_code','wall_seconds')}))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', required=True)
    parser.add_argument('--mode', choices=['cpu_tests','gpu_replay'], required=True)
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        sys.exit(launch(args))
