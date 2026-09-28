"""Bounded correction using all seven KSC observation components; diagnostic."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'docs/plans/sqmc-ksc-full-mixture-correction-20260929.md'
PRIOR = ROOT/'docs/plans/artifacts/sqmc-ksc-sv-20260928'
PRIOR_HASH = '2a7c3de939d0bd0ae6cb8cca9db6cc57ca9b1afe87a4d6b41d0d6041a2c2ef15'
SEEDS = tuple(range(213001, 213009))
ROUTES = ('iid_dual_cap', 'previous_inverse_cdf', 'repaired_permutation', 'repaired_permutation_ablation')
LADDER = ((401, 40.), (801, 40.), (1201, 40.), (1201, 48.))
PROGRAM = 'fp64_gpu_xla_ksc_seven_component_gaussian_sum_quadrature_reference'


class NumericalVeto(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_hashes():
    paths = [PLAN, 'bayesfilter/highdim/sqmc_ksc_gaussian_sum_reference_tf.py',
             'bayesfilter/highdim/sqmc_ksc_tf.py', 'bayesfilter/runtime/gpu_memory_policy.py',
             'docs/benchmarks/run_sqmc_expanded_comparison.py',
             'docs/benchmarks/run_sqmc_ksc_full_mixture.py',
             'tests/highdim/test_sqmc_ksc_gaussian_sum.py']
    return {p: digest(ROOT/p) for p in paths}


def require_window():
    ledger = read(PRIOR/'budget.json')
    if digest(PRIOR/'budget.json') != PRIOR_HASH:
        raise RuntimeError('Prior ledger changed; reconcile before continuing')
    if time.time() >= datetime.fromisoformat(ledger['deadline_utc']).timestamp():
        raise RuntimeError('Authorized elapsed window expired')
    return ledger


def saved_inputs():
    ledger = require_window()
    archived = read(PRIOR/'final-evidence-01/input-sha256.json')
    inputs = {}
    for path, expected in archived.items():
        if digest(path) != expected:
            raise RuntimeError('Archived particle evidence changed: '+path)
        inputs[path] = expected
    data, particles, scopes = {}, [], []
    for attempt in ledger['attempts']:
        if attempt['unit'] == 'check':
            continue
        if attempt['returncode'] != 0:
            raise RuntimeError('Unresolved original scope')
        directory = Path(attempt['output'])
        result, manifest = read(directory/'result.json'), read(directory/'manifest.json')
        saved, tuning = read(directory/'data.json'), read(directory/'tuning.json')['tuning_artifact']
        route, horizon = result['route'], result['horizon']
        if result['program'] != 'fp64_gpu_xla_ksc_seven_mixture' or result['tuning_status'] != 'exact_scope_frozen_controls':
            raise RuntimeError('Particle configuration is not the preserved tuned variant')
        scope = tuning['scope']
        if scope['theta'] != [1.5, 0.] or scope['route'] != route or scope['horizon'] != horizon:
            raise RuntimeError('Particle tuning scope changed')
        if scope['particle_count'] != 1008 or scope['parameter_count'] != 2:
            raise RuntimeError('Particle configuration changed')
        if digest(directory/'data.json') != manifest['data_sha256']:
            raise RuntimeError('Data hash mismatch')
        scopes.append(dict(horizon=horizon, route=route, program=result['program'],
                           tuning=result['tuning_path'], source_snapshot=attempt['source_snapshot'], path=str(directory)))
        for seed in SEEDS:
            datum = saved[str(seed)]
            key = (horizon, seed)
            if key in data and data[key] != datum:
                raise RuntimeError('Routes used different observations/references')
            if len(datum['observations']) != horizon:
                raise RuntimeError('Horizon/observation mismatch')
            data[key] = datum
        for row in result['rows']:
            if not row['valid'] or row['data_seed'] not in SEEDS:
                raise RuntimeError('Original candidate row invalid or wrong seed')
            particles.append(dict(row, horizon=horizon, program=result['program'], tuning=result['tuning_path']))
    if len(data) != 32 or len(particles) != 128 or len(scopes) != 16:
        raise RuntimeError('Incomplete saved campaign')
    keys = {(r['horizon'], r['route'], r['data_seed']) for r in particles}
    if len(keys) != 128:
        raise RuntimeError('Duplicate saved candidate cell')
    return data, particles, scopes, inputs


def materialize(result):
    record = {k: v.numpy().tolist() for k, v in result.items()}
    record['device'] = result['value'].device
    numbers = [record['value'], *record['score'], record['posterior_mean'],
               record['posterior_variance'], record['maximum_projection_mass_error']]
    if not record['valid'] or not all(math.isfinite(x) for x in numbers):
        raise NumericalVeto('Gaussian-sum calculation failed validity')
    if record['observation_components'] != 7:
        raise NumericalVeto('Observation mixture was not retained')
    return record


def errors(a, b):
    return abs(a['value']-b['value']), max(abs(x-y) for x, y in zip(a['score'], b['score']))


def worker(out, mode):
    out.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    start = time.monotonic()
    meta = dict(command=[sys.executable, *sys.argv], git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_sha256=before,
        plan=PLAN, program=PROGRAM, tuning='N/A deterministic resolution verification',
        started_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        production_deviations='FP64/TF32 off independent reference, not production/default/HMC')
    result = dict(status='running', program=PROGRAM, mode=mode, rows=[])
    tf = None
    dump(out/'manifest.json', meta)
    try:
        require_window()
        meta['pre_framework_nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
        import run_sqmc_expanded_comparison as base
        tf, gpu = base.configure_gpu()
        gpu['cpu_gpu_status'] = 'GPU full-seven-component Gaussian-sum quadrature reference; CPU exact diagnostic'
        meta.update(gpu)
        from bayesfilter.highdim.sqmc_ksc_gaussian_sum_reference_tf import gaussian_sum_reference
        from bayesfilter.highdim.sqmc_ksc_tf import enumeration_reference
        if mode == 'check':
            theta = tf.constant([.7, -.2], tf.float64)
            obs = tf.constant([[-4.], [1.], [-.6], [2.]], tf.float64)
            with tf.device('/GPU:0'):
                graph = materialize(gaussian_sum_reference(401, 40., False)(theta, obs))
                begin = time.monotonic()
                compiled = materialize(gaussian_sum_reference(401, 40., True)(theta, obs))
                first_wall = time.monotonic()-begin
                begin = time.monotonic()
                repeated = materialize(gaussian_sum_reference(401, 40., True)(theta, obs))
                repeat_wall = time.monotonic()-begin
                fd = []
                for i in (0, 1):
                    step = tf.one_hot(i, 2, dtype=tf.float64)*1e-5
                    plus = materialize(gaussian_sum_reference(401, 40., True)(theta+step, obs))
                    minus = materialize(gaussian_sum_reference(401, 40., True)(theta-step, obs))
                    fd.append((plus['value']-minus['value'])/2e-5)
                short = materialize(gaussian_sum_reference(401, 40., True)(theta, obs[:2]))
            with tf.device('/CPU:0'):
                ev, es = enumeration_reference(2)(theta, obs[:2])
                exact = dict(value=float(ev), score=es.numpy().tolist())
            parity = errors(graph, compiled)
            fd_error = max(abs(x-y) for x, y in zip(fd, compiled['score']))
            enumeration_error = errors(short, exact)
            result.update(graph=graph, compiled=compiled, repeated=repeated,
                          parity_errors=parity, finite_difference=fd, finite_difference_error=fd_error,
                          exact_T2=exact, gaussian_sum_T2=short, exact_errors=enumeration_error,
                          first_call_seconds=first_wall, repeat_seconds=repeat_wall)
            if max(parity) > 1e-8 or fd_error > 2e-7 or max(enumeration_error) > 1e-8:
                raise NumericalVeto('GPU parity/derivative/exact-mixture check failed')
            if 'GPU' not in compiled['device']:
                raise NumericalVeto('Reference did not execute on GPU')
        else:
            checkpoint = read(out.parent/'check-attempt-01/result.json')
            check_manifest = read(out.parent/'check-attempt-01/manifest.json')
            if checkpoint['status'] != 'pass' or check_manifest['source_sha256'] != before:
                raise RuntimeError('Matching-source GPU checks required')
            data, particles, scopes, inputs = saved_inputs()
            dump(out/'input-sha256.json', inputs)
            dump(out/'particle-inputs.json', particles)
            dump(out/'particle-scopes.json', scopes)
            dump(out/'observations.json', [dict(horizon=h, data_seed=s, **d) for (h,s), d in sorted(data.items())])
            theta = tf.constant([1.5, 0.], tf.float64)
            for (horizon, seed), datum in sorted(data.items()):
                require_window()
                observations = tf.constant(datum['observations'], tf.float64)
                resolutions = []
                for nodes, bound in LADDER:
                    begin = time.monotonic()
                    with tf.device('/GPU:0'):
                        record = materialize(gaussian_sum_reference(nodes, bound, True)(theta, observations))
                    record.update(nodes=nodes, bound=bound, wall_seconds=time.monotonic()-begin)
                    if 'GPU' not in record['device'] or record['maximum_gaussian_branches'] != 7*nodes:
                        raise NumericalVeto('Wrong reference lane or component count')
                    resolutions.append(record)
                final = resolutions[-1]
                refinement = [errors(final, r) for r in resolutions[-3:-1]]
                independent = errors(final, datum['reference'])
                delta_v = max(r[0] for r in refinement)
                delta_s = max(r[1] for r in refinement)
                valid = delta_v <= 1e-8 and delta_s <= 1e-7 and independent[0] <= 1e-8 and independent[1] <= 1e-7
                row = dict(horizon=horizon, data_seed=seed, theta=[1.5, 0.], resolutions=resolutions,
                           accepted=valid, reference=final, refinement_value_error=delta_v,
                           refinement_score_error=delta_s, independent_value_error=independent[0],
                           independent_score_error=independent[1])
                result['rows'].append(row)
                dump(out/'result.json', result)
                print(json.dumps({k: v for k, v in row.items() if k not in ('resolutions', 'reference')}), flush=True)
                if not valid:
                    raise NumericalVeto('Full-mixture reference convergence/agreement veto')
            meta['data_version'] = 'unchanged archived KSC final datasets; input-sha256.json'
        if source_hashes() != before:
            raise RuntimeError('Numerical source changed during worker')
        meta['source_unchanged'] = True
        result['status'] = 'pass'
    except NumericalVeto as error:
        result.update(status='numerical_reference_veto', error=str(error))
        traceback.print_exc()
    except BaseException as error:
        result.update(status='infrastructure_or_harness_failure', error=repr(error))
        traceback.print_exc()
    finally:
        meta.update(status=result['status'], wall_seconds=time.monotonic()-start,
                    finished_utc=datetime.now(timezone.utc).isoformat(), result=str(out/'result.json'))
        if tf is not None:
            meta['gpu_allocator_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
        dump(out/'result.json', result)
        dump(out/'manifest.json', meta)
    return 0 if result['status'] == 'pass' else (42 if result['status'] == 'numerical_reference_veto' else 1)


def supervise(out):
    old = require_window()
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out.parent/'budget.json'
    if ledger_path.exists():
        ledger = read(ledger_path)
        if ledger['prior_sha256'] != PRIOR_HASH or any(a['status']=='running' for a in ledger['attempts']):
            raise RuntimeError('Unreconciled previous correction attempt')
    else:
        ledger = dict(schema='ksc_full_mixture_correction_budget_v1', prior_ledger=str(PRIOR/'budget.json'),
                      prior_sha256=PRIOR_HASH, prior_charged_seconds=old['charged_seconds'],
                      correction_allocation_seconds=3600., correction_seconds=0., cap_seconds=43200.,
                      charged_seconds=old['charged_seconds'], remaining_gpu_seconds=old['remaining_gpu_seconds'],
                      deadline_utc=old['deadline_utc'], attempts=[], status='running')
    before = source_hashes()
    snapshot = out/'source'
    for path in before:
        target = snapshot/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, target)
    dump(out/'source-sha256.json', before)
    dump(ledger_path, ledger)
    for mode in ('check', 'compare'):
        failures = [a for a in ledger['attempts'] if a['unit']==mode and a.get('returncode') not in (None, 0)]
        if len(failures) > 2:
            raise RuntimeError('Original two-infrastructure-retry cap exhausted')
        available = min(ledger['remaining_gpu_seconds'], 3600.-ledger['correction_seconds'],
                        datetime.fromisoformat(ledger['deadline_utc']).timestamp()-time.time()-15.)
        if available <= 10.:
            raise RuntimeError('Insufficient remaining correction budget')
        target = out/f'{mode}-attempt-01'
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(target), '--mode', mode]
        entry = dict(unit=mode, command=command, status='running', output=str(target),
                     source_snapshot=str(snapshot), started_utc=datetime.now(timezone.utc).isoformat(),
                     timeout_seconds=available)
        ledger['attempts'].append(entry)
        ledger['status']='running'
        dump(ledger_path, ledger)
        start=time.monotonic()
        with (out/f'{mode}-attempt-01.log').open('w') as log:
            proc=subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid']=proc.pid
            dump(ledger_path, ledger)
            try:
                rc=proc.wait(timeout=available)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                rc=124
        wall=time.monotonic()-start
        entry.update(status='finished', returncode=rc, wall_seconds=wall,
                     result_status=read(target/'result.json')['status'] if (target/'result.json').exists() else 'missing_result')
        ledger['correction_seconds']+=wall
        ledger['charged_seconds']=ledger['prior_charged_seconds']+ledger['correction_seconds']
        ledger['remaining_gpu_seconds']=43200.-ledger['charged_seconds']
        dump(ledger_path, ledger)
        print(json.dumps(entry), flush=True)
        if rc:
            ledger['status']='numerical_reference_veto' if rc==42 else 'infrastructure_or_timeout_failure'
            dump(ledger_path, ledger)
            return rc
    ledger['status']='complete'
    dump(ledger_path, ledger)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('supervise','check','compare'), required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    return supervise(out) if args.mode=='supervise' else worker(out,args.mode)


if __name__=='__main__':
    raise SystemExit(main())
