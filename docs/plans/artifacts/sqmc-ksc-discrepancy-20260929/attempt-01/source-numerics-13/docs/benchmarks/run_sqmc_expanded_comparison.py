"""Bounded FP64 GPU/XLA diagnostic against Kalman; no method promotion.

The supervisor uses only the standard library. Each scope/route worker
configures GPU memory growth before importing algorithmic modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'docs/plans/sqmc-expanded-comparison-20260926.md'
RESULT_NOTE = 'docs/benchmarks/sqmc-expanded-results-20260926.md'
RENEWAL = 'docs/plans/sqmc-expanded-renewal-20260928.md'
GPU_UUID = 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b'
ROUTES = ('iid_dual_cap', 'previous_inverse_cdf', 'repaired_permutation',
          'repaired_permutation_ablation')
CAL = (195001, 195002)
VAL = (196001,)
CLAIM = (197001, 197002)
FILTER = (198001, 198002)
SCOPES = (('p44_d3_T10', 'p44', 3, 10, 1008),
          ('p44_d3_T120', 'p44', 3, 120, 1008),
          ('full_d3_T2', 'full_matrix', 3, 2, 1020),
          ('full_d3_T10', 'full_matrix', 3, 10, 1020),
          ('full_d3_T120', 'full_matrix', 3, 120, 1020),
          ('full_d10_T2', 'full_matrix', 10, 2, 1020),
          ('full_d10_T10', 'full_matrix', 10, 10, 1020),
          ('full_d10_T120', 'full_matrix', 10, 120, 1020))
CONTROLS = [dict(flow_substeps=steps, reset_epsilon=.4, reset_sinkhorn_steps=24,
                 reset_balance_steps=12, correction_steps=1, correction_strength=.12,
                 pairwise_steps=1, pairwise_strength=.03) for steps in (2, 8)]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


ENTRY_POINT = Path(__file__).resolve()
EXTRA_SOURCE_PATHS = ()
EXTRA_CONTRACT = {}
CONTROL_PREPARER = None


def source_hashes():
    paths = (list((ROOT/'bayesfilter').rglob('*.py')) +
             list((ROOT/'experiments').rglob('*.py')) + [Path(__file__), ROOT/PLAN, ROOT/RENEWAL, ROOT/'docs/benchmarks/check_sqmc_expanded_gpu.py', ROOT/'docs/benchmarks/control_sqmc_expanded.py'] + list(EXTRA_SOURCE_PATHS))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def comparison_contract():
    return json.loads(json.dumps(dict(scopes=SCOPES, routes=ROUTES, controls=CONTROLS,
        calibration=CAL, validation=VAL, final_data=CLAIM, final_filter=FILTER,
        program='fp64_gpu_xla_sqmc_reference_comparison', tf32=False, **EXTRA_CONTRACT)))


def numerical_sources(hashes):
    return {k:v for k,v in hashes.items() if k.startswith(('bayesfilter/', 'experiments/'))}


def final_pairs():
    return tuple(itertools.product(CLAIM, FILTER))


def l2_error(estimate, reference):
    if len(estimate) != len(reference):
        raise ValueError('score dimensions differ')
    return math.sqrt(math.fsum((a-b)**2 for a, b in zip(estimate, reference)))


def attach_heuristics(row, oracle, first):
    # Both baselines are estimators of the full-horizon score.
    row['heuristic_zero_score_l2_error'] = l2_error([0.]*len(oracle), oracle)
    row['heuristic_first_observation_only_l2_error'] = l2_error(first, oracle)
    if row.get('valid'):
        loss = row['score_l2_error']
        row['heuristic_dominance'] = {
            'zero_score': 'observed_loss' if loss > row['heuristic_zero_score_l2_error'] else 'no_observed_loss',
            'first_observation_only': 'observed_loss' if loss > row['heuristic_first_observation_only_l2_error'] else 'no_observed_loss',
            'evidence_class': 'descriptive_only; observed loss vetoes promotion, no superiority inference'}
    else:
        row['heuristic_dominance'] = {'verdict': 'not_evaluable_invalid_candidate'}
    return row


def manifest():
    return dict(schema='bayesfilter.sqmc_expanded_manifest.v2', started_utc=now(),
                command=[sys.executable, *sys.argv], cwd=str(ROOT), plan=PLAN,
                result_note=RESULT_NOTE, python=sys.executable,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                working_tree_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
                source_sha256=source_hashes(), status='running',
                comparison_contract=comparison_contract(), renewal=RENEWAL,
                seed_partitions=dict(calibration=CAL, validation=VAL, final_data=CLAIM, final_filter=FILTER),
                precision_exception='FP64 comparison; not FP32/TF32 production evidence',
                oracle_exception='independent CPU/non-XLA Kalman reference',
                final_partition_note='Two pilot diagonal pairs were already observed; no controls were adjusted using pilot final errors. Repetition is not independent evidence.')


def configure_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    growth = configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('trusted GPU worker has no visible GPU')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    with tf.device('/GPU:0'):
        probe = tf.reduce_sum(tf.ones([2, 2], tf.float64))
    if 'GPU' not in probe.device:
        raise RuntimeError('framework probe did not execute on GPU')
    gpu_meta = dict(tensorflow=tf.__version__, gpu_memory_policy=growth, gpu_uuid=GPU_UUID,
                cpu_gpu_status='GPU particle kernel; CPU Kalman oracle', tf32=False, jit_compile=True,
                framework_gpu_probe=dict(device=probe.device, value=float(probe.numpy())),
                gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip())
    return tf, gpu_meta


def worker(out, scope_name, route):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    meta = manifest()
    dump(out/'manifest.json', meta)
    result = dict(scope=scope_name, route=route, status='running', rows=[])
    dump(out/'result.json', result)
    tf = None
    try:
        tf, gpu_meta = configure_gpu()
        meta.update(gpu_meta)
        dump(out/'manifest.json', meta)
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
        from bayesfilter.highdim.sqmc_campaign_tf import random_inputs, route_settings
        from bayesfilter.highdim.sqmc_campaign_tuning import tune_campaign, evaluate_untouched
        _, family, d, horizon, n = next(s for s in SCOPES if s[0] == scope_name)
        spec = FullLGSSMSpec(family, d) if family == 'full_matrix' else LGSSMSpec(family, d)
        theta = spec.default_theta(tf.float64)
        names = (spec.parameter_names if family == 'full_matrix' else
                 ['persistence_coordinate', 'log_Q_scale', 'log_R_scale', 'initial_mean_scale'])
        result.update(target_id=spec.target_id, dimension=d, horizon=horizon, particle_count=n,
                      parameter_count=spec.parameter_count, parameter_names=names, theta=theta.numpy().tolist(),
                      program='fp64_gpu_xla_sqmc_reference_comparison',
                      production_deviations=dict(dtype='float64', tf32=False, **route_settings(route)),
                      tuning_status='pending_scope_specific_flow_grid_calibration',
                      nonclaims='No FP32/TF32 production, HMC, default or statistical-superiority admission')
        params = spec.kalman_parameters(theta)
        eigenvalues = tf.linalg.eigvalsh(params['process_covariance'])
        result['model_diagnostics'] = dict(
            transition_max_abs_row_sum=float(tf.reduce_max(tf.reduce_sum(tf.abs(params['transition_matrix']), axis=1)).numpy()),
            process_covariance_min_eigenvalue=float(eigenvalues[0].numpy()),
            process_covariance_max_eigenvalue=float(eigenvalues[-1].numpy()),
            process_covariance_condition_number=float((eigenvalues[-1]/eigenvalues[0]).numpy()))
        data = {}
        for seed in (*CAL, *VAL, *CLAIM):
            observations = spec.simulate(theta, horizon, seed, jit_compile=True)
            value, score = spec.reference_value_and_score(theta, observations)
            first = spec.reference_value_and_score(theta, observations[:1])[1]
            data[str(seed)] = dict(observations=observations.numpy().tolist(), oracle_value=float(value.numpy()),
                                   oracle_score=score.numpy().tolist(), first_observation_score=first.numpy().tolist())
        dump(out/'data.json', dict(theta=theta.numpy().tolist(), datasets=data))
        meta['data_version'] = dict(path='data.json', sha256=digest(out/'data.json'), generator=spec.target_id)
        designs = {}
        for seed in (*CAL, *VAL, *FILTER):
            tensors = random_inputs(route, seed, n, d, horizon, tf.float64)
            designs[str(seed)] = {key: dict(shape=t.shape.as_list(), dtype=t.dtype.name,
                sha256=hashlib.sha256(tf.io.serialize_tensor(t).numpy()).hexdigest())
                for key, t in zip(('initial', 'process_noise', 'ancestor_uniforms'), tensors)}
        dump(out/'random_designs.json', designs)
        meta['random_designs_sha256'] = digest(out/'random_designs.json')
        dump(out/'manifest.json', meta)

        def progress(stage, controls, row):
            record = dict(stage=stage, flow_substeps=controls['flow_substeps'], seed=row['seed'],
                          valid=row['valid'], wall_seconds=row['wall_seconds'], score_l2_error=row.get('score_l2_error'))
            with (out/'evaluations.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(stage=stage, controls=controls, row=row), allow_nan=False) + '\n')
            with (out/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record), flush=True)

        candidate_controls = CONTROLS
        if CONTROL_PREPARER is not None:
            candidate_controls = CONTROL_PREPARER(spec, route, theta, horizon, n, out)
            meta['transport_calibration_sha256'] = digest(out/'transport_calibration.json')
            dump(out/'manifest.json', meta)
        if candidate_controls:
            artifact, tuning = tune_campaign(spec, route, candidate_controls, theta, horizon, n, CAL, VAL, CLAIM,
                                             jit_compile=True, on_result=progress)
        else:
            artifact, tuning = None, dict(grid=[], decision='transport_calibration_failed')
        dump(out/'tuning.json', tuning)
        result['tuning_path'] = 'tuning.json'
        result['selected_controls'] = tuning.get('selected_controls')
        if artifact is None:
            result.update(status='tuning_failed', tuning_status='no_validated_controls', tuning_decision=tuning['decision'])
        else:
            result['tuning_status'] = ('exact_scope_frozen_controls; epsilon_validity_nomination_then_full_score_flow_grid'
                if CONTROL_PREPARER is not None else 'exact_scope_frozen_controls; flow_grid_only')
            for data_seed, filter_seed in final_pairs():
                row = evaluate_untouched(spec, route, artifact, theta, horizon, n, data_seed,
                                         jit_compile=True, filter_seed=filter_seed)
                row.pop('scope', None)  # Exact scope is checked by the evaluator and saved in tuning.json.
                reference = data[str(data_seed)]
                if row.get('oracle_score') is not None and l2_error(row['oracle_score'], reference['oracle_score']) > 1e-10:
                    raise RuntimeError('final evaluator disagrees with saved Kalman oracle')
                attach_heuristics(row, reference['oracle_score'], reference['first_observation_score'])
                row['evidence_role'] = ('untouched_repair_diagnostic_partition; no_admission' if CONTROL_PREPARER is not None
                    else 'reserved_diagnostic_partition; disclosed_pilot_reuse; no_admission')
                result['rows'].append(row)
                dump(out/'result.json', result)
                progress('final', result['selected_controls'], row)
            result['status'] = 'complete' if all(r['valid'] for r in result['rows']) else 'complete_with_invalid_cells'
        if source_hashes() != meta['source_sha256']:
            raise RuntimeError('source changed during worker execution')
        meta['source_unchanged'] = True
        meta['status'] = result['status']
    except BaseException as error:
        meta.update(status='infrastructure_or_harness_failure', error=repr(error))
        result.update(status=meta['status'], error=repr(error))
        traceback.print_exc()
        raise
    finally:
        meta.update(finished_utc=now(), wall_seconds=time.perf_counter()-started)
        if tf is not None:
            try:
                meta['gpu_allocator_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError, RuntimeError) as error:
                meta['gpu_allocator_unavailable'] = str(error)
        result['wall_seconds'] = meta['wall_seconds']
        dump(out/'result.json', result)
        meta['result_sha256'] = digest(out/'result.json')
        if (out/'tuning.json').exists():
            meta['tuning_sha256'] = digest(out/'tuning.json')
        dump(out/'manifest.json', meta)


def assemble(out, cases):
    lookup = {(case['scope'], row['data_seed'], row['filter_seed']): row
              for case in cases if case['route'] == 'iid_dual_cap' for row in case['rows']}
    fields = ('program', 'tuning_status', 'scope', 'route', 'data_seed', 'filter_seed', 'valid', 'coordinate', 'exact_score', 'estimated_score', 'signed_error', 'absolute_error')
    with (out/'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for row in case['rows']:
                iid = lookup.get((case['scope'], row['data_seed'], row['filter_seed']))
                if row['valid'] and iid and iid['valid']:
                    row['heuristic_dominance']['iid'] = ('baseline' if case['route']=='iid_dual_cap' else
                        'observed_loss' if row['score_l2_error'] > iid['score_l2_error'] else 'no_observed_loss')
                estimates = row.get('score') if row['valid'] else row.get('raw_score')
                oracle = row.get('oracle_score')
                if not estimates or not oracle or len(estimates) != case['parameter_count'] or len(oracle) != case['parameter_count']:
                    raise RuntimeError('missing score coordinates or oracle')
                if row['valid']:
                    errors = [a-b for a,b in zip(estimates, oracle)]
                    d = case['dimension']
                    blocks = ({'A': (0,d*d), 'Cholesky_Q': (d*d,len(errors)-2), 'R': (len(errors)-2,len(errors)-1), 'm0':(len(errors)-1,len(errors))}
                              if case['scope'].startswith('full_') else {name:(i,i+1) for i,name in enumerate(case['parameter_names'])})
                    row['block_score_rmse'] = {name: math.sqrt(statistics.fmean(e*e for e in errors[a:b])) for name,(a,b) in blocks.items()}
                for name, exact, estimate in zip(case['parameter_names'], oracle, estimates):
                    error = estimate-exact if row['valid'] and isinstance(estimate,(int,float)) else None
                    writer.writerow(dict(program=case['program'], tuning_status=case['tuning_status'],
                        scope=case['scope'], route=case['route'], data_seed=row['data_seed'],
                        filter_seed=row['filter_seed'], valid=row['valid'], coordinate=name,
                        exact_score=exact, estimated_score=estimate, signed_error=error,
                        absolute_error=abs(error) if error is not None else None))
            valid_rows = [r for r in case['rows'] if r['valid']]
            case['component_rmse'] = {name: math.sqrt(statistics.fmean(
                (r['score'][i]-r['oracle_score'][i])**2 for r in valid_rows)) if valid_rows else None
                for i,name in enumerate(case['parameter_names'])}
    dump(out/'results.json', dict(schema='bayesfilter.sqmc_expanded_results.v2', cases=cases,
         inference_status='descriptive_only; two data sets and two random designs do not establish ranking',
         expected_units=32, completed_units=len(cases), expected_final_cells=128,
         recorded_final_cells=sum(len(c['rows']) for c in cases)))


# Failed historical launches carried forward conservatively; successful pilot
# computations are never substituted for the revised Cartesian comparison.
HISTORICAL_FAILED_LAUNCHES = {
    'p44_d3_T10__iid_dual_cap': 1,
    'full_d10_T2__iid_dual_cap': 1,
    'full_d10_T2__previous_inverse_cdf': 1,
}


def reusable_units(paths, current_sources):
    completed, launches = {}, dict(HISTORICAL_FAILED_LAUNCHES)
    seen_launches = set()
    for previous in paths:
        previous = Path(previous).resolve()
        meta = json.loads((previous/'manifest.json').read_text())
        for entry in meta.get('units', []):
            if entry.get('reused_from'):
                continue
            identity = (str(previous), entry['unit'])
            if identity not in seen_launches:
                launches[entry['unit']] = launches.get(entry['unit'], 0) + 1
                seen_launches.add(identity)
            unit_path = previous/entry['unit']
            if not (unit_path/'result.json').exists() or not (unit_path/'manifest.json').exists():
                continue
            record = json.loads((unit_path/'manifest.json').read_text())
            if record.get('status') not in ('complete', 'complete_with_invalid_cells', 'tuning_failed'):
                continue
            if record.get('comparison_contract') != comparison_contract():
                raise ValueError('completed unit comparison specification changed')
            if numerical_sources(record['source_sha256']) != numerical_sources(current_sources):
                raise ValueError('completed unit numerical source changed')
            required_hashes = {'result.json': record['result_sha256'],
                               'tuning.json': record['tuning_sha256'],
                               'data.json': record['data_version']['sha256'],
                               'random_designs.json': record['random_designs_sha256']}
            if any(digest(unit_path/name) != expected for name,expected in required_hashes.items()):
                raise ValueError('completed unit artifact checksum mismatch')
            case = json.loads((unit_path/'result.json').read_text())
            if entry['unit'] != case['scope']+'__'+case['route']:
                raise ValueError('completed unit identity mismatch')
            if case['status'] != 'tuning_failed':
                if {(r['data_seed'],r['filter_seed']) for r in case['rows']} != set(final_pairs()):
                    raise ValueError('completed unit final pairs incomplete')
                if len(case['rows']) != len(final_pairs()):
                    raise ValueError('completed unit has duplicate final pairs')
                if any(len(r.get('score') or r.get('raw_score') or []) != case['parameter_count'] for r in case['rows']):
                    raise ValueError('completed unit score shape mismatch')
            case['artifact_directory'] = str(unit_path)
            completed[entry['unit']] = case
    return completed, launches


def supervisor(out, budget_seconds, elapsed_deadline, prior_gpu_seconds, reuse_from=()):
    if not 0 < budget_seconds <= 43200-prior_gpu_seconds:
        raise ValueError('requested budget exceeds remaining aggregate cap')
    deadline = datetime.fromisoformat(elapsed_deadline).timestamp()
    if deadline <= time.time():
        raise ValueError('renewed elapsed window expired')
    def stop_on_signal(signum, frame):
        raise InterruptedError('supervisor terminated; cleaning up current worker')
    signal.signal(signal.SIGTERM, stop_on_signal)
    current_sources = source_hashes()
    reused, launches = reusable_units(reuse_from, current_sources)
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    started = time.perf_counter()
    meta = manifest()
    meta.update(budget_seconds=budget_seconds, prior_gpu_seconds_charged=prior_gpu_seconds,
                elapsed_deadline_utc=elapsed_deadline, units=[], maximum_infrastructure_retries_per_unit=2)
    cases = []
    dump(out/'manifest.json', meta)
    try:
        for scope, *_ in SCOPES:
            for route in ROUTES:
                unit = f'{scope}__{route}'
                if unit in reused:
                    case = reused[unit]
                    cases.append(case)
                    meta['units'].append(dict(unit=unit, status=case['status'],
                        reused_from=case['artifact_directory'], wall_seconds=0.))
                    assemble(out, cases)
                    dump(out/'manifest.json', meta)
                    continue
                remaining = min(budget_seconds-(time.perf_counter()-started), deadline-time.time())
                if remaining < 60:
                    meta['status'] = 'budget_exhausted'
                    return
                if launches.get(unit, 0) >= 3:
                    meta.update(status='infrastructure_retry_limit', blocked_unit=unit)
                    return
                if source_hashes() != current_sources:
                    raise RuntimeError('source changed during campaign')
                command = [sys.executable, str(ENTRY_POINT), '--output', str(out/unit),
                           '--worker-scope', scope, '--worker-route', route]
                entry = dict(unit=unit, command=command, started_utc=now(), status='running',
                             launch_number=launches.get(unit, 0)+1)
                meta['units'].append(entry)
                dump(out/'manifest.json', meta)
                tick = time.perf_counter()
                with (out/'logs'/f'{unit}.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    entry['pid'] = process.pid
                    dump(out/'manifest.json', meta)
                    timed_out = False
                    try:
                        code = process.wait(timeout=min(5400, remaining-10))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        code = process.returncode
                    except BaseException:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                entry.update(return_code=code, wall_seconds=time.perf_counter()-tick, finished_utc=now())
                if code:
                    entry['status'] = 'timeout' if timed_out else 'worker_failure'
                    dump(out/'manifest.json', meta)
                    raise RuntimeError(f'{unit} exited {code}; preserve and repair before retry')
                case = json.loads((out/unit/'result.json').read_text())
                case['artifact_directory'] = str(out/unit)
                entry['status'] = case['status']
                cases.append(case)
                assemble(out, cases)
                dump(out/'manifest.json', meta)
                print(json.dumps(dict(unit=unit,status=case['status'],wall_seconds=entry['wall_seconds'])), flush=True)
        meta['status'] = 'complete' if all(c['status']=='complete' for c in cases) else 'complete_with_candidate_failures'
    except BaseException as error:
        meta.update(status='infrastructure_or_harness_failure', error=repr(error))
        raise
    finally:
        meta.update(finished_utc=now(), wall_seconds=time.perf_counter()-started)
        dump(out/'manifest.json', meta)
        assemble(out, cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker-scope', choices=[s[0] for s in SCOPES])
    parser.add_argument('--worker-route', choices=ROUTES)
    parser.add_argument('--budget-seconds', type=float)
    parser.add_argument('--prior-gpu-seconds', type=float)
    parser.add_argument('--elapsed-deadline')
    parser.add_argument('--reuse-from', action='append', default=[])
    args = parser.parse_args()
    if bool(args.worker_scope) != bool(args.worker_route):
        parser.error('worker requires both scope and route')
    if args.worker_scope:
        worker(Path(args.output).resolve(), args.worker_scope, args.worker_route)
    else:
        if args.budget_seconds is None or args.prior_gpu_seconds is None or not args.elapsed_deadline:
            parser.error('supervisor requires reconciled budget, prior charge and elapsed deadline')
        supervisor(Path(args.output).resolve(), args.budget_seconds, args.elapsed_deadline,
                   args.prior_gpu_seconds, args.reuse_from)


if __name__ == '__main__':
    main()
