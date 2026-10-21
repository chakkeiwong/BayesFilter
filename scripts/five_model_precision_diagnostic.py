#!/usr/bin/env python3
"""Five-model arithmetic diagnostic, not a tuner or HMC admission authority.

The shared TensorFlow/XLA filter computes every value and analytical score.
Only orchestration, serialization and descriptive analysis occur on the host.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_range_bearing_covariance_comparison as base

PLAN = ROOT / 'docs/plans/ledh-five-model-tf32-2026-10-10.md'
RESULT = ROOT / 'docs/plans/ledh-five-model-tf32-results-2026-10-10.md'
ARTIFACTS = ROOT / 'docs/plans/artifacts'
FOUR = ARTIFACTS / 'ledh-independent-validation-20261008-01'
RANGE = {
    1008: ARTIFACTS / 'ledh-range-bearing-covariance-20261009-07',
    2016: ARTIFACTS / 'ledh-range-bearing-particle-ladder-20261010-02/N2016',
    4032: ARTIFACTS / 'ledh-range-bearing-particle-ladder-20261010-02/N4032',
}
MODES = ('fp64', 'fp32', 'fp32_tf32')
SCOPES = (('range_bearing', 1008), ('lgssm', 1008), ('ksc', 1008),
          ('predator_prey', 1008), ('sir_d18', 1008),
          ('range_bearing', 2016), ('range_bearing', 4032))
BUDGET = 7200


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(base.clean(row), allow_nan=False) + '\n')


def label(model, n, mode):
    return f'{model}-N{n}-{mode}'


def source_paths():
    paths = [Path(__file__), Path(base.__file__), PLAN,
             ROOT / 'docs/benchmarks/run_ledh_covariance_proposal.py']
    for directory in ('bayesfilter/highdim', 'bayesfilter/testing',
                      'bayesfilter/runtime', 'experiments/dpf_implementation/tf_tfp/fixtures'):
        paths.extend((ROOT / directory).glob('*.py'))
    paths.extend((FOUR / 'frozen').glob('*-tuning.json'))
    paths.extend((FOUR / 'inputs').glob('*/dataset.json'))
    for source in RANGE.values():
        paths.append(source / 'tuning.json')
        paths.extend(source / 'data' / f'{seed}-t20.json' for seed in base.HELDOUT)
    return sorted(set(paths))


def provenance():
    return {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}


def cases(model):
    seeds = base.HELDOUT if model == 'range_bearing' else (26100831, 26100832)
    return [(ds, point, rep) for ds in range(len(seeds))
            for point in range(2 if model == 'range_bearing' else 1) for rep in range(8)]


def evaluate(tf, program, theta, obs, fixed, beta):
    from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
    start = time.monotonic()
    directions = tf.eye(theta.shape[0], dtype=theta.dtype)
    calls = [program(theta, directions[j], obs, *fixed, beta) for j in range(theta.shape[0])]
    if any("GPU:" not in r["value"].device for r in calls):
        raise RuntimeError("Filtering unexpectedly executed outside the GPU")
    values = [float(r['value']) for r in calls]
    scores = [float(r['score']) for r in calls]
    steps = [int(r['steps_completed']) for r in calls]
    finite = all(math.isfinite(x) for x in values + scores)
    primal_difference = max(values)-min(values) if finite else float('nan')
    # All directional passes compute the same primal. This check is a wiring
    # invariant, not an error tolerance for comparisons between precisions.
    tolerance = 1.e-4 if theta.dtype == tf.float32 else 1.e-9
    valid = finite and primal_difference <= tolerance
    valid = valid and all(bool(r['valid']) and k == obs.shape[0] for r, k in zip(calls, steps))
    elapsed = time.monotonic()-start  # Tensor materialization synchronizes GPU.
    return dict(value=values[0], directional_values=values, score=scores, execution_device=calls[0]["value"].device,
                valid=valid, steps_completed=steps, primal_direction_difference=primal_difference,
                wall_seconds=elapsed, trace_fields=TRACE_FIELDS,
                directional_traces=[r['trace'] for r in calls])


def worker(args):
    args.dtype, args.no_tf32 = dict(fp64=('float64', True), fp32=('float32', True),
                                  fp32_tf32=('float32', False))[args.mode]
    out = args.output / label(args.model, args.particles, args.mode)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = dict(status='starting', started_utc=utc(), command=[sys.executable, *sys.argv],
                    model=args.model, particles=args.particles, mode=args.mode,
                    dtype=args.dtype, jit_compile=True, tf32=args.mode == 'fp32_tf32',
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256=provenance(), python=sys.executable,
                    plan=str(PLAN), result=str(RESULT), output=str(out),
                    evidence_role='arithmetic_diagnostic_only_no_tuning_admission',
                    cpu_only=False, device='GPU', trust_basis='escalated_local_GPU_execution')
    base.save(out / 'manifest.json', manifest)
    program = tf = None
    try:
        tf, memory = base.prepare(args)
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
        from docs.benchmarks.run_ledh_covariance_proposal import spec_for
        dtype = tf.as_dtype(args.dtype)
        if args.model == 'range_bearing':
            source = RANGE[args.particles]
            tuning_path = source / 'tuning.json'
            tuning = base.load(tuning_path)
            assert tuning['scope']['particles'] == args.particles
            controls = tuning['selected']['new']['controls']
            weights = tuning['selected']['new']['beta']
            _, spec = base.specs(tf, dtype)
            with tf.device('/CPU:0'):
                spec64, _ = base.specs(tf, tf.float64)
            horizon = 20
            data_seeds = base.HELDOUT
            paths = [source / 'data' / f'{seed}-t20.json' for seed in data_seeds]
            target_id = 'm13-t20'
        else:
            tuning_path = FOUR / 'frozen' / f'{args.model}-tuning.json'
            tuning = base.load(tuning_path)
            controls, weights = tuning['scope']['controls'], tuning['beta']
            spec = spec_for(args.model, lgssm_family='p44')
            target_id = spec.target_id
            horizon = 50
            assert tuning['scope']['target_id'] == target_id
            assert tuning['scope']['n'] == args.particles and tuning['scope']['horizon'] == horizon
            data_seeds = (26100831, 26100832)
            paths = [FOUR / 'inputs' / f'{args.model}-d{seed}' / 'dataset.json' for seed in data_seeds]
        saved = [base.load(p) for p in paths]

        def common(value):
            with tf.device('/CPU:0'):
                return tf.cast(tf.cast(tf.convert_to_tensor(value, dtype=tf.float64), tf.float32), dtype)

        observations = [common(d['observations']) for d in saved]
        if args.model != 'range_bearing':
            for data in saved:
                assert data['target_id'] == target_id
                with tf.device('/CPU:0'):
                    tensor = tf.constant(data['observations'], tf.float64)
                    digest = hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()
                assert digest == data['observation_sha256'], 'Saved observation identity mismatch'
        beta = common(weights)
        with tf.device('/GPU:0'):
            program = make_filter(spec, args.particles, horizon, ProposalControls(**controls), dtype,
                                  jit_compile=True)

        def tensors(ds, point, rep):
            seed = (31001 if args.model == 'range_bearing' else 261008301) + 100*ds + rep
            with tf.device('/CPU:0'):
                if args.model == 'range_bearing':
                    fixed = base.inputs(tf, spec64, args.particles, seed, dtype)[:3]
                    theta = common(base.POINTS[point])
                else:
                    from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
                    design = random_inputs('iid_dual_cap', seed, args.particles, spec.dimension,
                                           horizon, tf.float64, jit_compile=True)
                    extra = tf.random.stateless_uniform([horizon, args.particles, 2], [seed, 991], dtype=tf.float64)
                    fixed = tuple(common(x) for x in (*design[:2], extra))
                    theta = common(spec.default_theta(tf.float64))
                obs = observations[ds]
                hashes = [hashlib.sha256(tf.io.serialize_tensor(tf.cast(x, tf.float32)).numpy()).hexdigest()
                          for x in (theta, obs, *fixed, beta)]
            return theta, obs, fixed, dict(data_seed=data_seeds[ds], point=point, seed=seed,
                                           input_sha256=hashes, theta=theta)

        manifest.update(status='running', horizon=horizon, target_id=target_id,
                        state_dimension=spec.dimension,
                        observation_dimension=getattr(spec, 'observation_dimension', spec.dimension),
                        parameter_count=spec.parameter_count, controls=controls, beta=beta,
                        tuning_path=str(tuning_path), tuning_sha256=sha(tuning_path),
                        tuning_role='saved_FP64_scope_controls_frozen_precision_comparator',
                        covariance_tolerance=.01 if dtype == tf.float32 else 100*2.220446049250313e-16*spec.dimension,
                        chunk=select_transport_chunks(args.particles).row_chunk_size,
                        data_sha256={str(p): sha(p) for p in paths}, tensorflow=tf.__version__,
                        memory_policy=memory, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                        tf32_effective=tf.config.experimental.tensor_float_32_execution_enabled(),
                        physical_devices=[tf.config.experimental.get_device_details(g) for g in tf.config.list_physical_devices('GPU')],
                        input_protocol='CPU_FP64_generation_then_common_FP32_rounding',
                        data_seeds=data_seeds,
                        design_seeds=sorted({(31001 if args.model == 'range_bearing' else 261008301)+100*d+r
                                             for d, _, r in cases(args.model)}))
        base.save(out / 'manifest.json', manifest)
        # Compile plus one warm call. Invalid cases are preserved and all planned
        # cases still run: one failed parameter point cannot hide another case.
        with tf.device('/GPU:0'):
            for index in range(2):
                theta, obs, fixed, metadata = tensors(0, 0, 0)
                row = evaluate(tf, program, theta, obs, fixed, beta)
                row.update(metadata, phase='cold_compile' if index == 0 else 'warmup')
                append(out / 'preflight.jsonl', row)
            for ds, point, rep in cases(args.model):
                theta, obs, fixed, metadata = tensors(ds, point, rep)
                row = evaluate(tf, program, theta, obs, fixed, beta)
                row.update(metadata, phase='measurement')
                append(out / 'measurements.jsonl', row)
                print(json.dumps(dict(data_seed=metadata['data_seed'], point=point, seed=metadata['seed'],
                                      valid=row['valid'], seconds=row['wall_seconds'])), flush=True)
        manifest['status'] = 'completed'
    except Exception:
        manifest.update(status='infrastructure_or_harness_failure', traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-start, finished_utc=utc())
        if program is not None:
            manifest['graph_trace_count'] = program.experimental_get_tracing_count()
        if tf is not None:
            manifest['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
        base.save(out / 'manifest.json', manifest)


def rms(values):
    return math.sqrt(statistics.mean(x*x for x in values))


def ratio(a, b):
    return a/b if b else None


def report(root):
    records, timing, invalid, cells = {}, [], [], []
    for model, n in SCOPES:
        for mode in MODES:
            path = root / label(model, n, mode)
            rows = read_rows(path / 'measurements.jsonl')
            records[model, n, mode] = rows
            manifest = base.load(path / 'manifest.json') if (path / 'manifest.json').exists() else {}
            good = [r for r in rows if r['valid']]
            preflight = read_rows(path / 'preflight.jsonl')
            cov_index = rows[0]['trace_fields'].index('reset_covariance_relative_error') if rows else None
            covariance = [trace[t][cov_index] for r in rows
                          for trace, steps in zip(r['directional_traces'], r['steps_completed'])
                          for t in range(steps) if isinstance(trace[t][cov_index], (int, float))]
            timing.append(dict(model=model, particles=n, mode=mode, rows=len(rows), valid=len(good),
                               expected=len(cases(model)), status=manifest.get('status', 'not_started'),
                               complete=manifest.get('status') == 'completed' and len(good) == len(cases(model)),
                               median_full_seconds=statistics.median(r['wall_seconds'] for r in good) if good else None,
                               cold_seconds=preflight[0]['wall_seconds'] if preflight else None,
                               wall_seconds=manifest.get('wall_seconds'),
                               max_covariance_relative_error=max(covariance) if covariance else None,
                               covariance_tolerance=manifest.get('covariance_tolerance'),
                               graph_trace_count=manifest.get('graph_trace_count')))
            invalid.extend(dict(model=model, particles=n, mode=mode,
                                **{k:r[k] for k in ('data_seed', 'point', 'seed', 'steps_completed')})
                           for r in rows if not r['valid'])
        for ds, point in sorted({(d, p) for d, p, _ in cases(model)}):
            data_seed = (base.HELDOUT if model == 'range_bearing' else (26100831, 26100832))[ds]
            ref = {r['seed']:r for r in records[model,n,'fp64'] if r['data_seed']==data_seed and r['point']==point}
            for mode in MODES[1:]:
                arm = {r['seed']:r for r in records[model,n,mode] if r['data_seed']==data_seed and r['point']==point}
                paired = set(ref) == set(arm) and len(ref) == 8
                if paired and any(ref[k]['input_sha256'] != arm[k]['input_sha256'] for k in ref):
                    raise RuntimeError('Unmatched precision inputs')
                complete = paired and all(r['valid'] for r in (*ref.values(), *arm.values()))
                cell = dict(model=model, particles=n, mode=mode, data_seed=data_seed, point=point,
                            complete=complete, materiality_screen_pass=False)
                if complete:
                    vd = [arm[k]['value']-ref[k]['value'] for k in ref]
                    sd = [math.dist(arm[k]['score'],ref[k]['score']) for k in ref]
                    p = len(next(iter(ref.values()))['score'])
                    components = [[arm[k]['score'][j]-ref[k]['score'][j] for k in ref] for j in range(p)]
                    value_spread = statistics.stdev(r['value'] for r in ref.values())
                    score_spread = math.sqrt(sum(statistics.variance(r['score'][j] for r in ref.values()) for j in range(p)))
                    value_scale = rms([r['value'] for r in ref.values()])
                    score_scale = rms([math.sqrt(sum(s*s for s in r['score'])) for r in ref.values()])
                    vr, sr = rms(vd), rms(sd)
                    vlimit, slimit = max(.01,.1*value_spread), max(.01,.1*score_spread)
                    cell.update(value_difference_rms=vr, value_difference_max=max(map(abs,vd)),
                                value_signed_difference_mean=statistics.mean(vd),
                                score_difference_rms=sr, score_difference_max=max(sd),
                                score_coordinate_difference_rms=[rms(c) for c in components],
                                score_coordinate_difference_max=[max(map(abs,c)) for c in components],
                                fp64_value_mean=statistics.mean(r['value'] for r in ref.values()),
                                fp64_score_mean=[statistics.mean(r['score'][j] for r in ref.values()) for j in range(p)],
                                candidate_value_mean=statistics.mean(r['value'] for r in arm.values()),
                                candidate_score_mean=[statistics.mean(r['score'][j] for r in arm.values()) for j in range(p)],
                                value_relative_rms=ratio(vr,value_scale), score_relative_rms=ratio(sr,score_scale),
                                fp64_value_sd=value_spread, fp64_score_spread=score_spread,
                                value_sd_ratio=ratio(vr,value_spread), score_spread_ratio=ratio(sr,score_spread),
                                value_screen_limit=vlimit, score_screen_limit=slimit,
                                materiality_screen_pass=vr<=vlimit and sr<=slimit)
                cells.append(cell)
    prior_path = ARTIFACTS / 'ledh-range-bearing-precision-20261010-01/report.json'
    if prior_path.exists():
        prior = {(r['particles'],r['data_seed'],r['point']):r for r in base.load(prior_path)['cells']}
        for cell in cells:
            key = cell['particles'],cell['data_seed'],cell['point']
            if cell['model']=='range_bearing' and cell['complete'] and key in prior:
                old = prior[key]
                for quantity in ('value','score'):
                    gap = old.get(f'fp64_n1008_n2016_{quantity}_change_rms')
                    cell[f'prior_n1008_n2016_{quantity}_change_rms'] = gap
                    cell[f'{quantity}_change_ratio'] = ratio(cell[f'{quantity}_difference_rms'],gap)
    decisions = []
    for model, n in SCOPES:
        for mode in MODES[1:]:
            subset = [c for c in cells if (c['model'],c['particles'],c['mode'])==(model,n,mode)]
            complete = all(c['complete'] for c in subset)
            passed = complete and all(c['materiality_screen_pass'] for c in subset)
            decisions.append(dict(model=model, particles=n, mode=mode,
                                  decision='passed_precision_screen' if passed else 'precision_not_accepted',
                                  primary_criterion_status=passed, veto_diagnostic_status='clear' if complete else 'invalid_or_missing_final_outputs',
                                  main_uncertainty='Two datasets or six range cases; frozen FP64 controls; eight designs per case',
                                  next_justified_action='Separate precision-specific tuning/HMC only for a downstream claim' if passed else 'Localize final numerical failure before using this precision',
                                  not_concluded='No algorithm ranking, posterior convergence or default readiness'))
    result = dict(cells=cells, timing=timing, invalid=invalid, decisions=decisions,
                  inference_status=dict(hard_veto_screen='Incomplete/nonfinite runs veto that precision; see invalid and timing',
                                        statistically_supported_ranking='None',
                                        descriptive_only_differences='All precision errors, design spread estimates and timings',
                                        default_readiness='Not assessed; saved FP64 controls are diagnostic comparators',
                                        next_evidence_needed='Scope-specific tuning and downstream posterior tests for HMC claims'))
    base.save(root / 'report.json', result)
    return result


def status(root):
    result = []
    for model,n in SCOPES:
        for mode in MODES:
            path = root / label(model,n,mode)
            if (path/'manifest.json').exists():
                rows = read_rows(path/'measurements.jsonl')
                result.append(dict(model=model, n=n, mode=mode, rows=len(rows), valid=sum(r['valid'] for r in rows),
                                   status=base.load(path/'manifest.json')['status']))
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    execution = dict(status='running', command=[sys.executable,*sys.argv], started_utc=utc(),
                     plan=str(PLAN), result=str(RESULT), budget_seconds=BUDGET,
                     source_sha256=provenance(), attempts=[])
    base.save(args.output/'execution.json',execution)
    for index,(model,n) in enumerate(SCOPES):
        for mode in MODES[index%3:]+MODES[:index%3]:
            remaining = BUDGET-(time.monotonic()-start)
            if remaining<=0:
                execution['status']='budget_exhausted'
                break
            if provenance()!=execution['source_sha256']:
                execution['status']='source_change_veto'
                break
            command = [sys.executable,'-B',str(Path(__file__).resolve()),'worker','--output',str(args.output),
                       '--gpu',args.gpu,'--model',model,'--particles',str(n),'--mode',mode]
            attempt = dict(model=model,particles=n,mode=mode,command=command,started_utc=utc())
            job_start=time.monotonic()
            execution['active']=attempt
            base.save(args.output/'execution.json',execution)
            with (args.output/(label(model,n,mode)+'.log')).open('w') as stream:
                try:
                    proc=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
                    attempt['exit_code']=proc.returncode
                except subprocess.TimeoutExpired:
                    attempt.update(exit_code=None,failure='wall_budget_timeout')
            attempt['wall_seconds']=time.monotonic()-job_start
            execution['attempts'].append(attempt)
            execution.pop('active',None)
            execution['wall_seconds']=time.monotonic()-start
            if attempt['exit_code']!=0:
                execution['status']='worker_failure_repair_required'
            base.save(args.output/'execution.json',execution)
            print(json.dumps(attempt),flush=True)
            if execution['status']!='running':
                break
        if execution['status']!='running':
            break
    if execution['status']=='running':
        execution['status']='completed'
    execution.update(wall_seconds=time.monotonic()-start,finished_utc=utc())
    base.save(args.output/'execution.json',execution)
    report(args.output)
    return 0 if execution['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','worker','status','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='GPU-68251639-fe82-8f81-3ccc-2953c32e805b')
    parser.add_argument('--model',choices=sorted({m for m,_ in SCOPES}))
    parser.add_argument('--particles',type=int,choices=(1008,2016,4032),default=1008)
    parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if args.command=='run':
        return run(args)
    if args.command=='worker':
        if (args.model,args.particles) not in SCOPES or args.mode is None:
            parser.error('worker requires a declared model/N scope and precision mode')
        worker(args)
    elif args.command=='status':
        print(json.dumps(status(args.output)))
    else:
        result=report(args.output)
        print(json.dumps(dict(cells=len(result['cells']),invalid=len(result['invalid']),
                              decisions=result['decisions'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
