"""Bounded, explicitly untuned KSC diagnostics; no algorithm implementation fork."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import functools
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
import traceback

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '023e106102c89ee9d4787df3a55fbf5f68b88ecf'
PLAN = 'docs/plans/sqmc-ksc-discrepancy-analysis-20260929.md'
RESULT = 'docs/benchmarks/sqmc-ksc-discrepancy-results-20260929.md'
CAMPAIGN = ROOT / 'docs/plans/artifacts/sqmc-ksc-discrepancy-20260929'
PRIOR = 'docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/budget.json'
PRIOR_HASH = 'eedf51ff268c027ed73ede76b5f25b74676de35d8d0a25ea58a51314744842cc'
OLD = 'docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-02'
ROUTES = ('iid_dual_cap', 'previous_inverse_cdf', 'repaired_permutation', 'repaired_permutation_ablation')
SEEDS = tuple(range(231001, 231009))
PROGRAM = 'UNTUNED FP64 GPU/XLA KSC discrepancy diagnostic; TF32 off; Contract E and dual-cap enabled'


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_bytes(path):
    return subprocess.check_output(['git', 'show', COMMIT + ':' + path], cwd=ROOT)


def source_hashes():
    paths = list((ROOT / 'bayesfilter').rglob('*.py'))
    paths += list((ROOT / 'experiments/dpf_implementation/tf_tfp').rglob('*.py'))
    paths += [Path(__file__), ROOT / PLAN, ROOT / 'docs/benchmarks/run_sqmc_expanded_comparison.py']
    paths += [ROOT / 'docs/benchmarks/ksc_reset_mechanism_diagnostic.py', ROOT / 'docs/plans/sqmc-ksc-reset-mechanism-addendum-20260929.md']
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(set(paths))}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    old_raw = git_bytes(PRIOR)
    if sha(old_raw) != PRIOR_HASH:
        raise RuntimeError('Prior ledger hash mismatch')
    old = json.loads(old_raw)
    if old['status'] != 'complete' or old['charged_seconds'] + old['remaining_gpu_seconds'] != 43200.:
        raise RuntimeError('Prior ledger is not reconciled')
    if time.time() >= datetime.fromisoformat(old['deadline_utc']).timestamp():
        raise RuntimeError('Elapsed window expired')
    ledger = dict(schema='sqmc_ksc_discrepancy_budget_v1', prior_ledger=PRIOR,
                  prior_commit=COMMIT, prior_sha256=PRIOR_HASH,
                  prior_charged_seconds=old['charged_seconds'], cap_seconds=43200.,
                  allocation_seconds=14400., diagnostic_seconds=0.,
                  charged_seconds=old['charged_seconds'], remaining_gpu_seconds=old['remaining_gpu_seconds'],
                  deadline_utc=old['deadline_utc'], attempts=[], status='prepared')
    if (CAMPAIGN / 'budget.json').exists():
        raise RuntimeError('Existing campaign ledger; reconcile instead of overwriting')
    inputs = out / 'inputs'
    inputs.mkdir()
    (inputs / 'prior-budget.json').write_bytes(old_raw)
    results, data, controls, hashes = {}, {}, {}, {PRIOR: PRIOR_HASH}
    for route in ROUTES:
        files = {}
        for name in ('result.json', 'data.json', 'tuning.json', 'manifest.json'):
            path = f'{OLD}/T120_{route}-attempt-01/{name}'
            raw = git_bytes(path)
            hashes[path] = sha(raw)
            (inputs / (route + '-' + name)).write_bytes(raw)
            files[name] = json.loads(raw)
        result = files['result.json']
        if result['status'] != 'complete' or result['tuning_status'] != 'exact_scope_frozen_controls':
            raise RuntimeError('Original scope is incomplete or lacks frozen controls')
        if sha((inputs / (route + '-data.json')).read_bytes()) != files['manifest.json']['data_sha256']:
            raise RuntimeError('Original data hash mismatch')
        results[route] = {r['data_seed']: r for r in result['rows']}
        if data and data != files['data.json']:
            raise RuntimeError('Original routes used different data/reference')
        data = files['data.json']
        controls[route] = result['selected_controls']
    errors = {seed: sum(results[r][seed]['score_l2_error'] for r in ROUTES) / len(ROUTES)
              for seed in results[ROUTES[0]]}
    worst = max((s for s in errors if s != 213001), key=lambda s: (errors[s], -s))
    cases = [dict(data_seed=s, selection='first_seed' if s == 213001 else 'largest_archived_four_route_mean_error',
                  archived_mean_l2_error=errors[s], observations=data[str(s)]['observations'],
                  old_reference=data[str(s)]['reference'], old_rows={r: results[r][s] for r in ROUTES})
             for s in (213001, worst)]
    dump(out / 'prepared.json', dict(source_commit=COMMIT, cases=cases, controls=controls,
         design_seeds=SEEDS, input_sha256=hashes, program=PROGRAM, tuning='UNTUNED diagnostic perturbations'))
    dump(CAMPAIGN / 'budget.json', ledger)
    print(json.dumps(dict(status='prepared', cases=[c['data_seed'] for c in cases], remaining_gpu_seconds=old['remaining_gpu_seconds'])))


def fd_classify(ad, rows):
    tolerance = 1e-4 * (1. + abs(ad))
    for a, b in zip(rows, rows[1:]):
        if not a['branch_changes'] and not b['branch_changes'] and abs(a['centered'] - b['centered']) <= tolerance:
            error = abs(b['centered'] - ad)
            return dict(status='pass' if error <= 2e-4 * (1. + abs(ad)) else 'stable_derivative_mismatch',
                        steps=[a['step'], b['step']], error=error)
    return dict(status='unresolved_branch_crossing_or_fd_instability')


def numerical_arms(base):
    yield 'baseline', dict(base)
    for steps in (16, 32, 64):
        yield f'flow_{steps}', dict(base, flow_substeps=steps)
    for s, b in ((96, 48), (384, 192)):
        yield f'transport_{s}_{b}', dict(base, reset_sinkhorn_steps=s, reset_balance_steps=b)
    for epsilon in (25.6, 409.6):
        yield f'epsilon_{epsilon}', dict(base, reset_epsilon=epsilon, reset_sinkhorn_steps=96, reset_balance_steps=48)
    yield 'combined_refined', dict(base, flow_substeps=64, reset_sinkhorn_steps=384, reset_balance_steps=192)


def configure():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                                         '--format=csv,noheader'], text=True)
    import run_sqmc_expanded_comparison as base
    tf, meta = base.configure_gpu()
    meta.update(pre_framework_nvidia_smi=inventory, cpu_gpu_status='GPU/XLA particle and mixture diagnostics; explicit CPU auxiliary references')
    return tf, meta


def trace_kernel(tf, spec, route, controls, n, horizon):
    from bayesfilter.highdim.sqmc_campaign_tf import numerical_settings, route_settings, reset_design
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    settings = dict(numerical_settings(controls), **route_settings(route))
    design = reset_design(n, 1, tf.float64)
    signature = [tf.TensorSpec([2], tf.float64), tf.TensorSpec([n, 1], tf.float64),
                 tf.TensorSpec([horizon, n, 1], tf.float64), tf.TensorSpec([horizon, n], tf.float64),
                 tf.TensorSpec([horizon, 1], tf.float64)]
    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def compute(theta, initial, noise, uniforms, obs):
        direction = tf.constant([1., 0.], tf.float64)
        model, _ = spec.model(theta, direction)
        states, covs, ds, dc = spec.initial_cloud(theta, initial, direction)
        value, score, trace = canonical_value_and_analytical_score(
            model, theta, states, covs, noise, obs, with_score=True, return_trace=True,
            initial_state_tangent=ds, initial_covariance_tangent=dc, reset_design=design,
            process_ancestor_uniforms=uniforms, **settings)
        return value, score, tf.stack([t['ancestor_indices'] for t in trace])
    return compute


def references(tf, prepared, result, save):
    from bayesfilter.highdim.sqmc_ksc_gaussian_sum_reference_tf import gaussian_sum_reference
    from bayesfilter.highdim.sqmc_ksc_tf import checked_reference, enumeration_reference, gaussian_approximation_reference
    theta = tf.constant([1.5, 0.], tf.float64)
    for case in prepared['cases']:
        for horizon in (10, 50, 120):
            obs = tf.constant(case['observations'][:horizon], tf.float64)
            levels = []
            for nodes, bound in ((801, 40.), (1201, 40.), (1201, 48.)):
                raw = gaussian_sum_reference(nodes, bound, True)(theta, obs)
                levels.append(dict(nodes=nodes, bound=bound, value=float(raw['value']), score=raw['score'].numpy().tolist()))
            grid = case['old_reference'] if horizon == 120 else checked_reference((1.5, 0.), tuple(x[0] for x in case['observations'][:horizon]))
            last = levels[-1]
            error = max(abs(last['value'] - grid['value']), *(abs(a-b) for a,b in zip(last['score'], grid['score'])),
                        *(abs(last['value']-r['value']) for r in levels[:-1]),
                        *(abs(a-b) for r in levels[:-1] for a,b in zip(last['score'],r['score'])))
            if not math.isfinite(error) or error > 1e-7:
                raise RuntimeError('reference_continuation_veto')
            with tf.device('/CPU:0'):
                first = enumeration_reference(1)(theta, obs[:1])
                gauss = gaussian_approximation_reference(theta, obs)
            row = dict(data_seed=case['data_seed'], horizon=horizon, **last, levels=levels,
                       independent_grid=grid, max_agreement_error=error,
                       heuristic_scores={'zero_score':[0.,0.], 'first_only':first[1].numpy().tolist(),
                                         'gaussian_kalman':gauss[1].numpy().tolist()})
            result['rows'].append(row); save()


def worker(out, unit, attempt_root):
    out.mkdir(parents=True, exist_ok=False)
    prepared = read(attempt_root / 'prepared.json')
    before = source_hashes()
    meta = dict(command=[sys.executable, *sys.argv], git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256=before, plan=PLAN, result=RESULT, program=PROGRAM, tuning='UNTUNED diagnostic',
                production_deviations='FP64 and TF32 off; frozen prior T120 controls, diagnostic scope only',
                started_utc=now(), data_sha256=sha((attempt_root/'prepared.json').read_bytes()),
                environment=sys.executable, cpu_gpu_status='pending', unit=unit, seeds=list(SEEDS))
    result = dict(status='running', unit=unit, program=PROGRAM, tuning='UNTUNED diagnostic', rows=[], fd_checks=[])
    start = time.monotonic()
    tf = None
    def save():
        dump(out/'result.json', result)
        print(json.dumps(dict(unit=unit, rows=len(result['rows']), fd_checks=len(result['fd_checks']),
                              last={k:v for k,v in (result['rows'][-1] if result['rows'] else {}).items()
                                    if k in ('data_seed','horizon','n','design_seed','arm','valid','score_l2_error')})),flush=True)
    dump(out/'manifest.json',meta); save()
    try:
        tf, gpu = configure();meta.update(gpu);dump(out/'manifest.json',meta)
        if unit == 'reference':
            references(tf, prepared, result, save)
        else:
            phase, route = unit.split('__',1)
            from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
            from bayesfilter.highdim.sqmc_campaign_tf import value_and_score, random_inputs
            from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
            spec=KSCSpec();theta=spec.default_theta();controls=prepared['controls'][route]
            ref_result=latest_result(attempt_root,'reference')
            refs={(r['data_seed'],r['horizon']):r for r in ref_result['rows']}
            @functools.lru_cache(maxsize=12)
            def traced(n,h):
                return trace_kernel(tf,spec,route,controls,n,h)
            def evaluate(case,n,seed,control,horizon=120,arm='baseline',trace=False):
                inputs=random_inputs(route,seed,n,1,horizon,tf.float64)
                obs=tf.constant(case['observations'][:horizon],tf.float64)
                started=time.monotonic(); diagnostics={}
                value,score,valid=value_and_score(spec,route,control,theta,obs,seed,n,jit_compile=True,inputs=inputs,diagnostics=diagnostics)
                value=float(value);score=score.numpy().tolist();valid=bool(valid)
                ref=refs[(case['data_seed'],horizon)]
                row=dict(data_seed=case['data_seed'],horizon=horizon,n=n,chunk_size=select_transport_chunk_size(n),
                         route=route,design_seed=seed,arm=arm,controls=control,valid=valid,
                         program=PROGRAM,tuning='UNTUNED diagnostic',claim_eligible=False,
                         value=value if math.isfinite(value) else None,
                         score=[x if math.isfinite(x) else None for x in score],
                         reference_value=ref['value'],reference_score=ref['score'],validity=diagnostics,
                         design_sha256={key:sha(tf.io.serialize_tensor(t).numpy()) for key,t in zip(('initial','noise','uniforms'),inputs)},
                         wall_seconds=time.monotonic()-started)
                if valid:
                    errors=[a-b for a,b in zip(score,ref['score'])]
                    row.update(value_error=value-ref['value'],score_error=errors,absolute_score_error=list(map(abs,errors)),
                               score_l2_error=math.sqrt(sum(e*e for e in errors)))
                    row['heuristic_errors']={key:math.sqrt(sum((a-b)**2 for a,b in zip(s,ref['score']))) for key,s in ref['heuristic_scores'].items()}
                    row['heuristic_dominance']={k:('observed_loss' if row['score_l2_error']>e else 'no_observed_loss') for k,e in row['heuristic_errors'].items()}
                else:
                    row['failure_class']='candidate_numerical_invalidity'
                result['rows'].append(row);save()
                return row,inputs,obs
            if phase=='replay':
                # First bound trace memory and verify it returns the same finite program.
                tiny_inputs=random_inputs(route,231999,16,1,2,tf.float64)
                tiny_obs=tf.constant([[-1.],[-.5]],tf.float64)
                normal=value_and_score(spec,route,controls,theta,tiny_obs,231999,16,inputs=tiny_inputs)
                tr=traced(16,2)(theta,*tiny_inputs,tiny_obs)
                err=max(abs(float(normal[0]-tr[0])),abs(float(normal[1][0]-tr[1][0])))
                if not bool(normal[2]) or not math.isfinite(err) or err>1e-8:
                    raise RuntimeError('tiny_trace_parity_veto')
                result['tiny_trace_parity_error']=err
                for case in prepared['cases']:
                    old=case['old_rows'][route];row,inputs,obs=evaluate(case,1008,old['filter_seed'],controls)
                    differences=[abs(row['value']-old['value']),*(abs(a-b) for a,b in zip(row['score'],old['score']))] if row['valid'] else [math.inf]
                    tolerances=[1e-8*(1+abs(x)) for x in [old['value'],*old['score']]]
                    if any(a>b for a,b in zip(differences,tolerances)):
                        raise RuntimeError('original_replay_veto')
                    tr=traced(1008,120)(theta,*inputs,obs)
                    trace_error=max(abs(float(tr[0])-row['value']),abs(float(tr[1][0])-row['score'][0]))
                    if not math.isfinite(trace_error) or trace_error>1e-8:
                        raise RuntimeError('full_trace_parity_veto')
                    row.update(replay_max_error=max(differences),trace_max_error=trace_error,
                               ancestry_sha256=sha(tf.io.serialize_tensor(tr[2]).numpy()))
                    save()
            elif phase=='derivative':
                for case in prepared['cases']:
                    for horizon in (10,50,120):
                        row,inputs,obs=evaluate(case,1008,case['old_rows'][route]['filter_seed'],controls,horizon)
                        if not row['valid']: continue
                        fn=traced(1008,horizon);tv,ts,anc=fn(theta,*inputs,obs)
                        if max(abs(float(tv)-row['value']),abs(float(ts[0])-row['score'][0]))>1e-8:
                            raise RuntimeError('derivative_trace_parity_veto')
                        for coordinate in range(2):
                            ladder=[]
                            for step in (1e-4,1e-5,1e-6,1e-7,1e-8):
                                delta=tf.one_hot(coordinate,2,dtype=tf.float64)*step
                                vp,sp,ap=fn(theta+delta,*inputs,obs)
                                vm,sm,am=fn(theta-delta,*inputs,obs)
                                plus=float(vp);minus=float(vm)
                                if not all(math.isfinite(x) for x in (plus,minus)):
                                    ladder.append(dict(step=step,branch_changes=True,centered=None,status='invalid_perturbation'))
                                    break
                                pc=tf.reduce_sum(tf.cast(ap!=anc,tf.int32),axis=1).numpy().tolist()
                                mc=tf.reduce_sum(tf.cast(am!=anc,tf.int32),axis=1).numpy().tolist()
                                first=next((i for i,(a,b) in enumerate(zip(pc,mc)) if a or b),None)
                                ladder.append(dict(step=step,plus_value=plus,minus_value=minus,
                                                   centered=(plus-minus)/(2*step),forward=(plus-float(tv))/step,
                                                   backward=(float(tv)-minus)/step,branch_changes=bool(sum(pc)+sum(mc)),
                                                   plus_changes_by_time=pc,minus_changes_by_time=mc,first_changed_time=first))
                                if step<=1e-6 and fd_classify(row['score'][coordinate],ladder)['status']!='unresolved_branch_crossing_or_fd_instability':
                                    break
                            classification=fd_classify(row['score'][coordinate],[r for r in ladder if r['centered'] is not None])
                            result['fd_checks'].append(dict(data_seed=case['data_seed'],route=route,horizon=horizon,n=1008,
                                      coordinate=coordinate,analytical_score=row['score'][coordinate],ladder=ladder,**classification))
                            save()
            elif phase=='mechanism':
                from ksc_reset_mechanism_diagnostic import make_kernel
                kernel=make_kernel(spec,route,controls,1008,120)
                for case in prepared['cases']:
                    seed=case['old_rows'][route]['filter_seed']
                    row,inputs,obs=evaluate(case,1008,seed,controls,arm='reset_mechanism_baseline')
                    if not row['valid']:
                        raise RuntimeError('mechanism_baseline_validity_veto')
                    traces=[];parity=[]
                    for coordinate in range(2):
                        tr=kernel(theta,tf.one_hot(coordinate,2,dtype=tf.float64),*inputs,obs)
                        if not bool(tr['valid']):
                            raise RuntimeError('mechanism_trace_validity_veto')
                        record={k:t.numpy().tolist() for k,t in tr.items()}
                        parity.extend([abs(record['value']-row['value']),abs(record['score']-row['score'][coordinate])])
                        if record['outgoing_nonuniformity']>1e-12:
                            raise RuntimeError('mechanism_uniform_weights_veto')
                        traces.append(record)
                    if not all(math.isfinite(x) for x in parity) or max(parity)>1e-8:
                        raise RuntimeError('mechanism_trace_parity_veto')
                    row['mechanism']=dict(direction_traces=traces,parity_max_error=max(parity),
                        diagnostic='Exact next-observation mixture integration before/after actual reset and correction; local effects only')
                    save()
            elif phase=='particles':
                for n in (1008,2016,4032):
                    for case in prepared['cases']:
                        for seed in SEEDS:
                            evaluate(case,n,seed,controls)
            elif phase=='numerics':
                for arm,control in numerical_arms(controls):
                    for case in prepared['cases']:
                        for seed in SEEDS[:2]:
                            evaluate(case,1008,seed,control,arm=arm)
            else:
                raise ValueError('Unknown phase')
        if before!=source_hashes():
            raise RuntimeError('source_drift_veto')
        result['status']='complete'
        meta['source_unchanged']=True
        return 0
    except BaseException as error:
        result.update(status='failed',error=repr(error),failure_class='scientific_or_harness_veto' if 'veto' in str(error) else 'infrastructure_or_harness_failure')
        traceback.print_exc();return 1
    finally:
        result['finished_utc']=now();save()
        meta.update(status=result['status'],wall_seconds=time.monotonic()-start,finished_utc=now())
        if tf is not None:
            try:meta['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError,RuntimeError) as error:meta['allocator_error']=str(error)
        dump(out/'manifest.json',meta)


def latest_result(out,unit):
    candidates=sorted(out.glob(unit+'-attempt-*/result.json'))
    if not candidates:raise RuntimeError('Missing prerequisite '+unit)
    result=read(candidates[-1])
    if result['status']!='complete':raise RuntimeError('Incomplete prerequisite '+unit)
    return result


def supervise(out,phase):
    ledger=read(CAMPAIGN/'budget.json')
    if any(a['status']=='running' for a in ledger['attempts']):
        raise RuntimeError('Reconcile running attempt first')
    if phase!='replay':
        for route in ROUTES:latest_result(out,'replay__'+route)
    units=(['reference'] if phase=='replay' else [])+[phase+'__'+r for r in ROUTES]
    before=source_hashes()
    snapshot=out/('source-'+phase+'-'+str(len(ledger['attempts'])))
    snapshot.mkdir()
    for path in before:
        dst=snapshot/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dst)
    dump(snapshot/'sha256.json',before)
    for unit in units:
        attempts=[a for a in ledger['attempts'] if a['unit']==unit]
        if attempts and attempts[-1].get('returncode')==0:continue
        if sum(a.get('returncode') not in (None,0) for a in attempts)>2:
            raise RuntimeError('Infrastructure retry allowance exhausted')
        unit_limit = {'replay':600., 'derivative':1800., 'particles':2400., 'numerics':1800., 'mechanism':600.}[phase]
        available=min(unit_limit,ledger['remaining_gpu_seconds'],ledger['allocation_seconds']-ledger['diagnostic_seconds'],
                      datetime.fromisoformat(ledger['deadline_utc']).timestamp()-time.time())-30.
        if available<=10:raise RuntimeError('Budget exhausted')
        target=out/f'{unit}-attempt-{len(attempts)+1:02d}'
        command=[sys.executable,str(Path(__file__).resolve()),'--mode','worker','--output',str(target),'--attempt-root',str(out),'--unit',unit]
        entry=dict(unit=unit,status='running',output=str(target),command=command,source_snapshot=str(snapshot),
                   started_utc=now(),timeout_seconds=available)
        ledger['attempts'].append(entry);ledger['status']='running';dump(CAMPAIGN/'budget.json',ledger)
        start=time.monotonic()
        with target.with_suffix('.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=proc.pid;dump(CAMPAIGN/'budget.json',ledger)
            try:rc=proc.wait(timeout=available)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                rc=124
        wall=time.monotonic()-start
        entry.update(status='finished',returncode=rc,wall_seconds=wall,finished_utc=now())
        ledger['diagnostic_seconds']+=wall
        ledger['charged_seconds']=ledger['prior_charged_seconds']+ledger['diagnostic_seconds']
        ledger['remaining_gpu_seconds']=43200.-ledger['charged_seconds']
        ledger['status']='failed_unit' if rc else 'between_units'
        dump(CAMPAIGN/'budget.json',ledger);print(json.dumps(entry),flush=True)
        if rc:return rc
    ledger['status']='phase_complete';dump(CAMPAIGN/'budget.json',ledger)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['prepare','supervise','worker'],required=True)
    parser.add_argument('--output',type=Path,default=CAMPAIGN/'attempt-01')
    parser.add_argument('--phase',choices=['replay','derivative','particles','numerics','mechanism'])
    parser.add_argument('--unit');parser.add_argument('--attempt-root',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output);return 0
    if args.mode=='supervise':return supervise(args.output,args.phase)
    return worker(args.output,args.unit,args.attempt_root)


if __name__=='__main__':
    raise SystemExit(main())
